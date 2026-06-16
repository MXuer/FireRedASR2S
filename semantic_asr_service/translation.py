import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from semantic_asr_service.artifacts import artifact_path
from semantic_asr_service.settings import ServiceSettings


LANGUAGE_NAMES = {
    "ar_sa": "Arabic",
    "de_de": "German",
    "en_us": "English",
    "hi_in": "Hindi",
    "ja_jp": "Japanese",
    "ko_kr": "Korean",
    "pt_br": "Portuguese",
    "ru_ru": "Russian",
    "th_th": "Thai",
    "vi_vn": "Vietnamese",
    "yue_hk": "Cantonese",
    "zh_cn": "Chinese",
    "zh_hant": "Traditional Chinese",
}


class Translator(Protocol):
    def translate(self, text: str, source_language: str, target_language: str) -> str:
        ...

    def complete(self, prompt: str) -> str:
        ...


@dataclass
class HunyuanMTClient:
    base_url: str
    model: str
    api_key: str = ""
    temperature: float = 0.2
    top_p: float = 0.6
    max_tokens: int = 512

    @classmethod
    def from_settings(cls, settings: ServiceSettings) -> "HunyuanMTClient":
        if not settings.translation_base_url:
            raise RuntimeError("SEMANTIC_ASR_TRANSLATION_BASE_URL is not configured")
        return cls(
            base_url=settings.translation_base_url,
            model=settings.translation_model,
            api_key=settings.translation_api_key,
        )

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        prompt = build_translation_prompt(text, source_language, target_language)
        return self.complete(prompt)

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        return _extract_chat_content(data)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def translate_job_result(
    job: dict,
    settings: ServiceSettings,
    target_language: str,
    translator: Translator | None = None,
) -> dict:
    if target_language not in settings.translation_targets:
        raise ValueError(f"Translation target is not allowed: {target_language}")

    result_path = artifact_path(job["outdir"], job["job_id"], "json")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"ASR JSON is not available: {result_path}")

    with open(result_path, encoding="utf-8") as fin:
        result = json.load(fin)

    cache_path = translation_cache_path(job["outdir"], target_language)
    source_hash = _source_signature(result_path, result)
    cached = _read_valid_cache(cache_path, source_hash)
    if cached:
        return cached

    source_language = str(result.get("language") or job.get("config") or "").lower()
    source_name = language_prompt_name(source_language)
    target_name = language_prompt_name(target_language)
    translator = translator or HunyuanMTClient.from_settings(settings)
    source_sentences = []
    for index, sentence in enumerate(result.get("sentences") or []):
        source_sentences.append({
            "index": index,
            "start_ms": sentence.get("start_ms"),
            "end_ms": sentence.get("end_ms"),
            "cut_start_ms": sentence.get("cut_start_ms", sentence.get("start_ms")),
            "cut_end_ms": sentence.get("cut_end_ms", sentence.get("end_ms")),
            "text": str(sentence.get("text") or "").strip(),
        })
    translated_sentences = translate_sentences_batched(
        source_sentences,
        translator,
        source_name,
        target_name,
        max(1, int(settings.translation_batch_size)),
        settings.translation_request_mode,
    )

    payload = {
        "job_id": job["job_id"],
        "source_language": source_language,
        "target_language": target_language,
        "target_language_name": target_name,
        "status": "succeeded",
        "model": settings.translation_model,
        "source_signature": source_hash,
        "sentences": translated_sentences,
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)
    return payload


def translation_cache_path(outdir: str, target_language: str) -> str:
    safe_target = re.sub(r"[^a-zA-Z0-9_-]+", "_", target_language)
    return os.path.join(outdir, "translations", f"{safe_target}.json")


def translate_sentences_batched(
    sentences: list[dict],
    translator: Translator,
    source_language: str,
    target_language: str,
    batch_size: int = 16,
    request_mode: str = "json_batch",
) -> list[dict]:
    translated = []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        if request_mode == "concurrent_single":
            batch_translations = _translate_concurrent_single(batch, translator, source_language, target_language)
        else:
            try:
                batch_translations = _translate_batch(batch, translator, source_language, target_language)
            except Exception:
                batch_translations = _translate_one_by_one(batch, translator, source_language, target_language)
        for sentence, translation in zip(batch, batch_translations):
            enriched = dict(sentence)
            enriched["translation"] = translation
            translated.append(enriched)
    return translated


def _translate_concurrent_single(
    batch: list[dict],
    translator: Translator,
    source_language: str,
    target_language: str,
) -> list[str]:
    if len(batch) <= 1:
        return _translate_one_by_one(batch, translator, source_language, target_language)
    try:
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = [
                executor.submit(_translate_sentence, sentence, translator, source_language, target_language)
                for sentence in batch
            ]
            return [future.result() for future in futures]
    except Exception:
        return _translate_one_by_one(batch, translator, source_language, target_language)


def _translate_batch(
    batch: list[dict],
    translator: Translator,
    source_language: str,
    target_language: str,
) -> list[str]:
    prompt = build_batch_translation_prompt(batch, source_language, target_language)
    if hasattr(translator, "complete"):
        raw = translator.complete(prompt)  # type: ignore[attr-defined]
    else:
        raw = translator.translate(prompt, source_language, target_language)
    parsed = _parse_batch_translation_json(raw)
    expected_indexes = [int(item["index"]) for item in batch]
    translations_by_index = {}
    for item in parsed:
        index = int(item.get("index"))
        translation = str(item.get("translation", "")).strip()
        translations_by_index[index] = translation
    if sorted(translations_by_index) != sorted(expected_indexes):
        raise ValueError("Batch translation indexes do not match source indexes")
    return [translations_by_index[index] for index in expected_indexes]


def _translate_one_by_one(
    batch: list[dict],
    translator: Translator,
    source_language: str,
    target_language: str,
) -> list[str]:
    translations = []
    for sentence in batch:
        translations.append(_translate_sentence(sentence, translator, source_language, target_language))
    return translations


def _translate_sentence(
    sentence: dict,
    translator: Translator,
    source_language: str,
    target_language: str,
) -> str:
    text = str(sentence.get("text") or "").strip()
    return translator.translate(text, source_language, target_language) if text else ""


def build_translation_prompt(text: str, source_language: str, target_language: str) -> str:
    if source_language == "Chinese" or target_language == "Chinese":
        return f"把下面的文本翻译成{target_language}，不要额外解释。\n\n{text}"
    return f"Translate the following segment into {target_language}, without additional explanation.\n\n{text}"


def build_batch_translation_prompt(sentences: list[dict], source_language: str, target_language: str) -> str:
    payload = [
        {"index": int(sentence["index"]), "text": str(sentence.get("text") or "")}
        for sentence in sentences
    ]
    source_json = json.dumps(payload, ensure_ascii=False)
    if source_language == "Chinese" or target_language == "Chinese":
        return (
            f"把下面 JSON 数组中的 text 翻译成{target_language}，不要额外解释。"
            "只返回 JSON 数组，每一项包含 index 和 translation，index 必须保持不变。\n\n"
            f"{source_json}"
        )
    return (
        f"Translate each text field in the following JSON array into {target_language}. "
        "Return only a JSON array. Each item must contain index and translation, and index must remain unchanged.\n\n"
        f"{source_json}"
    )


def _parse_batch_translation_json(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if not text.startswith("["):
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            raise ValueError("Batch translation output is not JSON")
        text = match.group(0)
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Batch translation output must be a list")
    for item in data:
        if not isinstance(item, dict) or "index" not in item or "translation" not in item:
            raise ValueError("Batch translation item must contain index and translation")
    return data


def language_prompt_name(language: str) -> str:
    normalized = language.strip().lower().replace("-", "_")
    return LANGUAGE_NAMES.get(normalized, language)


def _extract_chat_content(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict)).strip()
    return str(content).strip()


def _source_signature(result_path: str, result: dict) -> dict:
    stat = os.stat(result_path)
    sentences = result.get("sentences") or []
    return {
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "sentence_count": len(sentences),
    }


def _read_valid_cache(path: str, source_signature: dict) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fin:
        cached = json.load(fin)
    if cached.get("source_signature") == source_signature:
        return cached
    return None
