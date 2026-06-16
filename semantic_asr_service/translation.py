import json
import os
import re
import urllib.request
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
    translated_sentences = []
    for index, sentence in enumerate(result.get("sentences") or []):
        text = str(sentence.get("text") or "").strip()
        translation = translator.translate(text, source_name, target_name) if text else ""
        translated_sentences.append({
            "index": index,
            "start_ms": sentence.get("start_ms"),
            "end_ms": sentence.get("end_ms"),
            "cut_start_ms": sentence.get("cut_start_ms", sentence.get("start_ms")),
            "cut_end_ms": sentence.get("cut_end_ms", sentence.get("end_ms")),
            "text": text,
            "translation": translation,
        })

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


def build_translation_prompt(text: str, source_language: str, target_language: str) -> str:
    if source_language == "Chinese" or target_language == "Chinese":
        return f"把下面的文本翻译成{target_language}，不要额外解释。\n\n{text}"
    return f"Translate the following segment into {target_language}, without additional explanation.\n\n{text}"


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
