from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import urllib.request
from typing import Any, Callable, Sequence

from semantic_asr.punctuation import _join_tokens

logger = logging.getLogger("semantic_asr.qwen_semantic_boundary")


@dataclass
class QwenSemanticBoundaryConfig:
    base_url: str = "http://10.10.23.16:18000/v1"
    model: str = "Qwen3.6-27B"
    api_key: str = "EMPTY"
    language: str = "und"
    timeout_s: float = 120.0
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    max_retries: int = 1
    disable_thinking: bool = True
    response_format_json: bool = True
    fallback_max_span_s: float = 15.0


RequestFn = Callable[[dict[str, Any]], str]


class QwenSemanticBoundaryPunc:
    """Index-only semantic sentence-boundary component backed by Qwen chat API."""

    def __init__(
        self,
        config: QwenSemanticBoundaryConfig | None = None,
        requester: RequestFn | None = None,
    ):
        self.config = config or QwenSemanticBoundaryConfig()
        self._requester = requester or self._request_content

    def process_with_timestamp(self, batch_timestamp: Sequence[list], batch_uttid: Sequence[str]) -> list[dict]:
        results = []
        for timestamp, uttid in zip(batch_timestamp, batch_uttid):
            clean_timestamp = _clean_timestamp(timestamp)
            end_indices = self._predict_end_indices(clean_timestamp)
            results.append({
                "uttid": uttid,
                "punc_sentences": _sentences_from_end_indices(clean_timestamp, end_indices),
            })
        return results

    def _predict_end_indices(self, timestamp: Sequence[Sequence]) -> list[int]:
        token_count = len(timestamp)
        if token_count == 0:
            return []

        last_error = ""
        attempts = max(0, int(self.config.max_retries)) + 1
        for attempt in range(attempts):
            body = self._request_body(timestamp, last_error if attempt else "")
            content = ""
            try:
                content = self._requester(body)
                return parse_boundary_end_indices(content, token_count)
            except Exception as exc:  # noqa: BLE001 - retry/fallback boundary for model output.
                last_error = str(exc)
                logger.warning(
                    "Qwen semantic boundary attempt %s/%s failed: %s; content=%r",
                    attempt + 1,
                    attempts,
                    last_error,
                    content[:500],
                )

        return fallback_duration_end_indices(timestamp, self.config.fallback_max_span_s)

    def _request_body(self, timestamp: Sequence[Sequence], previous_error: str = "") -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You select semantic sentence boundaries for ASR timestamp tokens. "
                    "Return only a valid JSON object."
                ),
            },
            {
                "role": "user",
                "content": _build_prompt(timestamp, self.config.language, previous_error),
            },
        ]
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": int(self.config.max_tokens),
            "temperature": float(self.config.temperature),
            "top_p": float(self.config.top_p),
            "stream": False,
        }
        if self.config.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if self.config.response_format_json:
            body["response_format"] = {"type": "json_object"}
        return body

    def _request_content(self, body: dict[str, Any]) -> str:
        url = _chat_completion_url(self.config.base_url)
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=float(self.config.timeout_s)) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        return str(data["choices"][0]["message"]["content"])


def parse_boundary_end_indices(content: str, token_count: int) -> list[int]:
    payload = _load_json_object(content)
    reported_count = payload.get("token_count")
    if _coerce_index(reported_count, "token_count") != token_count:
        raise ValueError(f"token_count mismatch: expected {token_count}, got {reported_count!r}")

    if "end_indices" in payload:
        end_indices = [_coerce_index(value, "end_indices") for value in payload["end_indices"]]
    elif "spans" in payload:
        end_indices = _end_indices_from_spans(payload["spans"])
    else:
        raise ValueError("missing end_indices")
    return validate_end_indices(end_indices, token_count)


def validate_end_indices(end_indices: Sequence[int], token_count: int) -> list[int]:
    if token_count < 0:
        raise ValueError(f"invalid token_count: {token_count}")
    if token_count == 0:
        if list(end_indices):
            raise ValueError("empty token list must not have end_indices")
        return []
    if not end_indices:
        raise ValueError("end_indices must not be empty")

    validated = []
    previous = -1
    for raw_index in end_indices:
        index = _coerce_index(raw_index, "end_indices")
        if index <= previous:
            raise ValueError(f"end_indices must be strictly increasing: {end_indices}")
        if index < 0 or index >= token_count:
            raise ValueError(f"end index out of range: {index}")
        validated.append(index)
        previous = index
    if validated[-1] != token_count - 1:
        raise ValueError(f"last end index must be {token_count - 1}, got {validated[-1]}")
    return validated


def fallback_duration_end_indices(timestamp: Sequence[Sequence], max_span_s: float) -> list[int]:
    if not timestamp:
        return []
    if max_span_s <= 0:
        return [len(timestamp) - 1]

    end_indices = []
    group_start_s = float(timestamp[0][1])
    for index, item in enumerate(timestamp):
        token_end_s = float(item[2])
        is_last = index == len(timestamp) - 1
        if is_last or token_end_s - group_start_s >= max_span_s:
            end_indices.append(index)
            if not is_last:
                group_start_s = float(timestamp[index + 1][1])
    return end_indices


def _build_prompt(timestamp: Sequence[Sequence], language: str, previous_error: str = "") -> str:
    retry_instruction = ""
    if previous_error:
        retry_instruction = (
            "\nThe previous response was invalid: "
            f"{previous_error}. Return a corrected JSON object only.\n"
        )
    token_lines = "\n".join(
        f"{index}\t{_clean_token(item[0])}\t{float(item[1]):.3f}\t{float(item[2]):.3f}"
        for index, item in enumerate(timestamp)
    )
    token_count = len(timestamp)
    return (
        f"{retry_instruction}"
        "Task: split this ASR token sequence into semantic sentence spans.\n"
        f"Language id: {language}\n"
        "Input columns: index<TAB>token<TAB>start_s<TAB>end_s.\n"
        "Rules:\n"
        "1. Do not rewrite, correct, add, delete, normalize, or output token text.\n"
        "2. Return only JSON in this exact shape: "
        '{"token_count": TOKEN_COUNT, "end_indices": [END_INDEX, ...]}.\n'
        "3. end_indices are inclusive token indexes where each sentence ends.\n"
        "4. end_indices must be strictly increasing and the last value must be token_count - 1.\n"
        "5. Every token from 0 through token_count - 1 must be covered exactly once.\n"
        "6. If the whole sequence is one semantic sentence, return only the final index.\n"
        f"token_count: {token_count}\n"
        "Tokens:\n"
        f"{token_lines}"
    )


def _sentences_from_end_indices(timestamp: Sequence[Sequence], end_indices: Sequence[int]) -> list[dict]:
    sentences = []
    start_index = 0
    for end_index in end_indices:
        selected = timestamp[start_index:end_index + 1]
        if selected:
            sentences.append({
                "start_s": float(selected[0][1]),
                "end_s": float(selected[-1][2]),
                "punc_text": _join_tokens([str(item[0]) for item in selected]),
            })
        start_index = end_index + 1
    return sentences


def _clean_timestamp(timestamp: Sequence[Sequence]) -> list[list[Any]]:
    clean = []
    for item in timestamp:
        token = _clean_token(item[0])
        if not token:
            continue
        clean.append([token, float(item[1]), float(item[2])])
    return clean


def _clean_token(token: Any) -> str:
    return str(token).strip().replace("\t", " ").replace("\n", " ")


def _load_json_object(content: str) -> dict[str, Any]:
    text = str(content).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(text[start:])
    if not isinstance(data, dict):
        raise ValueError("response must be a JSON object")
    return data


def _end_indices_from_spans(spans: Any) -> list[int]:
    if not isinstance(spans, list):
        raise ValueError("spans must be a list")
    end_indices = []
    expected_start = 0
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError("span must be an object")
        start = _coerce_index(span.get("start"), "span.start")
        end = _coerce_index(span.get("end"), "span.end")
        if start != expected_start:
            raise ValueError(f"span start mismatch: expected {expected_start}, got {start}")
        if end < start:
            raise ValueError(f"span end before start: {span}")
        end_indices.append(end)
        expected_start = end + 1
    return end_indices


def _coerce_index(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"{field_name} must be an integer, got {value!r}")


def _chat_completion_url(base_url: str) -> str:
    base = str(base_url).rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"
