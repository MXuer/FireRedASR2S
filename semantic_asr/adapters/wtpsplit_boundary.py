from __future__ import annotations

from dataclasses import dataclass
import json
import urllib.request
from typing import Any, Callable, Sequence

from semantic_asr.punctuation import _join_tokens


@dataclass
class WtpsplitBoundaryConfig:
    base_url: str = "http://127.0.0.1:11001"
    language: str = "und"
    timeout_s: float = 120.0
    max_length: int | None = None
    min_span_s: float = 0.0


RequestFn = Callable[[dict[str, Any]], str]


class WtpsplitBoundaryPunc:
    def __init__(
        self,
        config: WtpsplitBoundaryConfig | None = None,
        requester: RequestFn | None = None,
    ):
        self.config = config or WtpsplitBoundaryConfig()
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
        if not timestamp:
            return []
        content = self._requester({
            "language": self.config.language,
            "tokens": [str(item[0]) for item in timestamp],
            "text": _join_tokens([str(item[0]) for item in timestamp]),
            "max_length": self.config.max_length,
        })
        end_indices = parse_wtpsplit_end_indices(content, len(timestamp))
        return _merge_short_spans(timestamp, end_indices, self.config.min_span_s)

    def _request_content(self, body: dict[str, Any]) -> str:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/v1/boundaries",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=float(self.config.timeout_s)) as response:
            return response.read().decode("utf-8")


def parse_wtpsplit_end_indices(content: str, token_count: int) -> list[int]:
    data = json.loads(str(content))
    reported_count = _coerce_index(data.get("token_count"), "token_count")
    if reported_count != token_count:
        raise ValueError(f"token_count mismatch: expected {token_count}, got {reported_count}")
    if "end_indices" in data:
        end_indices = [_coerce_index(value, "end_indices") for value in data["end_indices"]]
    elif "spans" in data:
        end_indices = _end_indices_from_spans(data["spans"])
    else:
        raise ValueError("missing end_indices")
    return _validate_end_indices(end_indices, token_count)


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
                "semantic_boundary": True,
                "boundary_source": "wtpsplit",
            })
        start_index = end_index + 1
    return sentences


def _merge_short_spans(timestamp: Sequence[Sequence], end_indices: Sequence[int], min_span_s: float) -> list[int]:
    if min_span_s <= 0 or len(end_indices) <= 1:
        return list(end_indices)
    merged = []
    start_index = 0
    for span_index, end_index in enumerate(end_indices):
        selected = timestamp[start_index:end_index + 1]
        duration_s = float(selected[-1][2]) - float(selected[0][1]) if selected else 0.0
        is_last = span_index == len(end_indices) - 1
        if duration_s < min_span_s:
            if merged:
                merged[-1] = end_index
            elif is_last:
                merged.append(end_index)
            start_index = end_index + 1
            continue
        merged.append(end_index)
        start_index = end_index + 1
    if not merged or merged[-1] != end_indices[-1]:
        merged.append(end_indices[-1])
    return merged


def _clean_timestamp(timestamp: Sequence[Sequence]) -> list[list[Any]]:
    clean = []
    for item in timestamp:
        token = str(item[0]).strip()
        if token:
            clean.append([token, float(item[1]), float(item[2])])
    return clean


def _end_indices_from_spans(spans: Any) -> list[int]:
    if not isinstance(spans, list):
        raise ValueError("spans must be a list")
    expected_start = 0
    end_indices = []
    for span in spans:
        start = _coerce_index(span.get("start"), "span.start")
        end = _coerce_index(span.get("end"), "span.end")
        if start != expected_start:
            raise ValueError(f"span start mismatch: expected {expected_start}, got {start}")
        if end < start:
            raise ValueError(f"span end before start: {span}")
        end_indices.append(end)
        expected_start = end + 1
    return end_indices


def _validate_end_indices(end_indices: Sequence[int], token_count: int) -> list[int]:
    if token_count == 0:
        return []
    if not end_indices:
        raise ValueError("end_indices must not be empty")
    previous = -1
    validated = []
    for index in end_indices:
        if index <= previous:
            raise ValueError(f"end_indices must be strictly increasing: {end_indices}")
        if index < 0 or index >= token_count:
            raise ValueError(f"end index out of range: {index}")
        validated.append(index)
        previous = index
    if validated[-1] != token_count - 1:
        raise ValueError(f"last end index must be {token_count - 1}, got {validated[-1]}")
    return validated


def _coerce_index(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"{field_name} must be an integer, got {value!r}")
