from dataclasses import dataclass
import logging
import re
from typing import Sequence

from semantic_asr.core import SpeechSegment
from semantic_asr.language_mapping import canonical_language_id, model_language
from semantic_asr.mms_runtime.aligner import MmsAligner, MmsAlignmentFeasibilityError


logger = logging.getLogger("semantic_asr.adapters.mms_forced_aligner")


@dataclass
class MmsForcedAlignerConfig:
    model_path: str = "pretrained_models/mmsalign/model.pt"
    device: str = "cuda:0"
    language: str = "zh_cn"
    use_star: bool = False
    normalize_text: bool = False
    uroman_path: str = "uroman/bin"
    num_workers: int = 1
    fallback_on_feasibility_error: bool = False
    short_hallucination_segment_s: float = 0.8
    short_hallucination_target_rate: float = 50.0
    estimated_frame_ms: float = 20.0


@dataclass
class _AlignmentCheck:
    reason: str | None
    frame_count: int
    target_count: int
    repeat_count: int
    required_frame_count: int


class MmsForcedAlignerTimestampProvider:
    supports_batch: bool = False

    def __init__(self, config: MmsForcedAlignerConfig | None = None):
        self.config = config or MmsForcedAlignerConfig()
        self.config.language = canonical_language_id(self.config.language)
        self.config.use_star = False
        model_language("mms_forced_aligner", self.config.language)
        self.aligner = MmsAligner(
            model_path=self.config.model_path,
            device=self.config.device,
            uroman_path=self.config.uroman_path,
        )
        self.last_discarded_segments: list[dict] = []

    def add_timestamps(self, batch_asr_result: Sequence[dict], batch_segments: Sequence[SpeechSegment]) -> list[dict]:
        results = []
        self.last_discarded_segments = []
        for asr_result, segment in zip(batch_asr_result, batch_segments):
            tokens = self._prepare_tokens(asr_result.get("text", ""))
            tokens, alignment_tokens = self._prepare_alignment_items(tokens)
            check = self._alignment_check(alignment_tokens, segment)
            if check.reason is not None:
                self._record_discard(asr_result, segment, tokens, check)
                continue

            names = [f"{asr_result['uttid']}_{i}" for i in range(len(tokens))]
            fallback = None
            try:
                aligned = self.aligner.align(
                    tokens,
                    segment.wav,
                    segment.sample_rate,
                    names,
                    use_star=self.config.use_star,
                    language=model_language("mms_forced_aligner", self.config.language),
                    raw_transcripts=tokens,
                    alignment_transcripts=alignment_tokens,
                )
            except MmsAlignmentFeasibilityError as exc:
                check = _AlignmentCheck(
                    reason=exc.reason,
                    frame_count=exc.frame_count,
                    target_count=exc.target_count,
                    repeat_count=exc.repeat_count,
                    required_frame_count=exc.target_count + exc.repeat_count,
                )
                if not self.config.fallback_on_feasibility_error:
                    self._record_discard(asr_result, segment, tokens, check)
                    continue
                logger.warning(
                    "MMS alignment fallback uttid=%s reason=%s frames=%s target_chars=%s repeats=%s",
                    asr_result.get("uttid"),
                    exc.reason,
                    exc.frame_count,
                    exc.target_count,
                    exc.repeat_count,
                )
                aligned = self._fallback_alignment(tokens, segment)
                fallback = {
                    "provider": "mms_forced_aligner",
                    "reason": exc.reason,
                    "frame_count": exc.frame_count,
                    "target_count": exc.target_count,
                    "repeat_count": exc.repeat_count,
                }

            timestamped = dict(asr_result)
            timestamped["timestamp"] = self._normalize_alignment(aligned)
            if fallback is not None:
                timestamped["timestamp_fallback"] = fallback
            results.append(timestamped)
        return results

    def _alignment_check(self, alignment_tokens: Sequence[str], segment: SpeechSegment) -> "_AlignmentCheck":
        if not alignment_tokens:
            return _AlignmentCheck("empty_text", self._estimate_frame_count(segment), 0, 0, 0)
        if not hasattr(self.aligner, "_uromanize_alignment_tokens") or not hasattr(self.aligner, "dictionary"):
            return _AlignmentCheck(None, self._estimate_frame_count(segment), 0, 0, 0)

        uroman_tokens = self.aligner._uromanize_alignment_tokens(
            [str(token).strip().lower() for token in alignment_tokens],
            model_language("mms_forced_aligner", self.config.language),
        )
        token_indices = [
            self.aligner.dictionary[token]
            for token in " ".join(uroman_tokens).split(" ")
            if token in self.aligner.dictionary
        ]
        frame_count = self._estimate_frame_count(segment)
        target_count = len(token_indices)
        repeat_count = _count_consecutive_repeats(token_indices)
        required = target_count + repeat_count
        if target_count == 0:
            return _AlignmentCheck("empty_target", frame_count, target_count, repeat_count, required)
        if required > frame_count:
            reason = "ctc_target_too_long"
            duration_s = max(float(segment.end_s) - float(segment.start_s), 0.001)
            target_rate = target_count / duration_s
            if (
                duration_s <= self.config.short_hallucination_segment_s
                and target_rate >= self.config.short_hallucination_target_rate
            ):
                reason = "short_segment_hallucination"
            return _AlignmentCheck(reason, frame_count, target_count, repeat_count, required)
        return _AlignmentCheck(None, frame_count, target_count, repeat_count, required)

    def _estimate_frame_count(self, segment: SpeechSegment) -> int:
        duration_s = max(float(segment.end_s) - float(segment.start_s), 0.0)
        frame_ms = max(float(self.config.estimated_frame_ms), 1.0)
        return int(duration_s * 1000.0 / frame_ms)

    def _record_discard(
        self,
        asr_result: dict,
        segment: SpeechSegment,
        tokens: Sequence[str],
        check: "_AlignmentCheck",
    ) -> None:
        item = {
            "provider": "mms_forced_aligner",
            "uttid": asr_result.get("uttid"),
            "start_ms": int(segment.start_s * 1000),
            "end_ms": int(segment.end_s * 1000),
            "duration_s": round(max(float(segment.end_s) - float(segment.start_s), 0.0), 3),
            "text": asr_result.get("text", ""),
            "tokens": list(tokens),
            "reason": check.reason,
            "frame_count": check.frame_count,
            "target_count": check.target_count,
            "repeat_count": check.repeat_count,
            "required_frame_count": check.required_frame_count,
        }
        logger.warning("Discarding ASR segment before MMS alignment: %s", item)
        self.last_discarded_segments.append(item)

    def _prepare_tokens(self, text: str) -> list[str]:
        text = str(text).strip()
        text = re.sub('[#]', '', text)
        if self.config.normalize_text:
            text = self._normalize_text(text)
        if self.config.language.startswith("zh"):
            text = re.sub(r"[\u4e00-\u9fa5]", lambda item: f" {item[0]} ", text)
        if self.config.language.startswith("ko"):
            text = re.sub(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]", lambda item: f" {item[0]} ", text)
        if self.config.language.startswith("ja"):
            text = re.sub(r"[\u3040-\u309f\u4E00-\u9FFF\u30a0-\u30ff]", lambda item: f" {item[0]} ", text)
        return [token for token in text.split() if token.strip()]

    def _prepare_alignment_tokens(self, tokens: Sequence[str]) -> list[str]:
        return self._prepare_alignment_items(tokens)[1]

    def _prepare_alignment_items(self, tokens: Sequence[str]) -> tuple[list[str], list[str]]:
        output_tokens = []
        alignment_tokens = []
        index = 0
        while index < len(tokens):
            span = _numeric_alignment_span(tokens, index)
            if span is not None:
                start, end = span
                surface = "".join(str(token) for token in tokens[start:end])
                output_tokens.append(surface)
                alignment_tokens.append("<star>")
                index = end
                continue

            token = str(tokens[index])
            output_tokens.append(token)
            alignment_tokens.append("<star>" if _is_numeric_alignment_token(token) else token)
            index += 1
        return output_tokens, alignment_tokens

    def _normalize_text(self, text: str) -> str:
        from semantic_asr.mms_runtime.text_normalize import LANG2TEXTNORMALIZER

        normalizer_cls = LANG2TEXTNORMALIZER.get(self.config.language) or LANG2TEXTNORMALIZER.get("tricky")
        return normalizer_cls().norm(text)

    @staticmethod
    def _normalize_alignment(aligned: Sequence[dict]) -> list[list]:
        timestamps = []
        for item in aligned:
            token = str(item.get("text", item.get("clean_text", ""))).strip()
            if token:
                timestamps.append([token, float(item["start"]), float(item["end"])])
        return timestamps

    @staticmethod
    def _fallback_alignment(tokens: Sequence[str], segment: SpeechSegment) -> list[dict]:
        tokens = [str(token).strip() for token in tokens if str(token).strip()]
        if not tokens:
            return []

        duration_s = max(float(segment.end_s) - float(segment.start_s), 0.001)
        step_s = duration_s / len(tokens)
        aligned = []
        for index, token in enumerate(tokens):
            start_s = index * step_s
            end_s = duration_s if index == len(tokens) - 1 else (index + 1) * step_s
            aligned.append({
                "start": round(start_s, 3),
                "end": round(max(end_s, start_s + 0.001), 3),
                "duration": round(max(end_s - start_s, 0.001), 3),
                "clean_text": token,
                "text": token,
                "name": f"{segment.uttid}_{index}",
            })
        return aligned


_NUMERIC_PREFIX_WORDS = {
    "aud",
    "brl",
    "cad",
    "chf",
    "cny",
    "eur",
    "gbp",
    "hkd",
    "idr",
    "inr",
    "jpy",
    "krw",
    "mop",
    "mxn",
    "myr",
    "php",
    "rmb",
    "rub",
    "sgd",
    "thb",
    "twd",
    "usd",
    "vnd",
}

_NUMERIC_SUFFIX_WORDS = _NUMERIC_PREFIX_WORDS | {
    "a",
    "b",
    "bps",
    "byte",
    "bytes",
    "c",
    "cm",
    "db",
    "f",
    "ft",
    "g",
    "gb",
    "ghz",
    "ha",
    "hour",
    "hours",
    "hz",
    "in",
    "kb",
    "kg",
    "khz",
    "km",
    "km/h",
    "kph",
    "kwh",
    "l",
    "lb",
    "lbs",
    "m",
    "m/s",
    "mb",
    "mbps",
    "mg",
    "mhz",
    "min",
    "mins",
    "ml",
    "mm",
    "ms",
    "pct",
    "percent",
    "s",
    "sec",
    "secs",
    "sqm",
    "v",
    "w",
}

_NUMERIC_CURRENCY_SYMBOLS = {
    "$",
    "＄",
    "€",
    "£",
    "¥",
    "￥",
    "₩",
    "₫",
    "₹",
    "₽",
    "₺",
    "₴",
    "฿",
    "₱",
    "₪",
    "₦",
    "₡",
    "₲",
    "₵",
    "₭",
    "₮",
    "₨",
    "﷼",
}

_NUMERIC_PREFIX_SYMBOLS = _NUMERIC_CURRENCY_SYMBOLS | {
    "±",
    "+",
    "-",
    "−",
    "~",
    "～",
    "<",
    ">",
    "≤",
    "≥",
}

_NUMERIC_SUFFIX_SYMBOLS = _NUMERIC_CURRENCY_SYMBOLS | {
    "%",
    "％",
    "‰",
    "‱",
    "℃",
    "℉",
    "°",
    "°c",
    "°C",
    "°f",
    "°F",
    "㎡",
    "m²",
    "m³",
}

_NUMERIC_INFIX_SYMBOLS = {
    ",",
    ".",
    ":",
    "/",
    "\\",
    "+",
    "-",
    "−",
    "±",
    "~",
    "～",
    "–",
    "—",
    "×",
    "*",
    "÷",
    "=",
    "<",
    ">",
    "≤",
    "≥",
}


def _is_numeric_alignment_token(token: str) -> bool:
    stripped = str(token).strip()
    return bool(stripped) and _contains_digit(stripped)


def _numeric_alignment_span(tokens: Sequence[str], index: int) -> tuple[int, int] | None:
    if not _contains_digit(tokens[index]) and not _is_numeric_prefix_token(tokens[index]):
        return None

    start = index
    end = index + 1
    if not _contains_digit(tokens[index]):
        end = _consume_numeric_tail(tokens, end, has_digit=False)
    else:
        while start > 0 and _is_numeric_prefix_token(tokens[start - 1]):
            start -= 1
        end = _consume_numeric_tail(tokens, end, has_digit=True)

    span = tokens[start:end]
    if not any(_contains_digit(token) for token in span):
        return None
    if start == index and end == index + 1 and _contains_digit(tokens[index]):
        return (start, end)
    return (start, end)


def _consume_numeric_tail(tokens: Sequence[str], index: int, has_digit: bool) -> int:
    end = index
    while end < len(tokens):
        token = tokens[end]
        previous = tokens[end - 1] if end > 0 else ""
        if _contains_digit(token):
            if has_digit and not _is_numeric_bridge_token(previous):
                break
            has_digit = True
            end += 1
            continue
        if _is_numeric_suffix_token(token):
            end += 1
            continue
        if _is_numeric_bridge_token(token) and end + 1 < len(tokens) and _contains_digit(tokens[end + 1]):
            end += 1
            continue
        break
    return end


def _contains_digit(token: str) -> bool:
    return any(char.isdigit() for char in str(token))


def _is_numeric_prefix_token(token: str) -> bool:
    stripped = str(token).strip()
    return stripped in _NUMERIC_PREFIX_SYMBOLS or stripped.lower() in _NUMERIC_PREFIX_WORDS


def _is_numeric_suffix_token(token: str) -> bool:
    stripped = str(token).strip()
    return stripped in _NUMERIC_SUFFIX_SYMBOLS or stripped.lower() in _NUMERIC_SUFFIX_WORDS


def _is_numeric_bridge_token(token: str) -> bool:
    return str(token).strip() in _NUMERIC_INFIX_SYMBOLS


def _count_consecutive_repeats(token_indices: Sequence[int]) -> int:
    return sum(1 for previous, current in zip(token_indices, token_indices[1:]) if previous == current)
