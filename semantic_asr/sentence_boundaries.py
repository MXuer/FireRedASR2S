import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


_TRAILING_SENTENCE_PUNCTUATION = re.compile(r"[。.!?！？]+\s*$")
_LEADING_BOUNDARY_PUNCTUATION = re.compile(r"^[\s,،，。.!?！？؛;:：]+")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")


@dataclass
class SentenceBoundaryFusionConfig:
    enabled: bool = False
    min_vad_silence_s: float = 0.2
    max_vad_snap_gap_s: float = 1.0
    merge_max_token_gap_s: float = 0.3
    acoustic_window_s: float = 0.05
    acoustic_context_s: float = 0.3
    acoustic_valley_ratio: float = 0.25
    target_sentence_s: float = 15.0
    max_sentence_s: float = 30.0


def fuse_sentence_boundaries(
    sentences: Sequence[dict],
    words: Sequence[dict],
    raw_vad_segments_ms: Sequence[tuple[int, int]],
    wav: Any,
    sample_rate: int,
    config: SentenceBoundaryFusionConfig,
) -> tuple[list[dict], list[dict]]:
    if not sentences:
        return [], []

    fused = [dict(sentences[0])]
    decisions = []
    raw_vad = sorted((int(start), int(end)) for start, end in raw_vad_segments_ms)
    sorted_words = sorted((dict(word) for word in words), key=lambda word: (word["start_ms"], word["end_ms"]))

    for candidate_index, current_source in enumerate(sentences[1:], start=1):
        previous = fused[-1]
        current = dict(current_source)
        previous_word = _last_word_before(sorted_words, previous["end_ms"])
        current_word = _first_word_after(sorted_words, current["start_ms"])
        previous_end_ms = previous_word["end_ms"] if previous_word else previous["end_ms"]
        current_start_ms = current_word["start_ms"] if current_word else current["start_ms"]
        token_gap_ms = current_start_ms - previous_end_ms
        combined_duration_ms = current["end_ms"] - previous["start_ms"]

        candidate_gap_ms = current["start_ms"] - previous["end_ms"]
        vad_silence = None
        if candidate_gap_ms <= int(config.max_vad_snap_gap_s * 1000):
            vad_silence = _supported_vad_silence(
                raw_vad,
                previous["end_ms"],
                current["start_ms"],
                int(config.min_vad_silence_s * 1000),
            )
        same_raw_vad = _same_raw_vad_segment(raw_vad, previous["end_ms"], current["start_ms"])
        boundary_ms = (previous_end_ms + current_start_ms) // 2
        valley_ratio = _acoustic_valley_ratio(
            wav,
            sample_rate,
            boundary_ms,
            config.acoustic_window_s,
            config.acoustic_context_s,
        )

        action = "keep"
        reason = "punctuation"
        if vad_silence is not None:
            reason = "vad_silence"
            boundary_ms = _snap_to_supported_silence(
                vad_silence,
                previous["end_ms"],
                current["start_ms"],
            )
            previous["end_ms"] = boundary_ms
            current["start_ms"] = boundary_ms
        elif combined_duration_ms > int(config.max_sentence_s * 1000):
            reason = "max_duration"
        elif (
            same_raw_vad
            and token_gap_ms < int(config.merge_max_token_gap_s * 1000)
            and valley_ratio > config.acoustic_valley_ratio
        ):
            action = "merge"
            reason = "merged_active_speech"
            previous["end_ms"] = max(previous["end_ms"], current["end_ms"])
            previous["text"] = _merge_text(previous["text"], current["text"])
            previous["asr_confidence"] = min(
                previous.get("asr_confidence", 0),
                current.get("asr_confidence", 0),
            )
        elif valley_ratio <= config.acoustic_valley_ratio:
            reason = "acoustic_valley"
        elif combined_duration_ms > int(config.target_sentence_s * 1000):
            reason = "target_duration"

        decisions.append({
            "candidate_index": candidate_index,
            "action": action,
            "reason": reason,
            "previous_end_ms": previous_end_ms,
            "current_start_ms": current_start_ms,
            "token_gap_ms": token_gap_ms,
            "boundary_ms": boundary_ms,
            "same_raw_vad": same_raw_vad,
            "vad_silence_ms": list(vad_silence) if vad_silence is not None else None,
            "acoustic_valley_ratio": round(valley_ratio, 4),
            "combined_duration_ms": combined_duration_ms,
        })
        if action == "keep":
            fused.append(current)

    return fused, decisions


def _last_word_before(words: Sequence[dict], boundary_ms: int) -> dict | None:
    candidates = [word for word in words if word["start_ms"] <= boundary_ms]
    return candidates[-1] if candidates else None


def _first_word_after(words: Sequence[dict], boundary_ms: int) -> dict | None:
    return next((word for word in words if word["end_ms"] >= boundary_ms), None)


def _same_raw_vad_segment(
    raw_vad_segments_ms: Sequence[tuple[int, int]],
    previous_end_ms: int,
    current_start_ms: int,
) -> bool:
    return any(
        start_ms <= previous_end_ms <= current_start_ms <= end_ms
        for start_ms, end_ms in raw_vad_segments_ms
    )


def _supported_vad_silence(
    raw_vad_segments_ms: Sequence[tuple[int, int]],
    previous_end_ms: int,
    current_start_ms: int,
    min_silence_ms: int,
) -> tuple[int, int] | None:
    for (_, previous_vad_end), (next_vad_start, _) in zip(raw_vad_segments_ms, raw_vad_segments_ms[1:]):
        if next_vad_start - previous_vad_end < min_silence_ms:
            continue
        if previous_vad_end <= current_start_ms and next_vad_start >= previous_end_ms:
            return previous_vad_end, next_vad_start
    return None


def _snap_to_supported_silence(
    vad_silence: tuple[int, int],
    previous_end_ms: int,
    current_start_ms: int,
) -> int:
    start_ms = max(vad_silence[0], previous_end_ms)
    end_ms = min(vad_silence[1], current_start_ms)
    if start_ms <= end_ms:
        return (start_ms + end_ms) // 2
    return (previous_end_ms + current_start_ms) // 2


def _acoustic_valley_ratio(
    wav: Any,
    sample_rate: int,
    boundary_ms: int,
    window_s: float,
    context_s: float,
) -> float:
    if sample_rate <= 0 or window_s <= 0 or context_s <= window_s:
        return 1.0
    center = int(boundary_ms / 1000 * sample_rate)
    window = max(int(window_s * sample_rate), 1)
    context = max(int(context_s * sample_rate), window + 1)
    local_rms = _rms(wav[max(0, center - window):center + window])
    context_rms = _rms(wav[max(0, center - context):center + context])
    if context_rms <= 0:
        return 0.0
    return local_rms / context_rms


def _rms(samples: Any) -> float:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return math.sqrt(float(np.mean(values * values)))


def _merge_text(previous: str, current: str) -> str:
    previous = _TRAILING_SENTENCE_PUNCTUATION.sub("", previous).rstrip()
    current = _LEADING_BOUNDARY_PUNCTUATION.sub("", current).lstrip()
    if not previous:
        return current
    if not current:
        return previous
    separator = "" if _CJK.search(previous[-1]) and _CJK.search(current[0]) else " "
    return previous + separator + current
