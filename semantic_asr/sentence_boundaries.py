import re
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np


_TRAILING_SENTENCE_PUNCTUATION = re.compile(r"[。.!?！？؟]+\s*$")
_TERMINAL_SENTENCE_PUNCTUATION = re.compile(r"[。.!?！？؟]+[\s\"'”’)]*$")
_TRAILING_CONTINUATION_PUNCTUATION = re.compile(r"[,،，、؛;:：]+[\s\"'”’)]*$")
_LEADING_BOUNDARY_PUNCTUATION = re.compile(r"^[\s,،，。.!?！？؛;:：]+")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")


@dataclass
class SentenceBoundaryFusionConfig:
    enabled: bool = False
    min_vad_silence_s: float = 0.2
    max_merge_vad_silence_s: float = 1.0
    max_vad_snap_gap_s: float = 1.0
    merge_max_token_gap_s: float = 0.3
    target_sentence_s: float = 15.0
    max_sentence_s: float = 30.0
    rolling_boundary_max_candidates: int = 8
    speech_prob_silence_threshold: float = 0.2
    speech_prob_silence_mean_threshold: float = 0.2
    speech_prob_active_threshold: float = 0.5
    speech_prob_window_s: float = 0.08
    speech_prob_search_window_s: float = 0.5
    preserve_sentence_gaps: bool = False


@dataclass
class BoundaryCandidate:
    candidate_index: int
    previous_end_ms: int
    current_start_ms: int
    token_gap_ms: int
    combined_duration_ms: int
    boundary_ms: int
    same_raw_vad: bool
    vad_silence: tuple[int, int] | None
    long_vad_silence: bool
    speech_stats: dict
    prob_supported_silence: bool
    prob_active_boundary: bool
    audio_safe: bool
    audio_reason: str | None
    semantic_complete: bool
    semantic_reason: str
    active_speech: bool
    snapped_boundary_ms: int


def fuse_sentence_boundaries(
    sentences: Sequence[dict],
    words: Sequence[dict],
    raw_vad_segments_ms: Sequence[tuple[int, int]],
    wav: Any,
    sample_rate: int,
    config: SentenceBoundaryFusionConfig,
    vad_frame_speech_probs: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    if not sentences:
        return [], []

    fused = [_new_fusion_group(dict(sentences[0]), 0)]
    decisions = []
    raw_vad = sorted((int(start), int(end)) for start, end in raw_vad_segments_ms)
    sorted_words = sorted((dict(word) for word in words), key=lambda word: (word["start_ms"], word["end_ms"]))

    for candidate_index, current_source in enumerate(sentences[1:], start=1):
        previous = fused[-1]
        current = _new_fusion_group(dict(current_source), candidate_index)
        candidate = _boundary_candidate(
            candidate_index,
            previous,
            current,
            sorted_words,
            raw_vad,
            config,
            vad_frame_speech_probs,
        )

        action, reason = _decide_boundary(candidate, config)
        rolling_candidate = None
        if action == "keep" and reason in {"max_duration_terminal_punctuation", "max_duration_semantic_boundary"}:
            rolling_candidate = _select_rolling_boundary(
                previous.get("_fusion_boundaries", []) + [candidate],
                config,
            )
            if rolling_candidate is not None:
                reason = "max_duration_rolling_boundary"

        if action == "keep" and rolling_candidate is not None and rolling_candidate.candidate_index != candidate.candidate_index:
            left, right, boundary_ms, preserved_gap_ms = _split_group_at_boundary(
                previous,
                current,
                candidate,
                rolling_candidate,
                config,
            )
            _mark_rolling_decision(
                decisions,
                rolling_candidate,
                reason,
                boundary_ms,
                preserved_gap_ms,
            )
            fused[-1] = left
            fused.append(right)
            action = "merge"
            reason = "merged_after_rolling_boundary"
            boundary_ms = candidate.boundary_ms
        else:
            boundary_ms = (rolling_candidate or candidate).snapped_boundary_ms
            if config.preserve_sentence_gaps:
                if action == "keep":
                    previous["end_ms"], current["start_ms"] = _preserved_gap_boundary(
                        previous,
                        current,
                        candidate.previous_end_ms,
                        candidate.current_start_ms,
                        candidate.vad_silence,
                    )
                    boundary_ms = previous["end_ms"]
            else:
                if action == "keep":
                    previous["end_ms"] = boundary_ms
                    current["start_ms"] = boundary_ms

            if action == "merge":
                boundary_ms = candidate.boundary_ms
                _merge_sentence_into_previous(previous, current)
                _append_fusion_parts(previous, current, candidate)

            if action == "keep":
                fused.append(current)

        decision_candidate = rolling_candidate if action == "keep" and rolling_candidate is not None else candidate
        decisions.append(_decision_record(
            decision_candidate,
            action,
            reason,
            boundary_ms,
            max(current["start_ms"] - previous["end_ms"], 0) if action == "keep" else None,
            rolling_candidate,
        ))
        if action == "keep":
            continue

    return [_strip_fusion_metadata(sentence) for sentence in fused], decisions


def _new_fusion_group(sentence: dict, start_index: int) -> dict:
    group = dict(sentence)
    group["_fusion_start_index"] = start_index
    group["_fusion_parts"] = [_strip_fusion_metadata(sentence)]
    group["_fusion_boundaries"] = []
    return group


def _append_fusion_parts(previous: dict, current: dict, candidate: BoundaryCandidate) -> None:
    previous.setdefault("_fusion_parts", [_strip_fusion_metadata(previous)])
    previous.setdefault("_fusion_boundaries", [])
    previous["_fusion_parts"].extend(
        _strip_fusion_metadata(part)
        for part in current.get("_fusion_parts", [current])
    )
    previous["_fusion_boundaries"].append(candidate)


def _strip_fusion_metadata(sentence: dict) -> dict:
    return {key: value for key, value in sentence.items() if not key.startswith("_fusion_")}


def _decision_record(
    candidate: BoundaryCandidate,
    action: str,
    reason: str,
    boundary_ms: int,
    preserved_gap_ms: int | None,
    rolling_candidate: BoundaryCandidate | None = None,
) -> dict:
    record = {
        "candidate_index": candidate.candidate_index,
        "action": action,
        "reason": reason,
        "previous_end_ms": candidate.previous_end_ms,
        "current_start_ms": candidate.current_start_ms,
        "token_gap_ms": candidate.token_gap_ms,
        "boundary_ms": boundary_ms,
        "same_raw_vad": candidate.same_raw_vad,
        "audio_safe": candidate.audio_safe,
        "audio_reason": candidate.audio_reason,
        "active_speech": candidate.active_speech,
        "semantic_complete": candidate.semantic_complete,
        "semantic_reason": candidate.semantic_reason,
        "vad_silence_ms": list(candidate.vad_silence) if candidate.vad_silence is not None else None,
        "long_vad_silence": candidate.long_vad_silence,
        "speech_prob_min": _round_optional(candidate.speech_stats["min"]),
        "speech_prob_mean": _round_optional(candidate.speech_stats["mean"]),
        "speech_prob_max": _round_optional(candidate.speech_stats["max"]),
        "speech_prob_boundary_ms": candidate.speech_stats["boundary_ms"],
        "speech_prob_supported_silence": candidate.prob_supported_silence,
        "combined_duration_ms": candidate.combined_duration_ms,
        "preserved_gap_ms": preserved_gap_ms,
    }
    if rolling_candidate is not None:
        record["rolling_selected_candidate_index"] = rolling_candidate.candidate_index
        record["rolling_selected_boundary_ms"] = rolling_candidate.snapped_boundary_ms
        record["rolling_selected_speech_prob_mean"] = _round_optional(rolling_candidate.speech_stats["mean"])
    return record


def _mark_rolling_decision(
    decisions: list[dict],
    selected: BoundaryCandidate,
    reason: str,
    boundary_ms: int,
    preserved_gap_ms: int | None,
) -> None:
    for decision in reversed(decisions):
        if decision.get("candidate_index") != selected.candidate_index:
            continue
        decision.update(_decision_record(selected, "keep", reason, boundary_ms, preserved_gap_ms, selected))
        return


def _split_group_at_boundary(
    previous: dict,
    current: dict,
    current_candidate: BoundaryCandidate,
    selected: BoundaryCandidate,
    config: SentenceBoundaryFusionConfig,
) -> tuple[dict, dict, int, int | None]:
    parts = previous.get("_fusion_parts", [_strip_fusion_metadata(previous)]) + current.get("_fusion_parts", [current])
    boundaries = previous.get("_fusion_boundaries", []) + [current_candidate]
    group_start_index = int(previous.get("_fusion_start_index", 0))
    split_pos = selected.candidate_index - group_start_index
    if split_pos <= 0 or split_pos >= len(parts):
        raise ValueError(f"Invalid rolling boundary split: candidate_index={selected.candidate_index}")

    left = _group_from_parts(
        parts[:split_pos],
        group_start_index,
        [boundary for boundary in boundaries if boundary.candidate_index < selected.candidate_index],
    )
    right_parts = parts[split_pos:]
    right_start_index = selected.candidate_index
    right = _group_from_parts(
        right_parts,
        right_start_index,
        _adjust_boundaries_for_group(
            [boundary for boundary in boundaries if boundary.candidate_index > selected.candidate_index],
            right_parts,
            right_start_index,
        ),
    )
    if config.preserve_sentence_gaps:
        left_end_ms, right_start_ms = _preserved_gap_boundary(
            left,
            right,
            selected.previous_end_ms,
            selected.current_start_ms,
            selected.vad_silence,
        )
        left["end_ms"] = left_end_ms
        right["start_ms"] = right_start_ms
        return left, right, left_end_ms, max(right_start_ms - left_end_ms, 0)

    boundary_ms = selected.snapped_boundary_ms
    left["end_ms"] = boundary_ms
    right["start_ms"] = boundary_ms
    return left, right, boundary_ms, max(right["start_ms"] - left["end_ms"], 0)


def _group_from_parts(parts: Sequence[dict], start_index: int, boundaries: Sequence[BoundaryCandidate]) -> dict:
    group = _new_fusion_group(_strip_fusion_metadata(parts[0]), start_index)
    group["_fusion_boundaries"] = list(boundaries)
    for part in parts[1:]:
        _merge_sentence_into_previous(group, _strip_fusion_metadata(part))
    group["_fusion_parts"] = [_strip_fusion_metadata(part) for part in parts]
    return group


def _adjust_boundaries_for_group(
    boundaries: Sequence[BoundaryCandidate],
    parts: Sequence[dict],
    start_index: int,
) -> list[BoundaryCandidate]:
    adjusted = []
    group_start_ms = int(parts[0]["start_ms"])
    for boundary in boundaries:
        part_index = boundary.candidate_index - start_index
        if part_index < 0 or part_index >= len(parts):
            continue
        adjusted.append(replace(
            boundary,
            combined_duration_ms=int(parts[part_index]["end_ms"]) - group_start_ms,
        ))
    return adjusted


def _select_rolling_boundary(
    boundaries: Sequence[BoundaryCandidate],
    config: SentenceBoundaryFusionConfig,
) -> BoundaryCandidate | None:
    target_ms = int(config.target_sentence_s * 1000)
    candidates = [
        boundary for boundary in boundaries
        if boundary.semantic_complete
        and boundary.token_gap_ms >= 0
        and boundary.combined_duration_ms >= target_ms
    ]
    max_candidates = max(1, int(config.rolling_boundary_max_candidates))
    candidates = candidates[-max_candidates:]
    if not candidates:
        return None
    return min(candidates, key=_rolling_boundary_score)


def _rolling_boundary_score(candidate: BoundaryCandidate) -> tuple[float, float, float, int]:
    mean = candidate.speech_stats.get("mean")
    max_prob = candidate.speech_stats.get("max")
    min_prob = candidate.speech_stats.get("min")
    return (
        1.0 if mean is None else float(mean),
        1.0 if max_prob is None else float(max_prob),
        1.0 if min_prob is None else float(min_prob),
        candidate.candidate_index,
    )


def _boundary_candidate(
    candidate_index: int,
    previous: dict,
    current: dict,
    sorted_words: Sequence[dict],
    raw_vad: Sequence[tuple[int, int]],
    config: SentenceBoundaryFusionConfig,
    vad_frame_speech_probs: dict | None,
) -> BoundaryCandidate:
    previous_word = _last_word_in_sentence(sorted_words, previous)
    current_word = _first_word_in_sentence(sorted_words, current)
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
    long_vad_silence = _supported_vad_silence(
        raw_vad,
        previous_end_ms,
        current_start_ms,
        int(config.max_merge_vad_silence_s * 1000),
    )
    vad_silence = vad_silence or long_vad_silence

    same_raw_vad = _same_raw_vad_segment(raw_vad, previous["end_ms"], current["start_ms"])
    boundary_ms = (previous_end_ms + current_start_ms) // 2
    speech_stats = _speech_prob_stats(
        vad_frame_speech_probs,
        boundary_ms,
        config.speech_prob_window_s,
        config.speech_prob_search_window_s,
    )
    prob_supported_silence = _probability_supported_silence(speech_stats, config)
    prob_active_boundary = (
        speech_stats["mean"] is not None
        and speech_stats["mean"] >= config.speech_prob_active_threshold
    )

    audio_safe = False
    audio_reason = None
    snapped_boundary_ms = boundary_ms
    if vad_silence is not None:
        audio_safe = True
        audio_reason = "vad_silence"
        snapped_boundary_ms = _snap_to_supported_silence(
            vad_silence,
            previous["end_ms"],
            current["start_ms"],
        )
    elif prob_supported_silence:
        audio_safe = True
        audio_reason = "vad_prob_silence"
        snapped_boundary_ms = _snap_to_probability_boundary(
            speech_stats["boundary_ms"],
            previous["end_ms"],
            current["start_ms"],
        )

    semantic_complete, semantic_reason = _semantic_boundary(
        previous["text"],
        current["text"],
        current["end_ms"] - current["start_ms"],
        combined_duration_ms,
        config,
    )
    active_speech = (
        prob_active_boundary
        or (
            same_raw_vad
            and token_gap_ms < int(config.merge_max_token_gap_s * 1000)
            and not audio_safe
        )
    )
    return BoundaryCandidate(
        candidate_index=candidate_index,
        previous_end_ms=previous_end_ms,
        current_start_ms=current_start_ms,
        token_gap_ms=token_gap_ms,
        combined_duration_ms=combined_duration_ms,
        boundary_ms=boundary_ms,
        same_raw_vad=same_raw_vad,
        vad_silence=vad_silence,
        long_vad_silence=long_vad_silence is not None,
        speech_stats=speech_stats,
        prob_supported_silence=prob_supported_silence,
        prob_active_boundary=prob_active_boundary,
        audio_safe=audio_safe,
        audio_reason=audio_reason,
        semantic_complete=semantic_complete,
        semantic_reason=semantic_reason,
        active_speech=active_speech,
        snapped_boundary_ms=snapped_boundary_ms,
    )


def _decide_boundary(candidate: BoundaryCandidate, config: SentenceBoundaryFusionConfig) -> tuple[str, str]:
    target_ms = int(config.target_sentence_s * 1000)
    max_ms = int(config.max_sentence_s * 1000)
    over_max = candidate.combined_duration_ms > max_ms
    reached_target = candidate.combined_duration_ms >= target_ms

    if candidate.token_gap_ms < 0:
        return "merge", "merged_non_monotonic_boundary"

    if candidate.long_vad_silence:
        return "keep", "long_vad_silence"

    if over_max:
        if candidate.audio_safe and not candidate.semantic_complete:
            return "keep", "max_duration_audio_safe"
        if candidate.audio_safe:
            return "keep", candidate.audio_reason or "max_duration_audio_safe"
        if candidate.active_speech and candidate.semantic_complete:
            if candidate.semantic_reason == "terminal_punctuation":
                return "keep", "max_duration_terminal_punctuation"
            if candidate.token_gap_ms >= 0:
                return "keep", "max_duration_semantic_boundary"
        if candidate.active_speech:
            return "merge", "max_duration_wait_for_silence"
        return "keep", "max_duration_forced_boundary"

    if candidate.active_speech:
        reason = "merged_active_speech_prob" if candidate.prob_active_boundary else "merged_active_speech"
        return "merge", reason

    if candidate.audio_safe and candidate.semantic_complete:
        return "keep", candidate.audio_reason or "semantic_boundary"

    if candidate.audio_safe and reached_target:
        return "keep", candidate.audio_reason or "target_duration_audio_safe"

    if candidate.audio_safe:
        return "merge", "merged_semantic_incomplete"

    if candidate.semantic_reason == "terminal_punctuation" and candidate.token_gap_ms >= 0 and reached_target:
        return "keep", "terminal_punctuation"

    if (
        candidate.semantic_complete
        and not candidate.same_raw_vad
        and candidate.token_gap_ms >= int(config.merge_max_token_gap_s * 1000)
    ):
        return "keep", "semantic_boundary"

    if candidate.same_raw_vad:
        return "merge", "merged_active_speech"

    return "merge", "merged_semantic_incomplete"


def _semantic_boundary(
    previous_text: str,
    current_text: str,
    current_duration_ms: int,
    combined_duration_ms: int,
    config: SentenceBoundaryFusionConfig,
) -> tuple[bool, str]:
    previous = previous_text.strip()
    current = _LEADING_BOUNDARY_PUNCTUATION.sub("", current_text).strip()
    if not previous:
        return False, "empty_previous"
    if not current:
        return True, "empty_current"
    if _TRAILING_CONTINUATION_PUNCTUATION.search(previous):
        return False, "previous_continuation_punctuation"
    if _TERMINAL_SENTENCE_PUNCTUATION.search(previous):
        return True, "terminal_punctuation"
    target_ms = int(config.target_sentence_s * 1000)
    if current_duration_ms < 3000 and combined_duration_ms < target_ms:
        return False, "short_incomplete_fragment"
    if combined_duration_ms < target_ms:
        return False, "previous_no_terminal_punctuation"
    return True, "duration_target_without_terminal"


def _merge_sentence_into_previous(previous: dict, current: dict) -> None:
    previous["end_ms"] = max(previous["end_ms"], current["end_ms"])
    previous["text"] = _merge_text(previous["text"], current["text"])
    previous["asr_confidence"] = min(
        previous.get("asr_confidence", 0),
        current.get("asr_confidence", 0),
    )


def _preserved_gap_boundary(
    previous: dict,
    current: dict,
    previous_end_ms: int,
    current_start_ms: int,
    vad_silence: tuple[int, int] | None,
) -> tuple[int, int]:
    if vad_silence is not None:
        previous_boundary_ms = max(previous["start_ms"], min(int(vad_silence[0]), current["start_ms"]))
        current_boundary_ms = min(current["end_ms"], max(int(vad_silence[1]), previous_boundary_ms))
        return previous_boundary_ms, current_boundary_ms
    if previous_end_ms <= current_start_ms:
        previous_boundary_ms = max(previous["start_ms"], min(previous_end_ms, current["start_ms"]))
        current_boundary_ms = min(current["end_ms"], max(current_start_ms, previous_boundary_ms))
        return previous_boundary_ms, current_boundary_ms
    boundary_ms = (previous_end_ms + current_start_ms) // 2
    return max(previous["start_ms"], boundary_ms), min(current["end_ms"], boundary_ms)


def _last_word_in_sentence(words: Sequence[dict], sentence: dict) -> dict | None:
    candidates = [
        word for word in words
        if word["start_ms"] < sentence["end_ms"] and word["end_ms"] > sentence["start_ms"]
    ]
    return candidates[-1] if candidates else None


def _first_word_in_sentence(words: Sequence[dict], sentence: dict) -> dict | None:
    return next(
        (
            word for word in words
            if word["end_ms"] > sentence["start_ms"] and word["start_ms"] < sentence["end_ms"]
        ),
        None,
    )


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


def _snap_to_boundary_gap(boundary_ms: int, previous_end_ms: int, current_start_ms: int) -> int:
    if previous_end_ms <= boundary_ms <= current_start_ms:
        return boundary_ms
    if previous_end_ms <= current_start_ms:
        return max(previous_end_ms, min(boundary_ms, current_start_ms))
    return (previous_end_ms + current_start_ms) // 2


def _snap_to_probability_boundary(boundary_ms: int, previous_end_ms: int, current_start_ms: int) -> int:
    if previous_end_ms == current_start_ms:
        return boundary_ms
    return _snap_to_boundary_gap(boundary_ms, previous_end_ms, current_start_ms)


def _speech_prob_stats(
    frame_speech_probs: dict | None,
    boundary_ms: int,
    window_s: float,
    search_window_s: float,
) -> dict:
    if not frame_speech_probs:
        return {"min": None, "mean": None, "max": None, "boundary_ms": boundary_ms}
    probs = frame_speech_probs.get("probs") or []
    frame_shift_ms = float(frame_speech_probs.get("frame_shift_ms") or 0)
    frame_length_ms = float(frame_speech_probs.get("frame_length_ms") or frame_shift_ms)
    if not probs or frame_shift_ms <= 0:
        return {"min": None, "mean": None, "max": None, "boundary_ms": boundary_ms}

    window_ms = max(float(window_s) * 1000, frame_shift_ms)
    search_ms = max(float(search_window_s) * 1000, window_ms)
    frames = [
        (
            int(round(index * frame_shift_ms + frame_length_ms / 2)),
            float(prob),
        )
        for index, prob in enumerate(probs)
    ]
    search_frames = [
        (center_ms, prob)
        for center_ms, prob in frames
        if abs(center_ms - boundary_ms) <= search_ms
    ]
    if not search_frames:
        return {"min": None, "mean": None, "max": None, "boundary_ms": boundary_ms}
    candidates = []
    for candidate_center_ms, _ in search_frames:
        window_frames = [
            prob
            for center_ms, prob in frames
            if abs(center_ms - candidate_center_ms) <= window_ms
        ]
        if not window_frames:
            continue
        candidates.append((
            float(np.mean(window_frames)),
            float(np.max(window_frames)),
            abs(candidate_center_ms - boundary_ms),
            candidate_center_ms,
            float(np.min(window_frames)),
            window_frames,
        ))
    if not candidates:
        return {"min": None, "mean": None, "max": None, "boundary_ms": boundary_ms}
    mean_prob, max_prob, _, best_center_ms, min_prob, _ = min(candidates)
    return {
        "min": min_prob,
        "mean": mean_prob,
        "max": max_prob,
        "boundary_ms": best_center_ms,
    }


def _probability_supported_silence(
    speech_stats: dict,
    config: SentenceBoundaryFusionConfig,
) -> bool:
    return (
        speech_stats["min"] is not None
        and speech_stats["mean"] is not None
        and speech_stats["min"] <= config.speech_prob_silence_threshold
        and speech_stats["mean"] <= config.speech_prob_silence_mean_threshold
    )


def _round_optional(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def _merge_text(previous: str, current: str) -> str:
    previous = previous.rstrip()
    current = _LEADING_BOUNDARY_PUNCTUATION.sub("", current).lstrip()
    if not previous:
        return current
    if not current:
        return previous
    separator = "" if _CJK.search(previous[-1]) and _CJK.search(current[0]) else " "
    return previous + separator + current
