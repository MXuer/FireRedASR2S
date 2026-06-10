import re
import logging
from dataclasses import dataclass, fields
from typing import Any, Protocol, Sequence

import soundfile as sf

from semantic_asr.punctuation import strip_timestamp_punctuation
from semantic_asr.sentence_boundaries import (
    SentenceBoundaryFusionConfig,
    fuse_sentence_boundaries,
)

logger = logging.getLogger("semantic_asr.core")
_FINAL_TERMINAL_PUNCTUATION = re.compile(r"[。.!?！？؟]+[\s\"'”’)]*$")


@dataclass
class PipelineConfig:
    asr_batch_size: int = 1
    punc_batch_size: int = 1
    sample_rate: int | None = None
    strip_punctuation_before_punc: bool = True
    merge_vad_segments: bool = False
    vad_min_segment_s: float = 10.0
    vad_max_segment_s: float = 30.0
    vad_max_merge_gap_s: float = 3.0
    output_vad_min_silence_merge_s: float = 0.2
    output_vad_pad_s: float = 0.2
    final_sentence_merge_max_gap_s: float = 2.0
    final_sentence_merge_max_duration_s: float = 15.0
    preserve_sentence_gaps: bool = False
    sentence_boundary_fusion: SentenceBoundaryFusionConfig | dict | None = None


@dataclass
class SpeechSegment:
    uttid: str
    start_s: float
    end_s: float
    sample_rate: int
    wav: Any


class VadModel(Protocol):
    def detect(self, wav_path: str) -> Any:
        """Return {"timestamps": [(start_s, end_s), ...]} or (that_dict, extra)."""


class AsrModel(Protocol):
    def transcribe(self, batch_uttid: Sequence[str], batch_wav: Sequence[tuple[int, Any]]) -> list[dict]:
        """Return dicts with uttid, text and optional confidence."""


class TimestampProvider(Protocol):
    def add_timestamps(self, batch_asr_result: Sequence[dict], batch_segments: Sequence[SpeechSegment]) -> list[dict]:
        """Return ASR dicts with timestamp filled as [(token, start_s, end_s), ...]."""


class PuncModel(Protocol):
    def process_with_timestamp(self, batch_timestamp: Sequence[list], batch_uttid: Sequence[str]) -> list[dict]:
        """Return dicts with uttid and punc_sentences."""


class SemanticAsrPipeline:
    def __init__(
        self,
        vad: VadModel,
        asr: AsrModel,
        timestamp_provider: TimestampProvider,
        punc: PuncModel,
        config: PipelineConfig,
    ):
        self.vad = vad
        self.asr = asr
        self.timestamp_provider = timestamp_provider
        self.punc = punc
        self.config = config

    def process(self, wav_path: str, uttid: str = "tmpid") -> dict:
        wav_np, sample_rate = sf.read(wav_path, dtype="int16")
        dur_s = wav_np.shape[0] / sample_rate
        if self.config.sample_rate is not None:
            assert sample_rate == self.config.sample_rate

        raw_vad_result = self._detect(wav_path)
        vad_result = self._postprocess_vad(raw_vad_result)
        segments = self._build_segments(uttid, wav_np, sample_rate, vad_result["timestamps"])
        asr_results, asr_segments = self._transcribe(segments)
        asr_results = self.timestamp_provider.add_timestamps(asr_results, asr_segments)
        self._require_timestamps(asr_results)
        timestamp_segments = self._format_timestamp_segments(asr_results)
        if self.config.strip_punctuation_before_punc:
            asr_results = self._strip_punctuation(asr_results)
            self._require_timestamps(asr_results)
        punc_results = self._punctuate(asr_results)
        sentences, words = self._format(asr_results, punc_results)
        semantic_sentences = [dict(sentence) for sentence in sentences]
        boundary_decisions = []
        boundary_config = self._sentence_boundary_fusion_config()
        if boundary_config.enabled:
            sentences, boundary_decisions = fuse_sentence_boundaries(
                sentences,
                words,
                self._segments_ms(raw_vad_result["timestamps"]),
                wav_np,
                sample_rate,
                boundary_config,
                raw_vad_result.get("frame_speech_probs"),
            )
        output_vad_segments = self._format_output_vad_segments(raw_vad_result["timestamps"], dur_s)
        if not self.config.preserve_sentence_gaps:
            sentences = align_sentences_to_output_vad(sentences, self._segments_ms(output_vad_segments))
        sentences = remove_sentence_overlaps(sentences)
        if not self.config.preserve_sentence_gaps and not boundary_config.enabled:
            sentences = merge_final_sentences_by_gap(
                sentences,
                max_gap_s=self.config.final_sentence_merge_max_gap_s,
                max_duration_s=self.config.final_sentence_merge_max_duration_s,
            )
            sentences = remove_sentence_overlaps(sentences)
        sentences = add_sentence_cut_segments(sentences, self._segments_ms(raw_vad_result["timestamps"]))

        text = "".join(s["text"] for s in sentences)
        text = re.sub(r"([.,!?])\s*([a-zA-Z])", r"\1 \2", text)

        result = {
            "uttid": uttid,
            "text": text,
            "sentences": sentences,
            "vad_segments_ms": self._segments_ms(output_vad_segments),
            "raw_vad_segments_ms": [
                (int(s * 1000), int(e * 1000)) for s, e in raw_vad_result["timestamps"]
            ],
            "asr_vad_segments_ms": self._segments_ms(vad_result["timestamps"]),
            "dur_s": dur_s,
            "words": words,
            "timestamp_segments": timestamp_segments,
            "vad_frame_speech_probs": raw_vad_result.get("frame_speech_probs"),
            "wav_path": wav_path,
        }
        if boundary_config.enabled:
            result["semantic_sentences"] = semantic_sentences
            result["sentence_boundary_decisions"] = boundary_decisions
        return result

    def _detect(self, wav_path: str) -> dict:
        result = self.vad.detect(wav_path)
        vad_result = result[0] if isinstance(result, tuple) else result
        logger.info("VAD: %s", _compact_vad_for_log(vad_result))
        if not vad_result.get("timestamps"):
            raise ValueError("VAD must return non-empty timestamps")
        return vad_result

    def _postprocess_vad(self, vad_result: dict) -> dict:
        if self.config.merge_vad_segments:
            segments = merge_vad_segments(
                vad_result["timestamps"],
                min_segment_s=self.config.vad_min_segment_s,
                max_segment_s=self.config.vad_max_segment_s,
                max_merge_gap_s=self.config.vad_max_merge_gap_s,
            )
        else:
            segments = _normalize_segments(vad_result["timestamps"])
        logger.info("VAD ASR segments: %s", segments)

        result = dict(vad_result)
        result["timestamps"] = segments
        return result

    def _format_output_vad_segments(
        self,
        raw_vad_segments: Sequence[tuple[float, float]],
        dur_s: float,
    ) -> list[tuple[float, float]]:
        segments = merge_close_vad_segments(
            raw_vad_segments,
            max_silence_s=self.config.output_vad_min_silence_merge_s,
        )
        segments = pad_vad_segments(segments, dur_s=dur_s, pad_s=self.config.output_vad_pad_s)
        logger.info("VAD output segments: %s", segments)
        return segments

    @staticmethod
    def _build_segments(
        uttid: str,
        wav_np: Any,
        sample_rate: int,
        vad_segments: Sequence[tuple[float, float]],
    ) -> list[SpeechSegment]:
        segments = []
        for start_s, end_s in vad_segments:
            segment_wav = wav_np[int(start_s * sample_rate):int(end_s * sample_rate)]
            segment_uttid = f"{uttid}_s{int(start_s * 1000)}_e{int(end_s * 1000)}"
            segments.append(SpeechSegment(segment_uttid, start_s, end_s, sample_rate, segment_wav))
        return segments

    def _transcribe(
        self,
        segments: Sequence[SpeechSegment],
    ) -> tuple[list[dict], list[SpeechSegment]]:
        asr_results = []
        asr_segments = []
        batch_segments = []
        asr_batch_size = max(
            int(self.config.asr_batch_size),
            int(getattr(self.asr, "recommended_batch_size", 1)),
        )

        for i, segment in enumerate(segments):
            batch_segments.append(segment)
            if len(batch_segments) < asr_batch_size and i != len(segments) - 1:
                continue

            batch_uttid = [s.uttid for s in batch_segments]
            batch_wav = [(s.sample_rate, s.wav) for s in batch_segments]
            batch_asr_results = self.asr.transcribe(batch_uttid, batch_wav)
            logger.info("ASR: %s", batch_asr_results)

            for asr_result in batch_asr_results:
                text = asr_result.get("text", "").strip()
                text = re.sub('<.*?>', '', text).strip()
                text = text.replace('*', '').replace('–', '')
                if not text or re.search(r"(<blank>)|(<sil>)", text):
                    continue
                if not re.sub('[,.?!，。‑？！]', '', text):
                    continue
                text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
                asr_result['text'] = text
                asr_results.append(asr_result)
                asr_segments.append(self._find_segment(asr_result["uttid"], batch_segments))

            batch_segments = []

        return asr_results, asr_segments

    @staticmethod
    def _find_segment(uttid: str, segments: Sequence[SpeechSegment]) -> SpeechSegment:
        for segment in segments:
            if segment.uttid == uttid:
                return segment
        raise ValueError(f"ASR returned unknown uttid: {uttid}")

    @staticmethod
    def _require_timestamps(asr_results: Sequence[dict]) -> None:
        for asr_result in asr_results:
            if not asr_result.get("timestamp"):
                raise ValueError(f"Timestamp provider must return timestamp for {asr_result.get('uttid')}")

    @staticmethod
    def _strip_punctuation(asr_results: Sequence[dict]) -> list[dict]:
        stripped_results = []
        for asr_result in asr_results:
            stripped = dict(asr_result)
            stripped["timestamp"] = strip_timestamp_punctuation(asr_result["timestamp"])
            stripped_results.append(stripped)
        return stripped_results

    def _punctuate(self, asr_results: Sequence[dict]) -> list[dict]:
        if hasattr(self.punc, "process_asr_results"):
            batch_result = self.punc.process_asr_results(asr_results)
            logger.info("Punc: %s", batch_result)
            return batch_result

        punc_results = []
        batch_uttid = []
        batch_timestamp = []
        for i, asr_result in enumerate(asr_results):
            batch_uttid.append(asr_result["uttid"])
            batch_timestamp.append(asr_result["timestamp"])
            if len(batch_uttid) < self.config.punc_batch_size and i != len(asr_results) - 1:
                continue

            batch_result = self.punc.process_with_timestamp(batch_timestamp, batch_uttid)
            logger.info("Punc: %s", batch_result)
            punc_results.extend(batch_result)

            batch_uttid = []
            batch_timestamp = []

        return punc_results

    def _format(
        self,
        asr_results: Sequence[dict],
        punc_results: Sequence[dict],
    ) -> tuple[list[dict], list[dict]]:
        sentences = []
        words = []
        for asr_result, punc_result in zip(asr_results, punc_results):
            assert asr_result["uttid"] == punc_result["uttid"], f"{asr_result} | {punc_result}"
            segment_start_ms, segment_end_ms = self._parse_uttid_ms(asr_result["uttid"])

            punc_sentences = punc_result["punc_sentences"]
            for i, punc_sentence in enumerate(punc_sentences):
                punc_text = str(punc_sentence.get("punc_text", "")).strip()
                if not punc_text or not re.search(r"[\w\u4e00-\u9fff]", punc_text):
                    continue
                start_ms = segment_start_ms + int(punc_sentence["start_s"] * 1000)
                end_ms = segment_start_ms + int(punc_sentence["end_s"] * 1000)
                if i == 0:
                    start_ms = segment_start_ms
                if i == len(punc_sentences) - 1:
                    end_ms = segment_end_ms
                sentences.append(self._sentence(start_ms, end_ms, punc_text, asr_result))

            for token, start_s, end_s in asr_result.get("timestamp", []):
                words.append({
                    "start_ms": int(start_s * 1000 + segment_start_ms),
                    "end_ms": int(end_s * 1000 + segment_start_ms),
                    "text": token,
                })

        return sentences, words

    def _format_timestamp_segments(self, asr_results: Sequence[dict]) -> list[dict]:
        timestamp_segments = []
        for asr_result in asr_results:
            segment_start_ms, segment_end_ms = self._parse_uttid_ms(asr_result["uttid"])
            timestamps = []
            for item in asr_result.get("timestamp", []):
                token, start_s, end_s = item[0], float(item[1]), float(item[2])
                timestamps.append({
                    "text": token,
                    "start_s": start_s,
                    "end_s": end_s,
                    "start_ms": int(start_s * 1000 + segment_start_ms),
                    "end_ms": int(end_s * 1000 + segment_start_ms),
                })
            timestamp_segments.append({
                "uttid": asr_result["uttid"],
                "start_ms": segment_start_ms,
                "end_ms": segment_end_ms,
                "text": asr_result.get("text", ""),
                "confidence": asr_result.get("confidence", 0),
                "timestamps": timestamps,
            })
        return timestamp_segments

    @staticmethod
    def _parse_uttid_ms(uttid: str) -> tuple[int, int]:
        start_ms, end_ms = uttid.split("_")[-2:]
        assert start_ms.startswith("s") and end_ms.startswith("e")
        return int(start_ms[1:]), int(end_ms[1:])

    @staticmethod
    def _sentence(start_ms: int, end_ms: int, text: str, asr_result: dict) -> dict:
        return {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text,
            "asr_confidence": asr_result.get("confidence", 0),
        }

    @staticmethod
    def _segments_ms(segments: Sequence[tuple[float, float]]) -> list[tuple[int, int]]:
        return [(int(s * 1000), int(e * 1000)) for s, e in segments]

    def _sentence_boundary_fusion_config(self) -> SentenceBoundaryFusionConfig:
        config = self.config.sentence_boundary_fusion
        if config is None:
            boundary_config = SentenceBoundaryFusionConfig()
            boundary_config.preserve_sentence_gaps = self.config.preserve_sentence_gaps
            return boundary_config
        if isinstance(config, SentenceBoundaryFusionConfig):
            boundary_config = config
            boundary_config.preserve_sentence_gaps = (
                boundary_config.preserve_sentence_gaps or self.config.preserve_sentence_gaps
            )
            return boundary_config
        if isinstance(config, dict):
            allowed = {field.name for field in fields(SentenceBoundaryFusionConfig)}
            boundary_config = SentenceBoundaryFusionConfig(**{
                key: value for key, value in config.items() if key in allowed
            })
            boundary_config.preserve_sentence_gaps = (
                boundary_config.preserve_sentence_gaps or self.config.preserve_sentence_gaps
            )
            return boundary_config
        raise TypeError("sentence_boundary_fusion must be a mapping or SentenceBoundaryFusionConfig")


def merge_close_vad_segments(
    timestamps: Sequence[tuple[float, float]],
    max_silence_s: float = 0.5,
) -> list[tuple[float, float]]:
    segments = _normalize_segments(timestamps)
    if not segments:
        return []

    merged = []
    cur_start, cur_end = segments[0]
    for start, end in segments[1:]:
        if start - cur_end < max_silence_s:
            cur_end = max(cur_end, end)
            continue
        merged.append((cur_start, cur_end))
        cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def merge_vad_segments(
    timestamps: Sequence[tuple[float, float]],
    min_segment_s: float = 10.0,
    max_segment_s: float = 40.0,
    max_merge_gap_s: float = 3.0,
) -> list[tuple[float, float]]:
    segments = _normalize_segments(timestamps)
    if not segments:
        return []

    merged = []
    cur_start, cur_end = segments[0]
    for start, end in segments[1:]:
        gap_s = start - cur_end
        can_merge = gap_s <= max_merge_gap_s and (end - cur_start) <= max_segment_s
        if can_merge:
            cur_end = max(cur_end, end)
            continue
        merged.append((cur_start, cur_end))
        cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    return merged


def pad_vad_segments(
    timestamps: Sequence[tuple[float, float]],
    dur_s: float,
    pad_s: float = 0.1,
) -> list[tuple[float, float]]:
    segments = _normalize_segments(timestamps)
    if not segments or pad_s <= 0:
        return segments

    padded = []
    for i, (start, end) in enumerate(segments):
        prev_end = segments[i - 1][1] if i > 0 else 0.0
        next_start = segments[i + 1][0] if i < len(segments) - 1 else dur_s

        left_gap = max(start - prev_end, 0.0)
        right_gap = max(next_start - end, 0.0)
        left_pad = min(pad_s, left_gap / 2 if i > 0 and left_gap < 2 * pad_s else left_gap)
        right_pad = min(pad_s, right_gap / 2 if i < len(segments) - 1 and right_gap < 2 * pad_s else right_gap)
        padded.append((
            max(start - left_pad, 0.0),
            min(end + right_pad, dur_s),
        ))
    return padded


def _normalize_segments(timestamps: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    segments = [(float(s), float(e)) for s, e in timestamps if float(e) > float(s)]
    segments.sort(key=lambda item: item[0])
    return segments


def _compact_vad_for_log(vad_result: dict) -> dict:
    result = dict(vad_result)
    frame_probs = result.get("frame_speech_probs")
    if isinstance(frame_probs, dict):
        compact = dict(frame_probs)
        compact["probs"] = f"<{len(frame_probs.get('probs') or [])} frame probabilities>"
        result["frame_speech_probs"] = compact
    return result


def align_sentences_to_output_vad(
    sentences: Sequence[dict],
    output_vad_segments_ms: Sequence[tuple[int, int]],
) -> list[dict]:
    adjusted = [dict(sentence) for sentence in sentences]
    if not adjusted:
        return adjusted

    for vad_start_ms, vad_end_ms in output_vad_segments_ms:
        indexes = [
            i for i, sentence in enumerate(adjusted)
            if sentence["start_ms"] < vad_end_ms and sentence["end_ms"] > vad_start_ms
        ]
        if not indexes:
            continue
        first = indexes[0]
        last = indexes[-1]
        adjusted[first]["start_ms"] = min(adjusted[first]["start_ms"], vad_start_ms)
        adjusted[last]["end_ms"] = max(adjusted[last]["end_ms"], vad_end_ms)

    return adjusted


def remove_sentence_overlaps(sentences: Sequence[dict]) -> list[dict]:
    adjusted = [dict(sentence) for sentence in sentences]
    for index in range(1, len(adjusted)):
        previous = adjusted[index - 1]
        current = adjusted[index]
        if current["start_ms"] >= previous["end_ms"]:
            continue
        boundary_ms = (previous["end_ms"] + current["start_ms"]) // 2
        previous["end_ms"] = max(previous["start_ms"], boundary_ms)
        current["start_ms"] = min(current["end_ms"], boundary_ms)
    return adjusted


def merge_final_sentences_by_gap(
    sentences: Sequence[dict],
    max_gap_s: float = 2.0,
    max_duration_s: float = 15.0,
) -> list[dict]:
    if not sentences:
        return []
    if max_gap_s <= 0 or max_duration_s <= 0:
        return [dict(sentence) for sentence in sentences]

    max_gap_ms = int(max_gap_s * 1000)
    max_duration_ms = int(max_duration_s * 1000)
    merged = [dict(sentences[0])]
    for sentence_source in sentences[1:]:
        previous = merged[-1]
        current = dict(sentence_source)
        gap_ms = current["start_ms"] - previous["end_ms"]
        combined_duration_ms = current["end_ms"] - previous["start_ms"]
        if (
            gap_ms < max_gap_ms
            and combined_duration_ms <= max_duration_ms
            and not _has_final_terminal_punctuation(previous.get("text", ""))
        ):
            previous["end_ms"] = max(previous["end_ms"], current["end_ms"])
            previous["text"] = _join_final_sentence_text(previous["text"], current["text"])
            previous["asr_confidence"] = min(
                previous.get("asr_confidence", 0),
                current.get("asr_confidence", 0),
            )
            continue
        merged.append(current)
    return merged


def _has_final_terminal_punctuation(text: str) -> bool:
    return bool(_FINAL_TERMINAL_PUNCTUATION.search(str(text).strip()))


def add_sentence_cut_segments(
    sentences: Sequence[dict],
    raw_vad_segments_ms: Sequence[tuple[int, int]],
) -> list[dict]:
    raw_vad = [(int(start), int(end)) for start, end in raw_vad_segments_ms if int(end) > int(start)]
    enriched = []
    for sentence_source in sentences:
        sentence = dict(sentence_source)
        cut_segments = [
            [max(vad_start_ms, int(sentence["start_ms"])), min(vad_end_ms, int(sentence["end_ms"]))]
            for vad_start_ms, vad_end_ms in raw_vad
            if vad_start_ms < sentence["end_ms"] and vad_end_ms > sentence["start_ms"]
        ]
        cut_segments = [segment for segment in cut_segments if segment[1] > segment[0]]
        if not cut_segments:
            cut_segments = [[int(sentence["start_ms"]), int(sentence["end_ms"])]]
        sentence["cut_segments_ms"] = cut_segments
        sentence["cut_start_ms"] = cut_segments[0][0]
        sentence["cut_end_ms"] = cut_segments[-1][1]
        enriched.append(sentence)
    return remove_sentence_cut_overlaps(enriched)


def remove_sentence_cut_overlaps(sentences: Sequence[dict]) -> list[dict]:
    adjusted = [dict(sentence) for sentence in sentences]
    for sentence in adjusted:
        sentence["cut_segments_ms"] = [list(segment) for segment in sentence.get("cut_segments_ms", [])]

    for index in range(1, len(adjusted)):
        previous = adjusted[index - 1]
        current = adjusted[index]
        if current.get("cut_start_ms", current["start_ms"]) >= previous.get("cut_end_ms", previous["end_ms"]):
            continue

        previous_end = int(previous.get("cut_end_ms", previous["end_ms"]))
        current_start = int(current.get("cut_start_ms", current["start_ms"]))
        boundary_ms = (previous_end + current_start) // 2
        _set_cut_end(previous, boundary_ms)
        _set_cut_start(current, boundary_ms)
    return adjusted


def _set_cut_start(sentence: dict, start_ms: int) -> None:
    cut_end_ms = int(sentence.get("cut_end_ms", sentence["end_ms"]))
    sentence["cut_start_ms"] = min(int(start_ms), cut_end_ms)
    segments = sentence.get("cut_segments_ms") or []
    if not segments:
        sentence["cut_segments_ms"] = [[sentence["cut_start_ms"], int(sentence.get("cut_end_ms", sentence["end_ms"]))]]
        return
    segments[0][0] = max(int(segments[0][0]), sentence["cut_start_ms"])
    sentence["cut_segments_ms"] = [segment for segment in segments if int(segment[1]) > int(segment[0])]
    if not sentence["cut_segments_ms"]:
        end_ms = int(sentence.get("cut_end_ms", sentence["end_ms"]))
        sentence["cut_segments_ms"] = [[sentence["cut_start_ms"], max(sentence["cut_start_ms"], end_ms)]]


def _set_cut_end(sentence: dict, end_ms: int) -> None:
    cut_start_ms = int(sentence.get("cut_start_ms", sentence["start_ms"]))
    sentence["cut_end_ms"] = max(int(end_ms), cut_start_ms)
    segments = sentence.get("cut_segments_ms") or []
    if not segments:
        sentence["cut_segments_ms"] = [[int(sentence.get("cut_start_ms", sentence["start_ms"])), sentence["cut_end_ms"]]]
        return
    segments[-1][1] = min(int(segments[-1][1]), sentence["cut_end_ms"])
    sentence["cut_segments_ms"] = [segment for segment in segments if int(segment[1]) > int(segment[0])]
    if not sentence["cut_segments_ms"]:
        start_ms = int(sentence.get("cut_start_ms", sentence["start_ms"]))
        sentence["cut_segments_ms"] = [[start_ms, max(start_ms, sentence["cut_end_ms"])]]


def _join_final_sentence_text(previous: str, current: str) -> str:
    previous = previous.rstrip()
    current = current.lstrip()
    if not previous:
        return current
    if not current:
        return previous
    separator = "" if _CJK_BOUNDARY.search(previous[-1]) and _CJK_BOUNDARY.search(current[0]) else " "
    return previous + separator + current


_CJK_BOUNDARY = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")


SentenceAsrPipeline = SemanticAsrPipeline
