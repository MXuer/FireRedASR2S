import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import soundfile as sf

logger = logging.getLogger("sentence_asr_pipeline.core")


@dataclass
class PipelineConfig:
    asr_batch_size: int = 1
    punc_batch_size: int = 1
    sample_rate: int = 16000


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


class SentenceAsrPipeline:
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
        assert sample_rate == self.config.sample_rate

        vad_result = self._detect(wav_path)
        segments = self._build_segments(uttid, wav_np, sample_rate, vad_result["timestamps"])
        asr_results, asr_segments = self._transcribe(segments)
        asr_results = self.timestamp_provider.add_timestamps(asr_results, asr_segments)
        self._require_timestamps(asr_results)
        punc_results = self._punctuate(asr_results)
        sentences, words = self._format(asr_results, punc_results)

        text = "".join(s["text"] for s in sentences)
        text = re.sub(r"([.,!?])\s*([a-zA-Z])", r"\1 \2", text)

        return {
            "uttid": uttid,
            "text": text,
            "sentences": sentences,
            "vad_segments_ms": [(int(s * 1000), int(e * 1000)) for s, e in vad_result["timestamps"]],
            "dur_s": dur_s,
            "words": words,
            "wav_path": wav_path,
        }

    def _detect(self, wav_path: str) -> dict:
        result = self.vad.detect(wav_path)
        vad_result = result[0] if isinstance(result, tuple) else result
        logger.info("VAD: %s", vad_result)
        if not vad_result.get("timestamps"):
            raise ValueError("VAD must return non-empty timestamps")
        return vad_result

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

        for i, segment in enumerate(segments):
            batch_segments.append(segment)
            if len(batch_segments) < self.config.asr_batch_size and i != len(segments) - 1:
                continue

            batch_uttid = [s.uttid for s in batch_segments]
            batch_wav = [(s.sample_rate, s.wav) for s in batch_segments]
            batch_asr_results = self.asr.transcribe(batch_uttid, batch_wav)
            logger.info("ASR: %s", batch_asr_results)

            for asr_result in batch_asr_results:
                text = asr_result.get("text", "").strip()
                if not text or re.search(r"(<blank>)|(<sil>)", text):
                    continue
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

    def _punctuate(self, asr_results: Sequence[dict]) -> list[dict]:
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
                start_ms = segment_start_ms + int(punc_sentence["start_s"] * 1000)
                end_ms = segment_start_ms + int(punc_sentence["end_s"] * 1000)
                if i == 0:
                    start_ms = segment_start_ms
                if i == len(punc_sentences) - 1:
                    end_ms = segment_end_ms
                sentences.append(self._sentence(start_ms, end_ms, punc_sentence["punc_text"], asr_result))

            for token, start_s, end_s in asr_result.get("timestamp", []):
                words.append({
                    "start_ms": int(start_s * 1000 + segment_start_ms),
                    "end_ms": int(end_s * 1000 + segment_start_ms),
                    "text": token,
                })

        return sentences, words

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
