import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

import soundfile as sf

logger = logging.getLogger("sentence_asr_pipeline.core")


@dataclass
class PipelineConfig:
    asr_batch_size: int = 1
    punc_batch_size: int = 1
    enable_vad: bool = True
    enable_lid: bool = False
    enable_punc: bool = True
    return_timestamp: bool = True
    sample_rate: int = 16000


class VadModel(Protocol):
    def detect(self, wav_path: str) -> Any:
        """Return {"timestamps": [(start_s, end_s), ...]} or (that_dict, extra)."""


class AsrModel(Protocol):
    def transcribe(self, batch_uttid: Sequence[str], batch_wav: Sequence[tuple[int, Any]]) -> list[dict]:
        """Return dicts with uttid, text, optional confidence and timestamp."""


class TimestampPredictorModel(Protocol):
    def predict(self, batch_asr_result: Sequence[dict]) -> list[dict]:
        """Return ASR dicts with timestamp filled as [(token, start_s, end_s), ...]."""


class PuncModel(Protocol):
    def process(self, batch_text: Sequence[str], batch_uttid: Optional[Sequence[str]] = None) -> list[dict]:
        """Return dicts with uttid and punc_text."""

    def process_with_timestamp(self, batch_timestamp: Sequence[list], batch_uttid: Optional[Sequence[str]] = None) -> list[dict]:
        """Return dicts with uttid and punc_sentences."""


class LidModel(Protocol):
    def process(self, batch_uttid: Sequence[str], batch_wav: Sequence[tuple[int, Any]]) -> list[dict]:
        """Return dicts with uttid, lang and optional confidence."""


class SentenceAsrPipeline:
    def __init__(
        self,
        asr: AsrModel,
        config: PipelineConfig,
        vad: Optional[VadModel] = None,
        timestamp_predictor: Optional[TimestampPredictorModel] = None,
        punc: Optional[PuncModel] = None,
        lid: Optional[LidModel] = None,
    ):
        self.asr = asr
        self.config = config
        self.vad = vad
        self.timestamp_predictor = timestamp_predictor
        self.punc = punc
        self.lid = lid

    def process(self, wav_path: str, uttid: str = "tmpid") -> dict:
        wav_np, sample_rate = sf.read(wav_path, dtype="int16")
        dur_s = wav_np.shape[0] / sample_rate
        assert sample_rate == self.config.sample_rate

        vad_result = self._detect(wav_path, dur_s)
        asr_results, lid_results = self._transcribe(uttid, wav_np, sample_rate, vad_result["timestamps"])
        asr_results = self._add_timestamps(asr_results)
        punc_results = self._punctuate(asr_results)
        sentences, words = self._format(asr_results, punc_results, lid_results)

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

    def _detect(self, wav_path: str, dur_s: float) -> dict:
        if not self.config.enable_vad or self.vad is None:
            return {"timestamps": [(0, dur_s)]}
        result = self.vad.detect(wav_path)
        vad_result = result[0] if isinstance(result, tuple) else result
        logger.info("VAD: %s", vad_result)
        return vad_result

    def _transcribe(
        self,
        uttid: str,
        wav_np: Any,
        sample_rate: int,
        vad_segments: Sequence[tuple[float, float]],
    ) -> tuple[list[dict], list[Optional[dict]]]:
        asr_results = []
        lid_results = []
        batch_uttid = []
        batch_wav = []

        for i, (start_s, end_s) in enumerate(vad_segments):
            segment_wav = wav_np[int(start_s * sample_rate):int(end_s * sample_rate)]
            segment_uttid = f"{uttid}_s{int(start_s * 1000)}_e{int(end_s * 1000)}"
            batch_uttid.append(segment_uttid)
            batch_wav.append((sample_rate, segment_wav))
            if len(batch_uttid) < self.config.asr_batch_size and i != len(vad_segments) - 1:
                continue

            batch_asr_results = self.asr.transcribe(batch_uttid, batch_wav)
            logger.info("ASR: %s", batch_asr_results)
            if self.config.enable_lid and self.lid is not None:
                batch_lid_results = self.lid.process(batch_uttid, batch_wav)
                logger.info("LID: %s", batch_lid_results)
            else:
                batch_lid_results = [None] * len(batch_asr_results)

            for asr_result, lid_result in zip(batch_asr_results, batch_lid_results):
                text = asr_result.get("text", "").strip()
                if not text or re.search(r"(<blank>)|(<sil>)", text):
                    continue
                asr_results.append(asr_result)
                lid_results.append(lid_result)

            batch_uttid = []
            batch_wav = []

        return asr_results, lid_results

    def _add_timestamps(self, asr_results: list[dict]) -> list[dict]:
        if self.config.return_timestamp and self.timestamp_predictor is not None:
            return self.timestamp_predictor.predict(asr_results)
        return asr_results

    def _punctuate(self, asr_results: Sequence[dict]) -> list[dict]:
        if not self.config.enable_punc or self.punc is None:
            return list(asr_results)

        punc_results = []
        batch_text = []
        batch_uttid = []
        batch_timestamp = []
        for i, asr_result in enumerate(asr_results):
            batch_text.append(asr_result["text"])
            batch_uttid.append(asr_result["uttid"])
            if self.config.return_timestamp:
                batch_timestamp.append(asr_result.get("timestamp", []))
            if len(batch_text) < self.config.punc_batch_size and i != len(asr_results) - 1:
                continue

            if self.config.return_timestamp:
                batch_result = self.punc.process_with_timestamp(batch_timestamp, batch_uttid)
            else:
                batch_result = self.punc.process(batch_text, batch_uttid)
            logger.info("Punc: %s", batch_result)
            punc_results.extend(batch_result)

            batch_text = []
            batch_uttid = []
            batch_timestamp = []

        return punc_results

    def _format(
        self,
        asr_results: Sequence[dict],
        punc_results: Sequence[dict],
        lid_results: Sequence[Optional[dict]],
    ) -> tuple[list[dict], list[dict]]:
        sentences = []
        words = []
        for asr_result, punc_result, lid_result in zip(asr_results, punc_results, lid_results):
            assert asr_result["uttid"] == punc_result["uttid"], f"{asr_result} | {punc_result}"
            segment_start_ms, segment_end_ms = self._parse_uttid_ms(asr_result["uttid"])

            if self.config.return_timestamp:
                if self.config.enable_punc and self.punc is not None:
                    punc_sentences = punc_result["punc_sentences"]
                    for i, punc_sentence in enumerate(punc_sentences):
                        start_ms = segment_start_ms + int(punc_sentence["start_s"] * 1000)
                        end_ms = segment_start_ms + int(punc_sentence["end_s"] * 1000)
                        if i == 0:
                            start_ms = segment_start_ms
                        if i == len(punc_sentences) - 1:
                            end_ms = segment_end_ms
                        sentences.append(self._sentence(start_ms, end_ms, punc_sentence["punc_text"], asr_result, lid_result))
                else:
                    sentences.append(self._sentence(segment_start_ms, segment_end_ms, asr_result["text"], asr_result, lid_result))
            else:
                text = punc_result["punc_text"] if self.config.enable_punc and self.punc is not None else asr_result["text"]
                sentences.append(self._sentence(segment_start_ms, segment_end_ms, text, asr_result, lid_result))

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
    def _sentence(start_ms: int, end_ms: int, text: str, asr_result: dict, lid_result: Optional[dict]) -> dict:
        sentence = {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text,
            "asr_confidence": asr_result.get("confidence", 0),
            "lang": None,
            "lang_confidence": 0,
        }
        if lid_result:
            sentence["lang"] = lid_result.get("lang")
            sentence["lang_confidence"] = lid_result.get("confidence", 0)
        return sentence
