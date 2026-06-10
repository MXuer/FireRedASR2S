from dataclasses import dataclass
import os
import tempfile
from typing import Any, Sequence

import soundfile as sf

from semantic_asr.core import SpeechSegment
from semantic_asr.language_mapping import model_language


@dataclass
class WhisperLargeConfig:
    model_name: str = "large-v3"
    device: str = "cuda:0"
    download_root: str | None = "/home/duhu/.cache/whisper"
    language: str | None = None
    task: str = "transcribe"
    fp16: bool = True
    condition_on_previous_text: bool = False
    word_timestamps: bool = True
    num_workers: int = 1


class WhisperLarge:
    supports_batch: bool = False

    def __init__(self, config: WhisperLargeConfig | None = None):
        import whisper

        self.config = config or WhisperLargeConfig()
        self.model = whisper.load_model(
            self.config.model_name,
            device=self.config.device,
            download_root=self.config.download_root,
        )

    def transcribe(self, batch_uttid: Sequence[str], batch_wav: Sequence[tuple[int, Any]]) -> list[dict]:
        results = []
        for uttid, (sample_rate, wav) in zip(batch_uttid, batch_wav):
            wav_path = self._write_temp_wav(wav, sample_rate)
            try:
                raw_result = self.model.transcribe(
                    wav_path,
                    language=model_language("whisper_large", self.config.language) if self.config.language else None,
                    task=self.config.task,
                    fp16=self.config.fp16,
                    word_timestamps=self.config.word_timestamps,
                    condition_on_previous_text=self.config.condition_on_previous_text,
                )
            finally:
                os.unlink(wav_path)
            results.append({
                "uttid": uttid,
                "text": raw_result.get("text", "").strip(),
                "confidence": 0,
                "timestamp": self._normalize_timestamps(raw_result),
                "sample_rate": sample_rate,
            })
        return results

    @staticmethod
    def _write_temp_wav(wav: Any, sample_rate: int) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, wav, sample_rate)
        return tmp.name

    @staticmethod
    def _normalize_timestamps(raw_result: dict) -> list[list]:
        timestamps = []
        for segment in raw_result.get("segments", []):
            for word in segment.get("words", []):
                token = str(word.get("word", "")).strip()
                if token:
                    timestamps.append([token, float(word["start"]), float(word["end"])])
        return timestamps


class WhisperLargeTimestampProvider:
    def add_timestamps(self, batch_asr_result: Sequence[dict], batch_segments: Sequence[SpeechSegment]) -> list[dict]:
        for asr_result in batch_asr_result:
            if not asr_result.get("timestamp"):
                raise ValueError(f"Whisper large must return word timestamps for {asr_result.get('uttid')}")
        return list(batch_asr_result)
