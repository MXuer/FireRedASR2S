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
    temperature: float | None = None
    beam_size: int | None = None
    best_of: int | None = None
    patience: float | None = None
    length_penalty: float | None = None
    short_audio_threshold_s: float = 1.0
    short_temperature: float | None = 0.0
    short_beam_size: int | None = 5
    short_length_penalty: float | None = 0.0
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
                decode_kwargs = self._decode_kwargs(sample_rate, wav)
                raw_result = self.model.transcribe(
                    wav_path,
                    language=model_language("whisper_large", self.config.language) if self.config.language else None,
                    task=self.config.task,
                    fp16=self.config.fp16,
                    word_timestamps=self.config.word_timestamps,
                    condition_on_previous_text=self.config.condition_on_previous_text,
                    **decode_kwargs,
                )
            finally:
                os.unlink(wav_path)
            results.append({
                "uttid": uttid,
                "text": raw_result.get("text", "").strip(),
                "confidence": 0,
                "timestamp": self._normalize_timestamps(raw_result),
                "sample_rate": sample_rate,
                "asr_metadata": self._metadata(raw_result),
            })
        return results

    def _decode_kwargs(self, sample_rate: int, wav: Any) -> dict:
        kwargs = {
            "temperature": self.config.temperature,
            "beam_size": self.config.beam_size,
            "best_of": self.config.best_of,
            "patience": self.config.patience,
            "length_penalty": self.config.length_penalty,
        }
        duration_s = _duration_s(sample_rate, wav)
        if duration_s <= self.config.short_audio_threshold_s:
            kwargs.update({
                "temperature": self.config.short_temperature,
                "beam_size": self.config.short_beam_size,
                "length_penalty": self.config.short_length_penalty,
            })
            if self.config.short_beam_size is not None:
                kwargs["best_of"] = None
        return {key: value for key, value in kwargs.items() if value is not None}

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

    @staticmethod
    def _metadata(raw_result: dict) -> dict:
        segments = raw_result.get("segments", [])
        return {
            "segments": [
                {
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "avg_logprob": segment.get("avg_logprob"),
                    "no_speech_prob": segment.get("no_speech_prob"),
                    "compression_ratio": segment.get("compression_ratio"),
                }
                for segment in segments
            ],
        }


def _duration_s(sample_rate: int, wav: Any) -> float:
    try:
        return len(wav) / float(sample_rate)
    except TypeError:
        return 0.0


class WhisperLargeTimestampProvider:
    def add_timestamps(self, batch_asr_result: Sequence[dict], batch_segments: Sequence[SpeechSegment]) -> list[dict]:
        for asr_result in batch_asr_result:
            if not asr_result.get("timestamp"):
                raise ValueError(f"Whisper large must return word timestamps for {asr_result.get('uttid')}")
        return list(batch_asr_result)
