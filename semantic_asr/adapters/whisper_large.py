from dataclasses import dataclass
import os
import tempfile
from typing import Any, Sequence

import soundfile as sf

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
    temperature: float | None = None
    beam_size: int | None = None
    best_of: int | None = None
    patience: float | None = None
    length_penalty: float | None = None
    short_audio_threshold_s: float = 1.0
    short_temperature: float | None = 0.0
    short_beam_size: int | None = 5
    short_length_penalty: float | None = 0.0
    batch_size: int = 96
    num_workers: int = 1


class WhisperLarge:
    supports_batch: bool = True

    def __init__(self, config: WhisperLargeConfig | None = None):
        import whisper

        self.config = config or WhisperLargeConfig()
        self.model = whisper.load_model(
            self.config.model_name,
            device=self.config.device,
            download_root=self.config.download_root,
        )
        self._set_recommended_batch_size()

    def _set_recommended_batch_size(self) -> None:
        self.recommended_batch_size = max(1, int(self.config.batch_size))

    def transcribe(self, batch_uttid: Sequence[str], batch_wav: Sequence[tuple[int, Any]]) -> list[dict]:
        decoded = self._decode_batch(batch_wav)
        results = []
        for uttid, (sample_rate, _wav), raw_result in zip(batch_uttid, batch_wav, decoded):
            results.append({
                "uttid": uttid,
                "text": str(getattr(raw_result, "text", "")).strip(),
                "confidence": 0,
                "timestamp": [],
                "sample_rate": sample_rate,
                "asr_metadata": self._metadata(raw_result),
            })
        return results

    def _decode_batch(self, batch_wav: Sequence[tuple[int, Any]]) -> list[Any]:
        import torch
        from whisper import DecodingOptions

        prepared = [(sample_rate, wav, self._prepare_mel(wav, sample_rate)) for sample_rate, wav in batch_wav]
        results: list[Any] = [None] * len(prepared)
        groups: dict[tuple[tuple[str, Any], ...], list[tuple[int, Any]]] = {}
        for index, (sample_rate, wav, mel) in enumerate(prepared):
            key = tuple(sorted(self._decode_kwargs(sample_rate, wav).items()))
            groups.setdefault(key, []).append((index, mel))

        for key, items in groups.items():
            decode_chunks = [[item] for item in items] if dict(key).get("beam_size") is not None else [items]
            options = DecodingOptions(
                language=model_language("whisper_large", self.config.language) if self.config.language else None,
                task=self.config.task,
                fp16=self.config.fp16,
                **dict(key),
            )
            for chunk in decode_chunks:
                mel_batch = torch.stack([torch.as_tensor(mel) for _index, mel in chunk]).to(self.model.device)
                decoded = self.model.decode(mel_batch, options)
                for (index, _mel), raw_result in zip(chunk, decoded):
                    results[index] = raw_result
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

    def _prepare_mel(self, wav: Any, sample_rate: int):
        import whisper
        from whisper.audio import N_FRAMES, N_SAMPLES

        wav_path = self._write_temp_wav(wav, sample_rate)
        try:
            return whisper.pad_or_trim(
                whisper.log_mel_spectrogram(
                    wav_path,
                    n_mels=self.model.dims.n_mels,
                    padding=N_SAMPLES,
                ),
                length=N_FRAMES,
            )
        finally:
            os.unlink(wav_path)

    @staticmethod
    def _metadata(raw_result: Any) -> dict:
        return {
            "avg_logprob": getattr(raw_result, "avg_logprob", None),
            "no_speech_prob": getattr(raw_result, "no_speech_prob", None),
            "compression_ratio": getattr(raw_result, "compression_ratio", None),
        }


def _duration_s(sample_rate: int, wav: Any) -> float:
    try:
        return len(wav) / float(sample_rate)
    except TypeError:
        return 0.0
