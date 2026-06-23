from dataclasses import dataclass
import os
import sys
import tempfile
from typing import Any, Sequence

import soundfile as sf


@dataclass
class GigaAmV3Config:
    language: str = "ru_ru"
    model_name: str = "v3_e2e_rnnt"
    model_dir: str = "pretrained_models/gigaam_v3"
    repo_dir: str | None = None
    device: str = "cuda:0"
    batch_size: int = 192
    num_workers: int = 0
    model_workers: int = 1
    fp16_encoder: bool = True
    use_flash: bool | None = False
    disable_torch_sdpa: bool = True


class GigaAmV3Asr:
    supports_batch: bool = True

    def __init__(self, config: GigaAmV3Config | None = None):
        self.config = config or GigaAmV3Config()
        if self.config.repo_dir and self.config.repo_dir not in sys.path:
            sys.path.insert(0, self.config.repo_dir)
        import gigaam

        self.model = gigaam.load_model(
            self.config.model_name,
            fp16_encoder=self.config.fp16_encoder,
            use_flash=self.config.use_flash,
            device=self.config.device,
            download_root=self.config.model_dir,
        )
        if self.config.disable_torch_sdpa:
            _disable_encoder_sdpa(self.model)
        self._set_recommended_batch_size()

    def _set_recommended_batch_size(self) -> None:
        self.recommended_batch_size = max(1, int(self.config.batch_size))

    def transcribe(self, batch_uttid: Sequence[str], batch_wav: Sequence[tuple[int, Any]]) -> list[dict]:
        wav_paths = []
        try:
            for sample_rate, wav in batch_wav:
                wav_paths.append(_write_temp_wav(wav, sample_rate))
            decoded = self._transcribe_paths(wav_paths)
        finally:
            for path in wav_paths:
                if os.path.exists(path):
                    os.unlink(path)

        return [
            {
                "uttid": uttid,
                "text": text.strip(),
                "confidence": 0,
                "timestamp": timestamps,
                "sample_rate": sample_rate,
            }
            for uttid, (sample_rate, _wav), (text, timestamps) in zip(batch_uttid, batch_wav, decoded)
        ]

    def _transcribe_paths(self, wav_paths: Sequence[str]) -> list[tuple[str, list[list]]]:
        if self.config.repo_dir and self.config.repo_dir not in sys.path:
            sys.path.insert(0, self.config.repo_dir)
        import torch
        from torch.utils.data import DataLoader
        from gigaam.utils import AudioDataset

        dataset = AudioDataset(list(wav_paths), tokenizer=None)
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(self.config.batch_size)),
            shuffle=False,
            collate_fn=AudioDataset.collate,
            num_workers=max(0, int(self.config.num_workers)),
        )
        results = []
        with torch.inference_mode():
            for wav_pad, wav_lens in loader:
                wav_pad = wav_pad.to(self.model._device).to(self.model._dtype)
                wav_lens = wav_lens.to(self.model._device)
                encoded, encoded_len = self.model.forward(wav_pad, wav_lens)
                for text, words in self.model._decode(encoded, encoded_len, wav_lens, word_timestamps=True):
                    results.append((str(text or ""), _normalize_words(words or [])))
        return results


class GigaAmV3TimestampProvider:
    def add_timestamps(self, batch_asr_result: Sequence[dict], batch_segments: Sequence[Any]) -> list[dict]:
        for asr_result in batch_asr_result:
            if not asr_result.get("timestamp"):
                raise ValueError(f"GigaAM-v3 must return word timestamps for {asr_result.get('uttid')}")
        return list(batch_asr_result)


def _normalize_words(words: Sequence[Any]) -> list[list]:
    timestamps = []
    for word in words:
        text = str(getattr(word, "text", "")).strip()
        if text:
            timestamps.append([text, float(word.start), float(word.end)])
    return timestamps


def _disable_encoder_sdpa(model: Any) -> None:
    for layer in getattr(getattr(model, "encoder", None), "layers", []):
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is not None and hasattr(self_attn, "torch_sdpa_attn"):
            self_attn.torch_sdpa_attn = False


def _write_temp_wav(wav: Any, sample_rate: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, wav, sample_rate)
    return tmp.name
