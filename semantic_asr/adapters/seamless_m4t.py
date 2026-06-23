from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torchaudio.functional as audio_functional

from semantic_asr.compat import patch_torch_pytree_for_transformers
from semantic_asr.language_mapping import model_language


@dataclass
class SeamlessM4TConfig:
    model: str = "facebook/seamless-m4t-v2-large"
    device: str = "cuda:0"
    language: str = "en_us"
    target_language: str | None = None
    task: str = "transcribe"
    max_new_tokens: int = 256
    batch_size: int = 128


class SeamlessM4TAsr:
    supports_batch: bool = True

    def __init__(self, config: SeamlessM4TConfig | None = None):
        patch_torch_pytree_for_transformers()
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        self.config = config or SeamlessM4TConfig()
        self.processor = AutoProcessor.from_pretrained(self.config.model)
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(self.config.model).to(self.config.device)
        self._set_recommended_batch_size()

    def _set_recommended_batch_size(self) -> None:
        self.recommended_batch_size = max(1, int(self.config.batch_size))

    def transcribe(self, batch_uttid: Sequence[str], batch_wav: Sequence[tuple[int, Any]]) -> list[dict]:
        raw_results = self._decode_batch(batch_wav)
        if len(raw_results) != len(batch_uttid):
            raise ValueError(f"Seamless returned {len(raw_results)} results for {len(batch_uttid)} inputs")
        return [
            {
                "uttid": uttid,
                "text": raw_result.strip(),
                "confidence": 0,
                "timestamp": [],
                "sample_rate": sample_rate,
            }
            for uttid, (sample_rate, _wav), raw_result in zip(batch_uttid, batch_wav, raw_results)
        ]

    def _decode_batch(self, batch_wav: Sequence[tuple[int, Any]]) -> list[str]:
        audios = [_to_16k_float32_mono(wav, sample_rate) for sample_rate, wav in batch_wav]
        inputs = self.processor(
            audios=audios,
            sampling_rate=16000,
            src_lang=model_language("seamless_m4t_v2_large", self.config.language),
            return_tensors="pt",
        )
        inputs = {key: value.to(self.config.device) for key, value in inputs.items()}
        kwargs = {"max_new_tokens": self.config.max_new_tokens}
        target_language = self.config.target_language or self.config.language
        if target_language:
            kwargs["tgt_lang"] = model_language("seamless_m4t_v2_large", target_language)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **kwargs)
        return [text.strip() for text in self.processor.batch_decode(output_ids, skip_special_tokens=True)]


def _to_16k_float32_mono(wav: Any, sample_rate: int) -> np.ndarray:
    array = np.asarray(wav)
    if array.ndim > 1:
        array = array.mean(axis=1)
    if np.issubdtype(array.dtype, np.integer):
        scale = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
        array = array.astype(np.float32) / float(scale)
    else:
        array = array.astype(np.float32, copy=False)
    if sample_rate != 16000:
        array = audio_functional.resample(torch.from_numpy(array), sample_rate, 16000).numpy()
    return array.astype(np.float32, copy=False)
