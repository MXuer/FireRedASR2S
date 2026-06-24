from dataclasses import dataclass
import os
from typing import Any, Sequence

import numpy as np

from semantic_asr.compat import patch_torch_pytree_for_transformers
from semantic_asr.language_mapping import model_language


@dataclass
class HfWhisperAsrConfig:
    model_name_or_path: str
    language: str | None = None
    task: str = "transcribe"
    device: str = "cuda:0"
    torch_dtype: str = "float16"
    batch_size: int = 64
    max_new_tokens: int | None = None


class HfWhisperAsr:
    supports_batch = True

    def __init__(self, config: HfWhisperAsrConfig):
        patch_torch_pytree_for_transformers()
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.config = config
        self.torch = torch
        dtype = getattr(torch, config.torch_dtype) if config.torch_dtype else None
        model_path = os.path.expanduser(config.model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(config.device)
        self.model.eval()
        self.recommended_batch_size = max(1, int(config.batch_size))

    def transcribe(self, batch_uttid: Sequence[str], batch_wav: Sequence[tuple[int, Any]]) -> list[dict]:
        audios = [_to_16k_float32(sample_rate, wav) for sample_rate, wav in batch_wav]
        inputs = self.processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).to(self.config.device)
        kwargs = {}
        if self.config.max_new_tokens is not None:
            kwargs["max_new_tokens"] = self.config.max_new_tokens
        forced_decoder_ids = self._forced_decoder_ids()
        if forced_decoder_ids is not None:
            kwargs["forced_decoder_ids"] = forced_decoder_ids

        with self.torch.no_grad():
            generated = self.model.generate(**inputs, **kwargs)
        texts = self.processor.batch_decode(generated, skip_special_tokens=True)
        return [
            {
                "uttid": uttid,
                "text": text.strip(),
                "confidence": 0,
                "timestamp": [],
                "sample_rate": sample_rate,
            }
            for uttid, (sample_rate, _wav), text in zip(batch_uttid, batch_wav, texts)
        ]

    def _forced_decoder_ids(self):
        if not self.config.language:
            return None
        getter = getattr(self.processor, "get_decoder_prompt_ids", None)
        if getter is None:
            return None
        return getter(language=model_language(self._mapping_name(), self.config.language), task=self.config.task)

    def _mapping_name(self) -> str:
        model = self.config.model_name_or_path.lower()
        if "phowhisper" in model:
            return "phowhisper_large"
        if "whisper-th" in model or "biodatlab" in model:
            return "whisper_th_large_v3_combined"
        return "whisper_large"


def _to_16k_float32(sample_rate: int, wav: Any) -> np.ndarray:
    array = np.asarray(wav, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1)
    if np.issubdtype(np.asarray(wav).dtype, np.integer):
        array = array / np.iinfo(np.asarray(wav).dtype).max
    if sample_rate == 16000:
        return array

    import torch
    import torchaudio.functional as F

    tensor = torch.as_tensor(array).unsqueeze(0)
    return F.resample(tensor, sample_rate, 16000).squeeze(0).numpy().astype(np.float32)
