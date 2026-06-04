from dataclasses import dataclass
import os
import tempfile
from typing import Any, Sequence

import soundfile as sf
import torch
import torchaudio.functional as audio_functional

from semantic_asr.compat import patch_torch_pytree_for_transformers


@dataclass
class SeamlessM4TConfig:
    model: str = "facebook/seamless-m4t-v2-large"
    device: str = "cuda:0"
    src_lang: str = "eng"
    tgt_lang: str | None = None
    task: str = "transcribe"
    max_new_tokens: int = 256


class SeamlessM4TAsr:
    supports_batch: bool = False

    def __init__(self, config: SeamlessM4TConfig | None = None):
        patch_torch_pytree_for_transformers()
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        self.config = config or SeamlessM4TConfig()
        self.processor = AutoProcessor.from_pretrained(self.config.model)
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(self.config.model).to(self.config.device)

    def transcribe(self, batch_uttid: Sequence[str], batch_wav: Sequence[tuple[int, Any]]) -> list[dict]:
        results = []
        for uttid, (sample_rate, wav) in zip(batch_uttid, batch_wav):
            wav_path = self._write_temp_wav(wav, sample_rate)
            try:
                raw_result = self._transcribe_file(wav_path)
            finally:
                os.unlink(wav_path)
            results.append({
                "uttid": uttid,
                "text": raw_result.strip(),
                "confidence": 0,
                "timestamp": [],
                "sample_rate": sample_rate,
            })
        return results

    def _transcribe_file(self, wav_path: str) -> str:
        wav, sample_rate = sf.read(wav_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sample_rate != 16000:
            wav = audio_functional.resample(torch.from_numpy(wav), sample_rate, 16000).numpy()
            sample_rate = 16000
        inputs = self.processor(
            audios=wav,
            sampling_rate=sample_rate,
            src_lang=self.config.src_lang,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.config.device) for key, value in inputs.items()}
        kwargs = {"max_new_tokens": self.config.max_new_tokens}
        target_language = self.config.tgt_lang or self.config.src_lang
        if target_language:
            kwargs["tgt_lang"] = target_language
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **kwargs)
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    @staticmethod
    def _write_temp_wav(wav: Any, sample_rate: int) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, wav, sample_rate)
        return tmp.name
