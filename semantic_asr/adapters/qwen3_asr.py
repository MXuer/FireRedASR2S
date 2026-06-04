from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from semantic_asr.compat import patch_torch_pytree_for_transformers


@dataclass
class Qwen3AsrConfig:
    model: str = "Qwen/Qwen3-ASR-1.7B"
    forced_aligner: str | None = None
    dtype: str = "bfloat16"
    device_map: str = "cuda:0"
    max_inference_batch_size: int = 8
    max_new_tokens: int | None = 512
    language: str | None = None
    return_time_stamps: bool = False


class Qwen3Asr:
    supports_batch: bool = True

    def __init__(self, config: Qwen3AsrConfig | None = None):
        patch_torch_pytree_for_transformers()
        from qwen_asr import Qwen3ASRModel

        self.config = config or Qwen3AsrConfig()
        kwargs = {}
        if self.config.dtype:
            kwargs["dtype"] = getattr(torch, self.config.dtype)
        if self.config.device_map:
            kwargs["device_map"] = self.config.device_map
        self.model = Qwen3ASRModel.from_pretrained(
            self.config.model,
            forced_aligner=self.config.forced_aligner,
            max_inference_batch_size=self.config.max_inference_batch_size,
            max_new_tokens=self.config.max_new_tokens,
            **kwargs,
        )

    def transcribe(self, batch_uttid: Sequence[str], batch_wav: Sequence[tuple[int, Any]]) -> list[dict]:
        audio = [(_to_float32(wav), sample_rate) for sample_rate, wav in batch_wav]
        raw_results = self.model.transcribe(
            audio=audio,
            language=[self.config.language] * len(audio) if self.config.language else None,
            return_time_stamps=self.config.return_time_stamps,
        )
        if len(raw_results) != len(batch_uttid):
            raise ValueError(f"Qwen3-ASR returned {len(raw_results)} results for {len(batch_uttid)} inputs")
        return [
            {
                "uttid": uttid,
                "text": _get_text(raw_result),
                "confidence": 0,
                "timestamp": _normalize_timestamps(raw_result),
                "sample_rate": sample_rate,
            }
            for uttid, (sample_rate, _), raw_result in zip(batch_uttid, batch_wav, raw_results)
        ]


def _get_text(raw_result: Any) -> str:
    if isinstance(raw_result, dict):
        return str(raw_result.get("text", raw_result.get("transcription", ""))).strip()
    return str(getattr(raw_result, "text", getattr(raw_result, "transcription", ""))).strip()


def _normalize_timestamps(raw_result: Any) -> list[list]:
    if isinstance(raw_result, dict):
        raw_timestamps = raw_result.get("timestamps") or raw_result.get("timestamp") or []
    else:
        raw_timestamps = getattr(raw_result, "timestamps", getattr(raw_result, "timestamp", []))

    timestamps = []
    for item in raw_timestamps or []:
        if isinstance(item, dict):
            token = item.get("token", item.get("text", ""))
            start_s = item.get("start_time", item.get("start", 0.0))
            end_s = item.get("end_time", item.get("end", start_s))
        else:
            token = getattr(item, "token", getattr(item, "text", None))
            if token is None and isinstance(item, (list, tuple)) and len(item) >= 3:
                token, start_s, end_s = item[0], item[1], item[2]
            else:
                start_s = getattr(item, "start_time", getattr(item, "start", 0.0))
                end_s = getattr(item, "end_time", getattr(item, "end", start_s))
        token = str(token).strip()
        if token:
            timestamps.append([token, float(start_s), float(end_s)])
    return timestamps


def _to_float32(wav: Any) -> np.ndarray:
    array = np.asarray(wav)
    if np.issubdtype(array.dtype, np.integer):
        scale = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
        return array.astype(np.float32) / float(scale)
    return array.astype(np.float32, copy=False)
