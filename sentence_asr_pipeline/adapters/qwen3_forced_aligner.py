from dataclasses import dataclass
from typing import Sequence

import torch

from sentence_asr_pipeline.compat import patch_torch_pytree_for_transformers
from sentence_asr_pipeline.core import SpeechSegment


@dataclass
class Qwen3ForcedAlignerConfig:
    model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    dtype: str = "bfloat16"
    device_map: str = "cuda:0"
    language: str = "Russian"
    batch_size: int = 4


class Qwen3ForcedAlignerTimestampProvider:
    supports_batch: bool = True

    def __init__(self, config: Qwen3ForcedAlignerConfig | None = None):
        patch_torch_pytree_for_transformers()
        from qwen_asr import Qwen3ForcedAligner

        self.config = config or Qwen3ForcedAlignerConfig()
        self.model = Qwen3ForcedAligner.from_pretrained(
            self.config.model,
            dtype=getattr(torch, self.config.dtype),
            device_map=self.config.device_map,
        )

    def add_timestamps(self, batch_asr_result: Sequence[dict], batch_segments: Sequence[SpeechSegment]) -> list[dict]:
        results = []
        for start in range(0, len(batch_asr_result), self.config.batch_size):
            asr_chunk = list(batch_asr_result[start:start + self.config.batch_size])
            segment_chunk = list(batch_segments[start:start + self.config.batch_size])
            aligned = self.model.align(
                audio=[(segment.wav, segment.sample_rate) for segment in segment_chunk],
                text=[asr_result.get("text", "") for asr_result in asr_chunk],
                language=[self.config.language] * len(asr_chunk),
            )
            for asr_result, align_result in zip(asr_chunk, aligned):
                timestamped = dict(asr_result)
                timestamped["timestamp"] = self._normalize_alignment(align_result)
                results.append(timestamped)
        return results

    @staticmethod
    def _normalize_alignment(align_result) -> list[list]:
        items = getattr(align_result, "items", align_result)
        timestamps = []
        for item in items:
            if isinstance(item, dict):
                text = item.get("text", "")
                start_s = item.get("start_time", item.get("start", 0))
                end_s = item.get("end_time", item.get("end", start_s))
            else:
                text = getattr(item, "text", "")
                start_s = getattr(item, "start_time", getattr(item, "start", 0))
                end_s = getattr(item, "end_time", getattr(item, "end", start_s))
            text = str(text).strip()
            if text:
                timestamps.append([text, float(start_s), float(end_s)])
        return timestamps
