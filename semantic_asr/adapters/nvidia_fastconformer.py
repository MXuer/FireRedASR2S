from dataclasses import dataclass
import os
import tempfile
from typing import Any, Sequence

import soundfile as sf


@dataclass
class NvidiaFastConformerConfig:
    model_name: str = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
    device: str = "cuda:0"
    batch_size: int = 8
    timestamps: bool = True


class NvidiaFastConformerAsr:
    supports_batch: bool = True

    def __init__(self, config: NvidiaFastConformerConfig | None = None):
        self.config = config or NvidiaFastConformerConfig()
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(self.config.model_name)
        self.model.to(self.config.device)
        self.model.eval()
        self._set_recommended_batch_size()

    def _set_recommended_batch_size(self) -> None:
        self.recommended_batch_size = max(1, int(self.config.batch_size))

    def transcribe(self, batch_uttid: Sequence[str], batch_wav: Sequence[tuple[int, Any]]) -> list[dict]:
        wav_paths = []
        try:
            for sample_rate, wav in batch_wav:
                wav_paths.append(_write_temp_wav(wav, sample_rate))
            raw_results = self.model.transcribe(
                wav_paths,
                batch_size=self.recommended_batch_size,
                timestamps=self.config.timestamps,
            )
        finally:
            for path in wav_paths:
                if os.path.exists(path):
                    os.unlink(path)

        return [
            {
                "uttid": uttid,
                "text": _extract_text(raw_result).strip(),
                "confidence": 0,
                "timestamp": _extract_word_timestamps(raw_result),
                "sample_rate": sample_rate,
            }
            for uttid, (sample_rate, _wav), raw_result in zip(batch_uttid, batch_wav, raw_results)
        ]


def _extract_text(raw_result: Any) -> str:
    if hasattr(raw_result, "text"):
        return str(raw_result.text)
    return str(raw_result or "")


def _extract_word_timestamps(raw_result: Any) -> list[list]:
    timestamp = getattr(raw_result, "timestamp", None)
    if not isinstance(timestamp, dict):
        return []
    words = timestamp.get("word") or []
    return [
        [str(item["word"]).strip(), float(item["start"]), float(item["end"])]
        for item in words
        if str(item.get("word", "")).strip()
    ]


class NvidiaFastConformerTimestampProvider:
    def add_timestamps(self, batch_asr_result: Sequence[dict], batch_segments: Sequence[Any]) -> list[dict]:
        for asr_result in batch_asr_result:
            if not asr_result.get("timestamp"):
                raise ValueError(f"NVIDIA FastConformer must return word timestamps for {asr_result.get('uttid')}")
        return list(batch_asr_result)


def _write_temp_wav(wav: Any, sample_rate: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, wav, sample_rate)
    return tmp.name
