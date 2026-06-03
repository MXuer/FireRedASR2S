from dataclasses import dataclass, field
import os
import re
import sys
import tempfile
from typing import Any, Sequence

import soundfile as sf

from sentence_asr_pipeline.core import SpeechSegment


@dataclass
class FunAsrNanoConfig:
    model: str = "FunAudioLLM/Fun-ASR-Nano-2512"
    device: str = "cuda:0"
    hub: str = "ms"
    language: str = "English"
    batch_size: int = 1
    trust_remote_code: bool = True
    remote_code: str | None = None
    hotwords: list[str] = field(default_factory=list)
    itn: bool = True
    disable_pbar: bool = True


class FunAsrNano:
    def __init__(self, config: FunAsrNanoConfig | None = None):
        from funasr import AutoModel

        self.config = config or FunAsrNanoConfig()
        remote_code = self.config.remote_code or self._default_remote_code()
        self.model = AutoModel(
            model=self.config.model,
            trust_remote_code=self.config.trust_remote_code,
            remote_code=remote_code,
            device=self.config.device,
            hub=self.config.hub,
            disable_pbar=self.config.disable_pbar,
        )

    @staticmethod
    def _default_remote_code() -> str:
        import funasr.models.fun_asr_nano as fun_asr_nano

        model_dir = os.path.dirname(fun_asr_nano.__file__)
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        return os.path.join(model_dir, "model.py")

    def transcribe(self, batch_uttid: Sequence[str], batch_wav: Sequence[tuple[int, Any]]) -> list[dict]:
        results = []
        for uttid, (sample_rate, wav) in zip(batch_uttid, batch_wav):
            wav_path = self._write_temp_wav(wav, sample_rate)
            try:
                raw_results = self.model.generate(
                    input=wav_path,
                    cache={},
                    batch_size=self.config.batch_size,
                    language=self.config.language,
                    hotwords=self.config.hotwords,
                    itn=self.config.itn,
                )
                raw_result = raw_results[0] if raw_results else {}
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
        timestamps = raw_result.get("timestamps") or raw_result.get("timestamp") or []
        normalized = []
        for item in timestamps:
            if isinstance(item, dict):
                token = item.get("token", "")
                start = item.get("start_time", item.get("start", 0))
                end = item.get("end_time", item.get("end", start))
            else:
                token, start, end = item[0], item[1], item[2]
            token = str(token).strip().lower()
            if token and not re.fullmatch(r"[^\w\u4e00-\u9fff]+", token):
                normalized.append([str(token), float(start), float(end)])
        return normalized


class FunAsrNanoTimestampProvider:
    def add_timestamps(self, batch_asr_result: Sequence[dict], batch_segments: Sequence[SpeechSegment]) -> list[dict]:
        for asr_result in batch_asr_result:
            if not asr_result.get("timestamp"):
                raise ValueError(f"Fun-ASR-Nano must return timestamp for {asr_result.get('uttid')}")
        return list(batch_asr_result)
