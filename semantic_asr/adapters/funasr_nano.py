from dataclasses import dataclass, field
import os
import re
import sys
import tempfile
from typing import Any, Sequence

import soundfile as sf

from semantic_asr.core import SpeechSegment
from semantic_asr.punctuation import strip_timestamp_punctuation


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
    preserve_punctuation: bool = False


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
        wav_paths = [self._write_temp_wav(wav, sample_rate) for sample_rate, wav in batch_wav]
        try:
            raw_results = self._generate(wav_paths)
        finally:
            for wav_path in wav_paths:
                os.unlink(wav_path)

        if len(batch_uttid) == 1 and isinstance(raw_results, dict):
            raw_results = [raw_results]
        if len(raw_results) != len(batch_uttid):
            raise ValueError(f"Fun-ASR-Nano returned {len(raw_results)} results for {len(batch_uttid)} inputs")

        results = []
        for uttid, (sample_rate, _), raw_result in zip(batch_uttid, batch_wav, raw_results):
            results.append({
                "uttid": uttid,
                "text": raw_result.get("text", "").strip(),
                "confidence": 0,
                "timestamp": self._normalize_timestamps(raw_result, self.config.preserve_punctuation),
                "sample_rate": sample_rate,
            })
        return results

    def _generate(self, wav_paths: Sequence[str]):
        generate_kwargs = {
            "cache": {},
            "batch_size": self.config.batch_size,
            "language": self.config.language,
            "hotwords": self.config.hotwords,
            "itn": self.config.itn,
        }
        if len(wav_paths) == 1:
            return self.model.generate(input=wav_paths[0], **generate_kwargs)
        try:
            return self.model.generate(input=list(wav_paths), **generate_kwargs)
        except NotImplementedError as error:
            if "batch decoding is not implemented" not in str(error):
                raise
            results = []
            for wav_path in wav_paths:
                single_kwargs = dict(generate_kwargs)
                single_kwargs["cache"] = {}
                single_kwargs["batch_size"] = 1
                results.append(_single_result(self.model.generate(input=wav_path, **single_kwargs)))
            return results

    @staticmethod
    def _write_temp_wav(wav: Any, sample_rate: int) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, wav, sample_rate)
        return tmp.name

    @staticmethod
    def _normalize_timestamps(raw_result: dict, preserve_punctuation: bool = False) -> list[list]:
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
            if preserve_punctuation:
                normalized.append([str(token), float(start), float(end)])
            elif token and not re.fullmatch(r"[^\w\u4e00-\u9fff]+", token):
                normalized.append([str(token), float(start), float(end)])
        if not preserve_punctuation:
            return strip_timestamp_punctuation(normalized)
        return normalized


class FunAsrNanoTimestampProvider:
    def add_timestamps(self, batch_asr_result: Sequence[dict], batch_segments: Sequence[SpeechSegment]) -> list[dict]:
        for asr_result in batch_asr_result:
            if not asr_result.get("timestamp"):
                raise ValueError(f"Fun-ASR-Nano must return timestamp for {asr_result.get('uttid')}")
        return list(batch_asr_result)


def _single_result(result):
    if isinstance(result, list) and len(result) == 1:
        return result[0]
    return result
