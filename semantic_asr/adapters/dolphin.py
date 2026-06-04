from dataclasses import dataclass
import os
import tempfile
from typing import Any, Sequence

import soundfile as sf

from semantic_asr.compat import patch_torch_pytree_for_transformers
from semantic_asr.language_mapping import dolphin_language


@dataclass
class DolphinAsrConfig:
    model_name: str = "small"
    device: str = "cuda:0"
    language: str = "zh_cn"
    word_timestamp: bool = True
    decoding_method: str = "attention_rescoring"
    beam_size: int = 10


class DolphinAsr:
    supports_batch: bool = False

    def __init__(self, config: DolphinAsrConfig | None = None):
        patch_torch_pytree_for_transformers()
        import dolphin

        self.dolphin = dolphin
        self.config = config or DolphinAsrConfig()
        self.model = dolphin.load_model(
            self.config.model_name,
            device=self.config.device,
        )

    def transcribe(self, batch_uttid: Sequence[str], batch_wav: Sequence[tuple[int, Any]]) -> list[dict]:
        results = []
        lang_sym, region_sym = dolphin_language(self.config.language)
        for uttid, (sample_rate, wav) in zip(batch_uttid, batch_wav):
            wav_path = self._write_temp_wav(wav, sample_rate)
            try:
                raw_result = self.dolphin.transcribe(
                    self.model,
                    wav_path,
                    lang_sym=lang_sym,
                    region_sym=region_sym,
                    word_timestamp=self.config.word_timestamp,
                    decoding_method=self.config.decoding_method,
                    beam_size=self.config.beam_size,
                )
            finally:
                os.unlink(wav_path)
            results.append({
                "uttid": uttid,
                "text": _get_text(raw_result),
                "confidence": 0,
                "timestamp": _normalize_timestamps(raw_result),
                "sample_rate": sample_rate,
            })
        return results

    @staticmethod
    def _write_temp_wav(wav: Any, sample_rate: int) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, wav, sample_rate)
        return tmp.name


def _get_text(raw_result: Any) -> str:
    if isinstance(raw_result, dict):
        return str(raw_result.get("text_nospecial", raw_result.get("text", raw_result.get("transcription", "")))).strip()
    return str(getattr(raw_result, "text_nospecial", getattr(raw_result, "text", getattr(raw_result, "transcription", "")))).strip()


def _normalize_timestamps(raw_result: Any) -> list[list]:
    if isinstance(raw_result, dict):
        raw_timestamps = raw_result.get("word_timestamps") or raw_result.get("timestamps") or raw_result.get("timestamp") or raw_result.get("words") or []
    else:
        raw_timestamps = getattr(raw_result, "word_timestamps", getattr(raw_result, "timestamps", getattr(raw_result, "timestamp", getattr(raw_result, "words", []))))

    timestamps = []
    for item in raw_timestamps or []:
        if isinstance(item, dict):
            token = item.get("token", item.get("word", item.get("text", "")))
            start_s = item.get("start_time", item.get("start", 0.0))
            end_s = item.get("end_time", item.get("end", start_s))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            token, start_s, end_s = item[0], item[1], item[2]
        else:
            token = getattr(item, "token", getattr(item, "word", getattr(item, "text", "")))
            start_s = getattr(item, "start_time", getattr(item, "start", 0.0))
            end_s = getattr(item, "end_time", getattr(item, "end", start_s))
        token = str(token).strip()
        if token:
            timestamps.append([token, float(start_s), float(end_s)])
    return timestamps
