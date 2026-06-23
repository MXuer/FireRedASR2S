from dataclasses import dataclass
import os
from typing import Any, Sequence

import numpy as np
import torch
import torchaudio.functional as audio_functional

from semantic_asr.compat import patch_torch_pytree_for_transformers
from semantic_asr.language_mapping import dolphin_language


@dataclass
class DolphinAsrConfig:
    model_name: str = "small"
    model_dir: str | None = None
    device: str = "cuda:0"
    language: str = "zh_cn"
    word_timestamp: bool = True
    decoding_method: str = "attention_rescoring"
    beam_size: int = 10
    batch_size: int = 16


class DolphinAsr:
    supports_batch: bool = True

    def __init__(self, config: DolphinAsrConfig | None = None):
        patch_torch_pytree_for_transformers()
        from dolphin.transcribe import init_tokenizer, load_model

        self.config = config or DolphinAsrConfig()
        model_dir = self.config.model_dir or os.path.expanduser(f"~/.cache/dolphin/{self.config.model_name}")
        self.model = load_model(
            self.config.model_name,
            model_dir,
            self.config.device,
        )
        self.tokenizer = init_tokenizer(self.model.model_configs)
        self._set_recommended_batch_size()

    def _set_recommended_batch_size(self) -> None:
        self.recommended_batch_size = max(1, int(self.config.batch_size))

    def transcribe(self, batch_uttid: Sequence[str], batch_wav: Sequence[tuple[int, Any]]) -> list[dict]:
        raw_results = self._decode_batch(batch_wav)
        if len(raw_results) != len(batch_uttid):
            raise ValueError(f"Dolphin returned {len(raw_results)} results for {len(batch_uttid)} inputs")
        return [
            {
                "uttid": uttid,
                "text": _get_text(raw_result),
                "confidence": 0,
                "timestamp": _normalize_timestamps(raw_result),
                "sample_rate": sample_rate,
            }
            for uttid, (sample_rate, _wav), raw_result in zip(batch_uttid, batch_wav, raw_results)
        ]

    def _decode_batch(self, batch_wav: Sequence[tuple[int, Any]]) -> list[Any]:
        from dolphin.processor import extract_feats

        lang_sym, region_sym = dolphin_language(self.config.language)
        waveforms = [
            torch.from_numpy(_to_16k_float32_mono(wav, sample_rate)).float().unsqueeze(0)
            for sample_rate, wav in batch_wav
        ]
        batch = extract_feats(waveforms, self.model.model_configs)
        batch["feats"] = batch["feats"].to(self.model.device)
        batch["feats_lengths"] = batch["feats_lengths"].to(self.model.device)
        ret = self.model.decode(
            methods=[self.config.decoding_method],
            speech=batch["feats"],
            speech_lengths=batch["feats_lengths"],
            beam_size=self.config.beam_size,
            infos={
                "tokenizer": self.tokenizer,
                "langs": [f"<{lang_sym}>"] * len(batch_wav),
                "regions": [f"<{region_sym}>"] * len(batch_wav),
                "need_timestamp": self.config.word_timestamp,
            },
        )
        decoded = ret[self.config.decoding_method]
        return [_dolphin_decode_result(item, self.tokenizer) for item in decoded]


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


def _dolphin_decode_result(raw_result: Any, tokenizer: Any) -> dict:
    text = ""
    tokens = getattr(raw_result, "tokens", None)
    if tokens is not None:
        text = str(tokenizer.detokenize(tokens)[0]).strip()
    timestamps = (
        getattr(raw_result, "word_timestamps", None)
        or getattr(raw_result, "timestamps", None)
        or getattr(raw_result, "timestamp", None)
        or []
    )
    return {
        "text": text,
        "text_nospecial": text,
        "word_timestamps": timestamps,
    }


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
