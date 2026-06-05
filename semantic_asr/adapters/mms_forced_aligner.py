from dataclasses import dataclass
import re
from typing import Sequence

from semantic_asr.core import SpeechSegment
from semantic_asr.language_mapping import canonical_language_id, model_language
from semantic_asr.mms_runtime.aligner import MmsAligner


@dataclass
class MmsForcedAlignerConfig:
    model_path: str = "pretrained_models/mmsalign/model.pt"
    device: str = "cuda:0"
    language: str = "zh_cn"
    use_star: bool = True
    normalize_text: bool = False
    uroman_path: str = "uroman/bin"


class MmsForcedAlignerTimestampProvider:
    supports_batch: bool = False

    def __init__(self, config: MmsForcedAlignerConfig | None = None):
        self.config = config or MmsForcedAlignerConfig()
        self.config.language = canonical_language_id(self.config.language)
        model_language("mms_forced_aligner", self.config.language)
        self.aligner = MmsAligner(
            model_path=self.config.model_path,
            device=self.config.device,
            uroman_path=self.config.uroman_path,
        )

    def add_timestamps(self, batch_asr_result: Sequence[dict], batch_segments: Sequence[SpeechSegment]) -> list[dict]:
        results = []
        for asr_result, segment in zip(batch_asr_result, batch_segments):
            tokens = self._prepare_tokens(asr_result.get("text", ""))
            names = [f"{asr_result['uttid']}_{i}" for i in range(len(tokens))]
            aligned = self.aligner.align(
                tokens,
                segment.wav,
                segment.sample_rate,
                names,
                use_star=self.config.use_star,
                language=model_language("mms_forced_aligner", self.config.language),
                raw_transcripts=tokens,
            )

            timestamped = dict(asr_result)
            timestamped["timestamp"] = self._normalize_alignment(aligned)
            results.append(timestamped)
        return results

    def _prepare_tokens(self, text: str) -> list[str]:
        text = str(text).strip()
        if self.config.normalize_text:
            text = self._normalize_text(text)
        if self.config.language.startswith("zh"):
            text = re.sub(r"[\u4e00-\u9fa5]", lambda item: f" {item[0]} ", text)
        if self.config.language.startswith("ko"):
            text = re.sub(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]", lambda item: f" {item[0]} ", text)
        if self.config.language.startswith("ja"):
            text = re.sub(r"[\u3040-\u309f\u4E00-\u9FFF\u30a0-\u30ff]", lambda item: f" {item[0]} ", text)
        return [token for token in text.split() if token.strip()]

    def _normalize_text(self, text: str) -> str:
        from semantic_asr.mms_runtime.text_normalize import LANG2TEXTNORMALIZER

        normalizer_cls = LANG2TEXTNORMALIZER.get(self.config.language) or LANG2TEXTNORMALIZER.get("tricky")
        return normalizer_cls().norm(text)

    @staticmethod
    def _normalize_alignment(aligned: Sequence[dict]) -> list[list]:
        timestamps = []
        for item in aligned:
            token = str(item.get("text", item.get("clean_text", ""))).strip()
            if token:
                timestamps.append([token, float(item["start"]), float(item["end"])])
        return timestamps
