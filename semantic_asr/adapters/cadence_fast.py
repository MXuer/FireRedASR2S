from dataclasses import dataclass
import os
from typing import Sequence

from semantic_asr.punctuation import split_text_by_punctuation


@dataclass
class CadenceFastConfig:
    model: str = "Cadence-Fast"
    model_path: str | None = None
    device: str = "cuda"
    max_length: int = 300
    sliding_window: bool = True
    batch_size: int = 32


class CadenceFastPunctuation:
    def __init__(self, config: CadenceFastConfig | None = None):
        self.config = config or CadenceFastConfig()
        try:
            from Cadence import PunctuationModel
        except ImportError as exc:
            raise ImportError(
                "Cadence-Fast requires the cadence-punctuation package; install it in the active model environment."
            ) from exc

        self.model = PunctuationModel(
            model=self.config.model,
            model_path=os.path.expanduser(self.config.model_path) if self.config.model_path else None,
            cpu=self.config.device == "cpu",
            gpu_id=_gpu_id(self.config.device),
            max_length=self.config.max_length,
            sliding_window=self.config.sliding_window,
        )

    def process_with_timestamp(self, batch_timestamp: Sequence[Sequence[Sequence]], batch_uttid: Sequence[str]):
        texts = [_timestamp_text(timestamp) for timestamp in batch_timestamp]
        outputs = self.model.punctuate(texts, batch_size=self.config.batch_size)
        return [
            {
                "uttid": uttid,
                "punc_sentences": split_text_by_punctuation(_output_text(output), timestamp),
            }
            for uttid, timestamp, output in zip(batch_uttid, batch_timestamp, outputs)
        ]


def _timestamp_text(timestamp: Sequence[Sequence]) -> str:
    return " ".join(str(item[0]).strip() for item in timestamp if str(item[0]).strip())


def _output_text(output) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        return str(output.get("text") or output.get("punctuated_text") or "")
    return str(output)


def _gpu_id(device: str) -> int | None:
    if device == "cpu":
        return None
    if ":" in device:
        return int(device.rsplit(":", 1)[1])
    return 0
