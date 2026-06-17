from dataclasses import dataclass
import re
from typing import Sequence

from semantic_asr.punctuation import split_text_by_punctuation


_ASCII_WORD = re.compile(r"[A-Za-z0-9#]")


@dataclass
class CtPuncConfig:
    model: str = "ct-punc"
    disable_update: bool = True
    batch_size: int = 1


class CtPunc:
    supports_batch: bool = True

    def __init__(self, config: CtPuncConfig | None = None):
        self.config = config or CtPuncConfig()
        from funasr import AutoModel

        self.model = AutoModel(model=self.config.model, disable_update=self.config.disable_update)

    def process_with_timestamp(self, batch_timestamp: Sequence[list], batch_uttid: Sequence[str]) -> list[dict]:
        texts = [_timestamp_to_text(timestamp) for timestamp in batch_timestamp]
        raw_outputs = self.model.generate(input=texts, batch_size=max(1, int(self.config.batch_size)))
        if isinstance(raw_outputs, dict):
            raw_outputs = [raw_outputs]
        return [
            {
                "uttid": uttid,
                "punc_sentences": split_text_by_punctuation(_output_text(raw_output), timestamp),
            }
            for uttid, raw_output, timestamp in zip(batch_uttid, raw_outputs, batch_timestamp)
        ]


def _timestamp_to_text(timestamp: Sequence[Sequence]) -> str:
    text = ""
    previous = ""
    for item in timestamp:
        token = str(item[0]).strip()
        if not token:
            continue
        if text and (_ASCII_WORD.search(previous) or _ASCII_WORD.search(token)):
            text += " "
        text += token
        previous = token
    return text


def _output_text(raw_output) -> str:
    if isinstance(raw_output, dict):
        return str(raw_output.get("text") or "").strip()
    return str(raw_output or "").strip()
