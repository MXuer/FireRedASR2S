from dataclasses import dataclass
import re
from typing import Sequence

from semantic_asr.punctuation import split_text_by_punctuation


_ASCII_WORD = re.compile(r"[A-Za-z0-9#]")


@dataclass
class CtPuncConfig:
    model: str = "ct-punc"
    disable_update: bool = True


class CtPunc:
    supports_batch: bool = False

    def __init__(self, config: CtPuncConfig | None = None):
        self.config = config or CtPuncConfig()
        from funasr import AutoModel

        self.model = AutoModel(model=self.config.model, disable_update=self.config.disable_update)

    def process_with_timestamp(self, batch_timestamp: Sequence[list], batch_uttid: Sequence[str]) -> list[dict]:
        results = []
        for timestamp, uttid in zip(batch_timestamp, batch_uttid):
            raw_output = self.model.generate(input=_timestamp_to_text(timestamp), batch_size=1)
            results.append({
                "uttid": uttid,
                "punc_sentences": split_text_by_punctuation(_output_text(raw_output), timestamp),
            })
        return results


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
    if isinstance(raw_output, (list, tuple)):
        if not raw_output:
            return ""
        return _output_text(raw_output[0])
    if isinstance(raw_output, dict):
        return str(raw_output.get("text") or "").strip()
    return str(raw_output or "").strip()
