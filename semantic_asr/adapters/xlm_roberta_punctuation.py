from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

from semantic_asr.punctuation import split_text_by_punctuation


@dataclass
class XlmRobertaPunctuationConfig:
    model: str = "pcs_47lang"
    local_model_dir: str | None = None
    device: str = "cuda"
    apply_sbd: bool = True


class XlmRobertaPunctuation:
    def __init__(self, config: XlmRobertaPunctuationConfig | None = None):
        self.config = config or XlmRobertaPunctuationConfig()
        from punctuators.models import PunctCapSegModelONNX
        from punctuators.models.punc_cap_seg_model import PunctCapSegConfigONNX

        local_model_dir = self.config.local_model_dir or _find_local_47lang_model()
        if local_model_dir:
            model_config = PunctCapSegConfigONNX(
                directory=os.path.expanduser(local_model_dir),
                spe_filename="spe_unigram_64k_lowercase_47lang.model",
                model_filename="punct_cap_seg_47lang.onnx",
            )
            self.model = PunctCapSegModelONNX(model_config)
        else:
            self.model = PunctCapSegModelONNX.from_pretrained(self.config.model)

    def process_with_timestamp(self, batch_timestamp: Sequence[list], batch_uttid: Sequence[str]) -> list[dict]:
        texts = [_timestamp_to_text(timestamp) for timestamp in batch_timestamp]
        raw_outputs = self.model.infer(texts, apply_sbd=self.config.apply_sbd)
        return [
            {
                "uttid": uttid,
                "punc_sentences": _sentences_from_output(raw_output, timestamp),
            }
            for uttid, raw_output, timestamp in zip(batch_uttid, raw_outputs, batch_timestamp)
        ]


def _timestamp_to_text(timestamp: Sequence[Sequence]) -> str:
    return " ".join(str(item[0]).strip() for item in timestamp if str(item[0]).strip())


def _sentences_from_output(raw_output, timestamp: Sequence[Sequence]) -> list[dict]:
    if isinstance(raw_output, str):
        text = raw_output
    elif isinstance(raw_output, (list, tuple)):
        text = " ".join(str(item).strip() for item in raw_output if str(item).strip())
    else:
        text = str(raw_output).strip()
    return split_text_by_punctuation(text, timestamp)


def _find_local_47lang_model() -> str | None:
    snapshots_dir = Path(os.path.expanduser(
        "~/.cache/huggingface/hub/models--1-800-BAD-CODE--punct_cap_seg_47_language/snapshots"
    ))
    snapshots = sorted(path for path in snapshots_dir.glob("*") if path.is_dir())
    return str(snapshots[-1]) if snapshots else None
