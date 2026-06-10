from dataclasses import dataclass, field
import os
from typing import Sequence

from semantic_asr.punctuation import split_text_by_punctuation


DEFAULT_NAQTA_LABEL_TO_PUNCTUATION = {
    "0": "",
    "O": "",
    "NONE": "",
    "NO_PUNCT": "",
    "NO_PUNC": "",
    "COMMA": "،",
    "ARABIC_COMMA": "،",
    "،": "،",
    ",": "،",
    "PERIOD": ".",
    "FULL_STOP": ".",
    "DOT": ".",
    ".": ".",
    "QUESTION": "؟",
    "QUESTION_MARK": "؟",
    "ARABIC_QUESTION_MARK": "؟",
    "؟": "؟",
    "?": "؟",
    "SEMICOLON": "؛",
    "ARABIC_SEMICOLON": "؛",
    "؛": "؛",
    ";": "؛",
    "COLON": ":",
    ":": ":",
}


@dataclass
class NaqtaPunctuationConfig:
    model_name_or_path: str = "MostafaMaroof/Naqta"
    local_model_dir: str | None = None
    device: str = "cuda"
    max_words_per_chunk: int = 256
    label_to_punctuation: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_NAQTA_LABEL_TO_PUNCTUATION)
    )


class NaqtaPunctuation:
    def __init__(self, config: NaqtaPunctuationConfig | None = None):
        self.config = config or NaqtaPunctuationConfig()
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        model_path = os.path.expanduser(self.config.local_model_dir or self.config.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.torch = torch
        self.device = _torch_device(torch, self.config.device)
        self.model.to(self.device)
        self.model.eval()

    def process_with_timestamp(self, batch_timestamp: Sequence[list], batch_uttid: Sequence[str]) -> list[dict]:
        results = []
        for timestamp, uttid in zip(batch_timestamp, batch_uttid):
            tokens = _timestamp_tokens(timestamp)
            punctuated_text = self.punctuate_tokens(tokens)
            results.append({
                "uttid": uttid,
                "punc_sentences": split_text_by_punctuation(punctuated_text, timestamp),
            })
        return results

    def punctuate_tokens(self, tokens: Sequence[str]) -> str:
        labels = []
        chunk_size = max(1, int(self.config.max_words_per_chunk))
        for start in range(0, len(tokens), chunk_size):
            labels.extend(self._predict_labels(tokens[start:start + chunk_size]))
        return punctuate_tokens_from_labels(tokens, labels, self.config.label_to_punctuation)

    def _predict_labels(self, tokens: Sequence[str]) -> list[str]:
        if not tokens:
            return []
        encoded = self.tokenizer(
            list(tokens),
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
        )
        word_ids = encoded.word_ids()
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            logits = self.model(**encoded).logits[0]
        pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
        id2label = self.model.config.id2label

        labels_by_word: dict[int, str] = {}
        for word_id, pred_id in zip(word_ids, pred_ids):
            if word_id is None:
                continue
            labels_by_word[int(word_id)] = str(id2label[int(pred_id)])
        return [labels_by_word.get(i, "O") for i in range(len(tokens))]


def punctuate_tokens_from_labels(
    tokens: Sequence[str],
    labels: Sequence[str],
    label_to_punctuation: dict[str, str] | None = None,
) -> str:
    mapping = label_to_punctuation or DEFAULT_NAQTA_LABEL_TO_PUNCTUATION
    punctuated = []
    for token, label in zip(tokens, labels):
        token = str(token).strip()
        if not token:
            continue
        punctuated.append(token + _punctuation_for_label(str(label), mapping))
    if len(labels) < len(tokens):
        punctuated.extend(str(token).strip() for token in tokens[len(labels):] if str(token).strip())
    return " ".join(punctuated)


def _punctuation_for_label(label: str, mapping: dict[str, str]) -> str:
    candidates = [
        label,
        label.upper(),
        label.lower(),
        label.replace("LABEL_", ""),
        label.replace("B-", "").replace("I-", ""),
        label.replace("B_", "").replace("I_", ""),
    ]
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return ""


def _timestamp_tokens(timestamp: Sequence[Sequence]) -> list[str]:
    return [str(item[0]).strip() for item in timestamp if str(item[0]).strip()]


def _torch_device(torch, device: str):
    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)
