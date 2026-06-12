from dataclasses import dataclass, field
import os
from typing import Sequence

from semantic_asr.punctuation import _join_tokens


DEFAULT_YUE_LABEL_TO_PUNCTUATION = {
    "0": "",
    "O": "",
    "NONE": "",
    "NO_PUNCT": "",
    "NO_PUNC": "",
    "LABEL_0": "",
    "COMMA": "，",
    "LABEL_1": "，",
    "，": "，",
    ",": "，",
    "PERIOD": "。",
    "FULL_STOP": "。",
    "FULLSTOP": "。",
    "STOP": "。",
    "DOT": "。",
    "LABEL_2": "。",
    "。": "。",
    ".": "。",
    "QUESTION": "？",
    "QUESTION_MARK": "？",
    "QMARK": "？",
    "LABEL_3": "？",
    "？": "？",
    "?": "？",
    "EXCLAMATION": "！",
    "EXCLAMATION_MARK": "！",
    "EXCLAMATION_POINT": "！",
    "EMARK": "！",
    "LABEL_4": "！",
    "！": "！",
    "!": "！",
    "SEMICOLON": "；",
    "SEMI_COLON": "；",
    "LABEL_5": "；",
    "；": "；",
    ";": "；",
    "COLON": "︰",
    "LABEL_6": "︰",
    "︰": "︰",
    "：": "︰",
    ":": "︰",
    "PAUSE": "、",
    "CAESURA": "、",
    "IDEOGRAPHIC_COMMA": "、",
    "LABEL_7": "、",
    "、": "、",
}


@dataclass
class YuePunctuationConfig:
    model_name_or_path: str = "nizzzo/zh-yue-punctuation-restore-v3"
    local_model_dir: str | None = None
    device: str = "cuda"
    max_tokens_per_chunk: int = 256
    label_to_punctuation: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_YUE_LABEL_TO_PUNCTUATION)
    )
    sentence_end_punctuation: tuple[str, ...] = ("。", "？", "！", "；")


class YuePunctuation:
    def __init__(self, config: YuePunctuationConfig | None = None):
        self.config = config or YuePunctuationConfig()
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
            labels = self.predict_labels(tokens)
            results.append({
                "uttid": uttid,
                "punc_sentences": punctuate_timestamp_from_labels(
                    timestamp,
                    labels,
                    self.config.label_to_punctuation,
                    self.config.sentence_end_punctuation,
                ),
            })
        return results

    def predict_labels(self, tokens: Sequence[str]) -> list[str]:
        labels = []
        chunk_size = max(1, int(self.config.max_tokens_per_chunk))
        for start in range(0, len(tokens), chunk_size):
            labels.extend(self._predict_chunk_labels(tokens[start:start + chunk_size]))
        return labels

    def _predict_chunk_labels(self, tokens: Sequence[str]) -> list[str]:
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


def punctuate_timestamp_from_labels(
    timestamp: Sequence[Sequence],
    labels: Sequence[str],
    label_to_punctuation: dict[str, str] | None = None,
    sentence_end_punctuation: Sequence[str] = ("。", "？", "！", "；"),
) -> list[dict]:
    mapping = label_to_punctuation or DEFAULT_YUE_LABEL_TO_PUNCTUATION
    end_marks = set(sentence_end_punctuation)
    sentences = []
    current_tokens = []
    current_start = None
    current_end = None
    label_cursor = 0

    for item in timestamp:
        if len(item) < 3:
            continue
        token = str(item[0]).strip()
        if not token:
            continue
        start_s = float(item[1])
        end_s = float(item[2])
        label = str(labels[label_cursor]) if label_cursor < len(labels) else "O"
        label_cursor += 1
        punctuation = _punctuation_for_label(label, mapping)
        current_start = start_s if current_start is None else current_start
        current_end = end_s
        current_tokens.append(token + punctuation)
        if punctuation in end_marks:
            sentences.append({
                "punc_text": _join_tokens(current_tokens),
                "start_s": current_start,
                "end_s": current_end,
            })
            current_tokens = []
            current_start = None
            current_end = None

    if current_tokens and current_start is not None and current_end is not None:
        sentences.append({
            "punc_text": _join_tokens(current_tokens),
            "start_s": current_start,
            "end_s": current_end,
        })
    return sentences


def punctuate_tokens_from_labels(
    tokens: Sequence[str],
    labels: Sequence[str],
    label_to_punctuation: dict[str, str] | None = None,
) -> str:
    mapping = label_to_punctuation or DEFAULT_YUE_LABEL_TO_PUNCTUATION
    punctuated = []
    for index, token in enumerate(tokens):
        token = str(token).strip()
        if not token:
            continue
        label = str(labels[index]) if index < len(labels) else "O"
        punctuated.append(token + _punctuation_for_label(label, mapping))
    return _join_tokens(punctuated)


def _punctuation_for_label(label: str, mapping: dict[str, str]) -> str:
    normalized = str(label).strip()
    candidates = [
        normalized,
        normalized.upper(),
        normalized.lower(),
        normalized.replace("LABEL_", ""),
        _strip_sequence_prefix(normalized, "-"),
        _strip_sequence_prefix(normalized, "_"),
    ]
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return ""


def _strip_sequence_prefix(label: str, separator: str) -> str:
    prefixes = ("B", "I", "E", "S", "U")
    for prefix in prefixes:
        marker = f"{prefix}{separator}"
        if label.startswith(marker):
            return label[len(marker):]
    return label


def _timestamp_tokens(timestamp: Sequence[Sequence]) -> list[str]:
    return [str(item[0]).strip() for item in timestamp if str(item[0]).strip()]


def _torch_device(torch, device: str):
    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)
