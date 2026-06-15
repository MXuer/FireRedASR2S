import re
from typing import Sequence

_PUNCT_ONLY = re.compile(r"^[^\w\u4e00-\u9fff]+$")
_EDGE_PUNCT = re.compile(r"(^[^\w\u4e00-\u9fff]+)|([^\w\u4e00-\u9fff]+$)")
_SENTENCE_END = re.compile(r"[。.!?！？؟]+$")
_SENTENCE_END_CHARS = set("。.!?！？؟")
_ASCII_WORD = re.compile(r"[a-zA-Z0-9#]")


def strip_timestamp_punctuation(timestamp: Sequence[Sequence]) -> list[list]:
    stripped = []
    for item in timestamp:
        token, start_s, end_s = item[0], item[1], item[2]
        token = str(token).strip().lower()
        if not token or _PUNCT_ONLY.fullmatch(token):
            continue
        token = _EDGE_PUNCT.sub("", token)
        if token:
            stripped.append([token, float(start_s), float(end_s)])
    return stripped


class AsrNativePunc:
    def process_with_timestamp(self, batch_timestamp: Sequence[list], batch_uttid: Sequence[str]) -> list[dict]:
        results = []
        for uttid, timestamp in zip(batch_uttid, batch_timestamp):
            results.append({
                "uttid": uttid,
                "punc_sentences": split_timestamp_by_native_punctuation(timestamp),
            })
        return results


class AsrTextPunc:
    def process_asr_results(self, batch_asr_result: Sequence[dict]) -> list[dict]:
        results = []
        for asr_result in batch_asr_result:
            results.append({
                "uttid": asr_result["uttid"],
                "punc_sentences": split_text_by_punctuation(
                    asr_result.get("text", ""),
                    asr_result.get("timestamp", []),
                ),
            })
        return results


def split_text_by_punctuation(text: str, timestamp: Sequence[Sequence]) -> list[dict]:
    sentence_texts = _split_punctuated_sentence_texts(text)
    if not sentence_texts:
        sentence_texts = [text.strip()] if text.strip() else []

    timestamps = [item for item in timestamp if str(item[0]).strip()]
    cursor = 0
    sentences = []
    for i, sentence_text in enumerate(sentence_texts):
        token_count = _count_timestamp_tokens_for_sentence(sentence_text, timestamps, cursor)
        if token_count is None:
            token_count = len(strip_timestamp_punctuation([[tok, 0, 0] for tok in sentence_text.split()]))
        if token_count == 0:
            if sentences:
                sentences[-1]["punc_text"] += sentence_text
            continue
        if i == len(sentence_texts) - 1:
            token_count = max(token_count, len(timestamps) - cursor)
        selected = timestamps[cursor:cursor + token_count] if token_count > 0 else []
        cursor += token_count

        if not selected:
            if sentences:
                sentences[-1]["punc_text"] = _join_tokens([sentences[-1]["punc_text"], sentence_text])
            continue
        start_s = float(selected[0][1])
        end_s = float(selected[-1][2])
        sentences.append(_sentence(start_s, end_s, [sentence_text]))
    return sentences


def _count_timestamp_tokens_for_sentence(
    sentence_text: str,
    timestamps: Sequence[Sequence],
    cursor: int,
) -> int | None:
    expected = _normalize_for_timestamp_match(sentence_text)
    if not expected:
        return 0

    consumed = ""
    for index in range(cursor, len(timestamps)):
        consumed += _normalize_for_timestamp_match(str(timestamps[index][0]))
        if not consumed:
            continue
        if expected == consumed:
            return index - cursor + 1
        if not expected.startswith(consumed):
            return None
    return None


def _normalize_for_timestamp_match(text: str) -> str:
    return "".join(char.lower() for char in text if _is_timestamp_match_char(char))


def _is_timestamp_match_char(char: str) -> bool:
    return char.isalnum() or char == "_" or char == "#"


def _split_punctuated_sentence_texts(text: str) -> list[str]:
    sentence_texts = []
    start = 0
    i = 0
    while i < len(text):
        if text[i] not in _SENTENCE_END_CHARS or not _is_sentence_boundary(text, i):
            i += 1
            continue
        end = i + 1
        while end < len(text) and text[end] in _SENTENCE_END_CHARS:
            end += 1
        _append_sentence_text(sentence_texts, text[start:end])
        start = end
        i = end

    _append_sentence_text(sentence_texts, text[start:])
    return sentence_texts


def _append_sentence_text(sentence_texts: list[str], sentence_text: str) -> None:
    sentence_text = sentence_text.strip()
    if sentence_text and not _PUNCT_ONLY.fullmatch(sentence_text):
        sentence_texts.append(sentence_text)


def _is_sentence_boundary(text: str, index: int) -> bool:
    if text[index] != ".":
        return True
    previous = text[index - 1] if index > 0 else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    return not (_is_ascii_token_char(previous) and _is_ascii_token_char(following))


def _is_ascii_token_char(char: str) -> bool:
    return bool(char and _ASCII_WORD.fullmatch(char))


def split_timestamp_by_native_punctuation(timestamp: Sequence[Sequence]) -> list[dict]:
    sentences = []
    tokens = []
    start_s = None
    end_s = 0.0

    for item in timestamp:
        token, token_start_s, token_end_s = item[0], float(item[1]), float(item[2])
        token = str(token).strip()
        if not token:
            continue
        if _PUNCT_ONLY.fullmatch(token):
            if tokens:
                tokens[-1] += token
                end_s = token_end_s
                if _SENTENCE_END.search(token):
                    sentence_start_s = start_s if start_s is not None else token_start_s
                    sentences.append(_sentence(sentence_start_s, end_s, tokens))
                    tokens = []
                    start_s = None
            elif sentences:
                sentences[-1]["punc_text"] += token
                sentences[-1]["end_s"] = token_end_s
            continue
        if start_s is None:
            start_s = token_start_s
        tokens.append(token)
        end_s = token_end_s
        if _SENTENCE_END.search(token):
            sentences.append(_sentence(start_s, end_s, tokens))
            tokens = []
            start_s = None

    if tokens:
        sentences.append(_sentence(start_s or 0.0, end_s, tokens))
    return sentences


def _sentence(start_s: float, end_s: float, tokens: Sequence[str]) -> dict:
    return {
        "start_s": start_s,
        "end_s": end_s,
        "punc_text": _join_tokens(tokens),
    }


def _join_tokens(tokens: Sequence[str]) -> str:
    text = ""
    last_token = ""
    for token in tokens:
        if text and _ASCII_WORD.search(last_token) and _ASCII_WORD.search(token):
            text += " "
        text += token
        last_token = token
    return text
