import argparse
import json
import os

import soundfile as sf

from semantic_asr.language_mapping import model_language
from semantic_asr.mms_runtime.aligner import MmsAligner, _insert_gap_stars
from semantic_asr.mms_runtime.align_utils import get_spans


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump MMS star/no-star alignment for a clipped region.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--wav-path", required=True)
    parser.add_argument("--start-ms", type=int, required=True)
    parser.add_argument("--end-ms", type=int, required=True)
    parser.add_argument("--language", default="vi_vn")
    parser.add_argument("--model-path", default="pretrained_models/mmsalign/model.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    with open(args.json_path, encoding="utf-8") as fin:
        result = json.load(fin)

    sentence_tokens = _tokens_from_sentences(result, args.start_ms, args.end_ms)
    wav, sample_rate = sf.read(args.wav_path, dtype="int16")
    clip_wav = wav[int(args.start_ms * sample_rate / 1000):int(args.end_ms * sample_rate / 1000)]
    language = model_language("mms_forced_aligner", args.language)

    aligner = MmsAligner(args.model_path, args.device)
    items, tokens = aligner._prepare_items_and_tokens(
        sentence_tokens,
        [f"tok_{index}" for index in range(len(sentence_tokens))],
        language,
        sentence_tokens,
        sentence_tokens,
    )
    expanded_items, expanded_tokens = _insert_gap_stars(items, tokens)

    first_pass_use_star = aligner.align(
        sentence_tokens,
        clip_wav,
        sample_rate,
        [f"tok_{index}" for index in range(len(sentence_tokens))],
        use_star=True,
        language=language,
        raw_transcripts=sentence_tokens,
        alignment_transcripts=sentence_tokens,
    )
    first_pass_expanded_debug = _align_expanded_items(aligner, expanded_items, expanded_tokens, clip_wav, sample_rate)
    second_pass = aligner.align(
        sentence_tokens,
        clip_wav,
        sample_rate,
        [f"tok_{index}" for index in range(len(sentence_tokens))],
        use_star=False,
        language=language,
        raw_transcripts=sentence_tokens,
        alignment_transcripts=sentence_tokens,
    )

    output = {
        "job_id": args.job_id,
        "source_json": args.json_path,
        "source_wav": args.wav_path,
        "clip_start_ms": args.start_ms,
        "clip_end_ms": args.end_ms,
        "language": args.language,
        "mms_language": language,
        "tokens": sentence_tokens,
        "first_pass_use_star_token_alignment": _with_absolute_ms(first_pass_use_star, args.start_ms),
        "first_pass_inferred_star_gaps": _inferred_gaps(first_pass_use_star, args.start_ms),
        "first_pass_expanded_star_span_debug": _with_absolute_ms(first_pass_expanded_debug, args.start_ms),
        "second_pass_no_star_alignment": _with_absolute_ms(second_pass, args.start_ms),
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fout:
        json.dump(output, fout, ensure_ascii=False, indent=2)
    print(args.out_json)


def _tokens_from_sentences(result: dict, start_ms: int, end_ms: int) -> list[str]:
    texts = [
        sentence["text"]
        for sentence in result.get("sentences", [])
        if int(sentence["start_ms"]) < end_ms and int(sentence["end_ms"]) > start_ms
    ]
    tokens = " ".join(texts).split()
    return [token for token in tokens if token.strip()]


def _align_expanded_items(
    aligner: MmsAligner,
    expanded_items: list[dict],
    expanded_tokens: list[str],
    wav,
    sample_rate: int,
) -> list[dict]:
    segments, stride = aligner.get_alignments(wav, sample_rate, expanded_tokens)
    spans = get_spans(expanded_tokens, segments)
    aligned = []
    for index, (item, token, span) in enumerate(zip(expanded_items, expanded_tokens, spans)):
        audio_start = round(span[0].start * stride / 1000, 3)
        audio_end = round(span[-1].end * stride / 1000, 3)
        aligned.append({
            "index": index,
            "inserted_star": bool(item.get("inserted_star")),
            "text": "<star>" if item.get("inserted_star") else item.get("raw_transcript"),
            "alignment_token": token,
            "token_index": item.get("token_index"),
            "before_token_index": item.get("before_token_index"),
            "after_token_index": item.get("after_token_index"),
            "start": audio_start,
            "end": audio_end,
            "duration": round(audio_end - audio_start, 3),
        })
    return aligned


def _with_absolute_ms(items: list[dict], clip_start_ms: int) -> list[dict]:
    output = []
    for item in items:
        row = dict(item)
        row["abs_start_ms"] = int(round(clip_start_ms + float(item["start"]) * 1000))
        row["abs_end_ms"] = int(round(clip_start_ms + float(item["end"]) * 1000))
        output.append(row)
    return output


def _inferred_gaps(items: list[dict], clip_start_ms: int) -> list[dict]:
    gaps = []
    for index, (previous, current) in enumerate(zip(items, items[1:])):
        gap_start = float(previous["end"])
        gap_end = float(current["start"])
        if gap_end < gap_start:
            continue
        gaps.append({
            "index": index,
            "kind": "inferred_star_gap",
            "before_token_index": index,
            "after_token_index": index + 1,
            "before_text": previous.get("text", previous.get("clean_text")),
            "after_text": current.get("text", current.get("clean_text")),
            "start": round(gap_start, 3),
            "end": round(gap_end, 3),
            "duration": round(gap_end - gap_start, 3),
            "abs_start_ms": int(round(clip_start_ms + gap_start * 1000)),
            "abs_end_ms": int(round(clip_start_ms + gap_end * 1000)),
        })
    return gaps


if __name__ == "__main__":
    main()
