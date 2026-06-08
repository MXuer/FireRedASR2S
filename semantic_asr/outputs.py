import datetime
import json
import os

import soundfile as sf
from textgrid import IntervalTier, TextGrid


def write_result_jsonl(outdir: str, result: dict) -> str:
    os.makedirs(outdir, exist_ok=True)
    output_path = os.path.join(outdir, "result.jsonl")
    with open(output_path, "a", encoding="utf-8") as fout:
        fout.write(f"{json.dumps(result, ensure_ascii=False)}\n")
    return output_path


def write_textgrid(tg_dir: str, name: str, wav_dur: float, sentences: list[dict], words: list[dict] | None = None) -> str:
    os.makedirs(tg_dir, exist_ok=True)
    output_path = os.path.join(tg_dir, name + ".TextGrid")
    textgrid = TextGrid(maxTime=wav_dur)

    tier = IntervalTier(name="sentence", maxTime=wav_dur)
    for sentence in sentences:
        start_s = max(sentence["start_ms"] / 1000.0, 0)
        end_s = min(sentence["end_ms"] / 1000.0, wav_dur)
        text = sentence["text"]
        if start_s != end_s and text.strip():
            tier.add(minTime=start_s, maxTime=end_s, mark=text)
    textgrid.append(tier)

    if words:
        tier = IntervalTier(name="token", maxTime=wav_dur)
        for word in words:
            start_s = max(word["start_ms"] / 1000.0, 0)
            end_s = min(word["end_ms"] / 1000.0, wav_dur)
            text = word["text"]
            if start_s != end_s and text.strip():
                tier.add(minTime=start_s, maxTime=end_s, mark=text)
        textgrid.append(tier)

    textgrid.write(output_path)
    return output_path


def write_srt(srt_dir: str, name: str, sentences: list[dict]) -> str:
    os.makedirs(srt_dir, exist_ok=True)
    output_path = os.path.join(srt_dir, name + ".srt")

    with open(output_path, "w", encoding="utf-8") as fout:
        index = 0
        for sentence in sentences:
            text = sentence["text"]
            if not text.strip():
                continue
            index += 1
            fout.write(f"{index}\n")
            fout.write(f"{_ms_to_srt_time(sentence['start_ms'])} --> {_ms_to_srt_time(sentence['end_ms'])}\n")
            fout.write(f"{text}\n\n")
    return output_path


def write_csv(csv_dir: str, name: str, wav_dur: float, sentences: list[dict]) -> str:
    os.makedirs(csv_dir, exist_ok=True)
    output_path = os.path.join(csv_dir, name + ".csv")

    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write("Name\tStart\tDuration\tTime Format\tType\tDescription\n")
        index = 0
        for sentence in sentences:
            text = sentence["text"]
            if not text.strip():
                continue
            start_s = max(sentence["start_ms"] / 1000.0, 0)
            end_s = min(sentence["end_ms"] / 1000.0, wav_dur)
            fout.write(
                f"{index:06d}\t{_seconds_to_time(start_s)}\t{_seconds_to_time(end_s - start_s)}\t"
                f"decimal\tCue\t{text}\n"
            )
            index += 1
    return output_path


def split_and_save_segments(wav_path: str, timestamps_ms: list[tuple[int, int]], save_segment_dir: str) -> list[str]:
    os.makedirs(save_segment_dir, exist_ok=True)
    wav_np, sample_rate = sf.read(wav_path, dtype="int16")
    uttid = os.path.basename(wav_path).replace(".wav", "")
    output_paths = []
    for i, (start_ms, end_ms) in enumerate(timestamps_ms):
        seg_id = f"{uttid}_{i}_{start_ms}_{end_ms}"
        output_path = os.path.join(save_segment_dir, seg_id + ".wav")
        start = int(start_ms / 1000 * sample_rate)
        end = int(end_ms / 1000 * sample_rate)
        sf.write(output_path, wav_np[start:end], samplerate=sample_rate)
        output_paths.append(output_path)
    return output_paths


def write_all_outputs(
    outdir: str,
    name: str,
    result: dict,
    write_textgrid_output: bool = True,
    write_srt_output: bool = True,
    write_csv_output: bool = True,
) -> dict:
    outputs = {"jsonl": write_result_jsonl(outdir, result)}
    if write_textgrid_output:
        outputs["textgrid"] = write_textgrid(
            os.path.join(outdir, "asr_tg"),
            name,
            result["dur_s"],
            result["sentences"],
            None,
        )
    if write_srt_output:
        outputs["srt"] = write_srt(os.path.join(outdir, "asr_srt"), name, result["sentences"])
    if write_csv_output:
        outputs["csv"] = write_csv(os.path.join(outdir, "asr_csv"), name, result["dur_s"], result["sentences"])
    return outputs


def _ms_to_srt_time(ms: int) -> str:
    h = ms // 1000 // 3600
    m = (ms // 1000 % 3600) // 60
    s = (ms // 1000 % 3600) % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms % 1000:03d}"


def _seconds_to_time(seconds: float) -> str:
    delta = datetime.timedelta(seconds=seconds)
    result_time = datetime.datetime(1, 1, 1) + delta
    return result_time.strftime("%H:%M:%S.%f")
