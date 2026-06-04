import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from semantic_asr.config import load_pipeline_profile
from semantic_asr.run_pipeline import run_profile


CONFIG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../semantic_asr/configs/fireredvad_whisper_qwenaligner_textpunc_ru.json",
    )
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", default="data/test/ru_ru.wav")
    parser.add_argument("--uttid", default="ru_ru")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--whisper_model_name", default="large-v3")
    parser.add_argument("--whisper_language", default="ru")
    parser.add_argument("--aligner_model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--aligner_language", default="Russian")
    parser.add_argument("--outdir", default="output/experiments/fireredvad_whisper_qwenaligner_textpunc_ru")
    parser.add_argument("--asr_batch_size", type=int, default=1)
    parser.add_argument("--aligner_batch_size", type=int, default=4)
    parser.add_argument("--max_seconds", type=float, default=60)
    parser.add_argument("--write_textgrid", type=int, default=1)
    parser.add_argument("--write_srt", type=int, default=1)
    parser.add_argument("--write_csv", type=int, default=1)
    args = parser.parse_args()

    profile = load_pipeline_profile(CONFIG_PATH)
    profile.components["asr"].params["device"] = args.device
    profile.components["asr"].params["model_name"] = args.whisper_model_name
    profile.components["asr"].params["language"] = args.whisper_language
    profile.components["asr"].params["word_timestamps"] = False
    profile.components["timestamp"].params["model"] = args.aligner_model
    profile.components["timestamp"].params["device_map"] = args.device
    profile.components["timestamp"].params["language"] = args.aligner_language
    profile.components["timestamp"].params["batch_size"] = args.aligner_batch_size
    profile.pipeline.asr_batch_size = args.asr_batch_size
    profile.output.outdir = args.outdir
    profile.output.write_textgrid = bool(args.write_textgrid)
    profile.output.write_srt = bool(args.write_srt)
    profile.output.write_csv = bool(args.write_csv)

    outputs = run_profile(
        profile=profile,
        wav_path=args.wav_path,
        uttid=args.uttid,
        max_seconds=args.max_seconds,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
