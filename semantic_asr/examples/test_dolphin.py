import argparse
import json
import os
import sys

import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from semantic_asr.adapters.dolphin import DolphinAsr, DolphinAsrConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", default="data/test/short.wav")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_name", default="small")
    parser.add_argument("--language", default="zh_cn")
    parser.add_argument("--word_timestamp", type=int, default=1)
    parser.add_argument("--max_seconds", type=float, default=30)
    parser.add_argument("--skip_model_load", type=int, default=0)
    args = parser.parse_args()

    config = DolphinAsrConfig(
        model_name=args.model_name,
        device=args.device,
        language=args.language,
        word_timestamp=bool(args.word_timestamp),
    )
    if args.skip_model_load:
        print(json.dumps({"config": config.__dict__}, ensure_ascii=False, indent=2))
        return

    wav, sample_rate = sf.read(args.wav_path, dtype="int16")
    if args.max_seconds > 0:
        wav = wav[: int(args.max_seconds * sample_rate)]
    asr = DolphinAsr(config)
    result = asr.transcribe(["test"], [(sample_rate, wav)])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
