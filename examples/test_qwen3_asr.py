import argparse
import json
import os
import sys

import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from semantic_asr.adapters.qwen3_asr import Qwen3Asr, Qwen3AsrConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", default="data/test/short.wav")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--language", default=None)
    parser.add_argument("--max_seconds", type=float, default=30)
    parser.add_argument("--skip_model_load", type=int, default=0)
    args = parser.parse_args()

    config = Qwen3AsrConfig(model=args.model, language=args.language)
    if args.skip_model_load:
        print(json.dumps({"config": config.__dict__}, ensure_ascii=False, indent=2))
        return

    wav, sample_rate = sf.read(args.wav_path, dtype="int16")
    if args.max_seconds > 0:
        wav = wav[: int(args.max_seconds * sample_rate)]
    asr = Qwen3Asr(config)
    result = asr.transcribe(["test"], [(sample_rate, wav)])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
