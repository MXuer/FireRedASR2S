import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import soundfile as sf

from semantic_asr.adapters.funasr_nano import (
    FunAsrNano,
    FunAsrNanoConfig,
    FunAsrNanoTimestampProvider,
)
from semantic_asr.core import SpeechSegment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hub", default="ms")
    parser.add_argument("--language", default="zh_cn")
    parser.add_argument("--max_seconds", type=float, default=30.0)
    args = parser.parse_args()

    wav, sample_rate = sf.read(args.wav_path, dtype="int16")
    if wav.ndim > 1:
        wav = wav.mean(axis=1).astype("int16")
    wav = wav[: int(args.max_seconds * sample_rate)]

    asr = FunAsrNano(FunAsrNanoConfig(device=args.device, hub=args.hub, language=args.language))
    result = asr.transcribe(["funasr_test_s0_e%d" % int(len(wav) / sample_rate * 1000)], [(sample_rate, wav)])
    result = FunAsrNanoTimestampProvider().add_timestamps(
        result,
        [SpeechSegment(result[0]["uttid"], 0, len(wav) / sample_rate, sample_rate, wav)],
    )
    assert result[0]["text"], "Fun-ASR-Nano returned empty text"
    assert result[0]["timestamp"], "Fun-ASR-Nano returned no timestamps"
    print(json.dumps(result[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
