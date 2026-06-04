import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import soundfile as sf

from semantic_asr.adapters.whisper_large import (
    WhisperLarge,
    WhisperLargeConfig,
    WhisperLargeTimestampProvider,
)
from semantic_asr.core import SpeechSegment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", default="data/test/short.wav")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_name", default="large-v3")
    parser.add_argument("--language", default="en_us")
    parser.add_argument("--max_seconds", type=float, default=30)
    args = parser.parse_args()

    wav_path = args.wav_path
    if args.max_seconds > 0:
        wav, sample_rate = sf.read(args.wav_path, dtype="int16")
        wav = wav[: int(args.max_seconds * sample_rate)]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, wav, sample_rate)
        wav_path = tmp.name

    wav, sample_rate = sf.read(wav_path, dtype="int16")
    model = WhisperLarge(WhisperLargeConfig(
        model_name=args.model_name,
        device=args.device,
        language=args.language,
    ))
    try:
        result = model.transcribe(["whisper_test"], [(sample_rate, wav)])
    finally:
        if wav_path != args.wav_path:
            os.unlink(wav_path)

    segment = SpeechSegment("whisper_test", 0.0, len(wav) / sample_rate, sample_rate, wav)
    timestamped = WhisperLargeTimestampProvider().add_timestamps(result, [segment])
    assert timestamped[0]["text"], timestamped
    assert timestamped[0]["timestamp"], timestamped
    print(json.dumps(timestamped[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
