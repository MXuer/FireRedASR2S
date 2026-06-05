import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import soundfile as sf

from semantic_asr.adapters.ten_vad import TenVadAdapter, TenVadConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_seconds", type=float, default=30.0)
    args = parser.parse_args()

    wav_path = args.wav_path
    if args.max_seconds > 0:
        wav, sample_rate = sf.read(args.wav_path, dtype="int16")
        wav = wav[: int(args.max_seconds * sample_rate)]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, wav, sample_rate)
        wav_path = tmp.name

    vad = TenVadAdapter(TenVadConfig(threshold=args.threshold))
    result = vad.detect(wav_path)
    assert result["timestamps"], "Ten-VAD returned no speech segments"
    assert all(start < end for start, end in result["timestamps"])
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

