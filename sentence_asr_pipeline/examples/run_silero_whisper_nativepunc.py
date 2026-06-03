import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import soundfile as sf

from sentence_asr_pipeline.adapters.silero import SileroVadConfig
from sentence_asr_pipeline.adapters.silero_whisper_nativepunc import (
    SileroWhisperNativePuncConfig,
    build_silero_whisper_nativepunc_pipeline,
)
from sentence_asr_pipeline.adapters.whisper_large import WhisperLargeConfig
from sentence_asr_pipeline.core import PipelineConfig
from sentence_asr_pipeline.outputs import write_all_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", default="data/test/short.wav")
    parser.add_argument("--uttid", default="short")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_name", default="large-v3")
    parser.add_argument("--language", default=None)
    parser.add_argument("--outdir", default="output/experiments/silero_whisper_nativepunc")
    parser.add_argument("--asr_batch_size", type=int, default=1)
    parser.add_argument("--punc_batch_size", type=int, default=1)
    parser.add_argument("--max_seconds", type=float, default=0)
    parser.add_argument("--write_textgrid", type=int, default=1)
    parser.add_argument("--write_srt", type=int, default=1)
    parser.add_argument("--write_csv", type=int, default=1)
    args = parser.parse_args()

    wav_path = args.wav_path
    if args.max_seconds > 0:
        wav, sample_rate = sf.read(args.wav_path, dtype="int16")
        wav = wav[: int(args.max_seconds * sample_rate)]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, wav, sample_rate)
        wav_path = tmp.name

    config = SileroWhisperNativePuncConfig(
        silero_config=SileroVadConfig(),
        whisper_config=WhisperLargeConfig(
            model_name=args.model_name,
            device=args.device,
            language=args.language,
        ),
        pipeline_config=PipelineConfig(
            asr_batch_size=args.asr_batch_size,
            punc_batch_size=args.punc_batch_size,
            strip_punctuation_before_punc=False,
        ),
    )
    pipeline = build_silero_whisper_nativepunc_pipeline(config)
    try:
        result = pipeline.process(wav_path, args.uttid)
    finally:
        if wav_path != args.wav_path:
            os.unlink(wav_path)

    os.makedirs(args.outdir, exist_ok=True)
    json_path = os.path.join(args.outdir, f"{args.uttid}.json")
    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)
    outputs = write_all_outputs(
        args.outdir,
        args.uttid,
        result,
        write_textgrid_output=bool(args.write_textgrid),
        write_srt_output=bool(args.write_srt),
        write_csv_output=bool(args.write_csv),
    )
    outputs["json"] = json_path
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

