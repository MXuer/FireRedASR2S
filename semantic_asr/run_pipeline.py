import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import soundfile as sf

from semantic_asr.config import (
    PipelineProfileConfig,
    build_pipeline_from_profile,
    load_pipeline_profile,
    write_resolved_config,
)
from semantic_asr.outputs import write_all_outputs
from semantic_asr.registry import ComponentRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--wav_path", required=True)
    parser.add_argument("--uttid", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--max_seconds", type=float, default=0)
    args = parser.parse_args()

    outputs = run_from_config(
        config_path=args.config,
        wav_path=args.wav_path,
        uttid=args.uttid,
        outdir=args.outdir,
        max_seconds=args.max_seconds,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


def run_from_config(
    config_path: str,
    wav_path: str,
    uttid: str | None = None,
    outdir: str | None = None,
    max_seconds: float = 0,
    registry: ComponentRegistry | None = None,
) -> dict:
    profile = load_pipeline_profile(config_path)
    return run_profile(
        profile=profile,
        wav_path=wav_path,
        uttid=uttid,
        outdir=outdir,
        max_seconds=max_seconds,
        registry=registry,
    )


def run_profile(
    profile: PipelineProfileConfig,
    wav_path: str,
    uttid: str | None = None,
    outdir: str | None = None,
    max_seconds: float = 0,
    registry: ComponentRegistry | None = None,
) -> dict:
    output_dir = outdir or profile.output.outdir
    resolved_uttid = uttid or os.path.splitext(os.path.basename(wav_path))[0]

    input_wav_path = _maybe_truncate_wav(wav_path, max_seconds)
    try:
        pipeline = build_pipeline_from_profile(profile, registry=registry)
        result = pipeline.process(input_wav_path, resolved_uttid)
    finally:
        if input_wav_path != wav_path:
            os.unlink(input_wav_path)

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{resolved_uttid}.json")
    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)

    outputs = write_all_outputs(
        output_dir,
        resolved_uttid,
        result,
        write_textgrid_output=profile.output.write_textgrid,
        write_srt_output=profile.output.write_srt,
        write_csv_output=profile.output.write_csv,
    )
    outputs["json"] = json_path
    if profile.output.copy_resolved_config:
        outputs["resolved_config"] = write_resolved_config(output_dir, profile)
    return outputs


def _maybe_truncate_wav(wav_path: str, max_seconds: float) -> str:
    if max_seconds <= 0:
        return wav_path

    wav, sample_rate = sf.read(wav_path, dtype="int16")
    wav = wav[: int(max_seconds * sample_rate)]
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, wav, sample_rate)
    return tmp.name


if __name__ == "__main__":
    main()
