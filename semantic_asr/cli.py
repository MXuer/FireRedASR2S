import argparse
import json
import os
from contextlib import contextmanager
from typing import Sequence

from semantic_asr.api import SemanticASR, list_model_languages, list_models
from semantic_asr.run_batch import run_batch


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "transcribe":
        result = _run_transcribe(args)
    elif args.command == "batch":
        result = _run_batch(args)
    elif args.command == "models":
        result = list_models(args.language, role=args.role)
    elif args.command == "model-languages":
        result = list_model_languages(args.model, role=args.role)
    else:
        parser.error(f"Unknown command: {args.command}")
        return

    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semantic-asr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe = subparsers.add_parser("transcribe", help="Run one audio file through a pipeline config.")
    transcribe.add_argument("--config", required=True)
    transcribe.add_argument("--wav-path", required=True)
    transcribe.add_argument("--uttid", default=None)
    transcribe.add_argument("--outdir", default=None)
    transcribe.add_argument("--formats", default="json,srt,csv,textgrid")
    transcribe.add_argument("--max-seconds", type=float, default=0)
    transcribe.add_argument("--devices", default=None)
    transcribe.add_argument("--no-cache", action="store_true")

    batch = subparsers.add_parser("batch", help="Run wav.scp batch transcription.")
    batch.add_argument("--config", required=True)
    batch.add_argument("--wav-scp", required=True)
    batch.add_argument("--outdir", required=True)
    batch.add_argument("--num-workers", type=int, default=8)
    batch.add_argument("--max-seconds", type=float, default=0)
    batch.add_argument("--devices", default=None)

    models = subparsers.add_parser("models", help="List components available for a language.")
    models.add_argument("language")
    models.add_argument("--role", default=None, choices=("vad", "asr", "timestamp", "punc"))

    model_languages = subparsers.add_parser("model-languages", help="List languages supported by a component.")
    model_languages.add_argument("model")
    model_languages.add_argument("--role", default=None, choices=("vad", "asr", "timestamp", "punc"))

    return parser


def _run_transcribe(args) -> dict:
    with _temporary_cuda_visible_devices(args.devices):
        sdk = SemanticASR.from_config(args.config)
        return sdk.transcribe(
            wav_path=args.wav_path,
            uttid=args.uttid,
            outdir=args.outdir,
            formats=_parse_formats(args.formats),
            max_seconds=args.max_seconds,
            use_cache=not args.no_cache,
        )


def _run_batch(args) -> list[dict]:
    with _temporary_cuda_visible_devices(args.devices):
        return run_batch(
            config_path=args.config,
            wav_scp=args.wav_scp,
            outdir=args.outdir,
            num_workers=args.num_workers,
            max_seconds=args.max_seconds,
        )


def _parse_formats(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


@contextmanager
def _temporary_cuda_visible_devices(devices: str | None):
    if devices is None:
        yield
        return
    old_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_value


if __name__ == "__main__":
    main()
