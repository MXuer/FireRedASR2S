import json
import os
from contextlib import contextmanager
from dataclasses import replace
from typing import Sequence

from semantic_asr.config import (
    OutputConfig,
    PipelineProfileConfig,
    build_pipeline_from_profile,
    load_pipeline_profile,
    parse_pipeline_profile,
    write_resolved_config,
)
from semantic_asr.core import SemanticAsrPipeline, validate_sentence_intervals
from semantic_asr.language_support import list_languages_by_model, list_models_by_language
from semantic_asr.outputs import write_csv, write_result_jsonl, write_srt, write_textgrid
from semantic_asr.registry import ComponentRegistry
from semantic_asr.run_batch import run_batch
from semantic_asr.run_pipeline import _maybe_truncate_wav, result_json_path

DEFAULT_FORMATS = ("json", "srt", "csv", "textgrid")
SUPPORTED_FORMATS = {"json", "jsonl", "srt", "csv", "textgrid", "resolved_config"}


class SemanticASR:
    """External Python SDK facade for the semantic ASR pipeline."""

    def __init__(
        self,
        profile: PipelineProfileConfig,
        pipeline: SemanticAsrPipeline,
        config_path: str | None = None,
    ):
        self.profile = profile
        self.pipeline = pipeline
        self.config_path = config_path

    @classmethod
    def from_config(
        cls,
        config_path: str,
        registry: ComponentRegistry | None = None,
    ) -> "SemanticASR":
        profile = load_pipeline_profile(config_path)
        pipeline = build_pipeline_from_profile(profile, registry=registry)
        return cls(profile=profile, pipeline=pipeline, config_path=config_path)

    @classmethod
    def from_profile(
        cls,
        profile: PipelineProfileConfig | dict,
        registry: ComponentRegistry | None = None,
    ) -> "SemanticASR":
        parsed = parse_pipeline_profile(profile) if isinstance(profile, dict) else profile
        pipeline = build_pipeline_from_profile(parsed, registry=registry)
        return cls(profile=parsed, pipeline=pipeline)

    def transcribe(
        self,
        wav_path: str,
        uttid: str | None = None,
        outdir: str | None = None,
        formats: Sequence[str] | None = DEFAULT_FORMATS,
        max_seconds: float = 0,
        use_cache: bool = True,
    ) -> dict:
        resolved_uttid = uttid or os.path.splitext(os.path.basename(wav_path))[0]
        selected_formats = _normalize_formats(formats)

        if outdir:
            os.makedirs(outdir, exist_ok=True)
            json_path = result_json_path(outdir, resolved_uttid)
            if use_cache and os.path.exists(json_path):
                with open(json_path, encoding="utf-8") as fin:
                    result = json.load(fin)
                validate_sentence_intervals(result.get("sentences", []))
                outputs = _write_selected_outputs(
                    outdir,
                    resolved_uttid,
                    result,
                    self.profile,
                    selected_formats,
                    write_json=False,
                )
                outputs["json"] = json_path
                return {"result": result, "outputs": outputs}

        input_wav_path = _maybe_truncate_wav(wav_path, max_seconds)
        try:
            result = self.pipeline.process(input_wav_path, resolved_uttid)
        finally:
            if input_wav_path != wav_path:
                os.unlink(input_wav_path)

        if not outdir:
            return {"result": result, "outputs": {}}

        outputs = _write_selected_outputs(
            outdir,
            resolved_uttid,
            result,
            self.profile,
            selected_formats,
            write_json=True,
        )
        return {"result": result, "outputs": outputs}

    def transcribe_batch(
        self,
        wav_scp: str,
        outdir: str,
        num_workers: int = 8,
        max_seconds: float = 0,
        devices: str | Sequence[str] | None = None,
    ) -> list[dict]:
        if not self.config_path:
            raise ValueError("transcribe_batch requires an instance created with SemanticASR.from_config()")
        with _temporary_cuda_visible_devices(devices):
            return run_batch(
                config_path=self.config_path,
                wav_scp=wav_scp,
                outdir=outdir,
                num_workers=num_workers,
                max_seconds=max_seconds,
            )


def list_models(language: str, role: str | None = None) -> dict:
    return list_models_by_language(language, role=role)


def list_model_languages(model: str, role: str | None = None) -> dict:
    return list_languages_by_model(model, role=role)


def suggest_components(language: str, role: str | None = None) -> dict:
    return list_models(language, role=role)


def _normalize_formats(formats: Sequence[str] | None) -> set[str]:
    if formats is None:
        return set(DEFAULT_FORMATS)
    selected = {str(item).lower() for item in formats}
    unknown = selected - SUPPORTED_FORMATS
    if unknown:
        raise ValueError(f"Unknown output formats: {', '.join(sorted(unknown))}")
    return selected


def _write_selected_outputs(
    outdir: str,
    uttid: str,
    result: dict,
    profile: PipelineProfileConfig,
    formats: set[str],
    write_json: bool,
) -> dict:
    outputs = {}
    if write_json or "json" in formats:
        json_path = result_json_path(outdir, uttid)
        with open(json_path, "w", encoding="utf-8") as fout:
            json.dump(result, fout, ensure_ascii=False, indent=2)
        outputs["json"] = json_path
    if "jsonl" in formats:
        outputs["jsonl"] = write_result_jsonl(outdir, result)
    if "textgrid" in formats:
        outputs["textgrid"] = write_textgrid(
            os.path.join(outdir, "asr_tg"),
            uttid,
            result["dur_s"],
            result["sentences"],
            result.get("words"),
        )
    if "srt" in formats:
        outputs["srt"] = write_srt(os.path.join(outdir, "asr_srt"), uttid, result["sentences"])
    if "csv" in formats:
        outputs["csv"] = write_csv(os.path.join(outdir, "asr_csv"), uttid, result["dur_s"], result["sentences"])
    if "resolved_config" in formats or profile.output.copy_resolved_config:
        resolved_profile = replace(
            profile,
            output=replace(profile.output, outdir=outdir) if isinstance(profile.output, OutputConfig) else profile.output,
        )
        outputs["resolved_config"] = write_resolved_config(outdir, resolved_profile)
    return outputs


@contextmanager
def _temporary_cuda_visible_devices(devices: str | Sequence[str] | None):
    if devices is None:
        yield
        return
    old_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if isinstance(devices, str):
        value = devices
    else:
        value = ",".join(str(item) for item in devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_value
