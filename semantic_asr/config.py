import copy
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from semantic_asr.core import PipelineConfig, SemanticAsrPipeline
from semantic_asr.language_mapping import require_canonical_language_id
from semantic_asr.registry import ComponentRegistry, create_default_registry

REQUIRED_ROLES = ("vad", "asr", "timestamp", "punc")
LANGUAGE_COMPONENTS = {
    "asr": {
        "dolphin",
        "funasr_nano",
        "qwen3_asr_1_7b",
        "seamless_m4t_v2_large",
        "whisper_large",
    },
    "timestamp": {
        "mms_forced_aligner",
        "qwen3_forced_aligner",
    },
    "punc": {
        "qwen_semantic_boundary",
    },
}


@dataclass
class ComponentSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    outdir: str = "output/experiments/config_runner"
    write_textgrid: bool = True
    write_srt: bool = True
    write_csv: bool = True
    copy_resolved_config: bool = True


@dataclass
class PipelineProfileConfig:
    name: str
    language: str
    components: dict[str, ComponentSpec]
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_pipeline_profile(path: str) -> PipelineProfileConfig:
    raw = _load_config_file(path)
    return parse_pipeline_profile(raw)


def parse_pipeline_profile(raw: dict[str, Any]) -> PipelineProfileConfig:
    if not isinstance(raw, dict):
        raise ValueError("Pipeline profile config must be a mapping")
    name = str(raw.get("name") or "").strip()
    if not name:
        raise ValueError("Pipeline profile config requires non-empty 'name'")

    raw_components = raw.get("components")
    if not isinstance(raw_components, dict):
        raise ValueError("Pipeline profile config requires 'components' mapping")
    language = str(raw.get("language") or "").strip()
    if not language:
        raise ValueError("Pipeline profile config requires canonical 'language'")

    components: dict[str, ComponentSpec] = {}
    for role in REQUIRED_ROLES:
        spec = raw_components.get(role)
        if not isinstance(spec, dict):
            raise ValueError(f"Pipeline profile config requires components.{role}")
        component_name = str(spec.get("name") or "").strip()
        if not component_name:
            raise ValueError(f"components.{role}.name must be non-empty")
        params = spec.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"components.{role}.params must be a mapping")
        components[role] = ComponentSpec(name=component_name, params=copy.deepcopy(params))

    return PipelineProfileConfig(
        name=name,
        language=require_canonical_language_id(language),
        components=components,
        pipeline=_dataclass_from_raw(PipelineConfig, raw.get("pipeline", {})),
        output=_dataclass_from_raw(OutputConfig, raw.get("output", {})),
    )


def build_pipeline_from_profile(
    profile: PipelineProfileConfig,
    registry: ComponentRegistry | None = None,
) -> SemanticAsrPipeline:
    registry = registry or create_default_registry()
    component_params = {
        role: _resolved_component_params(profile, role)
        for role in REQUIRED_ROLES
    }
    return SemanticAsrPipeline(
        vad=registry.build("vad", profile.components["vad"].name, component_params["vad"]),
        asr=registry.build("asr", profile.components["asr"].name, component_params["asr"]),
        timestamp_provider=registry.build(
            "timestamp",
            profile.components["timestamp"].name,
            component_params["timestamp"],
        ),
        punc=registry.build("punc", profile.components["punc"].name, component_params["punc"]),
        config=profile.pipeline,
    )


def profile_to_dict(profile: PipelineProfileConfig) -> dict[str, Any]:
    return {
        "name": profile.name,
        "language": profile.language,
        "components": {
            role: {"name": spec.name, "params": copy.deepcopy(spec.params)}
            for role, spec in profile.components.items()
        },
        "pipeline": asdict(profile.pipeline),
        "output": asdict(profile.output),
    }


def write_resolved_config(outdir: str, profile: PipelineProfileConfig) -> str:
    os.makedirs(outdir, exist_ok=True)
    output_path = os.path.join(outdir, "resolved_config.json")
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(profile_to_dict(profile), fout, ensure_ascii=False, indent=2)
    return output_path


def _load_config_file(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fin:
        if path.endswith((".yaml", ".yml")):
            import yaml

            data = yaml.safe_load(fin)
        else:
            data = json.load(fin)
    return data


def _dataclass_from_raw(cls, raw: dict[str, Any] | None):
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} config must be a mapping")
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown fields for {cls.__name__}: {', '.join(unknown)}")
    return cls(**copy.deepcopy(raw))


def _resolved_component_params(profile: PipelineProfileConfig, role: str) -> dict[str, Any]:
    spec = profile.components[role]
    params = copy.deepcopy(spec.params)
    if profile.language and spec.name in LANGUAGE_COMPONENTS.get(role, set()):
        params["language"] = profile.language
    return params
