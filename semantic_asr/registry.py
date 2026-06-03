from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from semantic_asr.adapters.firered import build_firered_punc
from semantic_asr.adapters.funasr_nano import (
    FunAsrNano,
    FunAsrNanoConfig,
    FunAsrNanoTimestampProvider,
)
from semantic_asr.adapters.qwen3_forced_aligner import (
    Qwen3ForcedAlignerConfig,
    Qwen3ForcedAlignerTimestampProvider,
)
from semantic_asr.adapters.silero import SileroVad, SileroVadConfig
from semantic_asr.adapters.whisper_large import (
    WhisperLarge,
    WhisperLargeConfig,
    WhisperLargeTimestampProvider,
)
from semantic_asr.core import AsrModel, PuncModel, TimestampProvider, VadModel
from semantic_asr.firered_runtime.fireredpunc import FireRedPuncConfig
from semantic_asr.firered_runtime.fireredvad import FireRedVad, FireRedVadConfig
from semantic_asr.punctuation import AsrNativePunc, AsrTextPunc

Factory = Callable[[Mapping[str, Any]], Any]


class ComponentRegistry:
    def __init__(self):
        self._factories: dict[str, dict[str, Factory]] = {
            "vad": {},
            "asr": {},
            "timestamp": {},
            "punc": {},
        }

    def register(self, role: str, name: str, factory: Factory) -> None:
        self._require_role(role)
        if name in self._factories[role]:
            raise ValueError(f"Duplicate component registration: {role}.{name}")
        self._factories[role][name] = factory

    def build(self, role: str, name: str, params: Mapping[str, Any] | None = None) -> Any:
        self._require_role(role)
        try:
            factory = self._factories[role][name]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories[role])) or "<none>"
            raise ValueError(f"Unknown {role} component: {name}. Available: {available}") from exc
        return factory(params or {})

    def names(self, role: str) -> list[str]:
        self._require_role(role)
        return sorted(self._factories[role])

    def _require_role(self, role: str) -> None:
        if role not in self._factories:
            roles = ", ".join(sorted(self._factories))
            raise ValueError(f"Unknown component role: {role}. Available roles: {roles}")


def create_default_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    registry.register("vad", "silero", _build_silero_vad)
    registry.register("vad", "firered_vad", _build_firered_vad)
    registry.register("asr", "funasr_nano", _build_funasr_nano)
    registry.register("asr", "whisper_large", _build_whisper_large)
    registry.register("timestamp", "funasr_native", lambda params: FunAsrNanoTimestampProvider())
    registry.register("timestamp", "whisper_native", lambda params: WhisperLargeTimestampProvider())
    registry.register("timestamp", "qwen3_forced_aligner", _build_qwen3_forced_aligner)
    registry.register("punc", "firered_punc", _build_firered_punc)
    registry.register("punc", "asr_native", lambda params: AsrNativePunc())
    registry.register("punc", "asr_text", lambda params: AsrTextPunc())
    return registry


def _build_silero_vad(params: Mapping[str, Any]) -> VadModel:
    return SileroVad(_dataclass_from_mapping(SileroVadConfig, params))


def _build_firered_vad(params: Mapping[str, Any]) -> VadModel:
    model_dir = str(params.get("model_dir", "pretrained_models/FireRedVAD/VAD"))
    config = _dataclass_from_mapping(FireRedVadConfig, params.get("config", {}))
    return FireRedVad.from_pretrained(model_dir, config)


def _build_funasr_nano(params: Mapping[str, Any]) -> AsrModel:
    return FunAsrNano(_dataclass_from_mapping(FunAsrNanoConfig, params))


def _build_whisper_large(params: Mapping[str, Any]) -> AsrModel:
    return WhisperLarge(_dataclass_from_mapping(WhisperLargeConfig, params))


def _build_qwen3_forced_aligner(params: Mapping[str, Any]) -> TimestampProvider:
    return Qwen3ForcedAlignerTimestampProvider(_dataclass_from_mapping(Qwen3ForcedAlignerConfig, params))


def _build_firered_punc(params: Mapping[str, Any]) -> PuncModel:
    model_dir = str(params.get("model_dir", "pretrained_models/FireRedPunc"))
    config = _dataclass_from_mapping(FireRedPuncConfig, params.get("config", {}))
    return build_firered_punc(model_dir, config)


def _dataclass_from_mapping(cls, values: Mapping[str, Any]):
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    allowed = {field.name for field in fields(cls)}
    kwargs = {key: value for key, value in dict(values).items() if key in allowed}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown fields for {cls.__name__}: {', '.join(unknown)}")
    return cls(**kwargs)
