from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from semantic_asr.core import AsrModel, PuncModel, TimestampProvider, VadModel

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
    registry.register("vad", "ten_vad", _build_ten_vad)
    registry.register("asr", "funasr_nano", _build_funasr_nano)
    registry.register("asr", "whisper_large", _build_whisper_large)
    registry.register("asr", "qwen3_asr_1_7b", _build_qwen3_asr)
    registry.register("asr", "dolphin", _build_dolphin)
    registry.register("asr", "seamless_m4t_v2_large", _build_seamless_m4t)
    registry.register("timestamp", "funasr_native", _build_funasr_native_timestamp)
    registry.register("timestamp", "whisper_native", _build_whisper_native_timestamp)
    registry.register("timestamp", "qwen3_forced_aligner", _build_qwen3_forced_aligner)
    registry.register("timestamp", "mms_forced_aligner", _build_mms_forced_aligner)
    registry.register("punc", "firered_punc", _build_firered_punc)
    registry.register("punc", "asr_native", _build_asr_native_punc)
    registry.register("punc", "asr_text", _build_asr_text_punc)
    registry.register("punc", "naqta", _build_naqta_punctuation)
    registry.register("punc", "qwen_semantic_boundary", _build_qwen_semantic_boundary)
    registry.register("punc", "xlm_roberta_punctuation", _build_xlm_roberta_punctuation)
    registry.register("punc", "yue_punctuation", _build_yue_punctuation)
    return registry


def _build_silero_vad(params: Mapping[str, Any]) -> VadModel:
    from semantic_asr.adapters.silero import SileroVad, SileroVadConfig

    return SileroVad(_dataclass_from_mapping(SileroVadConfig, params))


def _build_ten_vad(params: Mapping[str, Any]) -> VadModel:
    from semantic_asr.adapters.ten_vad import TenVadAdapter, TenVadConfig

    return TenVadAdapter(_dataclass_from_mapping(TenVadConfig, params))


def _build_firered_vad(params: Mapping[str, Any]) -> VadModel:
    from semantic_asr.firered_runtime.fireredvad import FireRedVad, FireRedVadConfig

    model_dir = str(params.get("model_dir", "pretrained_models/FireRedVAD/VAD"))
    config_params = _nested_or_direct_config(params, FireRedVadConfig)
    config = _dataclass_from_mapping(FireRedVadConfig, config_params)
    return FireRedVad.from_pretrained(model_dir, config)


def _build_funasr_nano(params: Mapping[str, Any]) -> AsrModel:
    from semantic_asr.adapters.funasr_nano import FunAsrNano, FunAsrNanoConfig

    return FunAsrNano(_dataclass_from_mapping(FunAsrNanoConfig, params))


def _build_whisper_large(params: Mapping[str, Any]) -> AsrModel:
    from semantic_asr.adapters.whisper_large import WhisperLarge, WhisperLargeConfig
    from semantic_asr.parallel_components import ParallelAsrModel

    config = _dataclass_from_mapping(WhisperLargeConfig, params)
    if config.num_workers > 1:
        return ParallelAsrModel(WhisperLarge, config, config.num_workers)
    return WhisperLarge(config)


def _build_qwen3_asr(params: Mapping[str, Any]) -> AsrModel:
    from semantic_asr.adapters.qwen3_asr import Qwen3Asr, Qwen3AsrConfig

    return Qwen3Asr(_dataclass_from_mapping(Qwen3AsrConfig, params))


def _build_dolphin(params: Mapping[str, Any]) -> AsrModel:
    from semantic_asr.adapters.dolphin import DolphinAsr, DolphinAsrConfig

    return DolphinAsr(_dataclass_from_mapping(DolphinAsrConfig, params))


def _build_seamless_m4t(params: Mapping[str, Any]) -> AsrModel:
    from semantic_asr.adapters.seamless_m4t import SeamlessM4TAsr, SeamlessM4TConfig

    return SeamlessM4TAsr(_dataclass_from_mapping(SeamlessM4TConfig, params))


def _build_funasr_native_timestamp(params: Mapping[str, Any]) -> TimestampProvider:
    from semantic_asr.adapters.funasr_nano import FunAsrNanoTimestampProvider

    return FunAsrNanoTimestampProvider()


def _build_whisper_native_timestamp(params: Mapping[str, Any]) -> TimestampProvider:
    from semantic_asr.adapters.whisper_large import WhisperLargeTimestampProvider

    return WhisperLargeTimestampProvider()


def _build_qwen3_forced_aligner(params: Mapping[str, Any]) -> TimestampProvider:
    from semantic_asr.adapters.qwen3_forced_aligner import (
        Qwen3ForcedAlignerConfig,
        Qwen3ForcedAlignerTimestampProvider,
    )

    return Qwen3ForcedAlignerTimestampProvider(_dataclass_from_mapping(Qwen3ForcedAlignerConfig, params))


def _build_mms_forced_aligner(params: Mapping[str, Any]) -> TimestampProvider:
    from semantic_asr.adapters.mms_forced_aligner import (
        MmsForcedAlignerConfig,
        MmsForcedAlignerTimestampProvider,
    )
    from semantic_asr.parallel_components import ParallelTimestampProvider

    config = _dataclass_from_mapping(MmsForcedAlignerConfig, params)
    if config.num_workers > 1:
        return ParallelTimestampProvider(MmsForcedAlignerTimestampProvider, config, config.num_workers)
    return MmsForcedAlignerTimestampProvider(config)


def _build_firered_punc(params: Mapping[str, Any]) -> PuncModel:
    from semantic_asr.adapters.firered import build_firered_punc
    from semantic_asr.firered_runtime.fireredpunc import FireRedPuncConfig

    model_dir = str(params.get("model_dir", "pretrained_models/FireRedPunc"))
    config_params = _nested_or_direct_config(params, FireRedPuncConfig)
    config = _dataclass_from_mapping(FireRedPuncConfig, config_params)
    return build_firered_punc(model_dir, config)


def _build_asr_native_punc(params: Mapping[str, Any]) -> PuncModel:
    from semantic_asr.punctuation import AsrNativePunc

    return AsrNativePunc()


def _build_asr_text_punc(params: Mapping[str, Any]) -> PuncModel:
    from semantic_asr.punctuation import AsrTextPunc

    return AsrTextPunc()


def _build_xlm_roberta_punctuation(params: Mapping[str, Any]) -> PuncModel:
    from semantic_asr.adapters.xlm_roberta_punctuation import (
        XlmRobertaPunctuation,
        XlmRobertaPunctuationConfig,
    )

    return XlmRobertaPunctuation(_dataclass_from_mapping(XlmRobertaPunctuationConfig, params))


def _build_naqta_punctuation(params: Mapping[str, Any]) -> PuncModel:
    from semantic_asr.adapters.naqta_punctuation import NaqtaPunctuation, NaqtaPunctuationConfig

    return NaqtaPunctuation(_dataclass_from_mapping(NaqtaPunctuationConfig, params))


def _build_yue_punctuation(params: Mapping[str, Any]) -> PuncModel:
    from semantic_asr.adapters.yue_punctuation import YuePunctuation, YuePunctuationConfig

    return YuePunctuation(_dataclass_from_mapping(YuePunctuationConfig, params))


def _build_qwen_semantic_boundary(params: Mapping[str, Any]) -> PuncModel:
    from semantic_asr.adapters.qwen_semantic_boundary import (
        QwenSemanticBoundaryConfig,
        QwenSemanticBoundaryPunc,
    )

    return QwenSemanticBoundaryPunc(_dataclass_from_mapping(QwenSemanticBoundaryConfig, params))


def _dataclass_from_mapping(cls, values: Mapping[str, Any]):
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    allowed = {field.name for field in fields(cls)}
    kwargs = {key: value for key, value in dict(values).items() if key in allowed}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown fields for {cls.__name__}: {', '.join(unknown)}")
    return cls(**kwargs)


def _nested_or_direct_config(params: Mapping[str, Any], config_cls) -> dict:
    values = dict(params.get("config", {}))
    config_fields = {field.name for field in fields(config_cls)}
    direct_unknown = sorted(set(params) - {"model_dir", "config"} - config_fields)
    if direct_unknown:
        raise ValueError(f"Unknown fields for {config_cls.__name__}: {', '.join(direct_unknown)}")
    for key, value in params.items():
        if key in {"model_dir", "config"}:
            continue
        if key in config_fields:
            values[key] = value
    return values
