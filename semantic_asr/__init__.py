from semantic_asr.compat import patch_torch_pytree_for_transformers
from importlib import import_module

patch_torch_pytree_for_transformers()

from semantic_asr.core import (
    AsrModel,
    PipelineConfig,
    PuncModel,
    SemanticAsrPipeline,
    SpeechSegment,
    TimestampProvider,
    VadModel,
)

_API_EXPORTS = {"SemanticASR", "list_model_languages", "list_models", "suggest_components"}


def __getattr__(name):
    if name in _API_EXPORTS:
        api = import_module("semantic_asr.api")
        return getattr(api, name)
    raise AttributeError(name)

__all__ = [
    "AsrModel",
    "PipelineConfig",
    "PuncModel",
    "SemanticASR",
    "SemanticAsrPipeline",
    "SpeechSegment",
    "TimestampProvider",
    "VadModel",
    "list_model_languages",
    "list_models",
    "patch_torch_pytree_for_transformers",
    "suggest_components",
]
