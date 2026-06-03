from dataclasses import dataclass, field

from semantic_asr.adapters.firered import build_firered_punc
from semantic_asr.adapters.funasr_nano import (
    FunAsrNano,
    FunAsrNanoConfig,
    FunAsrNanoTimestampProvider,
)
from semantic_asr.adapters.silero import SileroVad, SileroVadConfig
from semantic_asr.core import PipelineConfig, SemanticAsrPipeline
from semantic_asr.firered_runtime.fireredpunc import FireRedPuncConfig


@dataclass
class SileroFunAsrFireRedPuncConfig:
    silero_config: SileroVadConfig = field(default_factory=SileroVadConfig)
    funasr_config: FunAsrNanoConfig = field(default_factory=FunAsrNanoConfig)
    punc_model_dir: str = "pretrained_models/FireRedPunc"
    punc_config: FireRedPuncConfig = field(default_factory=FireRedPuncConfig)
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)


def build_silero_funasr_fireredpunc_pipeline(config: SileroFunAsrFireRedPuncConfig) -> SemanticAsrPipeline:
    return SemanticAsrPipeline(
        vad=SileroVad(config.silero_config),
        asr=FunAsrNano(config.funasr_config),
        timestamp_provider=FunAsrNanoTimestampProvider(),
        punc=build_firered_punc(config.punc_model_dir, config.punc_config),
        config=config.pipeline_config,
    )
