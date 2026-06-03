from dataclasses import dataclass, field

from fireredasr2s.fireredpunc import FireRedPuncConfig
from sentence_asr_pipeline.adapters.firered import build_firered_punc
from sentence_asr_pipeline.adapters.funasr_nano import (
    FunAsrNano,
    FunAsrNanoConfig,
    FunAsrNanoTimestampProvider,
)
from sentence_asr_pipeline.adapters.silero import SileroVad, SileroVadConfig
from sentence_asr_pipeline.core import PipelineConfig, SentenceAsrPipeline


@dataclass
class SileroFunAsrFireRedPuncConfig:
    silero_config: SileroVadConfig = field(default_factory=SileroVadConfig)
    funasr_config: FunAsrNanoConfig = field(default_factory=FunAsrNanoConfig)
    punc_model_dir: str = "pretrained_models/FireRedPunc"
    punc_config: FireRedPuncConfig = field(default_factory=FireRedPuncConfig)
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)


def build_silero_funasr_fireredpunc_pipeline(config: SileroFunAsrFireRedPuncConfig) -> SentenceAsrPipeline:
    return SentenceAsrPipeline(
        vad=SileroVad(config.silero_config),
        asr=FunAsrNano(config.funasr_config),
        timestamp_provider=FunAsrNanoTimestampProvider(),
        punc=build_firered_punc(config.punc_model_dir, config.punc_config),
        config=config.pipeline_config,
    )
