from dataclasses import dataclass, field

from sentence_asr_pipeline.adapters.silero import SileroVad, SileroVadConfig
from sentence_asr_pipeline.adapters.whisper_large import (
    WhisperLarge,
    WhisperLargeConfig,
    WhisperLargeTimestampProvider,
)
from sentence_asr_pipeline.core import PipelineConfig, SentenceAsrPipeline
from sentence_asr_pipeline.punctuation import AsrNativePunc


@dataclass
class SileroWhisperNativePuncConfig:
    silero_config: SileroVadConfig = field(default_factory=SileroVadConfig)
    whisper_config: WhisperLargeConfig = field(default_factory=WhisperLargeConfig)
    pipeline_config: PipelineConfig = field(
        default_factory=lambda: PipelineConfig(strip_punctuation_before_punc=False)
    )


def build_silero_whisper_nativepunc_pipeline(config: SileroWhisperNativePuncConfig) -> SentenceAsrPipeline:
    return SentenceAsrPipeline(
        vad=SileroVad(config.silero_config),
        asr=WhisperLarge(config.whisper_config),
        timestamp_provider=WhisperLargeTimestampProvider(),
        punc=AsrNativePunc(),
        config=config.pipeline_config,
    )
