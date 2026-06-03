from dataclasses import dataclass, field

from sentence_asr_pipeline.adapters.qwen3_forced_aligner import (
    Qwen3ForcedAlignerConfig,
    Qwen3ForcedAlignerTimestampProvider,
)
from sentence_asr_pipeline.adapters.whisper_large import WhisperLarge, WhisperLargeConfig
from sentence_asr_pipeline.core import PipelineConfig, SentenceAsrPipeline
from sentence_asr_pipeline.firered_runtime.fireredvad import FireRedVad, FireRedVadConfig
from sentence_asr_pipeline.punctuation import AsrTextPunc


@dataclass
class FireRedVadWhisperQwenAlignerTextPuncConfig:
    vad_model_dir: str = "pretrained_models/FireRedVAD/VAD"
    vad_config: FireRedVadConfig = field(default_factory=FireRedVadConfig)
    whisper_config: WhisperLargeConfig = field(
        default_factory=lambda: WhisperLargeConfig(language="ru", word_timestamps=False)
    )
    aligner_config: Qwen3ForcedAlignerConfig = field(default_factory=Qwen3ForcedAlignerConfig)
    pipeline_config: PipelineConfig = field(
        default_factory=lambda: PipelineConfig(strip_punctuation_before_punc=False)
    )


def build_fireredvad_whisper_qwenaligner_textpunc_pipeline(
    config: FireRedVadWhisperQwenAlignerTextPuncConfig,
) -> SentenceAsrPipeline:
    return SentenceAsrPipeline(
        vad=FireRedVad.from_pretrained(config.vad_model_dir, config.vad_config),
        asr=WhisperLarge(config.whisper_config),
        timestamp_provider=Qwen3ForcedAlignerTimestampProvider(config.aligner_config),
        punc=AsrTextPunc(),
        config=config.pipeline_config,
    )
