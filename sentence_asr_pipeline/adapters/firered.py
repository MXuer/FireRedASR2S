from dataclasses import dataclass, field

from sentence_asr_pipeline.firered_runtime.fireredasr2 import FireRedAsr2, FireRedAsr2Config
from sentence_asr_pipeline.firered_runtime.fireredpunc import FireRedPunc, FireRedPuncConfig
from sentence_asr_pipeline.firered_runtime.fireredvad import FireRedVad, FireRedVadConfig
from sentence_asr_pipeline.core import PipelineConfig, SentenceAsrPipeline


@dataclass
class FireRedPipelineConfig:
    vad_model_dir: str = "pretrained_models/FireRedVAD/VAD"
    asr_type: str = "aed"
    asr_model_dir: str = "pretrained_models/FireRedASR2-AED"
    punc_model_dir: str = "pretrained_models/FireRedPunc"
    vad_config: FireRedVadConfig = field(default_factory=FireRedVadConfig)
    asr_config: FireRedAsr2Config = field(default_factory=FireRedAsr2Config)
    punc_config: FireRedPuncConfig = field(default_factory=FireRedPuncConfig)
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)


class AsrTimestampProvider:
    def add_timestamps(self, batch_asr_result: list[dict], batch_segments: list) -> list[dict]:
        for asr_result in batch_asr_result:
            if not asr_result.get("timestamp"):
                raise ValueError(f"FireRed ASR must return timestamp for {asr_result.get('uttid')}")
        return batch_asr_result


def build_firered_pipeline(config: FireRedPipelineConfig) -> SentenceAsrPipeline:
    config.asr_config.return_timestamp = True
    vad = FireRedVad.from_pretrained(config.vad_model_dir, config.vad_config)
    asr = FireRedAsr2.from_pretrained(config.asr_type, config.asr_model_dir, config.asr_config)
    punc = FireRedPunc.from_pretrained(config.punc_model_dir, config.punc_config)
    return SentenceAsrPipeline(
        vad=vad,
        asr=asr,
        timestamp_provider=AsrTimestampProvider(),
        punc=punc,
        config=config.pipeline_config,
    )


def build_firered_punc(model_dir: str = "pretrained_models/FireRedPunc", config: FireRedPuncConfig | None = None):
    return FireRedPunc.from_pretrained(model_dir, config or FireRedPuncConfig())
