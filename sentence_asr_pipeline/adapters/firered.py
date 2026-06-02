from dataclasses import dataclass, field

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
from fireredasr2s.fireredpunc import FireRedPunc, FireRedPuncConfig
from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig
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


class ExistingTimestampPredictor:
    def predict(self, batch_asr_result: list[dict]) -> list[dict]:
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
        timestamp_predictor=ExistingTimestampPredictor(),
        punc=punc,
        config=config.pipeline_config,
    )
