from dataclasses import dataclass, field

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
from fireredasr2s.fireredlid import FireRedLid, FireRedLidConfig
from fireredasr2s.fireredpunc import FireRedPunc, FireRedPuncConfig
from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig
from sentence_asr_pipeline.core import PipelineConfig, SentenceAsrPipeline


@dataclass
class FireRedPipelineConfig:
    vad_model_dir: str = "pretrained_models/FireRedVAD/VAD"
    lid_model_dir: str = "pretrained_models/FireRedLID"
    asr_type: str = "aed"
    asr_model_dir: str = "pretrained_models/FireRedASR2-AED"
    punc_model_dir: str = "pretrained_models/FireRedPunc"
    vad_config: FireRedVadConfig = field(default_factory=FireRedVadConfig)
    lid_config: FireRedLidConfig = field(default_factory=FireRedLidConfig)
    asr_config: FireRedAsr2Config = field(default_factory=FireRedAsr2Config)
    punc_config: FireRedPuncConfig = field(default_factory=FireRedPuncConfig)
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)


def build_firered_pipeline(config: FireRedPipelineConfig) -> SentenceAsrPipeline:
    vad = FireRedVad.from_pretrained(config.vad_model_dir, config.vad_config) if config.pipeline_config.enable_vad else None
    lid = FireRedLid.from_pretrained(config.lid_model_dir, config.lid_config) if config.pipeline_config.enable_lid else None
    asr = FireRedAsr2.from_pretrained(config.asr_type, config.asr_model_dir, config.asr_config)
    punc = FireRedPunc.from_pretrained(config.punc_model_dir, config.punc_config) if config.pipeline_config.enable_punc else None
    config.pipeline_config.return_timestamp = config.asr_config.return_timestamp
    return SentenceAsrPipeline(asr=asr, config=config.pipeline_config, vad=vad, punc=punc, lid=lid)
