from dataclasses import dataclass, field
from typing import Any, Sequence

from semantic_asr.firered_runtime.fireredasr2 import FireRedAsr2, FireRedAsr2Config
from semantic_asr.firered_runtime.fireredpunc import FireRedPunc, FireRedPuncConfig
from semantic_asr.firered_runtime.fireredvad import FireRedVad, FireRedVadConfig
from semantic_asr.core import PipelineConfig, SemanticAsrPipeline, SpeechSegment


@dataclass
class FireRedAsrAdapterConfig:
    asr_type: str = "aed"
    model_dir: str = "pretrained_models/FireRedASR2-AED"
    return_timestamp: bool = True
    batch_size: int = 128
    config: FireRedAsr2Config = field(default_factory=FireRedAsr2Config)


class FireRedAsrAdapter:
    def __init__(self, config: FireRedAsrAdapterConfig | None = None):
        self.config = config or FireRedAsrAdapterConfig()
        self.config.config.return_timestamp = self.config.return_timestamp
        self.model = FireRedAsr2.from_pretrained(
            self.config.asr_type,
            self.config.model_dir,
            self.config.config,
        )
        self.recommended_batch_size = max(1, int(self.config.batch_size))

    def transcribe(self, batch_uttid: Sequence[str], batch_wav: Sequence[tuple[int, Any]]) -> list[dict]:
        return self.model.transcribe(batch_uttid, batch_wav)


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
    def add_timestamps(
        self,
        batch_asr_result: Sequence[dict],
        batch_segments: Sequence[SpeechSegment],
    ) -> list[dict]:
        for asr_result in batch_asr_result:
            if not asr_result.get("timestamp"):
                raise ValueError(f"FireRed ASR must return timestamp for {asr_result.get('uttid')}")
        return list(batch_asr_result)


def build_firered_pipeline(config: FireRedPipelineConfig) -> SemanticAsrPipeline:
    config.asr_config.return_timestamp = True
    vad = FireRedVad.from_pretrained(config.vad_model_dir, config.vad_config)
    asr = FireRedAsr2.from_pretrained(config.asr_type, config.asr_model_dir, config.asr_config)
    punc = FireRedPunc.from_pretrained(config.punc_model_dir, config.punc_config)
    return SemanticAsrPipeline(
        vad=vad,
        asr=asr,
        timestamp_provider=AsrTimestampProvider(),
        punc=punc,
        config=config.pipeline_config,
    )


def build_firered_punc(model_dir: str = "pretrained_models/FireRedPunc", config: FireRedPuncConfig | None = None):
    return FireRedPunc.from_pretrained(model_dir, config or FireRedPuncConfig())
