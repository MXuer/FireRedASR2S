from dataclasses import dataclass, field
from typing import Any

from semantic_asr.core import PipelineConfig
from semantic_asr.language_mapping import canonical_language_id


@dataclass(frozen=True)
class LanguagePipelineProfile:
    language: str
    aliases: tuple[str, ...]
    builder: str
    pipeline_config: PipelineConfig
    component_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)


LANGUAGE_PROFILES = {
    "zh_cn": LanguagePipelineProfile(
        language="zh_cn",
        aliases=("zh_cn", "zh", "zh-cn", "cn", "chinese"),
        builder="silero_funasr_fireredpunc",
        pipeline_config=PipelineConfig(asr_batch_size=8, punc_batch_size=8),
        component_kwargs={
            "funasr": {"language": "zh_cn"},
        },
    ),
    "en_us": LanguagePipelineProfile(
        language="en_us",
        aliases=("en_us", "en", "english"),
        builder="silero_whisper_nativepunc",
        pipeline_config=PipelineConfig(strip_punctuation_before_punc=False),
        component_kwargs={
            "whisper": {"language": "en_us"},
        },
    ),
    "ru_ru": LanguagePipelineProfile(
        language="ru_ru",
        aliases=("ru_ru", "ru", "ru-ru", "russian"),
        builder="fireredvad_whisper_qwenaligner_textpunc",
        pipeline_config=PipelineConfig(
            asr_batch_size=1,
            punc_batch_size=1,
            strip_punctuation_before_punc=False,
        ),
        component_kwargs={
            "whisper": {"language": "ru_ru", "word_timestamps": False},
            "qwen_aligner": {"language": "ru_ru"},
        },
    ),
}


def resolve_language_profile(language: str) -> LanguagePipelineProfile:
    canonical = canonical_language_id(language)
    try:
        return LANGUAGE_PROFILES[canonical]
    except KeyError as exc:
        raise ValueError(f"Unsupported language profile: {canonical}") from exc
