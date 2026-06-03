from dataclasses import dataclass, field
from typing import Any

from semantic_asr.core import PipelineConfig


@dataclass(frozen=True)
class LanguagePipelineProfile:
    language: str
    aliases: tuple[str, ...]
    builder: str
    pipeline_config: PipelineConfig
    component_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)


LANGUAGE_PROFILES = {
    "zh": LanguagePipelineProfile(
        language="zh",
        aliases=("zh", "zh-cn", "cn", "chinese"),
        builder="silero_funasr_fireredpunc",
        pipeline_config=PipelineConfig(asr_batch_size=8, punc_batch_size=8),
        component_kwargs={
            "funasr": {"language": "Chinese"},
        },
    ),
    "en": LanguagePipelineProfile(
        language="en",
        aliases=("en", "english"),
        builder="silero_whisper_nativepunc",
        pipeline_config=PipelineConfig(strip_punctuation_before_punc=False),
        component_kwargs={
            "whisper": {"language": "en"},
        },
    ),
    "ru": LanguagePipelineProfile(
        language="ru",
        aliases=("ru", "ru-ru", "russian"),
        builder="fireredvad_whisper_qwenaligner_textpunc",
        pipeline_config=PipelineConfig(
            asr_batch_size=1,
            punc_batch_size=1,
            strip_punctuation_before_punc=False,
        ),
        component_kwargs={
            "whisper": {"language": "ru", "word_timestamps": False},
            "qwen_aligner": {"language": "Russian"},
        },
    ),
}


def resolve_language_profile(language: str) -> LanguagePipelineProfile:
    normalized = language.strip().lower()
    for profile in LANGUAGE_PROFILES.values():
        if normalized in profile.aliases:
            return profile
    raise ValueError(f"Unsupported language profile: {language}")

