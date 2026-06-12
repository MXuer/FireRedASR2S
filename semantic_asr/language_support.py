from dataclasses import dataclass, field
from typing import Any


ANY_LANGUAGE = "*"


@dataclass(frozen=True)
class ModelLanguageSupport:
    role: str
    name: str
    languages: tuple[str, ...]
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    notes: str = ""
    has_native_timestamps: bool = False
    has_native_punctuation: bool = False
    supports_batch: bool = False


def normalize_language(language: str) -> str:
    return language.strip().lower().replace("_", "-")


MODEL_LANGUAGE_SUPPORT: tuple[ModelLanguageSupport, ...] = (
    ModelLanguageSupport(
        role="vad",
        name="silero",
        languages=(ANY_LANGUAGE,),
        notes="Language-agnostic speech activity detector.",
    ),
    ModelLanguageSupport(
        role="vad",
        name="firered_vad",
        languages=(ANY_LANGUAGE,),
        notes="Language-agnostic speech activity detector in this pipeline contract.",
    ),
    ModelLanguageSupport(
        role="vad",
        name="ten_vad",
        languages=(ANY_LANGUAGE,),
        notes="Language-agnostic TEN VAD adapter using 16 kHz, 256-sample frames.",
    ),
    ModelLanguageSupport(
        role="asr",
        name="funasr_nano",
        languages=("en", "ja", "zh"),
        aliases={
            "en": ("english", "en-us", "en-gb"),
            "ja": ("japanese", "ja-jp"),
            "zh": ("chinese", "zh-cn", "cn"),
        },
        notes="Fun-ASR-Nano-2512 supports Chinese, English and Japanese with native timestamps.",
        has_native_timestamps=True,
        has_native_punctuation=True,
        supports_batch=True,
    ),
    ModelLanguageSupport(
        role="asr",
        name="whisper_large",
        languages=(
            "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br",
            "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu",
            "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr",
            "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km",
            "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk",
            "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc",
            "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl",
            "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th",
            "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh",
        ),
        aliases={
            "en": ("english", "en-us", "en-gb"),
        },
        notes="Whisper large multilingual ASR. Word timestamps are requested by adapter config.",
        has_native_timestamps=True,
        has_native_punctuation=True,
    ),
    ModelLanguageSupport(
        role="asr",
        name="qwen3_asr_1_7b",
        languages=(
            "ar", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fil", "fr",
            "hi", "hu", "id", "it", "ja", "ko", "mk", "ms", "nl", "pl", "pt",
            "ro", "ru", "sv", "th", "tr", "vi", "yue", "zh",
        ),
        aliases={
            "ar": ("arabic",),
            "cs": ("czech",),
            "da": ("danish",),
            "de": ("german",),
            "el": ("greek",),
            "en": ("english", "en-us", "en-gb"),
            "es": ("spanish",),
            "fa": ("persian",),
            "fi": ("finnish",),
            "fil": ("filipino",),
            "fr": ("french",),
            "hi": ("hindi",),
            "hu": ("hungarian",),
            "id": ("indonesian",),
            "it": ("italian",),
            "ja": ("japanese",),
            "ko": ("korean",),
            "mk": ("macedonian",),
            "ms": ("malay",),
            "nl": ("dutch",),
            "pl": ("polish",),
            "pt": ("portuguese",),
            "ro": ("romanian",),
            "ru": ("russian",),
            "sv": ("swedish",),
            "th": ("thai",),
            "tr": ("turkish",),
            "vi": ("vietnamese",),
            "yue": ("cantonese", "zh-hk"),
            "zh": ("chinese", "zh-cn", "cn"),
        },
        notes="Qwen3-ASR supports 30 languages; the model API receives full language names such as English, Thai and Cantonese.",
        has_native_punctuation=True,
        supports_batch=True,
    ),
    ModelLanguageSupport(
        role="asr",
        name="dolphin",
        languages=(
            "ar", "az", "ba", "bn", "ct", "fa", "fil", "gu", "hi", "id", "ja",
            "jv", "kab", "kk", "km", "ko", "ks", "ky", "lo", "mn", "mr", "ms",
            "my", "ne", "or", "pa", "ps", "ru", "si", "su", "ta", "te", "tg",
            "th", "tl", "ug", "ur", "uz", "vi", "zh",
        ),
        aliases={
            "zh": ("chinese", "zh-cn", "zh-tw", "cn"),
            "ct": ("yue", "cantonese", "zh-hk"),
            "fil": ("filipino",),
        },
        notes="Dolphin supports 40 Eastern languages plus Chinese dialect region codes; configure a canonical language-region id and the adapter derives native symbols.",
        has_native_timestamps=True,
        has_native_punctuation=True,
    ),
    ModelLanguageSupport(
        role="asr",
        name="seamless_m4t_v2_large",
        languages=("ar", "en", "hi", "ja", "ko", "pt", "ru", "th", "vi", "zh"),
        aliases={
            "eng": ("en", "en-us", "en-gb", "english"),
            "cmn": ("zh", "zh-cn", "chinese", "mandarin"),
            "rus": ("ru", "ru-ru", "russian"),
        },
        notes="The current adapter mapping covers ten validated languages; configure a canonical language-region id and the adapter derives the Seamless/FLORES code.",
        has_native_punctuation=True,
    ),
    ModelLanguageSupport(
        role="timestamp",
        name="funasr_native",
        languages=("en", "ja", "zh"),
        aliases={
            "en": ("english", "en-us", "en-gb"),
            "ja": ("japanese", "ja-jp"),
            "zh": ("chinese", "zh-cn", "cn"),
        },
        notes="Validates Fun-ASR-Nano native token timestamps.",
    ),
    ModelLanguageSupport(
        role="timestamp",
        name="whisper_native",
        languages=(ANY_LANGUAGE,),
        notes="Validates Whisper word timestamps when word_timestamps=True.",
    ),
    ModelLanguageSupport(
        role="timestamp",
        name="qwen3_forced_aligner",
        languages=("de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "th", "zh"),
        aliases={
            "en": ("english", "en-us", "en-gb"),
            "zh": ("chinese", "zh-cn", "cn"),
        },
        notes="Qwen3 forced aligner official package supports 11 languages.",
        supports_batch=True,
    ),
    ModelLanguageSupport(
        role="timestamp",
        name="mms_forced_aligner",
        languages=(
            "ar-sa", "bg-bg", "bn-bd", "bo-lhasa", "ct-hk", "de-de", "en-gb",
            "en-us", "es-mx", "et-ee", "fa-ir", "fil-ph", "fr-fr", "he-il",
            "hi-in", "hr-hr", "ht-ht", "hu-hu", "id-id", "it-it", "ja-jp",
            "kk-kz", "km-kh", "ko-kr", "lo-la", "mai-in", "min-cn", "mn-mn",
            "my-mm", "nb-no", "ne-in", "ne-np", "nl-nl", "nn-no", "pl-pl",
            "pt-br", "ro-ro", "ru-ru", "sr-rs", "sv-se", "sw-ke", "tg-tj",
            "th-th", "tr-tr", "ug-cn", "uk-ua", "uz-uz", "vi-in", "zh-cn",
        ),
        aliases={
            "ct-hk": ("yue", "yue-hk", "cantonese"),
            "en-us": ("en", "english"),
            "zh-cn": ("zh", "chinese", "cn"),
            "ru-ru": ("ru", "russian"),
            "vi-in": ("vi", "vi-vn", "vietnamese"),
        },
        notes="MMS forced aligner via vendored runtime; configure a canonical language-region id and the adapter maps it through MMS_CODE_MAP.",
    ),
    ModelLanguageSupport(
        role="punc",
        name="firered_punc",
        languages=("zh",),
        aliases={"zh": ("chinese", "zh-cn", "cn")},
        notes="Chinese punctuation model.",
    ),
    ModelLanguageSupport(
        role="punc",
        name="asr_native",
        languages=(ANY_LANGUAGE,),
        notes="Uses punctuation emitted by the ASR model.",
    ),
    ModelLanguageSupport(
        role="punc",
        name="asr_text",
        languages=(ANY_LANGUAGE,),
        notes="Splits ASR text by existing punctuation.",
    ),
    ModelLanguageSupport(
        role="punc",
        name="qwen_semantic_boundary",
        languages=(ANY_LANGUAGE,),
        notes=(
            "Uses Qwen3.6 text understanding to return index-only semantic "
            "sentence boundaries. It does not rewrite ASR text and should be "
            "validated per target language/domain."
        ),
        supports_batch=False,
    ),
    ModelLanguageSupport(
        role="punc",
        name="naqta",
        languages=("ar",),
        aliases={"ar": ("arabic", "ar-sa", "ar-eg", "ar-ae")},
        notes="Arabic punctuation restoration model MostafaMaroof/Naqta.",
    ),
    ModelLanguageSupport(
        role="punc",
        name="yue_punctuation",
        languages=("yue",),
        aliases={"yue": ("cantonese", "yue-hk", "ct-hk")},
        notes="Cantonese punctuation restoration model nizzzo/zh-yue-punctuation-restore-v3.",
    ),
    ModelLanguageSupport(
        role="punc",
        name="xlm_roberta_punctuation",
        languages=(
            "af", "am", "ar", "bg", "bn", "de", "el", "en", "es", "et",
            "fa", "fi", "fr", "gu", "hi", "hr", "hu", "id", "is", "it",
            "ja", "kk", "kn", "ko", "ky", "lt", "lv", "mk", "ml", "mr",
            "nl", "or", "pa", "pl", "ps", "pt", "ro", "ru", "rw", "so",
            "sr", "sw", "ta", "te", "tr", "uk", "zh",
        ),
        aliases={
            "en": ("english", "en-us", "en-gb"),
            "zh": ("chinese", "zh-cn", "cn"),
            "rw": ("kinyarwanda",),
            "pa": ("panjabi", "punjabi"),
            "or": ("oriya", "odia"),
        },
        notes="XLM-R punctuation/fullstop/truecase model supports 47 languages including Chinese. Use standalone tests to verify quality for a specific domain.",
    ),
)


def list_models_by_language(language: str, role: str | None = None) -> dict[str, list[dict[str, Any]]]:
    normalized = normalize_language(language)
    result: dict[str, list[dict[str, Any]]] = {}
    for support in MODEL_LANGUAGE_SUPPORT:
        if role is not None and support.role != role:
            continue
        if supports_language(support, normalized):
            result.setdefault(support.role, []).append(_support_to_dict(support))
    return {key: sorted(value, key=lambda item: item["name"]) for key, value in sorted(result.items())}


def list_languages_by_model(model_name: str, role: str | None = None) -> dict[str, Any]:
    normalized_name = model_name.strip().lower()
    matches = [
        support for support in MODEL_LANGUAGE_SUPPORT
        if support.name == normalized_name and (role is None or support.role == role)
    ]
    if not matches:
        raise ValueError(f"Unknown model language support entry: {model_name}")
    if len(matches) == 1:
        return _support_to_dict(matches[0])
    return {"matches": [_support_to_dict(match) for match in matches]}


def supports_language(support: ModelLanguageSupport, language: str) -> bool:
    language_base = language.split("-", 1)[0]
    if ANY_LANGUAGE in support.languages:
        return True
    if language in support.languages:
        return True
    if language_base in support.languages:
        return True
    for canonical, aliases in support.aliases.items():
        canonical_base = canonical.split("-", 1)[0]
        if language == canonical or language_base == canonical_base or language in aliases or language_base in aliases:
            return True
    return False


def _support_to_dict(support: ModelLanguageSupport) -> dict[str, Any]:
    return {
        "role": support.role,
        "name": support.name,
        "configuration_language_format": "canonical language-region id, for example zh_cn",
        "languages": list(support.languages),
        "aliases": {key: list(value) for key, value in support.aliases.items()},
        "notes": support.notes,
        "has_native_timestamps": support.has_native_timestamps,
        "has_native_punctuation": support.has_native_punctuation,
        "supports_batch": support.supports_batch,
    }
