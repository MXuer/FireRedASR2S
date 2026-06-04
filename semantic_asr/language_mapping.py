from semantic_asr.mms_runtime.model_registry import MMS_CODE_MAP


DEFAULT_LANGUAGE_IDS = {
    "ar": "ar_sa",
    "de": "de_de",
    "en": "en_us",
    "es": "es_mx",
    "fa": "fa_ir",
    "fil": "fil_ph",
    "fr": "fr_fr",
    "hi": "hi_in",
    "id": "id_id",
    "it": "it_it",
    "ja": "ja_jp",
    "ko": "ko_kr",
    "nl": "nl_nl",
    "pl": "pl_pl",
    "pt": "pt_br",
    "ro": "ro_ro",
    "ru": "ru_ru",
    "sv": "sv_se",
    "th": "th_th",
    "tr": "tr_tr",
    "vi": "vi_vn",
    "yue": "yue_hk",
    "zh": "zh_cn",
}

LANGUAGE_ALIASES = {
    "arabic": "ar_sa",
    "cantonese": "yue_hk",
    "chinese": "zh_cn",
    "english": "en_us",
    "japanese": "ja_jp",
    "korean": "ko_kr",
    "russian": "ru_ru",
    "thai": "th_th",
    "vietnamese": "vi_vn",
}

QWEN_LANGUAGE_NAMES = {
    "ar": "Arabic", "cs": "Czech", "da": "Danish", "de": "German",
    "el": "Greek", "en": "English", "es": "Spanish", "fa": "Persian",
    "fi": "Finnish", "fil": "Filipino", "fr": "French", "hi": "Hindi",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "mk": "Macedonian", "ms": "Malay", "nl": "Dutch",
    "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sv": "Swedish", "th": "Thai", "tr": "Turkish", "vi": "Vietnamese",
    "yue": "Cantonese", "zh": "Chinese",
}

QWEN_ALIGNER_LANGUAGE_NAMES = {
    key: value for key, value in QWEN_LANGUAGE_NAMES.items()
    if key in {"de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "th", "zh"}
}

FUNASR_LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
}

SEAMLESS_LANGUAGE_CODES = {
    "ar_sa": "arb",
    "en_us": "eng",
    "en_gb": "eng",
    "hi_in": "hin",
    "ja_jp": "jpn",
    "ko_kr": "kor",
    "pt_br": "por",
    "ru_ru": "rus",
    "th_th": "tha",
    "vi_vn": "vie",
    "zh_cn": "cmn",
    "zh_tw": "cmn",
}

DOLPHIN_LANGUAGE_CODES = {
    "ar", "az", "ba", "bn", "ct", "fa", "fil", "gu", "hi", "id", "ja",
    "jv", "kab", "kk", "km", "ko", "ks", "ky", "lo", "mn", "mr", "ms",
    "my", "ne", "or", "pa", "ps", "ru", "si", "su", "ta", "te", "tg",
    "th", "tl", "ug", "ur", "uz", "vi", "zh",
}


def canonical_language_id(language: str) -> str:
    normalized = language.strip().lower().replace("-", "_")
    normalized = LANGUAGE_ALIASES.get(normalized, normalized)
    normalized = DEFAULT_LANGUAGE_IDS.get(normalized, normalized)
    if "_" not in normalized:
        raise ValueError(
            f"Language must use a canonical language-region id such as zh_cn: {language}"
        )
    return normalized


def require_canonical_language_id(language: str) -> str:
    canonical = canonical_language_id(language)
    if language != canonical:
        raise ValueError(f"Language must be configured as canonical id {canonical}, not {language}")
    return canonical


def model_language(model_name: str, language: str) -> str:
    canonical = canonical_language_id(language)
    base = canonical.split("_", 1)[0]
    mappings = {
        "funasr_nano": FUNASR_LANGUAGE_NAMES,
        "mms_forced_aligner": MMS_CODE_MAP,
        "qwen3_asr_1_7b": QWEN_LANGUAGE_NAMES,
        "qwen3_forced_aligner": QWEN_ALIGNER_LANGUAGE_NAMES,
        "seamless_m4t_v2_large": SEAMLESS_LANGUAGE_CODES,
        "whisper_large": None,
    }
    if model_name not in mappings:
        raise ValueError(f"Unknown model language mapping: {model_name}")
    mapping = mappings[model_name]
    if mapping is None:
        return base
    key = canonical if model_name in {"mms_forced_aligner", "seamless_m4t_v2_large"} else base
    if model_name == "mms_forced_aligner":
        key = {"vi_vn": "vi_in", "yue_hk": "ct_hk"}.get(key, key)
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"{model_name} does not support language {canonical}") from exc


def dolphin_language(language: str) -> tuple[str, str]:
    canonical = canonical_language_id(language)
    base, region = canonical.split("_", 1)
    if base == "yue":
        base = "ct"
    if base not in DOLPHIN_LANGUAGE_CODES:
        raise ValueError(f"dolphin does not support language {canonical}")
    return base, region.upper()
