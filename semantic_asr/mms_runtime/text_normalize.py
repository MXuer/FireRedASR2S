class BaseTextNormalizer:
    def norm(self, text: str) -> str:
        return text

    def batch_norm(self, texts: list[str]) -> list[str]:
        return [self.norm(text) for text in texts]


class ZH_CN_TextNormalizer(BaseTextNormalizer):
    def __init__(self, overwrite_cache: bool = False) -> None:
        try:
            from tn.chinese.normalizer import Normalizer

            self.normalizer = Normalizer(overwrite_cache=overwrite_cache)
        except Exception:
            self.normalizer = None

    def norm(self, text: str) -> str:
        if self.normalizer is None:
            return text
        return self.normalizer.normalize(text)


class EN_TextNormalizer(BaseTextNormalizer):
    def __init__(self, overwrite_cache: bool = False) -> None:
        try:
            from tn.english.normalizer import Normalizer

            self.normalizer = Normalizer(overwrite_cache=overwrite_cache)
        except Exception:
            self.normalizer = None

    def norm(self, text: str) -> str:
        if self.normalizer is None:
            return text
        return self.normalizer.normalize(text)


class TRICKY_TextNormalizer(EN_TextNormalizer):
    pass


LANG2TEXTNORMALIZER = {
    "zh_cn": ZH_CN_TextNormalizer,
    "en_us": EN_TextNormalizer,
    "tricky": TRICKY_TextNormalizer,
}
