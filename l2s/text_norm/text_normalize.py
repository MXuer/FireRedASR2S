import os
import re
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer


class BaseTextNormalizer:
    def __init__(self) -> None:
        pass
    
    def norm(self, text):
        raise NotImplementedError
    def batch_norm(self, texts):
        norm_texts = []
        for text in texts:
            norm_texts.append(self.norm(text))
        return norm_texts
    

class ZH_CN_TextNormalizer(BaseTextNormalizer):
    def __init__(self, overwrite_cache=False) -> None:
        super().__init__()
        self.normalizer = ZhNormalizer(overwrite_cache=overwrite_cache)

    def norm(self, text):
        norm_text = self.normalizer.normalize(text)
        return norm_text
    

class EN_TextNormalizer(BaseTextNormalizer):
    def __init__(self, overwrite_cache=False) -> None:
        super().__init__()
        self.normalizer = EnNormalizer(overwrite_cache=overwrite_cache)

    def norm(self, text):
        norm_text = self.normalizer.normalize(text)
        return norm_text
    

class TRICKY_TextNormalizer(BaseTextNormalizer):
    """
    很多其他的语种，我们没有对应的TN程序，
    为了避免开头和结尾的部分存在数字之类的特殊符号，
    对于这些语种，我们暂时会把这些数据变成英文（所有的不支持TN的语种，都使用英文）
    """

    def __init__(self, overwrite_cache=False) -> None:
        super().__init__()
        self.normalizer = EnNormalizer(overwrite_cache=overwrite_cache)

    def norm(self, text):
        # TODO: fix this @duhu
        norm_text = self.normalizer.normalize(text)
        return norm_text
        

LANG2TEXTNORMALIZER = {
    'zh_cn': ZH_CN_TextNormalizer,
    'en_us': EN_TextNormalizer,
    'tricky': TRICKY_TextNormalizer
}