import unittest

from semantic_asr.adapters.mms_forced_aligner import MmsForcedAlignerConfig, MmsForcedAlignerTimestampProvider
from semantic_asr.core import SpeechSegment
from semantic_asr.language_mapping import model_language


class MmsForcedAlignerTest(unittest.TestCase):
    def test_prepare_tokens_splits_chinese_characters(self):
        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="zh_cn")

        self.assertEqual(provider._prepare_tokens("你好 world"), ["你", "好", "world"])

    def test_prepare_tokens_splits_korean_characters(self):
        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="ko_kr")

        self.assertEqual(provider._prepare_tokens("한국어 test"), ["한", "국", "어", "test"])

    def test_prepare_tokens_splits_japanese_kana_and_kanji(self):
        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="ja_jp")

        self.assertEqual(provider._prepare_tokens("日本語かな test"), ["日", "本", "語", "か", "な", "test"])

    def test_mms_language_uses_canonical_id_and_maps_to_native_code(self):
        self.assertEqual(MmsForcedAlignerConfig().language, "zh_cn")
        self.assertEqual(model_language("mms_forced_aligner", "zh_cn"), "cmn")
        self.assertEqual(model_language("mms_forced_aligner", "vi_vn"), "vie")

    def test_provider_passes_native_language_code_to_mms_runtime(self):
        class FakeAligner:
            def align(inner_self, *args, **kwargs):
                inner_self.language = kwargs["language"]
                return []

        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="zh_cn")
        provider.aligner = FakeAligner()
        segment = SpeechSegment("test", 0.0, 1.0, 16000, [0.0] * 16000)

        provider.add_timestamps([{"uttid": "test", "text": "你好"}], [segment])

        self.assertEqual(provider.aligner.language, "cmn")

    def test_provider_forces_use_star_false(self):
        class FakeAligner:
            def __init__(inner_self, *args, **kwargs):
                pass

        config = MmsForcedAlignerConfig(use_star=True)

        original_aligner = MmsForcedAlignerTimestampProvider.__init__.__globals__["MmsAligner"]
        try:
            MmsForcedAlignerTimestampProvider.__init__.__globals__["MmsAligner"] = FakeAligner
            provider = MmsForcedAlignerTimestampProvider(config)
        finally:
            MmsForcedAlignerTimestampProvider.__init__.__globals__["MmsAligner"] = original_aligner

        self.assertFalse(provider.config.use_star)

    def test_normalize_alignment(self):
        aligned = [{"text": "hello", "start": 0.1, "end": 0.3}]

        self.assertEqual(
            MmsForcedAlignerTimestampProvider._normalize_alignment(aligned),
            [["hello", 0.1, 0.3]],
        )


if __name__ == "__main__":
    unittest.main()
