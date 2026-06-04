import unittest

from semantic_asr.language_mapping import canonical_language_id, dolphin_language, model_language, require_canonical_language_id
from semantic_asr.language_configs import resolve_language_profile


class LanguageMappingTest(unittest.TestCase):
    def test_canonicalizes_common_aliases(self):
        self.assertEqual(canonical_language_id("zh"), "zh_cn")
        self.assertEqual(canonical_language_id("en-US"), "en_us")
        self.assertEqual(canonical_language_id("Cantonese"), "yue_hk")
        self.assertEqual(require_canonical_language_id("zh_cn"), "zh_cn")
        with self.assertRaisesRegex(ValueError, "canonical id zh_cn"):
            require_canonical_language_id("Chinese")

    def test_maps_canonical_id_to_model_native_values(self):
        self.assertEqual(model_language("qwen3_asr_1_7b", "th_th"), "Thai")
        self.assertEqual(model_language("qwen3_forced_aligner", "ru_ru"), "Russian")
        self.assertEqual(model_language("funasr_nano", "ja_jp"), "Japanese")
        self.assertEqual(model_language("whisper_large", "zh_cn"), "zh")
        self.assertEqual(model_language("seamless_m4t_v2_large", "ar_sa"), "arb")
        self.assertEqual(dolphin_language("th_th"), ("th", "TH"))
        self.assertEqual(dolphin_language("yue_hk"), ("ct", "HK"))

    def test_rejects_unsupported_model_language(self):
        with self.assertRaisesRegex(ValueError, "funasr_nano does not support"):
            model_language("funasr_nano", "th_th")

    def test_legacy_language_profile_resolver_uses_canonical_ids(self):
        self.assertEqual(resolve_language_profile("zh").language, "zh_cn")
        self.assertEqual(resolve_language_profile("ru-ru").language, "ru_ru")


if __name__ == "__main__":
    unittest.main()
