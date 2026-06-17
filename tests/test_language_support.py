import unittest

from semantic_asr.language_support import list_languages_by_model, list_models_by_language
from semantic_asr.registry import create_default_registry


class LanguageSupportTest(unittest.TestCase):
    def test_language_query_normalizes_region_underscore(self):
        result = list_models_by_language("en_us")

        self.assertIn("vad", result)
        self.assertIn("asr", result)
        self.assertIn("timestamp", result)
        self.assertIn("punc", result)
        self.assertIn("qwen3_asr_1_7b", {item["name"] for item in result["asr"]})
        self.assertIn("whisper_large", {item["name"] for item in result["asr"]})
        self.assertIn("xlm_roberta_punctuation", {item["name"] for item in result["punc"]})

    def test_region_language_falls_back_to_base_language(self):
        ja_result = list_models_by_language("ja_jp")
        hi_result = list_models_by_language("hi_in")
        vi_result = list_models_by_language("vi_vn")

        self.assertIn("qwen3_asr_1_7b", {item["name"] for item in ja_result["asr"]})
        self.assertIn("whisper_large", {item["name"] for item in ja_result["asr"]})
        self.assertIn("xlm_roberta_punctuation", {item["name"] for item in ja_result["punc"]})
        self.assertIn("qwen3_asr_1_7b", {item["name"] for item in hi_result["asr"]})
        self.assertIn("mms_forced_aligner", {item["name"] for item in hi_result["timestamp"]})
        self.assertIn("mms_forced_aligner", {item["name"] for item in vi_result["timestamp"]})

    def test_corrected_qwen_and_funasr_language_support(self):
        th_result = list_models_by_language("th_th", role="asr")
        ja_result = list_models_by_language("ja_jp", role="asr")
        uk_result = list_models_by_language("uk_ua", role="asr")
        yue_result = list_models_by_language("Cantonese", role="asr")

        self.assertIn("qwen3_asr_1_7b", {item["name"] for item in th_result["asr"]})
        self.assertNotIn("funasr_nano", {item["name"] for item in th_result["asr"]})
        self.assertIn("funasr_nano", {item["name"] for item in ja_result["asr"]})
        self.assertNotIn("qwen3_asr_1_7b", {item["name"] for item in uk_result["asr"]})
        self.assertIn("qwen3_asr_1_7b", {item["name"] for item in yue_result["asr"]})

        qwen = list_languages_by_model("qwen3_asr_1_7b", role="asr")
        funasr = list_languages_by_model("funasr_nano", role="asr")
        funasr_timestamp = list_languages_by_model("funasr_native", role="timestamp")
        self.assertEqual(len(qwen["languages"]), 30)
        self.assertEqual(set(funasr["languages"]), {"zh", "en", "ja"})
        self.assertEqual(set(funasr_timestamp["languages"]), {"zh", "en", "ja"})

    def test_model_query_returns_languages(self):
        result = list_languages_by_model("dolphin", role="asr")

        self.assertEqual(result["role"], "asr")
        self.assertIn("zh", result["languages"])
        self.assertTrue(result["has_native_timestamps"])

    def test_new_asr_models_are_registered(self):
        registry = create_default_registry()
        names = set(registry.names("asr"))

        self.assertIn("qwen3_asr_1_7b", names)
        self.assertIn("dolphin", names)
        self.assertIn("seamless_m4t_v2_large", names)
        self.assertIn("firered_asr", names)

    def test_ten_vad_is_registered(self):
        registry = create_default_registry()
        names = set(registry.names("vad"))
        pt_result = list_models_by_language("pt_br", role="vad")

        self.assertIn("ten_vad", names)
        self.assertIn("ten_vad", {item["name"] for item in pt_result["vad"]})

    def test_new_punctuation_model_is_registered(self):
        registry = create_default_registry()
        names = set(registry.names("punc"))

        self.assertIn("xlm_roberta_punctuation", names)
        self.assertIn("naqta", names)
        self.assertIn("yue_punctuation", names)
        self.assertIn("ct_punc", names)

    def test_ct_punc_is_chinese_only(self):
        zh_result = list_models_by_language("zh_cn", role="punc")
        en_result = list_models_by_language("en_us", role="punc")

        self.assertIn("ct_punc", {item["name"] for item in zh_result["punc"]})
        self.assertNotIn("ct_punc", {item["name"] for item in en_result["punc"]})

    def test_naqta_punctuation_is_arabic_only(self):
        ar_result = list_models_by_language("ar_sa", role="punc")
        en_result = list_models_by_language("en_us", role="punc")

        self.assertIn("naqta", {item["name"] for item in ar_result["punc"]})
        self.assertNotIn("naqta", {item["name"] for item in en_result["punc"]})

    def test_yue_punctuation_is_cantonese_only(self):
        yue_result = list_models_by_language("yue_hk", role="punc")
        alias_result = list_models_by_language("Cantonese", role="punc")
        zh_result = list_models_by_language("zh_cn", role="punc")

        self.assertIn("yue_punctuation", {item["name"] for item in yue_result["punc"]})
        self.assertIn("yue_punctuation", {item["name"] for item in alias_result["punc"]})
        self.assertNotIn("yue_punctuation", {item["name"] for item in zh_result["punc"]})

    def test_mms_forced_aligner_is_registered_for_timestamp(self):
        registry = create_default_registry()
        names = set(registry.names("timestamp"))
        zh_result = list_models_by_language("zh_cn", role="timestamp")

        self.assertIn("mms_forced_aligner", names)
        self.assertIn("mms_forced_aligner", {item["name"] for item in zh_result["timestamp"]})

    def test_firered_asr_and_native_timestamp_are_chinese_components(self):
        registry = create_default_registry()
        asr_names = set(registry.names("asr"))
        timestamp_names = set(registry.names("timestamp"))
        zh_asr = list_models_by_language("zh_cn", role="asr")
        zh_timestamp = list_models_by_language("zh_cn", role="timestamp")
        en_asr = list_models_by_language("en_us", role="asr")
        en_timestamp = list_models_by_language("en_us", role="timestamp")

        self.assertIn("firered_asr", asr_names)
        self.assertIn("firered_asr_native", timestamp_names)
        self.assertIn("firered_asr", {item["name"] for item in zh_asr["asr"]})
        self.assertIn("firered_asr_native", {item["name"] for item in zh_timestamp["timestamp"]})
        self.assertNotIn("firered_asr", {item["name"] for item in en_asr["asr"]})
        self.assertNotIn("firered_asr_native", {item["name"] for item in en_timestamp["timestamp"]})

    def test_xlm_roberta_punctuation_supports_full_language_list(self):
        zh_result = list_models_by_language("zh_cn", role="punc")
        am_result = list_models_by_language("am", role="punc")
        rw_result = list_models_by_language("kinyarwanda", role="punc")

        self.assertIn("xlm_roberta_punctuation", {item["name"] for item in zh_result["punc"]})
        self.assertIn("xlm_roberta_punctuation", {item["name"] for item in am_result["punc"]})
        self.assertIn("xlm_roberta_punctuation", {item["name"] for item in rw_result["punc"]})


if __name__ == "__main__":
    unittest.main()
