import unittest

import numpy as np

from semantic_asr.adapters.dolphin import _get_text, _normalize_timestamps
from semantic_asr.adapters.funasr_nano import FunAsrNano, FunAsrNanoConfig
from semantic_asr.adapters.qwen3_asr import Qwen3Asr, Qwen3AsrConfig, _to_float32, normalize_qwen3_asr_language


class _NoBatchFunAsrModel:
    def generate(self, input, **kwargs):
        if isinstance(input, list) or kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")
        return [{"text": input}]


class _QwenModel:
    def __init__(self):
        self.language = None

    def transcribe(self, audio, language, return_time_stamps):
        self.language = language
        return [{"text": "ok"} for _ in audio]


class AdapterInputTest(unittest.TestCase):
    def test_dolphin_uses_nonspecial_text_and_word_timestamps(self):
        raw_result = {
            "text": "<ru><RU><asr><notimestamp> привет",
            "text_nospecial": "привет",
            "word_timestamps": [{"word": "привет", "start": 0.1, "end": 0.5}],
        }

        self.assertEqual(_get_text(raw_result), "привет")
        self.assertEqual(_normalize_timestamps(raw_result), [["привет", 0.1, 0.5]])

    def test_qwen_integer_audio_is_normalized_to_float32(self):
        wav = np.array([-32768, 0, 32767], dtype=np.int16)

        normalized = _to_float32(wav)

        self.assertEqual(normalized.dtype, np.float32)
        self.assertGreaterEqual(float(normalized.min()), -1.0)
        self.assertLessEqual(float(normalized.max()), 1.0)

    def test_qwen_language_code_is_converted_to_full_model_name(self):
        self.assertEqual(normalize_qwen3_asr_language("th_th"), "Thai")
        self.assertEqual(normalize_qwen3_asr_language("yue"), "Cantonese")
        self.assertEqual(normalize_qwen3_asr_language("English"), "English")

        with self.assertRaisesRegex(ValueError, "Unsupported Qwen3-ASR language"):
            normalize_qwen3_asr_language("uk")

    def test_qwen_adapter_sends_full_language_name_to_model(self):
        adapter = Qwen3Asr.__new__(Qwen3Asr)
        adapter.config = Qwen3AsrConfig(language="th_th")
        adapter.model = _QwenModel()

        adapter.transcribe(["test"], [(16000, np.zeros(160, dtype=np.float32))])

        self.assertEqual(adapter.model.language, ["Thai"])

    def test_funasr_falls_back_when_batch_decoding_is_unavailable(self):
        adapter = object.__new__(FunAsrNano)
        adapter.config = FunAsrNanoConfig(batch_size=4)
        adapter.model = _NoBatchFunAsrModel()

        results = adapter._generate(["a.wav", "b.wav"])

        self.assertEqual(results, [{"text": "a.wav"}, {"text": "b.wav"}])


if __name__ == "__main__":
    unittest.main()
