import unittest

import numpy as np

from semantic_asr.adapters.dolphin import DolphinAsr, DolphinAsrConfig, _get_text, _normalize_timestamps
from semantic_asr.adapters.funasr_nano import FunAsrNano, FunAsrNanoConfig
from semantic_asr.adapters.gigaam_v3 import (
    GigaAmV3Asr,
    GigaAmV3Config,
    _disable_encoder_sdpa,
    _normalize_words,
)
from semantic_asr.adapters.hf_whisper import HfWhisperAsr, HfWhisperAsrConfig
from semantic_asr.adapters.nvidia_fastconformer import (
    NvidiaFastConformerAsr,
    NvidiaFastConformerConfig,
    NvidiaFastConformerTimestampProvider,
)
from semantic_asr.adapters.qwen3_asr import Qwen3Asr, Qwen3AsrConfig, _to_float32, normalize_qwen3_asr_language
from semantic_asr.adapters.seamless_m4t import SeamlessM4TAsr, SeamlessM4TConfig
from semantic_asr.adapters.whisper_large import WhisperLarge, WhisperLargeConfig


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


class _WhisperModel:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, wav_path, **kwargs):
        self.kwargs = kwargs
        return {"text": "ok", "segments": []}


class _BatchWhisperModel:
    def __init__(self):
        self.decode_batch_size = None
        self.options = None
        self.device = "cpu"

    def decode(self, mel_batch, options):
        self.decode_batch_size = len(mel_batch)
        self.options = options
        return [
            type("Result", (), {
                "text": f"text-{index}",
                "avg_logprob": -0.1,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
            })()
            for index in range(len(mel_batch))
        ]


class _BeamSensitiveWhisperModel:
    device = "cpu"

    def __init__(self):
        self.decode_batch_sizes = []

    def decode(self, mel_batch, options):
        batch_size = len(mel_batch)
        self.decode_batch_sizes.append(batch_size)
        if options.beam_size and batch_size > 1:
            raise RuntimeError("batched beam decode is unsafe")
        return [
            type("Result", (), {
                "text": f"text-{index}",
                "avg_logprob": -0.1,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
            })()
            for index in range(batch_size)
        ]


class _HfWhisperProcessor:
    def __init__(self):
        self.padding = None

    def __call__(self, audios, sampling_rate, return_tensors, padding):
        self.padding = padding
        return _HfWhisperInputs()

    def batch_decode(self, generated, skip_special_tokens):
        return ["ok" for _ in generated]


class _HfWhisperInputs(dict):
    def to(self, device, dtype):
        return self


class _HfWhisperModel:
    def generate(self, **kwargs):
        return [[1]]


class _NvidiaFastConformerModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, paths, batch_size, timestamps):
        self.calls.append((paths, batch_size, timestamps))
        return [
            type("Hypothesis", (), {
                "text": f"نص {index}.",
                "timestamp": {
                    "word": [
                        {"word": "نص", "start": 0.1, "end": 0.2},
                        {"word": f"{index}.", "start": 0.2, "end": 0.3},
                    ],
                },
            })()
            for index in range(len(paths))
        ]


class _NoopContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class AdapterInputTest(unittest.TestCase):
    def test_dolphin_uses_nonspecial_text_and_word_timestamps(self):
        raw_result = {
            "text": "<ru><RU><asr><notimestamp> привет",
            "text_nospecial": "привет",
            "word_timestamps": [{"word": "привет", "start": 0.1, "end": 0.5}],
        }

        self.assertEqual(_get_text(raw_result), "привет")
        self.assertEqual(_normalize_timestamps(raw_result), [["привет", 0.1, 0.5]])

    def test_dolphin_transcribes_batch_in_one_native_call(self):
        adapter = object.__new__(DolphinAsr)
        adapter.config = DolphinAsrConfig(language="ru_ru", batch_size=16)
        adapter._set_recommended_batch_size()
        calls = []
        adapter._decode_batch = lambda batch_wav: calls.append(batch_wav) or [
            {"text_nospecial": "привет", "word_timestamps": [["привет", 0.0, 0.4]]},
            {"text_nospecial": "пока", "word_timestamps": [["пока", 0.0, 0.3]]},
        ]

        results = adapter.transcribe(
            ["utt1", "utt2"],
            [(16000, np.zeros(16000, dtype=np.float32)), (16000, np.zeros(16000, dtype=np.float32))],
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 2)
        self.assertEqual(adapter.recommended_batch_size, 16)
        self.assertEqual([item["text"] for item in results], ["привет", "пока"])
        self.assertEqual(results[0]["timestamp"], [["привет", 0.0, 0.4]])

    def test_seamless_transcribes_batch_in_one_native_call(self):
        adapter = object.__new__(SeamlessM4TAsr)
        adapter.config = SeamlessM4TConfig(language="ru_ru", batch_size=128)
        adapter._set_recommended_batch_size()
        calls = []
        adapter._decode_batch = lambda batch_wav: calls.append(batch_wav) or ["первый", "второй"]

        results = adapter.transcribe(
            ["utt1", "utt2"],
            [(16000, np.zeros(16000, dtype=np.float32)), (16000, np.zeros(16000, dtype=np.float32))],
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 2)
        self.assertEqual(adapter.recommended_batch_size, 128)
        self.assertEqual([item["text"] for item in results], ["первый", "второй"])

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

        with self.assertRaisesRegex(ValueError, "canonical language-region id"):
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

    def test_whisper_uses_short_audio_decode_settings(self):
        adapter = object.__new__(WhisperLarge)
        adapter.config = WhisperLargeConfig(
            language="ar_sa",
            short_audio_threshold_s=1.0,
            short_beam_size=5,
            short_length_penalty=0.0,
            short_temperature=0.0,
        )
        adapter.model = _BatchWhisperModel()
        adapter._prepare_mel = lambda wav, sample_rate: np.zeros((80, 3000), dtype=np.float32)

        [result] = adapter.transcribe(["utt"], [(16000, np.zeros(8000, dtype=np.float32))])

        self.assertEqual(result["text"], "text-0")
        self.assertEqual(adapter.model.options.language, "ar")
        self.assertEqual(adapter.model.options.beam_size, 5)
        self.assertEqual(adapter.model.options.length_penalty, 0.0)
        self.assertEqual(adapter.model.options.temperature, 0.0)

    def test_whisper_batches_decode_without_word_timestamps(self):
        adapter = object.__new__(WhisperLarge)
        adapter.config = WhisperLargeConfig(language="ru_ru", short_audio_threshold_s=0.0)
        adapter.model = _BatchWhisperModel()
        adapter._prepare_mel = lambda wav, sample_rate: np.zeros((80, 3000), dtype=np.float32)

        results = adapter.transcribe(
            ["utt1", "utt2"],
            [(16000, np.zeros(16000, dtype=np.float32)), (16000, np.zeros(16000, dtype=np.float32))],
        )

        self.assertEqual(adapter.model.decode_batch_size, 2)
        self.assertEqual([item["text"] for item in results], ["text-0", "text-1"])
        self.assertEqual(results[0]["timestamp"], [])

    def test_whisper_splits_batched_beam_decode(self):
        adapter = object.__new__(WhisperLarge)
        adapter.config = WhisperLargeConfig(
            language="ar_sa",
            short_audio_threshold_s=10.0,
            short_beam_size=5,
        )
        adapter.model = _BeamSensitiveWhisperModel()
        adapter._prepare_mel = lambda wav, sample_rate: np.zeros((80, 3000), dtype=np.float32)

        results = adapter.transcribe(
            ["utt1", "utt2"],
            [(16000, np.zeros(16000, dtype=np.float32)), (16000, np.zeros(16000, dtype=np.float32))],
        )

        self.assertEqual(adapter.model.decode_batch_sizes, [1, 1])
        self.assertEqual([item["text"] for item in results], ["text-0", "text-0"])

    def test_whisper_recommends_configured_batch_size(self):
        adapter = object.__new__(WhisperLarge)
        adapter.config = WhisperLargeConfig(batch_size=24)
        adapter._set_recommended_batch_size()

        self.assertEqual(adapter.recommended_batch_size, 24)

    def test_hf_whisper_pads_features_to_model_context(self):
        adapter = object.__new__(HfWhisperAsr)
        adapter.config = HfWhisperAsrConfig(model_name_or_path="mock", device="cpu")
        adapter.dtype = None
        adapter.processor = _HfWhisperProcessor()
        adapter.model = _HfWhisperModel()
        adapter.torch = type("Torch", (), {"no_grad": staticmethod(lambda: _NoopContext())})

        adapter.transcribe(["utt"], [(16000, np.zeros(8000, dtype=np.float32))])

        self.assertEqual(adapter.processor.padding, "max_length")

    def test_gigaam_adapter_batches_text_and_word_timestamps(self):
        adapter = object.__new__(GigaAmV3Asr)
        adapter.config = GigaAmV3Config(batch_size=2)
        adapter._set_recommended_batch_size()
        calls = []
        adapter._transcribe_paths = lambda paths: calls.append(paths) or [
            ("привет.", [["привет", 0.0, 0.4]]),
            ("пока.", [["пока", 0.0, 0.3]]),
        ]

        results = adapter.transcribe(
            ["utt1", "utt2"],
            [(16000, np.zeros(16000, dtype=np.float32)), (16000, np.zeros(16000, dtype=np.float32))],
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(adapter.recommended_batch_size, 2)
        self.assertEqual([item["text"] for item in results], ["привет.", "пока."])
        self.assertEqual(results[0]["timestamp"], [["привет", 0.0, 0.4]])

    def test_gigaam_normalizes_word_timestamp_objects(self):
        word = type("Word", (), {"text": "тест", "start": 0.1, "end": 0.5})()

        self.assertEqual(_normalize_words([word]), [["тест", 0.1, 0.5]])

    def test_gigaam_disables_encoder_sdpa_for_variable_length_batches(self):
        attn = type("Attention", (), {"torch_sdpa_attn": True})()
        layer = type("Layer", (), {"self_attn": attn})()
        encoder = type("Encoder", (), {"layers": [layer]})()
        model = type("Model", (), {"encoder": encoder})()

        _disable_encoder_sdpa(model)

        self.assertFalse(attn.torch_sdpa_attn)

    def test_nvidia_fastconformer_transcribes_batch_with_native_punctuation(self):
        adapter = object.__new__(NvidiaFastConformerAsr)
        adapter.config = NvidiaFastConformerConfig(batch_size=2)
        adapter.model = _NvidiaFastConformerModel()
        adapter._set_recommended_batch_size()

        results = adapter.transcribe(
            ["utt1", "utt2"],
            [(16000, np.zeros(16000, dtype=np.float32)), (16000, np.zeros(16000, dtype=np.float32))],
        )

        self.assertEqual(adapter.model.calls[0][1], 2)
        self.assertTrue(adapter.model.calls[0][2])
        self.assertEqual(len(adapter.model.calls[0][0]), 2)
        self.assertEqual([item["text"] for item in results], ["نص 0.", "نص 1."])
        self.assertEqual(results[0]["timestamp"], [["نص", 0.1, 0.2], ["0.", 0.2, 0.3]])

    def test_nvidia_fastconformer_native_timestamp_provider_requires_words(self):
        provider = NvidiaFastConformerTimestampProvider()

        with self.assertRaisesRegex(ValueError, "must return word timestamps"):
            provider.add_timestamps([{"uttid": "utt", "timestamp": []}], [])


if __name__ == "__main__":
    unittest.main()
