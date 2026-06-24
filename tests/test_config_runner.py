import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

from semantic_asr.config import build_pipeline_from_profile, parse_pipeline_profile
from semantic_asr.registry import ComponentRegistry
from semantic_asr.registry import create_default_registry
from semantic_asr.run_pipeline import run_from_config


class FakeVad:
    def detect(self, wav_path: str):
        return {
            "timestamps": [(0.0, 0.4), (0.5, 1.0)],
            "frame_speech_probs": {
                "frame_shift_ms": 10,
                "frame_length_ms": 25,
                "probs": [0.0, 0.8, 0.1],
            },
        }


class FakeAsr:
    def __init__(self):
        self.batch_wav = []

    def transcribe(self, batch_uttid, batch_wav):
        self.batch_wav.extend(batch_wav)
        return [
            {
                "uttid": uttid,
                "text": "hello.",
                "confidence": 0.9,
                "timestamp": [["hello", 0.0, 0.7], [".", 0.7, 0.8]],
            }
            for uttid in batch_uttid
        ]


class FakeTimestampProvider:
    def __init__(self):
        self.segment_ranges = []

    def add_timestamps(self, batch_asr_result, batch_segments):
        self.segment_ranges.extend((segment.start_s, segment.end_s) for segment in batch_segments)
        return list(batch_asr_result)


class FakePunc:
    def process_with_timestamp(self, batch_timestamp, batch_uttid):
        return [
            {
                "uttid": uttid,
                "punc_sentences": [{"start_s": 0.0, "end_s": 0.8, "punc_text": "hello."}],
            }
            for uttid in batch_uttid
        ]


def fake_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    registry.register("vad", "fake_vad", lambda params: FakeVad())
    registry.register("asr", "fake_asr", lambda params: FakeAsr())
    registry.register("timestamp", "fake_timestamp", lambda params: FakeTimestampProvider())
    registry.register("punc", "fake_punc", lambda params: FakePunc())
    return registry


def fake_registry_with_instances(fake_asr: FakeAsr, fake_timestamp: FakeTimestampProvider) -> ComponentRegistry:
    registry = ComponentRegistry()
    registry.register("vad", "fake_vad", lambda params: FakeVad())
    registry.register("asr", "fake_asr", lambda params: fake_asr)
    registry.register("timestamp", "fake_timestamp", lambda params: fake_timestamp)
    registry.register("punc", "fake_punc", lambda params: FakePunc())
    return registry


def fake_profile(outdir: str = "unused") -> dict:
    return {
        "name": "fake_profile",
        "language": "en_us",
        "components": {
            "vad": {"name": "fake_vad", "params": {}},
            "asr": {"name": "fake_asr", "params": {}},
            "timestamp": {"name": "fake_timestamp", "params": {}},
            "punc": {"name": "fake_punc", "params": {}},
        },
        "pipeline": {
            "asr_batch_size": 2,
            "punc_batch_size": 2,
            "strip_punctuation_before_punc": False,
            "output_vad_pad_s": 0.0,
        },
        "output": {
            "outdir": outdir,
            "write_textgrid": False,
            "write_srt": False,
            "write_csv": False,
            "copy_resolved_config": True,
        },
    }


class ConfigRunnerTest(unittest.TestCase):
    def test_build_pipeline_from_registered_components(self):
        profile = parse_pipeline_profile(fake_profile())
        pipeline = build_pipeline_from_profile(profile, registry=fake_registry())

        self.assertEqual(pipeline.config.asr_batch_size, 2)
        self.assertIsInstance(pipeline.vad, FakeVad)

    def test_sentence_boundary_fusion_config_is_preserved(self):
        raw = fake_profile()
        raw["pipeline"]["preserve_sentence_gaps"] = True
        raw["pipeline"]["sentence_boundary_fusion"] = {
            "enabled": True,
            "merge_max_token_gap_s": 0.25,
        }

        profile = parse_pipeline_profile(raw)
        pipeline = build_pipeline_from_profile(profile, registry=fake_registry())

        resolved = pipeline._sentence_boundary_fusion_config()
        self.assertTrue(resolved.enabled)
        self.assertEqual(resolved.merge_max_token_gap_s, 0.25)
        self.assertTrue(resolved.preserve_sentence_gaps)

    def test_missing_required_component_fails_validation(self):
        raw = fake_profile()
        del raw["components"]["timestamp"]

        with self.assertRaisesRegex(ValueError, "components.timestamp"):
            parse_pipeline_profile(raw)

    def test_profile_requires_canonical_language(self):
        raw = fake_profile()
        raw["language"] = "English"

        with self.assertRaisesRegex(ValueError, "canonical id en_us"):
            parse_pipeline_profile(raw)

    def test_profile_language_is_injected_into_language_aware_components(self):
        raw = fake_profile()
        raw["language"] = "zh_cn"
        raw["components"]["asr"]["name"] = "whisper_large"
        raw["components"]["timestamp"]["name"] = "mms_forced_aligner"
        captured = {}
        registry = fake_registry()
        registry.register("asr", "whisper_large", lambda params: captured.setdefault("asr", dict(params)) or FakeAsr())
        registry.register("timestamp", "mms_forced_aligner", lambda params: captured.setdefault("timestamp", dict(params)) or FakeTimestampProvider())

        build_pipeline_from_profile(parse_pipeline_profile(raw), registry=registry)

        self.assertEqual(captured["asr"]["language"], "zh_cn")
        self.assertEqual(captured["timestamp"]["language"], "zh_cn")

    def test_gigaam_profile_uses_native_timestamps(self):
        with open("configs/ru_ru_gigaam_v3.json", encoding="utf-8") as fin:
            raw = json.load(fin)

        profile = parse_pipeline_profile(raw)

        self.assertEqual(profile.components["asr"].name, "gigaam_v3")
        self.assertEqual(profile.components["asr"].params["batch_size"], 192)
        self.assertEqual(profile.components["timestamp"].name, "gigaam_v3_native")
        self.assertEqual(profile.components["punc"].name, "asr_text")

    def test_new_minimal_model_profiles_parse(self):
        for path, asr_name, punc_name in [
            ("configs/vi_vn_phowhisper.json", "phowhisper_large", "asr_text"),
            ("configs/th_th_whisper_th.json", "whisper_th_large_v3_combined", "asr_text"),
            ("configs/hi_in_cadence.json", "seamless_m4t_v2_large", "cadence_fast"),
        ]:
            with self.subTest(path=path):
                with open(path, encoding="utf-8") as fin:
                    profile = parse_pipeline_profile(json.load(fin))
                self.assertEqual(profile.components["asr"].name, asr_name)
                self.assertEqual(profile.components["punc"].name, punc_name)

    def test_firered_asr_registry_builds_with_native_timestamps_enabled(self):
        registry = create_default_registry()
        fake_model = mock.Mock()
        fake_model.transcribe.return_value = [
            {
                "uttid": "utt",
                "text": "你好",
                "timestamp": [["你", 0.0, 0.1], ["好", 0.1, 0.2]],
            }
        ]

        with mock.patch("semantic_asr.adapters.firered.FireRedAsr2.from_pretrained", return_value=fake_model) as build:
            asr = registry.build(
                "asr",
                "firered_asr",
                {
                    "asr_type": "aed",
                    "model_dir": "mock_firered",
                    "beam_size": 5,
                },
            )

        args = build.call_args.args
        self.assertEqual(args[0], "aed")
        self.assertEqual(args[1], "mock_firered")
        self.assertEqual(args[2].beam_size, 5)
        self.assertTrue(args[2].return_timestamp)
        self.assertEqual(asr.transcribe(["utt"], [(16000, np.zeros(1600, dtype=np.int16))])[0]["text"], "你好")

    def test_firered_asr_native_timestamp_provider_validates_timestamps(self):
        registry = create_default_registry()
        provider = registry.build("timestamp", "firered_asr_native")

        result = provider.add_timestamps(
            [{"uttid": "utt", "timestamp": [["你", 0.0, 0.1]]}],
            [],
        )

        self.assertEqual(result[0]["timestamp"], [["你", 0.0, 0.1]])
        with self.assertRaisesRegex(ValueError, "FireRed ASR must return timestamp"):
            provider.add_timestamps([{"uttid": "utt", "text": "你好"}], [])

    def test_runner_writes_json_jsonl_and_resolved_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "sample.wav")
            config_path = os.path.join(tmpdir, "config.json")
            outdir = os.path.join(tmpdir, "out")

            sf.write(wav_path, np.zeros(16000, dtype=np.int16), 16000)
            with open(config_path, "w", encoding="utf-8") as fout:
                json.dump(fake_profile(outdir), fout)

            outputs = run_from_config(
                config_path,
                wav_path,
                uttid="sample",
                registry=fake_registry(),
            )

            self.assertTrue(os.path.exists(outputs["json"]))
            self.assertTrue(os.path.exists(outputs["jsonl"]))
            self.assertTrue(os.path.exists(outputs["resolved_config"]))
            with open(outputs["json"], "r", encoding="utf-8") as fin:
                result = json.load(fin)
            self.assertEqual(result["text"], "hello.")
            self.assertEqual(len(result["timestamp_segments"]), 1)
            self.assertEqual(result["timestamp_segments"][0]["timestamps"][0]["text"], "hello")
            self.assertEqual(result["timestamp_segments"][0]["timestamps"][0]["start_ms"], 0)
            self.assertEqual(result["raw_vad_segments_ms"], [[0, 400], [500, 1000]])
            self.assertEqual(result["asr_vad_segments_ms"], [[0, 1000]])
            self.assertEqual(result["vad_frame_speech_probs"]["frame_shift_ms"], 10)
            self.assertEqual(result["vad_frame_speech_probs"]["probs"], [0.0, 0.8, 0.1])

    def test_default_pipeline_postprocesses_vad_segments_for_asr_and_timestamp(self):
        profile = parse_pipeline_profile(fake_profile())
        fake_asr = FakeAsr()
        fake_timestamp = FakeTimestampProvider()
        pipeline = build_pipeline_from_profile(
            profile,
            registry=fake_registry_with_instances(fake_asr, fake_timestamp),
        )

        result = pipeline.process(self._write_silence_wav(), "sample")

        self.assertEqual(fake_timestamp.segment_ranges, [(0.0, 1.0)])
        self.assertEqual(result["raw_vad_segments_ms"], [(0, 400), (500, 1000)])
        self.assertEqual(result["asr_vad_segments_ms"], [(0, 1000)])

    @staticmethod
    def _write_silence_wav() -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, np.zeros(16000, dtype=np.int16), 16000)
        return tmp.name


if __name__ == "__main__":
    unittest.main()
