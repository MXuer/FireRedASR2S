import json
import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from semantic_asr.config import build_pipeline_from_profile, parse_pipeline_profile
from semantic_asr.registry import ComponentRegistry
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
            self.assertEqual(result["text"], "hello. hello.")
            self.assertEqual(len(result["timestamp_segments"]), 2)
            self.assertEqual(result["timestamp_segments"][0]["timestamps"][0]["text"], "hello")
            self.assertEqual(result["timestamp_segments"][0]["timestamps"][0]["start_ms"], 0)
            self.assertEqual(result["raw_vad_segments_ms"], [[0, 400], [500, 1000]])
            self.assertEqual(result["asr_vad_segments_ms"], [[0, 400], [500, 1000]])
            self.assertEqual(result["vad_frame_speech_probs"]["frame_shift_ms"], 10)
            self.assertEqual(result["vad_frame_speech_probs"]["probs"], [0.0, 0.8, 0.1])

    def test_default_pipeline_uses_raw_vad_segments_for_asr_and_timestamp(self):
        profile = parse_pipeline_profile(fake_profile())
        fake_asr = FakeAsr()
        fake_timestamp = FakeTimestampProvider()
        pipeline = build_pipeline_from_profile(
            profile,
            registry=fake_registry_with_instances(fake_asr, fake_timestamp),
        )

        result = pipeline.process(self._write_silence_wav(), "sample")

        self.assertEqual(fake_timestamp.segment_ranges, [(0.0, 0.4), (0.5, 1.0)])
        self.assertEqual(result["raw_vad_segments_ms"], [(0, 400), (500, 1000)])
        self.assertEqual(result["asr_vad_segments_ms"], [(0, 400), (500, 1000)])

    @staticmethod
    def _write_silence_wav() -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, np.zeros(16000, dtype=np.int16), 16000)
        return tmp.name


if __name__ == "__main__":
    unittest.main()
