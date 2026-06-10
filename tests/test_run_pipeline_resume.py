import json
import os
import tempfile
import unittest
from unittest import mock

from semantic_asr.config import ComponentSpec, OutputConfig, PipelineProfileConfig
from semantic_asr.run_pipeline import output_artifacts_complete, run_profile


def _profile(outdir: str) -> PipelineProfileConfig:
    return PipelineProfileConfig(
        name="test",
        language="en_us",
        components={
            "vad": ComponentSpec(name="fake"),
            "asr": ComponentSpec(name="fake"),
            "timestamp": ComponentSpec(name="fake"),
            "punc": ComponentSpec(name="fake"),
        },
        output=OutputConfig(outdir=outdir, copy_resolved_config=False),
    )


def _result() -> dict:
    return {
        "uttid": "sample",
        "dur_s": 2.0,
        "text": "hello",
        "sentences": [{"start_ms": 0, "end_ms": 1000, "text": "hello", "asr_confidence": 0}],
        "words": [],
        "vad_segments_ms": [],
        "raw_vad_segments_ms": [],
        "asr_vad_segments_ms": [],
        "timestamp_segments": [],
        "wav_path": "sample.wav",
    }


class RunPipelineResumeTest(unittest.TestCase):
    def test_existing_json_rebuilds_missing_exports_without_model_inference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _profile(tmpdir)
            json_path = os.path.join(tmpdir, "sample.json")
            with open(json_path, "w", encoding="utf-8") as fout:
                json.dump(_result(), fout)

            with mock.patch("semantic_asr.run_pipeline.build_pipeline_from_profile") as build:
                outputs = run_profile(profile, "sample.wav", uttid="sample")

            build.assert_not_called()
            self.assertEqual(outputs["json"], json_path)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "asr_tg", "sample.TextGrid")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "asr_srt", "sample.srt")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "asr_csv", "sample.csv")))

    def test_existing_complete_outputs_skip_export_rewrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _profile(tmpdir)
            json_path = os.path.join(tmpdir, "sample.json")
            with open(json_path, "w", encoding="utf-8") as fout:
                json.dump(_result(), fout)
            os.makedirs(os.path.join(tmpdir, "asr_tg"))
            os.makedirs(os.path.join(tmpdir, "asr_srt"))
            os.makedirs(os.path.join(tmpdir, "asr_csv"))
            paths = [
                os.path.join(tmpdir, "asr_tg", "sample.TextGrid"),
                os.path.join(tmpdir, "asr_srt", "sample.srt"),
                os.path.join(tmpdir, "asr_csv", "sample.csv"),
            ]
            for path in paths:
                with open(path, "w", encoding="utf-8") as fout:
                    fout.write("sentinel")

            self.assertTrue(output_artifacts_complete(tmpdir, "sample", profile))
            with mock.patch("semantic_asr.run_pipeline.build_pipeline_from_profile") as build:
                outputs = run_profile(profile, "sample.wav", uttid="sample")

            build.assert_not_called()
            self.assertTrue(output_artifacts_complete(tmpdir, "sample", profile))
            for path in paths:
                with open(path, encoding="utf-8") as fin:
                    self.assertEqual(fin.read(), "sentinel")
            self.assertEqual(outputs["json"], json_path)


if __name__ == "__main__":
    unittest.main()
