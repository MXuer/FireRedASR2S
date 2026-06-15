import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

from semantic_asr import SemanticASR, list_model_languages, list_models, suggest_components
from tests.test_config_runner import fake_profile, fake_registry


class SemanticAsrApiTest(unittest.TestCase):
    def test_transcribe_reuses_loaded_pipeline_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "sample.wav")
            config_path = os.path.join(tmpdir, "config.json")
            outdir = os.path.join(tmpdir, "out")
            sf.write(wav_path, np.zeros(16000, dtype=np.int16), 16000)
            with open(config_path, "w", encoding="utf-8") as fout:
                json.dump(fake_profile(outdir), fout)

            sdk = SemanticASR.from_config(config_path, registry=fake_registry())
            first = sdk.transcribe(wav_path, uttid="sample", outdir=outdir, formats=("json", "srt", "csv"))
            second = sdk.transcribe(wav_path, uttid="sample", outdir=outdir, formats=("json", "srt", "csv"))

            self.assertEqual(first["result"]["text"], "hello.")
            self.assertEqual(second["result"]["text"], "hello.")
            self.assertTrue(os.path.exists(first["outputs"]["json"]))
            self.assertTrue(os.path.exists(first["outputs"]["srt"]))
            self.assertTrue(os.path.exists(first["outputs"]["csv"]))
            self.assertTrue(os.path.exists(first["outputs"]["resolved_config"]))
            self.assertEqual(first["outputs"]["json"], second["outputs"]["json"])

    def test_transcribe_without_outdir_returns_result_only(self):
        sdk = SemanticASR.from_profile(fake_profile(), registry=fake_registry())

        result = sdk.transcribe(self._write_silence_wav(), uttid="sample", outdir=None)

        self.assertEqual(result["result"]["uttid"], "sample")
        self.assertEqual(result["outputs"], {})

    def test_transcribe_rejects_unknown_output_format(self):
        sdk = SemanticASR.from_profile(fake_profile(), registry=fake_registry())

        with self.assertRaisesRegex(ValueError, "Unknown output formats"):
            sdk.transcribe(self._write_silence_wav(), formats=("pdf",))

    def test_transcribe_batch_delegates_to_batch_runner_with_devices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w", encoding="utf-8") as fout:
                json.dump(fake_profile(os.path.join(tmpdir, "out")), fout)
            sdk = SemanticASR.from_config(config_path, registry=fake_registry())

            seen_env = {}

            def fake_run_batch(**kwargs):
                seen_env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
                return [{"ok": True}]

            with mock.patch("semantic_asr.api.run_batch", side_effect=fake_run_batch) as mocked:
                with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True):
                    outputs = sdk.transcribe_batch(
                        wav_scp="/tmp/wav.scp",
                        outdir="/tmp/out",
                        num_workers=4,
                        devices=["4", "5"],
                    )
                    self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")

        self.assertEqual(outputs, [{"ok": True}])
        self.assertEqual(seen_env["CUDA_VISIBLE_DEVICES"], "4,5")
        mocked.assert_called_once_with(
            config_path=config_path,
            wav_scp="/tmp/wav.scp",
            outdir="/tmp/out",
            num_workers=4,
            max_seconds=0,
        )

    def test_transcribe_batch_requires_config_path(self):
        sdk = SemanticASR.from_profile(fake_profile(), registry=fake_registry())

        with self.assertRaisesRegex(ValueError, "from_config"):
            sdk.transcribe_batch("/tmp/wav.scp", "/tmp/out")

    def test_model_query_helpers_are_exported(self):
        self.assertIn("asr", list_models("en_us"))
        self.assertIn("asr", suggest_components("en_us"))
        self.assertEqual(list_model_languages("funasr_nano", role="asr")["role"], "asr")

    @staticmethod
    def _write_silence_wav() -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, np.zeros(16000, dtype=np.int16), 16000)
        return tmp.name


if __name__ == "__main__":
    unittest.main()
