import json
import os
import unittest
from unittest import mock

from semantic_asr.cli import _parse_formats, main


class SemanticAsrCliTest(unittest.TestCase):
    def test_transcribe_command_uses_sdk_and_devices_before_loading(self):
        sdk = mock.Mock()
        sdk.transcribe.return_value = {"result": {"text": "ok"}, "outputs": {}}
        seen_env = {}

        def fake_from_config(config_path):
            seen_env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
            self.assertEqual(config_path, "config.json")
            return sdk

        with mock.patch("semantic_asr.cli.SemanticASR.from_config", side_effect=fake_from_config):
            with mock.patch("semantic_asr.cli.print") as mocked_print:
                with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True):
                    main([
                        "transcribe",
                        "--config", "config.json",
                        "--wav-path", "audio.wav",
                        "--uttid", "utt",
                        "--outdir", "out",
                        "--formats", "json,srt",
                        "--max-seconds", "60",
                        "--devices", "4",
                        "--no-cache",
                    ])
                    self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")

        self.assertEqual(seen_env["CUDA_VISIBLE_DEVICES"], "4")
        sdk.transcribe.assert_called_once_with(
            wav_path="audio.wav",
            uttid="utt",
            outdir="out",
            formats=("json", "srt"),
            max_seconds=60.0,
            use_cache=False,
        )
        payload = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(payload["result"]["text"], "ok")

    def test_batch_command_exposes_num_workers_and_devices(self):
        sdk = mock.Mock()
        sdk.transcribe_batch.return_value = [{"ok": True}]

        with mock.patch("semantic_asr.cli.SemanticASR.from_config", return_value=sdk):
            with mock.patch("semantic_asr.cli.print") as mocked_print:
                main([
                    "batch",
                    "--config", "config.json",
                    "--wav-scp", "wav.scp",
                    "--outdir", "out",
                    "--num-workers", "12",
                    "--devices", "4,5,6,7",
                    "--max-seconds", "300",
                ])

        sdk.transcribe_batch.assert_called_once_with(
            wav_scp="wav.scp",
            outdir="out",
            num_workers=12,
            max_seconds=300.0,
            devices="4,5,6,7",
        )
        self.assertEqual(json.loads(mocked_print.call_args.args[0]), [{"ok": True}])

    def test_models_command_prints_language_query(self):
        with mock.patch("semantic_asr.cli.list_models", return_value={"asr": [{"name": "x"}]}) as mocked:
            with mock.patch("semantic_asr.cli.print") as mocked_print:
                main(["models", "en_us", "--role", "asr"])

        mocked.assert_called_once_with("en_us", role="asr")
        self.assertEqual(json.loads(mocked_print.call_args.args[0]), {"asr": [{"name": "x"}]})

    def test_model_languages_command_prints_model_query(self):
        with mock.patch("semantic_asr.cli.list_model_languages", return_value={"languages": ["en"]}) as mocked:
            with mock.patch("semantic_asr.cli.print") as mocked_print:
                main(["model-languages", "whisper_large", "--role", "asr"])

        mocked.assert_called_once_with("whisper_large", role="asr")
        self.assertEqual(json.loads(mocked_print.call_args.args[0]), {"languages": ["en"]})

    def test_parse_formats_allows_empty_list(self):
        self.assertEqual(_parse_formats("json,srt"), ("json", "srt"))
        self.assertEqual(_parse_formats(""), ())


if __name__ == "__main__":
    unittest.main()
