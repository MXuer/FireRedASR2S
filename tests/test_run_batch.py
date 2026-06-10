import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

from semantic_asr.run_batch import (
    BatchItem,
    assign_worker_devices,
    read_wav_scp,
    run_batch_items,
    split_round_robin,
    visible_cuda_devices,
)


class RunBatchTest(unittest.TestCase):
    def test_read_wav_scp_uses_audio_basename_as_uttid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "sample.wav")
            scp_path = os.path.join(tmpdir, "wav.scp")
            sf.write(wav_path, np.zeros(160, dtype=np.int16), 16000)
            with open(scp_path, "w", encoding="utf-8") as fout:
                fout.write(wav_path + "\n")

            [item] = read_wav_scp(scp_path)

        self.assertEqual(item.wav_path, wav_path)
        self.assertEqual(item.uttid, "sample")

    def test_visible_cuda_devices_defaults_to_eight_slots_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(visible_cuda_devices(), ["0", "1", "2", "3", "4", "5", "6", "7"])

    def test_visible_cuda_devices_reads_current_environment(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}, clear=True):
            self.assertEqual(visible_cuda_devices(), ["4", "5", "6", "7"])

    def test_assign_worker_devices_round_robins_over_visible_devices(self):
        self.assertEqual(
            assign_worker_devices(8, ["4", "5", "6", "7"]),
            ["4", "5", "6", "7", "4", "5", "6", "7"],
        )

    def test_split_round_robin_distributes_items(self):
        items = list(range(8))

        self.assertEqual(split_round_robin(items, 4), [[0, 4], [1, 5], [2, 6], [3, 7]])

    def test_run_batch_items_writes_error_json_and_continues(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            items = [
                BatchItem(wav_path="/tmp/fail.wav", uttid="fail"),
                BatchItem(wav_path="/tmp/ok.wav", uttid="ok"),
            ]

            def runner(**kwargs):
                if kwargs["uttid"] == "fail":
                    raise RuntimeError("boom")
                return {"json": os.path.join(tmpdir, "ok.json")}

            with mock.patch("semantic_asr.run_batch.logger.exception"):
                outputs = run_batch_items(
                    items=items,
                    config_path="config.json",
                    outdir=tmpdir,
                    max_seconds=0,
                    device="4",
                    runner=runner,
                )

            self.assertFalse(outputs[0]["ok"])
            self.assertTrue(outputs[1]["ok"])
            error_path = os.path.join(tmpdir, "fail.error.json")
            self.assertEqual(outputs[0]["error_json"], error_path)
            with open(error_path, encoding="utf-8") as fin:
                payload = json.load(fin)
            self.assertEqual(payload["uttid"], "fail")
            self.assertEqual(payload["wav_path"], "/tmp/fail.wav")
            self.assertIn("RuntimeError", payload["error"])
            self.assertIn("boom", payload["traceback"])


if __name__ == "__main__":
    unittest.main()
