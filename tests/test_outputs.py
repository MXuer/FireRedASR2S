import os
import tempfile
import unittest

from textgrid import TextGrid

from semantic_asr.outputs import write_all_outputs, write_csv, write_srt, write_textgrid


class OutputWriterTest(unittest.TestCase):
    def test_sentence_outputs_use_cut_times_when_present(self):
        sentences = [
            {
                "start_ms": 10410,
                "end_ms": 15450,
                "cut_start_ms": 10330,
                "cut_end_ms": 16059,
                "text": "hello",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            srt_path = write_srt(tmpdir, "sample", sentences)
            csv_path = write_csv(tmpdir, "sample", 20.0, sentences)
            tg_path = write_textgrid(tmpdir, "sample", 20.0, sentences)

            with open(srt_path, encoding="utf-8") as fin:
                self.assertIn("00:00:10,330 --> 00:00:16,059", fin.read())
            with open(csv_path, encoding="utf-8") as fin:
                csv_text = fin.read()
            self.assertIn("00:00:10.330000", csv_text)
            self.assertIn("00:00:05.729000", csv_text)

            tg = TextGrid.fromFile(tg_path)
            interval = next(item for item in tg.getFirst("sentence") if item.mark == "hello")
            self.assertAlmostEqual(interval.minTime, 10.33)
            self.assertAlmostEqual(interval.maxTime, 16.059)

    def test_sentence_outputs_fall_back_to_sentence_times_without_cut_times(self):
        sentences = [{"start_ms": 1000, "end_ms": 2500, "text": "hello"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            srt_path = write_srt(tmpdir, "sample", sentences)

            with open(srt_path, encoding="utf-8") as fin:
                self.assertIn("00:00:01,000 --> 00:00:02,500", fin.read())

    def test_textgrid_accepts_adjacent_clipped_cut_times(self):
        sentences = [
            {
                "start_ms": 275192,
                "end_ms": 286542,
                "cut_start_ms": 275290,
                "cut_end_ms": 286542,
                "text": "a",
            },
            {
                "start_ms": 286542,
                "end_ms": 296212,
                "cut_start_ms": 286542,
                "cut_end_ms": 293690,
                "text": "b",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tg_path = write_textgrid(tmpdir, "sample", 300.0, sentences)

            self.assertTrue(os.path.exists(tg_path))

    def test_textgrid_raises_on_overlapping_sentence_cut_times(self):
        sentences = [
            {
                "start_ms": 30904,
                "end_ms": 149659,
                "cut_start_ms": 30904,
                "cut_end_ms": 149659,
                "text": "a",
            },
            {
                "start_ms": 135887,
                "end_ms": 186697,
                "cut_start_ms": 135887,
                "cut_end_ms": 186697,
                "text": "b",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                write_textgrid(tmpdir, "sample", 200.0, sentences)

    def test_textgrid_raises_on_overlapping_token_intervals(self):
        words = [
            {"start_ms": 1000, "end_ms": 2200, "text": "a"},
            {"start_ms": 2000, "end_ms": 2600, "text": "b"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                write_textgrid(
                    tmpdir,
                    "sample",
                    5.0,
                    [{"start_ms": 0, "end_ms": 3000, "text": "sentence"}],
                    words=words,
                )

    def test_write_all_outputs_includes_token_tier(self):
        result = {
            "dur_s": 2.0,
            "sentences": [{"start_ms": 0, "end_ms": 1000, "text": "hello"}],
            "words": [{"start_ms": 0, "end_ms": 500, "text": "hello"}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = write_all_outputs(tmpdir, "sample", result, write_srt_output=False, write_csv_output=False)
            tg = TextGrid.fromFile(outputs["textgrid"])

        self.assertIsNotNone(tg.getFirst("token"))


if __name__ == "__main__":
    unittest.main()
