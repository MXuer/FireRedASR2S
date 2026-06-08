import unittest

from semantic_asr.core import merge_final_sentences_by_gap, remove_sentence_overlaps
from semantic_asr.punctuation import AsrNativePunc, AsrTextPunc


class PunctuationStrategyTest(unittest.TestCase):
    def test_asr_native_punctuation_splits_timestamp_tokens(self):
        result = AsrNativePunc().process_with_timestamp(
            [[["hello", 0.0, 0.2], ["world.", 0.2, 0.4], ["next", 0.5, 0.7]]],
            ["sample"],
        )[0]

        self.assertEqual(len(result["punc_sentences"]), 2)
        self.assertEqual(result["punc_sentences"][0]["punc_text"], "hello world.")

    def test_asr_text_punctuation_maps_sentences_to_timestamps(self):
        result = AsrTextPunc().process_asr_results([{
            "uttid": "sample",
            "text": "hello world. next line.",
            "timestamp": [
                ["hello", 0.0, 0.2],
                ["world", 0.2, 0.4],
                ["next", 0.5, 0.7],
                ["line", 0.7, 0.9],
            ],
        }])[0]

        self.assertEqual(len(result["punc_sentences"]), 2)
        self.assertEqual(result["punc_sentences"][-1]["end_s"], 0.9)

    def test_sentence_overlap_is_resolved_at_overlap_midpoint(self):
        sentences = remove_sentence_overlaps([
            {"start_ms": 181440, "end_ms": 182035, "text": "a"},
            {"start_ms": 181910, "end_ms": 182359, "text": "b"},
        ])

        self.assertEqual(sentences[0]["end_ms"], 181972)
        self.assertEqual(sentences[1]["start_ms"], 181972)

    def test_final_sentences_merge_when_gap_is_short_and_duration_fits(self):
        sentences = merge_final_sentences_by_gap([
            {"start_ms": 0, "end_ms": 5000, "text": "first.", "asr_confidence": 0.8},
            {"start_ms": 6500, "end_ms": 12000, "text": "second.", "asr_confidence": 0.7},
        ])

        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0]["start_ms"], 0)
        self.assertEqual(sentences[0]["end_ms"], 12000)
        self.assertEqual(sentences[0]["text"], "first. second.")
        self.assertEqual(sentences[0]["asr_confidence"], 0.7)

    def test_final_sentences_do_not_merge_when_combined_duration_exceeds_limit(self):
        sentences = merge_final_sentences_by_gap([
            {"start_ms": 0, "end_ms": 9000, "text": "first.", "asr_confidence": 0.8},
            {"start_ms": 10000, "end_ms": 16000, "text": "second.", "asr_confidence": 0.7},
        ])

        self.assertEqual(len(sentences), 2)

    def test_final_sentences_do_not_merge_when_gap_reaches_limit(self):
        sentences = merge_final_sentences_by_gap([
            {"start_ms": 0, "end_ms": 5000, "text": "first.", "asr_confidence": 0.8},
            {"start_ms": 7000, "end_ms": 12000, "text": "second.", "asr_confidence": 0.7},
        ])

        self.assertEqual(len(sentences), 2)

    def test_final_sentence_merge_can_be_disabled_with_nonpositive_gap(self):
        sentences = merge_final_sentences_by_gap(
            [
                {"start_ms": 0, "end_ms": 5000, "text": "first.", "asr_confidence": 0.8},
                {"start_ms": 5500, "end_ms": 9000, "text": "second.", "asr_confidence": 0.7},
            ],
            max_gap_s=0,
        )

        self.assertEqual(len(sentences), 2)


if __name__ == "__main__":
    unittest.main()
