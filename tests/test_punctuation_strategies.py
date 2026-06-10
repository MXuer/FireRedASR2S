import unittest

from semantic_asr.core import (
    add_sentence_cut_segments,
    merge_final_sentences_by_gap,
    remove_sentence_cut_overlaps,
    remove_sentence_overlaps,
    SemanticAsrPipeline,
)
from semantic_asr.adapters.naqta_punctuation import punctuate_tokens_from_labels
from semantic_asr.punctuation import AsrNativePunc, AsrTextPunc, split_text_by_punctuation


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

    def test_text_punctuation_does_not_create_standalone_punctuation_sentence(self):
        sentences = split_text_by_punctuation(
            "first sentence. . second sentence.",
            [
                ["first", 0.0, 0.3],
                ["sentence", 0.3, 0.8],
                ["second", 1.0, 1.3],
                ["sentence", 1.3, 1.8],
            ],
        )

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0]["punc_text"], "first sentence.")
        self.assertEqual(sentences[0]["start_s"], 0.0)
        self.assertEqual(sentences[0]["end_s"], 0.8)
        self.assertEqual(sentences[1]["punc_text"], "second sentence.")
        self.assertEqual(sentences[1]["start_s"], 1.0)
        self.assertEqual(sentences[1]["end_s"], 1.8)

    def test_arabic_question_mark_splits_sentences(self):
        sentences = split_text_by_punctuation(
            "كيف الحال؟ بخير.",
            [
                ["كيف", 0.0, 0.2],
                ["الحال", 0.2, 0.4],
                ["بخير", 0.5, 0.8],
            ],
        )

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0]["punc_text"], "كيف الحال؟")
        self.assertEqual(sentences[1]["punc_text"], "بخير.")

    def test_naqta_label_mapping_adds_arabic_punctuation(self):
        text = punctuate_tokens_from_labels(
            ["هذا", "اختبار", "هل", "تسمعني"],
            ["O", "COMMA", "O", "QUESTION_MARK"],
        )

        self.assertEqual(text, "هذا اختبار، هل تسمعني؟")

    def test_format_skips_punctuation_only_punc_sentence(self):
        pipeline = object.__new__(SemanticAsrPipeline)

        sentences, _ = pipeline._format(
            [
                {
                    "uttid": "sample_s1000_e5000",
                    "text": "hello",
                    "confidence": 0,
                    "timestamp": [["hello", 0.0, 1.0]],
                }
            ],
            [
                {
                    "uttid": "sample_s1000_e5000",
                    "punc_sentences": [
                        {"start_s": 0.0, "end_s": 1.0, "punc_text": "."},
                        {"start_s": 1.0, "end_s": 2.0, "punc_text": "hello."},
                    ],
                }
            ],
        )

        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0]["text"], "hello.")

    def test_sentence_overlap_is_resolved_at_overlap_midpoint(self):
        sentences = remove_sentence_overlaps([
            {"start_ms": 181440, "end_ms": 182035, "text": "a"},
            {"start_ms": 181910, "end_ms": 182359, "text": "b"},
        ])

        self.assertEqual(sentences[0]["end_ms"], 181972)
        self.assertEqual(sentences[1]["start_ms"], 181972)

    def test_final_sentences_merge_incomplete_sentence_when_gap_is_short_and_duration_fits(self):
        sentences = merge_final_sentences_by_gap([
            {"start_ms": 0, "end_ms": 5000, "text": "first,", "asr_confidence": 0.8},
            {"start_ms": 6500, "end_ms": 12000, "text": "second.", "asr_confidence": 0.7},
        ])

        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0]["start_ms"], 0)
        self.assertEqual(sentences[0]["end_ms"], 12000)
        self.assertEqual(sentences[0]["text"], "first, second.")
        self.assertEqual(sentences[0]["asr_confidence"], 0.7)

    def test_final_sentences_do_not_merge_after_terminal_punctuation(self):
        sentences = merge_final_sentences_by_gap([
            {"start_ms": 17355, "end_ms": 25916, "text": "وما أدراك ما الحق؟", "asr_confidence": 0},
            {"start_ms": 25916, "end_ms": 31935, "text": "كذبت ثمود وعاد بالقارعة", "asr_confidence": 0},
        ])

        self.assertEqual(len(sentences), 2)

    def test_final_sentences_do_not_merge_when_combined_duration_exceeds_limit(self):
        sentences = merge_final_sentences_by_gap([
            {"start_ms": 0, "end_ms": 9000, "text": "first.", "asr_confidence": 0.8},
            {"start_ms": 10000, "end_ms": 16000, "text": "second.", "asr_confidence": 0.7},
        ])

        self.assertEqual(len(sentences), 2)

    def test_final_sentences_overlap_is_resolved_after_failed_duration_merge(self):
        sentences = merge_final_sentences_by_gap(
            [
                {"start_ms": 71280, "end_ms": 82150, "text": "first.", "asr_confidence": 0.8},
                {"start_ms": 82080, "end_ms": 92412, "text": "second.", "asr_confidence": 0.7},
            ],
            max_gap_s=2.0,
            max_duration_s=15.0,
        )
        sentences = remove_sentence_overlaps(sentences)

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0]["end_ms"], 82115)
        self.assertEqual(sentences[1]["start_ms"], 82115)

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

    def test_sentence_cut_segments_expand_to_overlapping_raw_vad_speech_islands(self):
        [sentence] = add_sentence_cut_segments(
            [
                {
                    "start_ms": 10410,
                    "end_ms": 15450,
                    "text": "涯唔晓你讲的什么，唔係啊，以前涯话你讲的。",
                    "asr_confidence": 0,
                }
            ],
            [(10330, 11460), (12710, 13930), (14540, 16059)],
        )

        self.assertEqual(sentence["start_ms"], 10410)
        self.assertEqual(sentence["end_ms"], 15450)
        self.assertEqual(sentence["cut_start_ms"], 10410)
        self.assertEqual(sentence["cut_end_ms"], 15450)
        self.assertEqual(sentence["cut_segments_ms"], [[10410, 11460], [12710, 13930], [14540, 15450]])

    def test_sentence_cut_segments_clip_shared_raw_vad_to_sentence_bounds(self):
        sentences = add_sentence_cut_segments(
            [
                {"start_ms": 275192, "end_ms": 286542, "text": "a", "asr_confidence": 0},
                {"start_ms": 286542, "end_ms": 296212, "text": "b", "asr_confidence": 0},
            ],
            [(285040, 286580), (286700, 287690)],
        )

        self.assertEqual(sentences[0]["cut_segments_ms"], [[285040, 286542]])
        self.assertEqual(sentences[1]["cut_segments_ms"], [[286542, 286580], [286700, 287690]])
        self.assertLessEqual(sentences[0]["cut_end_ms"], sentences[1]["cut_start_ms"])

    def test_sentence_cut_overlaps_are_resolved_for_textgrid_intervals(self):
        sentences = remove_sentence_cut_overlaps(
            [
                {
                    "start_ms": 71280,
                    "end_ms": 82150,
                    "cut_start_ms": 71280,
                    "cut_end_ms": 82150,
                    "cut_segments_ms": [[71280, 82150]],
                    "text": "a",
                    "asr_confidence": 0,
                },
                {
                    "start_ms": 82080,
                    "end_ms": 92412,
                    "cut_start_ms": 82080,
                    "cut_end_ms": 92412,
                    "cut_segments_ms": [[82080, 92412]],
                    "text": "b",
                    "asr_confidence": 0,
                },
            ]
        )

        self.assertEqual(sentences[0]["cut_end_ms"], 82115)
        self.assertEqual(sentences[1]["cut_start_ms"], 82115)
        self.assertLessEqual(sentences[0]["cut_end_ms"], sentences[1]["cut_start_ms"])

    def test_sentence_cut_segments_fall_back_to_sentence_bounds_without_vad_overlap(self):
        [sentence] = add_sentence_cut_segments(
            [{"start_ms": 1000, "end_ms": 2000, "text": "a", "asr_confidence": 0}],
            [(3000, 4000)],
        )

        self.assertEqual(sentence["cut_start_ms"], 1000)
        self.assertEqual(sentence["cut_end_ms"], 2000)
        self.assertEqual(sentence["cut_segments_ms"], [[1000, 2000]])


if __name__ == "__main__":
    unittest.main()
