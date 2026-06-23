import unittest

from semantic_asr.core import (
    add_sentence_cut_segments,
    remove_sentence_cut_overlaps,
    remove_sentence_overlaps,
    SemanticAsrPipeline,
    validate_sentence_intervals,
)
from semantic_asr.adapters.naqta_punctuation import punctuate_tokens_from_labels
from semantic_asr.adapters.ct_punc import CtPunc
from semantic_asr.adapters.yue_punctuation import (
    punctuate_timestamp_from_labels as punctuate_yue_timestamp_from_labels,
    punctuate_tokens_from_labels as punctuate_yue_tokens_from_labels,
)
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

    def test_asr_text_punctuation_maps_korean_character_timestamps(self):
        sentences = split_text_by_punctuation(
            "감회가 새롭습니다. 중국 공산당은 이에 대해.",
            [
                ["감", 0.0, 0.28],
                ["회", 0.3, 0.38],
                ["가", 0.38, 0.5],
                ["새", 0.5, 0.62],
                ["롭", 0.62, 0.76],
                ["습", 0.76, 0.92],
                ["니", 0.94, 1.001],
                ["다", 1.001, 1.301],
                ["중", 1.301, 1.681],
                ["국", 1.681, 1.801],
                ["공", 1.801, 1.941],
                ["산", 1.941, 2.101],
                ["당", 2.101, 2.281],
                ["은", 2.281, 2.461],
                ["이", 2.461, 2.541],
                ["에", 2.541, 2.641],
                ["대", 2.641, 2.761],
                ["해", 2.761, 2.922],
            ],
        )

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0]["punc_text"], "감회가 새롭습니다.")
        self.assertEqual(sentences[0]["start_s"], 0.0)
        self.assertEqual(sentences[0]["end_s"], 1.301)
        self.assertEqual(sentences[1]["start_s"], 1.301)
        self.assertEqual(sentences[1]["end_s"], 2.922)

    def test_asr_text_punctuation_maps_japanese_character_timestamps(self):
        sentences = split_text_by_punctuation(
            "今日は晴れです。明日も行きます。",
            [
                ["今", 0.0, 0.1],
                ["日", 0.1, 0.2],
                ["は", 0.2, 0.3],
                ["晴", 0.3, 0.4],
                ["れ", 0.4, 0.5],
                ["で", 0.5, 0.6],
                ["す", 0.6, 0.7],
                ["明", 0.8, 0.9],
                ["日", 0.9, 1.0],
                ["も", 1.0, 1.1],
                ["行", 1.1, 1.2],
                ["き", 1.2, 1.3],
                ["ま", 1.3, 1.4],
                ["す", 1.4, 1.5],
            ],
        )

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0]["end_s"], 0.7)
        self.assertEqual(sentences[1]["start_s"], 0.8)
        self.assertEqual(sentences[1]["end_s"], 1.5)

    def test_asr_text_punctuation_maps_chinese_character_timestamps(self):
        sentences = split_text_by_punctuation(
            "今天很好。明天继续。",
            [
                ["今", 0.0, 0.1],
                ["天", 0.1, 0.2],
                ["很", 0.2, 0.3],
                ["好", 0.3, 0.4],
                ["明", 0.5, 0.6],
                ["天", 0.6, 0.7],
                ["继", 0.7, 0.8],
                ["续", 0.8, 0.9],
            ],
        )

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0]["end_s"], 0.4)
        self.assertEqual(sentences[1]["start_s"], 0.5)
        self.assertEqual(sentences[1]["end_s"], 0.9)

    def test_ct_punc_maps_funasr_text_output_to_timestamps(self):
        class FakeModel:
            def __init__(self):
                self.calls = []

            def generate(self, input, batch_size=1):
                self.calls.append((input, batch_size))
                return [{"text": "那今天的会就到这里吧，happy new year,明年见。"}]

        adapter = object.__new__(CtPunc)
        adapter.model = FakeModel()
        timestamp = [
            ["那", 0.0, 0.1],
            ["今", 0.1, 0.2],
            ["天", 0.2, 0.3],
            ["的", 0.3, 0.4],
            ["会", 0.4, 0.5],
            ["就", 0.5, 0.6],
            ["到", 0.6, 0.7],
            ["这", 0.7, 0.8],
            ["里", 0.8, 0.9],
            ["吧", 0.9, 1.0],
            ["happy", 1.1, 1.3],
            ["new", 1.3, 1.5],
            ["year", 1.5, 1.7],
            ["明", 1.8, 1.9],
            ["年", 1.9, 2.0],
            ["见", 2.0, 2.1],
        ]

        result = adapter.process_with_timestamp([timestamp, timestamp], ["sample", "sample2"])[0]

        self.assertEqual(
            adapter.model.calls,
            [
                ("那今天的会就到这里吧 happy new year 明年见", 1),
                ("那今天的会就到这里吧 happy new year 明年见", 1),
            ],
        )
        self.assertEqual(len(result["punc_sentences"]), 1)
        self.assertEqual(result["punc_sentences"][0]["punc_text"], "那今天的会就到这里吧，happy new year,明年见。")
        self.assertEqual(result["punc_sentences"][0]["end_s"], 2.1)

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

    def test_text_punctuation_keeps_periods_inside_token_like_strings(self):
        sentences = split_text_by_punctuation(
            "يبقى console.write console.write ونخلي بالنا كويس قوي لإن C sharp.",
            [
                ["يبقى", 0.0, 0.2],
                ["console.write", 0.2, 1.0],
                ["console.write", 1.0, 1.7],
                ["ونخلي", 1.7, 2.1],
                ["بالنا", 2.1, 2.5],
                ["كويس", 2.5, 2.9],
                ["قوي", 2.9, 3.2],
                ["لإن", 3.2, 3.5],
                ["C", 3.5, 3.7],
                ["sharp", 3.7, 4.0],
            ],
        )

        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0]["start_s"], 0.0)
        self.assertEqual(sentences[0]["end_s"], 4.0)

    def test_text_punctuation_keeps_decimal_and_url_periods_inside_tokens(self):
        decimal_sentences = split_text_by_punctuation(
            "منطقة الـ 1.33 أو 3.330 ثم منطقة الـ 1.3400.",
            [
                ["منطقة", 0.0, 0.2],
                ["الـ", 0.2, 0.4],
                ["1.33", 0.4, 0.7],
                ["أو", 0.7, 0.8],
                ["3.330", 0.8, 1.1],
                ["ثم", 1.1, 1.3],
                ["منطقة", 1.3, 1.5],
                ["الـ", 1.5, 1.7],
                ["1.3400", 1.7, 2.0],
            ],
        )
        url_sentences = split_text_by_punctuation(
            "الموقع اسمه getlink.io وهو سهل.",
            [
                ["الموقع", 0.0, 0.2],
                ["اسمه", 0.2, 0.4],
                ["getlink.io", 0.4, 1.0],
                ["وهو", 1.0, 1.2],
                ["سهل", 1.2, 1.5],
            ],
        )

        self.assertEqual(len(decimal_sentences), 1)
        self.assertEqual(decimal_sentences[0]["end_s"], 2.0)
        self.assertEqual(len(url_sentences), 1)
        self.assertEqual(url_sentences[0]["end_s"], 1.5)

    def test_text_punctuation_keeps_german_christian_era_abbreviation_with_continuation(self):
        sentences = split_text_by_punctuation(
            "Im Jahre 66 n. Chr. soll Nero Rom in Brand gesteckt haben. Danach floh er.",
            [
                ["Im", 0.0, 0.1],
                ["Jahre", 0.1, 0.2],
                ["66", 0.2, 0.3],
                ["n.", 0.3, 0.4],
                ["Chr.", 0.4, 0.5],
                ["soll", 0.5, 0.6],
                ["Nero", 0.6, 0.7],
                ["Rom", 0.7, 0.8],
                ["in", 0.8, 0.9],
                ["Brand", 0.9, 1.0],
                ["gesteckt", 1.0, 1.1],
                ["haben.", 1.1, 1.2],
                ["Danach", 1.3, 1.4],
                ["floh", 1.4, 1.5],
                ["er.", 1.5, 1.6],
            ],
        )

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0]["punc_text"], "Im Jahre 66 n. Chr. soll Nero Rom in Brand gesteckt haben.")
        self.assertEqual(sentences[0]["end_s"], 1.2)

    def test_text_punctuation_keeps_german_christian_era_abbreviation_before_comma(self):
        sentences = split_text_by_punctuation(
            "Plinius der Jüngere, 61-113 n. Chr., der römische Stadthalter. Danach.",
            [
                ["Plinius", 0.0, 0.1],
                ["der", 0.1, 0.2],
                ["Jüngere,", 0.2, 0.3],
                ["61-113", 0.3, 0.4],
                ["n.", 0.4, 0.5],
                ["Chr.,", 0.5, 0.6],
                ["der", 0.6, 0.7],
                ["römische", 0.7, 0.8],
                ["Stadthalter.", 0.8, 0.9],
                ["Danach.", 1.0, 1.1],
            ],
        )

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0]["punc_text"], "Plinius der Jüngere, 61-113 n. Chr., der römische Stadthalter.")
        self.assertEqual(sentences[0]["end_s"], 0.9)

    def test_text_punctuation_allows_german_christian_era_abbreviation_before_new_sentence(self):
        sentences = split_text_by_punctuation(
            "Tacitus ca. 55-115 n. Chr. Der Historiker berichtet.",
            [
                ["Tacitus", 0.0, 0.1],
                ["ca.", 0.1, 0.2],
                ["55-115", 0.2, 0.3],
                ["n.", 0.3, 0.4],
                ["Chr.", 0.4, 0.5],
                ["Der", 0.6, 0.7],
                ["Historiker", 0.7, 0.8],
                ["berichtet.", 0.8, 0.9],
            ],
        )

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0]["punc_text"], "Tacitus ca. 55-115 n. Chr.")
        self.assertEqual(sentences[1]["punc_text"], "Der Historiker berichtet.")

    def test_text_punctuation_does_not_fallback_empty_timestamp_slice_to_whole_segment(self):
        sentences = split_text_by_punctuation(
            "first. second. third.",
            [["first", 0.0, 0.4]],
        )

        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0]["start_s"], 0.0)
        self.assertEqual(sentences[0]["end_s"], 0.4)
        self.assertEqual(sentences[0]["punc_text"], "first. second. third.")

    def test_naqta_label_mapping_adds_arabic_punctuation(self):
        text = punctuate_tokens_from_labels(
            ["هذا", "اختبار", "هل", "تسمعني"],
            ["O", "COMMA", "O", "QUESTION_MARK"],
        )

        self.assertEqual(text, "هذا اختبار، هل تسمعني؟")

    def test_yue_label_mapping_adds_cantonese_punctuation_without_spaces(self):
        text = punctuate_yue_tokens_from_labels(
            ["我", "今日", "返工", "你", "去", "邊"],
            ["O", "O", "S-，", "O", "O", "S-？"],
        )

        self.assertEqual(text, "我今日返工，你去邊？")

    def test_yue_timestamp_punctuation_maps_sentences_by_token_index(self):
        sentences = punctuate_yue_timestamp_from_labels(
            [
                ["我", 0.0, 0.1],
                ["今日", 0.1, 0.4],
                ["返工", 0.4, 0.7],
                ["", 0.7, 0.8],
                ["你", 0.8, 0.9],
                ["去", 0.9, 1.0],
                ["邊", 1.0, 1.2],
            ],
            ["O", "O", "S-。", "O", "O", "S-？"],
        )

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0]["punc_text"], "我今日返工。")
        self.assertEqual(sentences[0]["start_s"], 0.0)
        self.assertEqual(sentences[0]["end_s"], 0.7)
        self.assertEqual(sentences[1]["punc_text"], "你去邊？")
        self.assertEqual(sentences[1]["start_s"], 0.8)
        self.assertEqual(sentences[1]["end_s"], 1.2)

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

    def test_sentence_overlap_merges_when_split_would_create_zero_width_sentence(self):
        sentences = remove_sentence_overlaps([
            {"start_ms": 22000, "end_ms": 22092, "text": "감회가", "asr_confidence": 0},
            {"start_ms": 22092, "end_ms": 22092, "text": "새롭습니다.", "asr_confidence": 0},
        ])

        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0]["start_ms"], 22000)
        self.assertEqual(sentences[0]["end_ms"], 22092)
        self.assertEqual(sentences[0]["text"], "감회가새롭습니다.")

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

    def test_sentence_cut_segments_coalesce_overlapping_padded_vad_islands(self):
        [sentence] = add_sentence_cut_segments(
            [
                {
                    "start_ms": 43000,
                    "end_ms": 47000,
                    "text": "a",
                    "asr_confidence": 0,
                }
            ],
            [(43000, 45160), (44888, 47000)],
        )

        self.assertEqual(sentence["cut_segments_ms"], [[43000, 47000]])
        validate_sentence_intervals([sentence])

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

    def test_sentence_interval_validation_rejects_reversed_and_overlapping_times(self):
        with self.assertRaisesRegex(ValueError, "Invalid sentence interval"):
            validate_sentence_intervals([
                {"start_ms": 242606, "end_ms": 241745, "text": "bad"},
            ])
        with self.assertRaisesRegex(ValueError, "Overlapping sentence cut interval"):
            validate_sentence_intervals([
                {
                    "start_ms": 1000,
                    "end_ms": 2000,
                    "cut_start_ms": 1000,
                    "cut_end_ms": 2200,
                    "cut_segments_ms": [[1000, 2200]],
                    "text": "a",
                },
                {
                    "start_ms": 2000,
                    "end_ms": 3000,
                    "cut_start_ms": 2100,
                    "cut_end_ms": 3000,
                    "cut_segments_ms": [[2100, 3000]],
                    "text": "b",
                },
            ])

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
