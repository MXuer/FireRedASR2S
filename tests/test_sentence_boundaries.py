import unittest

import numpy as np

from semantic_asr.sentence_boundaries import (
    SentenceBoundaryFusionConfig,
    fuse_sentence_boundaries,
)


class SentenceBoundaryFusionTest(unittest.TestCase):
    def setUp(self):
        self.config = SentenceBoundaryFusionConfig(enabled=True)
        self.wav = np.ones(16000 * 12, dtype=np.float32)

    def test_merges_short_boundary_inside_active_speech(self):
        sentences = [
            {"start_ms": 1000, "end_ms": 3494, "text": "first,", "asr_confidence": 0.8},
            {"start_ms": 3634, "end_ms": 6000, "text": "، second.", "asr_confidence": 0.7},
        ]
        words = [
            {"start_ms": 3000, "end_ms": 3494, "text": "first"},
            {"start_ms": 3634, "end_ms": 4000, "text": "second"},
        ]

        fused, decisions = fuse_sentence_boundaries(
            sentences,
            words,
            [(500, 7000)],
            self.wav,
            16000,
            self.config,
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["text"], "first, second.")
        self.assertEqual(decisions[0]["action"], "merge")
        self.assertEqual(decisions[0]["reason"], "merged_active_speech")

    def test_keeps_and_snaps_boundary_supported_by_vad_silence(self):
        sentences = [
            {"start_ms": 1000, "end_ms": 7396, "text": "first.", "asr_confidence": 0.8},
            {"start_ms": 7496, "end_ms": 9000, "text": "second.", "asr_confidence": 0.7},
        ]
        words = [
            {"start_ms": 7000, "end_ms": 7396, "text": "first"},
            {"start_ms": 7496, "end_ms": 7800, "text": "second"},
        ]

        fused, decisions = fuse_sentence_boundaries(
            sentences,
            words,
            [(500, 7180), (7440, 10000)],
            self.wav,
            16000,
            self.config,
        )

        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0]["end_ms"], 7418)
        self.assertEqual(fused[1]["start_ms"], 7418)
        self.assertEqual(decisions[0]["reason"], "vad_silence")
        self.assertEqual(decisions[0]["vad_silence_ms"], [7180, 7440])

    def test_preserve_sentence_gaps_keeps_vad_silence_unassigned(self):
        config = SentenceBoundaryFusionConfig(enabled=True, preserve_sentence_gaps=True)
        sentences = [
            {"start_ms": 1000, "end_ms": 7396, "text": "first.", "asr_confidence": 0.8},
            {"start_ms": 7496, "end_ms": 9000, "text": "second.", "asr_confidence": 0.7},
        ]
        words = [
            {"start_ms": 7000, "end_ms": 7396, "text": "first"},
            {"start_ms": 7496, "end_ms": 7800, "text": "second"},
        ]

        fused, decisions = fuse_sentence_boundaries(
            sentences,
            words,
            [(500, 7180), (7440, 10000)],
            self.wav,
            16000,
            config,
        )

        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0]["end_ms"], 7180)
        self.assertEqual(fused[1]["start_ms"], 7440)
        self.assertEqual(decisions[0]["preserved_gap_ms"], 260)

    def test_preserve_sentence_gaps_uses_token_gap_without_vad_silence(self):
        config = SentenceBoundaryFusionConfig(enabled=True, preserve_sentence_gaps=True)
        sentences = [
            {"start_ms": 1000, "end_ms": 5000, "text": "first.", "asr_confidence": 0.8},
            {"start_ms": 5200, "end_ms": 9000, "text": "second.", "asr_confidence": 0.7},
        ]
        words = [
            {"start_ms": 4500, "end_ms": 5000, "text": "first"},
            {"start_ms": 5200, "end_ms": 5600, "text": "second"},
        ]
        frame_probs = {
            "frame_shift_ms": 100,
            "frame_length_ms": 100,
            "probs": [0.9] * 46 + [0.05] * 9 + [0.9] * 46,
        }

        fused, decisions = fuse_sentence_boundaries(
            sentences,
            words,
            [(0, 10000)],
            self.wav,
            16000,
            config,
            frame_probs,
        )

        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0]["end_ms"], 5000)
        self.assertEqual(fused[1]["start_ms"], 5200)
        self.assertEqual(decisions[0]["reason"], "vad_prob_silence")
        self.assertEqual(decisions[0]["preserved_gap_ms"], 200)

    def test_max_duration_waits_for_silence_at_active_boundary(self):
        sentences = [
            {"start_ms": 0, "end_ms": 29500, "text": "first.", "asr_confidence": 0.8},
            {"start_ms": 31000, "end_ms": 35000, "text": "second.", "asr_confidence": 0.7},
        ]
        words = [
            {"start_ms": 29000, "end_ms": 29500, "text": "first"},
            {"start_ms": 31000, "end_ms": 31500, "text": "second"},
        ]
        frame_probs = {
            "frame_shift_ms": 100,
            "frame_length_ms": 100,
            "probs": [0.8] * 400,
        }

        fused, decisions = fuse_sentence_boundaries(
            sentences,
            words,
            [(0, 40000)],
            np.ones(16000 * 40, dtype=np.float32),
            16000,
            self.config,
            frame_probs,
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(decisions[0]["reason"], "max_duration_wait_for_silence")

    def test_max_duration_fallback_without_probability_or_raw_vad_support(self):
        sentences = [
            {"start_ms": 0, "end_ms": 29500, "text": "first.", "asr_confidence": 0.8},
            {"start_ms": 29600, "end_ms": 35000, "text": "second.", "asr_confidence": 0.7},
        ]
        words = [
            {"start_ms": 29000, "end_ms": 29500, "text": "first"},
            {"start_ms": 29600, "end_ms": 30100, "text": "second"},
        ]

        fused, decisions = fuse_sentence_boundaries(
            sentences,
            words,
            [(0, 29500), (29600, 35000)],
            np.ones(16000 * 40, dtype=np.float32),
            16000,
            self.config,
        )

        self.assertEqual(len(fused), 2)
        self.assertEqual(decisions[0]["reason"], "max_duration_forced_boundary")

    def test_probability_silence_keeps_and_snaps_boundary(self):
        sentences = [
            {"start_ms": 1000, "end_ms": 5000, "text": "first.", "asr_confidence": 0.8},
            {"start_ms": 5200, "end_ms": 9000, "text": "second.", "asr_confidence": 0.7},
        ]
        words = [
            {"start_ms": 4500, "end_ms": 5000, "text": "first"},
            {"start_ms": 5200, "end_ms": 5600, "text": "second"},
        ]
        frame_probs = {
            "frame_shift_ms": 100,
            "frame_length_ms": 100,
            "probs": [0.9] * 46 + [0.05] * 9 + [0.9] * 46,
        }

        fused, decisions = fuse_sentence_boundaries(
            sentences,
            words,
            [(0, 10000)],
            self.wav,
            16000,
            self.config,
            frame_probs,
        )

        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0]["end_ms"], 5050)
        self.assertEqual(fused[1]["start_ms"], 5050)
        self.assertEqual(decisions[0]["reason"], "vad_prob_silence")
        self.assertTrue(decisions[0]["speech_prob_supported_silence"])
        self.assertEqual(decisions[0]["speech_prob_boundary_ms"], 5050)

    def test_high_probability_boundary_waits_when_past_max_duration(self):
        sentences = [
            {"start_ms": 0, "end_ms": 29950, "text": "first.", "asr_confidence": 0.8},
            {"start_ms": 30050, "end_ms": 35000, "text": "second.", "asr_confidence": 0.7},
        ]
        words = [
            {"start_ms": 29000, "end_ms": 29950, "text": "first"},
            {"start_ms": 30050, "end_ms": 31000, "text": "second"},
        ]
        frame_probs = {
            "frame_shift_ms": 100,
            "frame_length_ms": 100,
            "probs": [0.8] * 400,
        }

        fused, decisions = fuse_sentence_boundaries(
            sentences,
            words,
            [(0, 40000)],
            np.ones(16000 * 40, dtype=np.float32),
            16000,
            self.config,
            frame_probs,
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(decisions[0]["reason"], "max_duration_wait_for_silence")
        self.assertGreaterEqual(decisions[0]["speech_prob_mean"], 0.5)

    def test_audio_safe_boundary_over_max_duration_keeps_even_if_semantic_incomplete(self):
        config = SentenceBoundaryFusionConfig(enabled=True, max_sentence_s=30.0)
        sentences = [
            {
                "start_ms": 0,
                "end_ms": 29500,
                "text": "first clause,",
                "asr_confidence": 0.8,
            },
            {
                "start_ms": 30300,
                "end_ms": 36000,
                "text": "second fragment",
                "asr_confidence": 0.7,
            },
        ]
        words = [
            {"start_ms": 29000, "end_ms": 29500, "text": "clause"},
            {"start_ms": 30300, "end_ms": 30800, "text": "second"},
        ]

        fused, decisions = fuse_sentence_boundaries(
            sentences,
            words,
            [(0, 29500), (30300, 36000)],
            np.ones(16000 * 40, dtype=np.float32),
            16000,
            config,
        )

        self.assertEqual(len(fused), 2)
        self.assertEqual(decisions[0]["action"], "keep")
        self.assertEqual(decisions[0]["reason"], "max_duration_audio_safe")
        self.assertEqual(decisions[0]["semantic_reason"], "previous_continuation_punctuation")

    def test_audio_safe_boundary_below_max_duration_still_merges_semantic_fragment(self):
        config = SentenceBoundaryFusionConfig(enabled=True, max_sentence_s=30.0)
        sentences = [
            {
                "start_ms": 0,
                "end_ms": 10000,
                "text": "first clause,",
                "asr_confidence": 0.8,
            },
            {
                "start_ms": 11200,
                "end_ms": 18000,
                "text": "second fragment",
                "asr_confidence": 0.7,
            },
        ]
        words = [
            {"start_ms": 9500, "end_ms": 10000, "text": "clause"},
            {"start_ms": 11200, "end_ms": 11700, "text": "second"},
        ]

        fused, decisions = fuse_sentence_boundaries(
            sentences,
            words,
            [(0, 10000), (11200, 18000)],
            np.ones(16000 * 20, dtype=np.float32),
            16000,
            config,
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(decisions[0]["action"], "merge")
        self.assertEqual(decisions[0]["reason"], "merged_semantic_incomplete")

    def test_active_speech_safety_overrides_target_duration_for_non_terminal_boundary(self):
        sentences = [
            {"start_ms": 0, "end_ms": 14950, "text": "first,", "asr_confidence": 0.8},
            {"start_ms": 15050, "end_ms": 20000, "text": "second.", "asr_confidence": 0.7},
        ]
        words = [
            {"start_ms": 14000, "end_ms": 14950, "text": "first"},
            {"start_ms": 15050, "end_ms": 16000, "text": "second"},
        ]

        fused, decisions = fuse_sentence_boundaries(
            sentences,
            words,
            [(0, 25000)],
            np.ones(16000 * 25, dtype=np.float32),
            16000,
            self.config,
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(decisions[0]["reason"], "merged_active_speech")

    def test_reported_arabic_active_speech_boundaries_merge_below_max_duration(self):
        for start_ms, previous_end_ms, current_start_ms, end_ms in [
            (186430, 202797, 203037, 211500),
            (337200, 350485, 350525, 364789),
        ]:
            with self.subTest(boundary_ms=(previous_end_ms, current_start_ms)):
                sentences = [
                    {"start_ms": start_ms, "end_ms": previous_end_ms, "text": "first,", "asr_confidence": 0},
                    {"start_ms": current_start_ms, "end_ms": end_ms, "text": "second.", "asr_confidence": 0},
                ]
                words = [
                    {"start_ms": previous_end_ms - 200, "end_ms": previous_end_ms, "text": "first"},
                    {"start_ms": current_start_ms, "end_ms": current_start_ms + 200, "text": "second"},
                ]

                fused, decisions = fuse_sentence_boundaries(
                    sentences,
                    words,
                    [(start_ms, end_ms)],
                    np.ones(16000 * 400, dtype=np.float32),
                    16000,
                    self.config,
                )

                self.assertEqual(len(fused), 1)
                self.assertEqual(decisions[0]["reason"], "merged_active_speech")

    def test_short_terminal_punctuation_boundary_merges_without_audio_safe_boundary(self):
        sentences = [
            {"start_ms": 17370, "end_ms": 25916, "text": "وما أدراك ما الحق؟", "asr_confidence": 0},
            {"start_ms": 25916, "end_ms": 31940, "text": "كذبت ثمود وعاد بالقارعة", "asr_confidence": 0},
        ]
        words = [
            {"start_ms": 21033, "end_ms": 25916, "text": "الحق"},
            {"start_ms": 25916, "end_ms": 26816, "text": "كذبت"},
        ]
        frame_probs = {
            "frame_shift_ms": 10,
            "frame_length_ms": 25,
            "probs": [0.95] * 4000,
        }

        fused, decisions = fuse_sentence_boundaries(
            sentences,
            words,
            [(17370, 31940)],
            np.ones(16000 * 40, dtype=np.float32),
            16000,
            self.config,
            frame_probs,
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["text"], "وما أدراك ما الحق؟ كذبت ثمود وعاد بالقارعة")
        self.assertEqual(decisions[0]["action"], "merge")
        self.assertEqual(decisions[0]["reason"], "merged_active_speech_prob")
        self.assertEqual(decisions[0]["semantic_reason"], "terminal_punctuation")

    def test_terminal_punctuation_boundary_uses_sentence_local_words(self):
        sentences = [
            {"start_ms": 83210, "end_ms": 90174, "text": "فعصوا رسول ربهم فأخذهم أخذة رابية.", "asr_confidence": 0},
            {"start_ms": 90400, "end_ms": 100040, "text": "إنا لما طغ الماء حملناكم في الجارية.", "asr_confidence": 0},
        ]
        words = [
            {"start_ms": 88273, "end_ms": 90174, "text": "رابية"},
            {"start_ms": 90400, "end_ms": 92195, "text": "إنا"},
            {"start_ms": 98199, "end_ms": 100020, "text": "الجارية"},
        ]

        fused, decisions = fuse_sentence_boundaries(
            sentences,
            words,
            [(83210, 90174), (90400, 100040)],
            np.ones(16000 * 110, dtype=np.float32),
            16000,
            self.config,
        )

        self.assertEqual(len(fused), 2)
        self.assertEqual(decisions[0]["token_gap_ms"], 226)
        self.assertEqual(decisions[0]["reason"], "vad_silence")

    def test_german_short_terminal_sentences_merge_until_target_duration(self):
        sentences = [
            {
                "start_ms": 300099,
                "end_ms": 305081,
                "text": "Du bekommst also dein erstes Set von diesen transparenten Schienen.",
                "asr_confidence": 0,
            },
            {
                "start_ms": 305081,
                "end_ms": 308062,
                "text": "Und diese Schiene trägst du zwei Wochen.",
                "asr_confidence": 0,
            },
            {
                "start_ms": 308062,
                "end_ms": 312563,
                "text": "Du solltest die 22 Stunden am Tag während der Korrekturphase tragen.",
                "asr_confidence": 0,
            },
            {
                "start_ms": 312563,
                "end_ms": 316585,
                "text": "Dadurch hast du zwei Stunden Zeit eben, um zu essen.",
                "asr_confidence": 0,
            },
        ]
        words = [
            {"start_ms": 304000, "end_ms": 305081, "text": "Schienen"},
            {"start_ms": 305081, "end_ms": 305500, "text": "Und"},
            {"start_ms": 307500, "end_ms": 308062, "text": "Wochen"},
            {"start_ms": 308062, "end_ms": 308600, "text": "Du"},
            {"start_ms": 312000, "end_ms": 312563, "text": "tragen"},
            {"start_ms": 312563, "end_ms": 313000, "text": "Dadurch"},
        ]
        frame_probs = {
            "frame_shift_ms": 10,
            "frame_length_ms": 16,
            "probs": [0.9] * 40000,
        }

        fused, decisions = fuse_sentence_boundaries(
            sentences,
            words,
            [(300099, 316585)],
            np.ones(16000 * 330, dtype=np.float32),
            16000,
            self.config,
            frame_probs,
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["start_ms"], 300099)
        self.assertEqual(fused[0]["end_ms"], 316585)
        self.assertIn("Schienen. Und diese", fused[0]["text"])
        self.assertEqual(decisions[0]["action"], "merge")
        self.assertEqual(decisions[1]["action"], "merge")
        self.assertEqual(decisions[2]["action"], "merge")
        self.assertEqual(decisions[2]["reason"], "merged_active_speech_prob")

    def test_vad_snap_does_not_use_distant_silence_inside_word_gap(self):
        sentences = [
            {"start_ms": 1000, "end_ms": 3000, "text": "first.", "asr_confidence": 0.8},
            {"start_ms": 7000, "end_ms": 9000, "text": "second.", "asr_confidence": 0.7},
        ]
        words = [
            {"start_ms": 1000, "end_ms": 3000, "text": "first"},
            {"start_ms": 7000, "end_ms": 9000, "text": "second"},
        ]

        fused, decisions = fuse_sentence_boundaries(
            sentences,
            words,
            [(500, 3200), (3500, 10000)],
            self.wav,
            16000,
            self.config,
        )

        self.assertEqual(fused[0]["end_ms"], 5000)
        self.assertEqual(fused[1]["start_ms"], 5000)
        self.assertIsNone(decisions[0]["vad_silence_ms"])

    def test_audio_safe_gap_merges_reported_portuguese_semantic_fragment(self):
        sentences = [
            {
                "start_ms": 47895,
                "end_ms": 53200,
                "text": "O primeiro ponto que gostaria de destacaréque, ao alinhar estratégia de tecnologia e negócios,",
                "asr_confidence": 0,
            },
            {
                "start_ms": 53200,
                "end_ms": 55408,
                "text": "as empresas podem criar",
                "asr_confidence": 0,
            },
            {
                "start_ms": 55600,
                "end_ms": 58600,
                "text": "uma conexão harmoniosa entre diferentes setores.",
                "asr_confidence": 0,
            },
        ]
        words = [
            {"start_ms": 52800, "end_ms": 52848, "text": "negócios"},
            {"start_ms": 53552, "end_ms": 54000, "text": "as"},
            {"start_ms": 55000, "end_ms": 55408, "text": "criar"},
            {"start_ms": 55600, "end_ms": 56000, "text": "uma"},
            {"start_ms": 58400, "end_ms": 58600, "text": "setores"},
        ]

        fused, decisions = fuse_sentence_boundaries(
            sentences,
            words,
            [(48096, 52848), (53552, 55408), (55600, 58480)],
            np.ones(16000 * 70, dtype=np.float32),
            16000,
            self.config,
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["start_ms"], 47895)
        self.assertEqual(fused[0]["end_ms"], 58600)
        self.assertIn("negócios, as empresas podem criar uma conexão", fused[0]["text"])
        self.assertEqual(decisions[0]["action"], "merge")
        self.assertEqual(decisions[0]["reason"], "merged_semantic_incomplete")
        self.assertEqual(decisions[0]["semantic_reason"], "previous_continuation_punctuation")
        self.assertEqual(decisions[1]["action"], "merge")
        self.assertEqual(decisions[1]["semantic_reason"], "previous_no_terminal_punctuation")


if __name__ == "__main__":
    unittest.main()
