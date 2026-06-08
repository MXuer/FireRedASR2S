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
            {"start_ms": 1000, "end_ms": 3494, "text": "first.", "asr_confidence": 0.8},
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
        self.assertEqual(fused[0]["text"], "first second.")
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
            "probs": [0.9] * 50 + [0.05] + [0.9] * 50,
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
        self.assertEqual(decisions[0]["reason"], "vad_prob_valley")
        self.assertEqual(decisions[0]["preserved_gap_ms"], 200)

    def test_max_duration_waits_for_silence_inside_active_speech(self):
        sentences = [
            {"start_ms": 0, "end_ms": 29500, "text": "first.", "asr_confidence": 0.8},
            {"start_ms": 31000, "end_ms": 35000, "text": "second.", "asr_confidence": 0.7},
        ]
        words = [
            {"start_ms": 29000, "end_ms": 29500, "text": "first"},
            {"start_ms": 31000, "end_ms": 31500, "text": "second"},
        ]

        fused, decisions = fuse_sentence_boundaries(
            sentences,
            words,
            [(0, 40000)],
            np.ones(16000 * 40, dtype=np.float32),
            16000,
            self.config,
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(decisions[0]["reason"], "max_duration_wait_for_silence")

    def test_probability_valley_keeps_and_snaps_boundary(self):
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
            "probs": [0.9] * 50 + [0.05] + [0.9] * 50,
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
        self.assertEqual(decisions[0]["reason"], "vad_prob_valley")
        self.assertTrue(decisions[0]["speech_prob_supported_silence"])
        self.assertEqual(decisions[0]["speech_prob_boundary_ms"], 5050)

    def test_high_probability_boundary_merges_even_past_max_duration(self):
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
        self.assertEqual(decisions[0]["reason"], "merged_active_speech_prob")
        self.assertGreaterEqual(decisions[0]["speech_prob_mean"], 0.5)

    def test_active_speech_safety_overrides_target_duration(self):
        sentences = [
            {"start_ms": 0, "end_ms": 14950, "text": "first.", "asr_confidence": 0.8},
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
                    {"start_ms": start_ms, "end_ms": previous_end_ms, "text": "first.", "asr_confidence": 0},
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

        self.assertEqual(fused[0]["end_ms"], 3000)
        self.assertEqual(fused[1]["start_ms"], 7000)
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
