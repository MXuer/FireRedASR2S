import unittest

from semantic_asr.adapters.ten_vad import get_speech_timestamps


class TenVadPostprocessTest(unittest.TestCase):
    def test_get_speech_timestamps_merges_short_silence(self):
        probabilities = [0.0] * 5 + [0.8] * 20 + [0.1] * 3 + [0.8] * 20 + [0.0] * 20

        segments = get_speech_timestamps(
            probabilities,
            sampling_rate=16000,
            threshold=0.5,
            min_speech_duration_ms=100,
            min_silence_duration_ms=100,
            window_size_samples=256,
        )

        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0][0], 0.08, places=3)
        self.assertGreater(segments[0][1], 0.7)

    def test_get_speech_timestamps_drops_too_short_speech(self):
        probabilities = [0.0] * 5 + [0.8] * 3 + [0.0] * 20

        segments = get_speech_timestamps(
            probabilities,
            sampling_rate=16000,
            threshold=0.5,
            min_speech_duration_ms=100,
            min_silence_duration_ms=100,
            window_size_samples=256,
        )

        self.assertEqual(segments, [])


if __name__ == "__main__":
    unittest.main()
