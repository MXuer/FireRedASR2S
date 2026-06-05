import unittest

import numpy as np
import soundfile as sf

from semantic_asr.adapters.ten_vad import TenVadAdapter, TenVadConfig
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

    def test_adapter_returns_frame_speech_probabilities(self):
        class FakeModel:
            def __init__(self):
                self.values = [0.0, 0.8, 0.8, 0.0]

            def process(self, frame):
                return self.values.pop(0), int(frame.size > 0)

        adapter = object.__new__(TenVadAdapter)
        adapter.config = TenVadConfig(min_speech_duration_ms=10, min_silence_duration_ms=10)
        adapter.model = FakeModel()

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, np.zeros(256 * 4, dtype=np.int16), 16000)
            result = adapter.detect(tmp.name)

        self.assertEqual(result["frame_speech_probs"]["frame_shift_ms"], 16.0)
        self.assertEqual(result["frame_speech_probs"]["frame_length_ms"], 16.0)
        self.assertEqual(result["frame_speech_probs"]["probs"], [0.0, 0.8, 0.8, 0.0])


if __name__ == "__main__":
    unittest.main()
