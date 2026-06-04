import unittest

from semantic_asr.adapters.funasr_nano import FunAsrNano, FunAsrNanoConfig


class _FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {"text": "ok", "timestamp": [["ok", 0, 1]]}


class FunAsrNanoTest(unittest.TestCase):
    def test_single_input_forces_batch_size_one(self):
        adapter = FunAsrNano.__new__(FunAsrNano)
        adapter.config = FunAsrNanoConfig(batch_size=8)
        adapter.model = _FakeModel()

        adapter._generate(["audio.wav"])

        self.assertEqual(adapter.model.calls[0]["batch_size"], 1)


if __name__ == "__main__":
    unittest.main()
