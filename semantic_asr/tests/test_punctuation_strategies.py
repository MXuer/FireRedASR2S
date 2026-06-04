import unittest

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


if __name__ == "__main__":
    unittest.main()
