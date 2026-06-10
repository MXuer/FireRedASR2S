import unittest

from semantic_asr.adapters.mms_forced_aligner import MmsForcedAlignerConfig, MmsForcedAlignerTimestampProvider
from semantic_asr.core import SpeechSegment
from semantic_asr.language_mapping import model_language
from semantic_asr.mms_runtime import aligner as mms_aligner_module
from semantic_asr.mms_runtime.align_utils import Segment


class MmsForcedAlignerTest(unittest.TestCase):
    def test_prepare_tokens_splits_chinese_characters(self):
        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="zh_cn")

        self.assertEqual(provider._prepare_tokens("你好 world"), ["你", "好", "world"])

    def test_prepare_tokens_splits_korean_characters(self):
        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="ko_kr")

        self.assertEqual(provider._prepare_tokens("한국어 test"), ["한", "국", "어", "test"])

    def test_prepare_alignment_tokens_replaces_numeric_tokens_with_star(self):
        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="ko_kr")
        tokens = provider._prepare_tokens("1952년 합포구에.")

        self.assertEqual(tokens, ["1952", "년", "합", "포", "구", "에", "."])
        self.assertEqual(provider._prepare_alignment_tokens(tokens), ["<star>", "년", "합", "포", "구", "에", "."])

    def test_prepare_alignment_items_groups_currency_percent_with_number(self):
        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="zh_cn")
        tokens = ["$", "100", "%", "涨"]

        output_tokens, alignment_tokens = provider._prepare_alignment_items(tokens)

        self.assertEqual(output_tokens, ["$100%", "涨"])
        self.assertEqual(alignment_tokens, ["<star>", "涨"])

    def test_prepare_alignment_items_groups_currency_code_and_units(self):
        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="en_us")
        tokens = ["USD", "100", "and", "20", "kg"]

        output_tokens, alignment_tokens = provider._prepare_alignment_items(tokens)

        self.assertEqual(output_tokens, ["USD100", "and", "20kg"])
        self.assertEqual(alignment_tokens, ["<star>", "and", "<star>"])

    def test_prepare_alignment_items_groups_numeric_ranges_but_keeps_sentence_punctuation(self):
        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="en_us")
        tokens = ["100", "-", "200", ".", "done"]

        output_tokens, alignment_tokens = provider._prepare_alignment_items(tokens)

        self.assertEqual(output_tokens, ["100-200", ".", "done"])
        self.assertEqual(alignment_tokens, ["<star>", ".", "done"])

    def test_prepare_alignment_items_does_not_absorb_terminal_math_symbol(self):
        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="en_us")
        tokens = ["100", "-", "done"]

        output_tokens, alignment_tokens = provider._prepare_alignment_items(tokens)

        self.assertEqual(output_tokens, ["100", "-", "done"])
        self.assertEqual(alignment_tokens, ["<star>", "-", "done"])

    def test_prepare_tokens_splits_japanese_kana_and_kanji(self):
        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="ja_jp")

        self.assertEqual(provider._prepare_tokens("日本語かな test"), ["日", "本", "語", "か", "な", "test"])

    def test_mms_language_uses_canonical_id_and_maps_to_native_code(self):
        self.assertEqual(MmsForcedAlignerConfig().language, "zh_cn")
        self.assertEqual(model_language("mms_forced_aligner", "zh_cn"), "cmn")
        self.assertEqual(model_language("mms_forced_aligner", "vi_vn"), "vie")

    def test_provider_passes_native_language_code_to_mms_runtime(self):
        class FakeAligner:
            def align(inner_self, *args, **kwargs):
                inner_self.language = kwargs["language"]
                return []

        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="zh_cn")
        provider.aligner = FakeAligner()
        segment = SpeechSegment("test", 0.0, 1.0, 16000, [0.0] * 16000)

        provider.add_timestamps([{"uttid": "test", "text": "你好"}], [segment])

        self.assertEqual(provider.aligner.language, "cmn")

    def test_provider_passes_numeric_star_alignment_tokens_to_mms_runtime(self):
        class FakeAligner:
            def align(inner_self, transcripts, *args, **kwargs):
                inner_self.transcripts = transcripts
                inner_self.alignment_transcripts = kwargs["alignment_transcripts"]
                inner_self.raw_transcripts = kwargs["raw_transcripts"]
                return [
                    {"text": token, "start": index * 0.1, "end": index * 0.1 + 0.05}
                    for index, token in enumerate(kwargs["raw_transcripts"])
                ]

        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="ko_kr")
        provider.aligner = FakeAligner()
        segment = SpeechSegment("test", 0.0, 1.0, 16000, [0.0] * 16000)

        [result] = provider.add_timestamps([{"uttid": "test", "text": "1952년 합포구에."}], [segment])

        self.assertEqual(provider.aligner.transcripts[0], "1952")
        self.assertEqual(provider.aligner.raw_transcripts[0], "1952")
        self.assertEqual(provider.aligner.alignment_transcripts[0], "<star>")
        self.assertEqual(result["timestamp"][0][0], "1952")

    def test_provider_passes_grouped_numeric_span_to_mms_runtime(self):
        class FakeAligner:
            def align(inner_self, transcripts, *args, **kwargs):
                inner_self.transcripts = transcripts
                inner_self.alignment_transcripts = kwargs["alignment_transcripts"]
                inner_self.raw_transcripts = kwargs["raw_transcripts"]
                return [
                    {"text": token, "start": index * 0.1, "end": index * 0.1 + 0.05}
                    for index, token in enumerate(kwargs["raw_transcripts"])
                ]

        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="en_us")
        provider.aligner = FakeAligner()
        segment = SpeechSegment("test", 0.0, 1.0, 16000, [0.0] * 16000)

        [result] = provider.add_timestamps([{"uttid": "test", "text": "$ 100 % rose"}], [segment])

        self.assertEqual(provider.aligner.transcripts, ["$100%", "rose"])
        self.assertEqual(provider.aligner.raw_transcripts, ["$100%", "rose"])
        self.assertEqual(provider.aligner.alignment_transcripts, ["<star>", "rose"])
        self.assertEqual(result["timestamp"][0][0], "$100%")

    def test_provider_forces_use_star_false(self):
        class FakeAligner:
            def __init__(inner_self, *args, **kwargs):
                pass

        config = MmsForcedAlignerConfig(use_star=True)

        original_aligner = MmsForcedAlignerTimestampProvider.__init__.__globals__["MmsAligner"]
        try:
            MmsForcedAlignerTimestampProvider.__init__.__globals__["MmsAligner"] = FakeAligner
            provider = MmsForcedAlignerTimestampProvider(config)
        finally:
            MmsForcedAlignerTimestampProvider.__init__.__globals__["MmsAligner"] = original_aligner

        self.assertFalse(provider.config.use_star)

    def test_normalize_alignment(self):
        aligned = [{"text": "hello", "start": 0.1, "end": 0.3}]

        self.assertEqual(
            MmsForcedAlignerTimestampProvider._normalize_alignment(aligned),
            [["hello", 0.1, 0.3]],
        )

    def test_mms_runtime_keeps_numeric_star_placeholder_when_use_star_inserts_noise_stars(self):
        aligner = object.__new__(mms_aligner_module.MmsAligner)
        aligner.uroman_path = "uroman/bin"
        aligner.device = "cpu"
        captured = {}

        def fake_uromanize(tokens, language):
            return ["<star>" if token == "<star>" else token for token in tokens]

        def fake_get_alignments(waveform, sample_rate, tokens):
            captured["tokens"] = tokens
            return [], 10.0

        def fake_get_spans(tokens, segments):
            return [[Segment(token, index * 10, index * 10 + 9)] for index, token in enumerate(tokens)]

        original_get_spans = mms_aligner_module.get_spans
        try:
            aligner._uromanize_alignment_tokens = fake_uromanize
            aligner.get_alignments = fake_get_alignments
            mms_aligner_module.get_spans = fake_get_spans
            aligned = aligner.align(
                ["1952", "년"],
                [0.0] * 16000,
                16000,
                ["sample_0", "sample_1"],
                use_star=True,
                language="kor",
                raw_transcripts=["1952", "년"],
                alignment_transcripts=["<star>", "년"],
            )
        finally:
            mms_aligner_module.get_spans = original_get_spans

        self.assertEqual(captured["tokens"], ["<star>", "<star>", "<star>", "년", "<star>"])
        self.assertEqual([item["text"] for item in aligned], ["1952", "년"])


if __name__ == "__main__":
    unittest.main()
