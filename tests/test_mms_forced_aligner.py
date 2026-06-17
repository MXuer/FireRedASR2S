import unittest

import torch

from semantic_asr.adapters.mms_forced_aligner import MmsForcedAlignerConfig, MmsForcedAlignerTimestampProvider
from semantic_asr.core import SpeechSegment
from semantic_asr.language_mapping import model_language
from semantic_asr.mms_runtime import aligner as mms_aligner_module
from semantic_asr.mms_runtime.aligner import MmsAlignmentFeasibilityError
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

    def test_provider_falls_back_for_mms_feasibility_error(self):
        class FakeAligner:
            def align(inner_self, *args, **kwargs):
                raise MmsAlignmentFeasibilityError("ctc_target_too_long", 24, 52, 3)

        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="en_us", fallback_on_feasibility_error=True)
        provider.aligner = FakeAligner()
        provider.last_discarded_segments = []
        segment = SpeechSegment("test", 10.0, 12.0, 16000, [0.0] * 32000)

        [result] = provider.add_timestamps([{"uttid": "test", "text": "hello world"}], [segment])

        self.assertEqual(result["timestamp"], [["hello", 0.0, 1.0], ["world", 1.0, 2.0]])
        self.assertEqual(result["timestamp_fallback"]["reason"], "ctc_target_too_long")
        self.assertEqual(result["timestamp_fallback"]["frame_count"], 24)
        self.assertEqual(result["timestamp_fallback"]["target_count"], 52)
        self.assertEqual(result["timestamp_fallback"]["repeat_count"], 3)

    def test_provider_discards_mms_feasibility_error_by_default(self):
        class FakeAligner:
            def align(inner_self, *args, **kwargs):
                raise MmsAlignmentFeasibilityError("ctc_target_too_long", 24, 52, 3)

        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="en_us")
        provider.aligner = FakeAligner()
        provider.last_discarded_segments = []
        segment = SpeechSegment("test", 0.0, 0.5, 16000, [0.0] * 8000)

        results = provider.add_timestamps([{"uttid": "test", "text": "hello world"}], [segment])

        self.assertEqual(results, [])
        self.assertEqual(provider.last_discarded_segments[0]["reason"], "ctc_target_too_long")
        self.assertEqual(provider.last_discarded_segments[0]["frame_count"], 24)

    def test_provider_prechecks_short_hallucination_before_mms_align(self):
        class FakeAligner:
            dictionary = {"<blank>": 0, "a": 1}

            def _uromanize_alignment_tokens(inner_self, tokens, language):
                return tokens

            def align(inner_self, *args, **kwargs):
                raise AssertionError("align should not be called")

        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="en_us")
        provider.aligner = FakeAligner()
        provider.last_discarded_segments = []
        segment = SpeechSegment("test", 0.0, 0.04, 16000, [0.0] * 640)

        results = provider.add_timestamps([{"uttid": "test", "text": "a a a"}], [segment])

        self.assertEqual(results, [])
        discard = provider.last_discarded_segments[0]
        self.assertEqual(discard["reason"], "short_segment_hallucination")
        self.assertEqual(discard["frame_count"], 2)
        self.assertEqual(discard["target_count"], 3)
        self.assertEqual(discard["repeat_count"], 2)

    def test_provider_prechecks_empty_target_before_mms_align(self):
        class FakeAligner:
            dictionary = {"<blank>": 0, "a": 1}

            def _uromanize_alignment_tokens(inner_self, tokens, language):
                return tokens

            def align(inner_self, *args, **kwargs):
                raise AssertionError("align should not be called")

        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="en_us")
        provider.aligner = FakeAligner()
        provider.last_discarded_segments = []
        segment = SpeechSegment("test", 0.0, 1.0, 16000, [0.0] * 16000)

        results = provider.add_timestamps([{"uttid": "test", "text": "!!!"}], [segment])

        self.assertEqual(results, [])
        self.assertEqual(provider.last_discarded_segments[0]["reason"], "empty_target")

    def test_provider_does_not_swallow_unexpected_mms_errors(self):
        class FakeAligner:
            def align(inner_self, *args, **kwargs):
                raise RuntimeError("boom")

        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="en_us")
        provider.aligner = FakeAligner()
        segment = SpeechSegment("test", 0.0, 1.0, 16000, [0.0] * 16000)

        with self.assertRaisesRegex(RuntimeError, "boom"):
            provider.add_timestamps([{"uttid": "test", "text": "hello"}], [segment])

    def test_provider_uses_star_probe_gap_to_realign_no_star_islands(self):
        class FakeAligner:
            def __init__(inner_self):
                inner_self.calls = []

            def probe_star_gaps(inner_self, *args, **kwargs):
                return [{
                    "kind": "gap_star",
                    "start": 1.0,
                    "end": 2.0,
                    "duration": 1.0,
                    "before_token_index": 0,
                    "after_token_index": 1,
                }]

            def align(inner_self, transcripts, waveform, sample_rate, names, **kwargs):
                inner_self.calls.append({
                    "transcripts": list(transcripts),
                    "sample_count": len(waveform),
                    "alignment_transcripts": list(kwargs["alignment_transcripts"]),
                    "use_star": kwargs["use_star"],
                })
                return [
                    {"text": token, "start": index * 0.1, "end": index * 0.1 + 0.05}
                    for index, token in enumerate(transcripts)
                ]

        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="en_us")
        provider.aligner = FakeAligner()
        provider.last_discarded_segments = []
        provider.set_vad_evidence({"timestamps": [(0.0, 0.8), (2.2, 6.0)]})
        segment = SpeechSegment("test", 0.0, 6.0, 16000, [0.0] * (6 * 16000))

        [result] = provider.add_timestamps([{"uttid": "test", "text": "hello world again"}], [segment])

        self.assertEqual([call["transcripts"] for call in provider.aligner.calls], [["hello"], ["world", "again"]])
        self.assertTrue(all(call["use_star"] is False for call in provider.aligner.calls))
        self.assertEqual(result["timestamp"], [["hello", 0.0, 0.05], ["world", 1.9, 1.95], ["again", 2.0, 2.05]])
        self.assertEqual(result["mms_star_probe_gaps"][0]["before_token_index"], 0)
        self.assertTrue(result["mms_star_probe_gaps"][0]["raw_vad_supported_silence"])

    def test_provider_ignores_short_star_probe_gap(self):
        class FakeAligner:
            def __init__(inner_self):
                inner_self.calls = []

            def probe_star_gaps(inner_self, *args, **kwargs):
                return [{
                    "kind": "gap_star",
                    "start": 1.0,
                    "end": 1.1,
                    "duration": 0.1,
                    "before_token_index": 0,
                    "after_token_index": 1,
                }]

            def align(inner_self, transcripts, *args, **kwargs):
                inner_self.calls.append(list(transcripts))
                return [
                    {"text": token, "start": index * 0.1, "end": index * 0.1 + 0.05}
                    for index, token in enumerate(transcripts)
                ]

        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = MmsForcedAlignerConfig(language="en_us")
        provider.aligner = FakeAligner()
        provider.last_discarded_segments = []
        provider.set_vad_evidence({})
        segment = SpeechSegment("test", 0.0, 3.0, 16000, [0.0] * (3 * 16000))

        [result] = provider.add_timestamps([{"uttid": "test", "text": "hello world"}], [segment])

        self.assertEqual(provider.aligner.calls, [["hello", "world"]])
        self.assertNotIn("mms_star_probe_gaps", result)

    def test_mms_runtime_rejects_empty_target_before_torchaudio(self):
        aligner = object.__new__(mms_aligner_module.MmsAligner)
        aligner.device = "cpu"
        aligner.dictionary = {"<blank>": 0, "a": 1}
        aligner.generate_emissions = lambda waveform, sample_rate: (torch.zeros(4, 2), 20.0)

        with self.assertRaisesRegex(MmsAlignmentFeasibilityError, "empty_target") as ctx:
            aligner.get_alignments([0.0] * 16000, 16000, ["!"])

        self.assertEqual(ctx.exception.reason, "empty_target")
        self.assertEqual(ctx.exception.frame_count, 4)
        self.assertEqual(ctx.exception.target_count, 0)

    def test_mms_runtime_rejects_ctc_impossible_target_before_torchaudio(self):
        aligner = object.__new__(mms_aligner_module.MmsAligner)
        aligner.device = "cpu"
        aligner.dictionary = {"<blank>": 0, "a": 1}
        aligner.generate_emissions = lambda waveform, sample_rate: (torch.zeros(2, 2), 20.0)

        with self.assertRaisesRegex(MmsAlignmentFeasibilityError, "ctc_target_too_long") as ctx:
            aligner.get_alignments([0.0] * 16000, 16000, ["a a a"])

        self.assertEqual(ctx.exception.reason, "ctc_target_too_long")
        self.assertEqual(ctx.exception.frame_count, 2)
        self.assertEqual(ctx.exception.target_count, 3)
        self.assertEqual(ctx.exception.repeat_count, 2)

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

    def test_mms_runtime_normalizes_uroman_token_spaces_before_alignment_and_spans(self):
        aligner = object.__new__(mms_aligner_module.MmsAligner)
        aligner.uroman_path = "uroman/bin"
        aligner.device = "cpu"
        captured = {}

        def fake_uromanize(tokens, language):
            return [" a  b "]

        def fake_get_alignments(waveform, sample_rate, tokens):
            captured["align_tokens"] = tokens
            return [], 10.0

        def fake_get_spans(tokens, segments):
            captured["span_tokens"] = tokens
            return [[Segment("a", 0, 1)]]

        original_get_spans = mms_aligner_module.get_spans
        try:
            aligner._uromanize_alignment_tokens = fake_uromanize
            aligner.get_alignments = fake_get_alignments
            mms_aligner_module.get_spans = fake_get_spans
            aligner.align(
                ["raw"],
                [0.0] * 16000,
                16000,
                ["sample_0"],
                use_star=False,
                language="deu",
                raw_transcripts=["raw"],
                alignment_transcripts=["raw"],
            )
        finally:
            mms_aligner_module.get_spans = original_get_spans

        self.assertEqual(captured["align_tokens"], ["a b"])
        self.assertEqual(captured["span_tokens"], ["a b"])

    def test_mms_runtime_drops_empty_uroman_tokens_before_alignment_and_spans(self):
        aligner = object.__new__(mms_aligner_module.MmsAligner)
        aligner.uroman_path = "uroman/bin"
        aligner.device = "cpu"
        captured = {}

        def fake_uromanize(tokens, language):
            return ["", " o "]

        def fake_get_alignments(waveform, sample_rate, tokens):
            captured["align_tokens"] = tokens
            return [], 10.0

        def fake_get_spans(tokens, segments):
            captured["span_tokens"] = tokens
            return [[Segment("o", 0, 1)]]

        original_get_spans = mms_aligner_module.get_spans
        try:
            aligner._uromanize_alignment_tokens = fake_uromanize
            aligner.get_alignments = fake_get_alignments
            mms_aligner_module.get_spans = fake_get_spans
            aligned = aligner.align(
                ["?", "ok"],
                [0.0] * 16000,
                16000,
                ["sample_0", "sample_1"],
                use_star=False,
                language="deu",
                raw_transcripts=["?", "ok"],
                alignment_transcripts=["?", "ok"],
            )
        finally:
            mms_aligner_module.get_spans = original_get_spans

        self.assertEqual(captured["align_tokens"], ["o"])
        self.assertEqual(captured["span_tokens"], ["o"])
        self.assertEqual([item["text"] for item in aligned], ["ok"])

    def test_mms_runtime_probe_star_gaps_returns_inserted_gap_spans(self):
        aligner = object.__new__(mms_aligner_module.MmsAligner)
        aligner.uroman_path = "uroman/bin"
        aligner.device = "cpu"

        def fake_uromanize(tokens, language):
            return ["<star>" if token == "<star>" else token for token in tokens]

        def fake_get_alignments(waveform, sample_rate, tokens):
            return [], 10.0

        def fake_get_spans(tokens, segments):
            return [[Segment(token, index * 10, index * 10 + 9)] for index, token in enumerate(tokens)]

        original_get_spans = mms_aligner_module.get_spans
        try:
            aligner._uromanize_alignment_tokens = fake_uromanize
            aligner.get_alignments = fake_get_alignments
            mms_aligner_module.get_spans = fake_get_spans
            gaps = aligner.probe_star_gaps(
                ["hello", "world"],
                [0.0] * 16000,
                16000,
                ["sample_0", "sample_1"],
                language="eng",
                raw_transcripts=["hello", "world"],
                alignment_transcripts=["hello", "world"],
            )
        finally:
            mms_aligner_module.get_spans = original_get_spans

        self.assertEqual(len(gaps), 3)
        self.assertIsNone(gaps[0]["before_token_index"])
        self.assertEqual(gaps[1]["before_token_index"], 0)
        self.assertEqual(gaps[1]["after_token_index"], 1)
        self.assertIsNone(gaps[2]["after_token_index"])


if __name__ == "__main__":
    unittest.main()
