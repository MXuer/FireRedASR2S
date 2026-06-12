import tempfile
import unittest

import numpy as np
import soundfile as sf

from semantic_asr.core import (
    PipelineConfig,
    SemanticAsrPipeline,
    prepare_asr_vad_segments,
)


class FakeVad:
    def detect(self, wav_path: str):
        return {"timestamps": [(0.0, 1.0), (1.2, 1.35), (1.6, 2.4), (5.0, 5.2)]}


class RecordingAsr:
    def __init__(self):
        self.uttids = []

    def transcribe(self, batch_uttid, batch_wav):
        self.uttids.extend(batch_uttid)
        return [
            {
                "uttid": uttid,
                "text": "hello",
                "confidence": 0,
                "timestamp": [],
            }
            for uttid in batch_uttid
        ]


class FakeTimestamp:
    def __init__(self):
        self.segment_ranges = []

    def add_timestamps(self, batch_asr_result, batch_segments):
        self.segment_ranges.extend((segment.start_s, segment.end_s) for segment in batch_segments)
        results = []
        for asr_result, segment in zip(batch_asr_result, batch_segments):
            result = dict(asr_result)
            result["timestamp"] = [["hello", 0.0, segment.end_s - segment.start_s]]
            results.append(result)
        return results


class DiscardingTimestamp(FakeTimestamp):
    def add_timestamps(self, batch_asr_result, batch_segments):
        self.last_discarded_segments = [{
            "uttid": batch_asr_result[0]["uttid"],
            "reason": "short_segment_hallucination",
        }]
        return []


class FakePunc:
    def process_with_timestamp(self, batch_timestamp, batch_uttid):
        return [
            {
                "uttid": uttid,
                "punc_sentences": [{"start_s": 0.0, "end_s": timestamp[-1][2], "punc_text": "hello."}],
            }
            for uttid, timestamp in zip(batch_uttid, batch_timestamp)
        ]


class AsrVadPostprocessTest(unittest.TestCase):
    def test_merges_tiny_island_to_nearest_neighbor(self):
        segments = prepare_asr_vad_segments(
            [(0.0, 1.0), (1.2, 1.35), (1.6, 2.4)],
            min_segment_s=0.5,
            max_merge_silence_s=1.0,
        )

        self.assertEqual(segments, [(0.0, 1.35), (1.6, 2.4)])

    def test_drops_isolated_tiny_island(self):
        segments = prepare_asr_vad_segments(
            [(0.0, 1.0), (3.0, 3.2), (5.0, 6.0)],
            min_segment_s=0.5,
            max_merge_silence_s=1.0,
        )

        self.assertEqual(segments, [(0.0, 1.0), (5.0, 6.0)])

    def test_treats_exact_min_duration_as_tiny(self):
        segments = prepare_asr_vad_segments(
            [(0.0, 1.0), (1.2, 1.7), (2.0, 3.0)],
            min_segment_s=0.5,
            max_merge_silence_s=1.0,
        )

        self.assertEqual(segments, [(0.0, 1.7), (2.0, 3.0)])

    def test_merges_tiny_chain_without_overlap(self):
        segments = prepare_asr_vad_segments(
            [(0.0, 0.2), (0.3, 0.45), (0.55, 1.2)],
            min_segment_s=0.5,
            max_merge_silence_s=1.0,
        )

        self.assertEqual(segments, [(0.0, 1.2)])

    def test_pipeline_uses_postprocessed_asr_vad_but_preserves_raw_vad(self):
        asr = RecordingAsr()
        timestamp = FakeTimestamp()
        pipeline = SemanticAsrPipeline(
            vad=FakeVad(),
            asr=asr,
            timestamp_provider=timestamp,
            punc=FakePunc(),
            config=PipelineConfig(
                asr_vad_min_segment_s=0.5,
                asr_vad_max_merge_silence_s=1.0,
                output_vad_pad_s=0.0,
            ),
        )

        result = pipeline.process(self._write_silence_wav(), "sample")

        self.assertEqual(
            result["raw_vad_segments_ms"],
            [(0, 1000), (1200, 1350), (1600, 2400), (5000, 5200)],
        )
        self.assertEqual(result["asr_vad_segments_ms"], [(0, 1350), (1600, 2400)])
        self.assertEqual(timestamp.segment_ranges, [(0.0, 1.35), (1.6, 2.4)])
        self.assertEqual(asr.uttids, ["sample_s0_e1350", "sample_s1600_e2400"])

    def test_output_vad_padding_reaches_final_cut_times(self):
        pipeline = SemanticAsrPipeline(
            vad=FakeVad(),
            asr=RecordingAsr(),
            timestamp_provider=FakeTimestamp(),
            punc=FakePunc(),
            config=PipelineConfig(
                asr_vad_min_segment_s=0.5,
                asr_vad_max_merge_silence_s=1.0,
                output_vad_min_silence_merge_s=0.0,
                output_vad_pad_s=0.2,
            ),
        )

        result = pipeline.process(self._write_silence_wav(), "sample")

        self.assertEqual(result["raw_vad_segments_ms"][1], (1200, 1350))
        self.assertIn((1100, 1475), result["vad_segments_ms"])
        self.assertEqual(result["sentences"][0]["cut_start_ms"], 0)
        self.assertEqual(result["sentences"][0]["cut_end_ms"], 1475)

    def test_timestamp_segments_preserve_fallback_metadata(self):
        pipeline = object.__new__(SemanticAsrPipeline)

        [segment] = pipeline._format_timestamp_segments([
            {
                "uttid": "sample_s1000_e2000",
                "text": "hello",
                "confidence": 0,
                "timestamp": [["hello", 0.0, 1.0]],
                "timestamp_fallback": {"provider": "mms_forced_aligner", "reason": "empty_target"},
            }
        ])

        self.assertEqual(segment["timestamp_fallback"]["provider"], "mms_forced_aligner")
        self.assertEqual(segment["timestamp_fallback"]["reason"], "empty_target")

    def test_pipeline_records_discarded_asr_segments_from_timestamp_provider(self):
        pipeline = SemanticAsrPipeline(
            vad=FakeVad(),
            asr=RecordingAsr(),
            timestamp_provider=DiscardingTimestamp(),
            punc=FakePunc(),
            config=PipelineConfig(
                asr_vad_min_segment_s=0.5,
                asr_vad_max_merge_silence_s=1.0,
                output_vad_pad_s=0.0,
            ),
        )

        result = pipeline.process(self._write_silence_wav(), "sample")

        self.assertEqual(result["sentences"], [])
        self.assertEqual(result["discarded_asr_segments"][0]["reason"], "short_segment_hallucination")

    @staticmethod
    def _write_silence_wav() -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, np.zeros(6 * 16000, dtype=np.int16), 16000)
        return tmp.name


if __name__ == "__main__":
    unittest.main()
