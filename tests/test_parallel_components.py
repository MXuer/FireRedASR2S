import unittest
from dataclasses import dataclass

from semantic_asr.core import PipelineConfig, SemanticAsrPipeline, SpeechSegment
from semantic_asr.parallel_components import ParallelAsrModel, ParallelTimestampProvider
from semantic_asr.registry import create_default_registry
from semantic_asr.adapters.whisper_large import WhisperLarge
from semantic_asr.adapters.mms_forced_aligner import MmsForcedAlignerTimestampProvider


@dataclass
class FakeConfig:
    marker: str = "x"


class FakeAsr:
    def __init__(self, config: FakeConfig):
        self.config = config

    def transcribe(self, batch_uttid, batch_wav):
        return [
            {
                "uttid": uttid,
                "text": f"{self.config.marker}:{uttid}",
                "confidence": 0,
                "timestamp": [],
                "sample_rate": sample_rate,
            }
            for uttid, (sample_rate, _) in zip(batch_uttid, batch_wav)
        ]


class FakeTimestampProvider:
    def __init__(self, config: FakeConfig):
        self.config = config

    def add_timestamps(self, batch_asr_result, batch_segments):
        results = []
        for asr_result, segment in zip(batch_asr_result, batch_segments):
            result = dict(asr_result)
            result["timestamp"] = [[self.config.marker, 0.0, segment.end_s - segment.start_s]]
            results.append(result)
        return results


class DiscardingTimestampProvider:
    def __init__(self, config: FakeConfig):
        self.last_discarded_segments = []

    def add_timestamps(self, batch_asr_result, batch_segments):
        self.last_discarded_segments = [
            {"uttid": asr_result["uttid"], "reason": "discarded"}
            for asr_result in batch_asr_result
        ]
        return []


class RecordingAsr:
    recommended_batch_size = 2

    def __init__(self):
        self.batch_sizes = []

    def transcribe(self, batch_uttid, batch_wav):
        self.batch_sizes.append(len(batch_uttid))
        return [
            {
                "uttid": uttid,
                "text": uttid,
                "confidence": 0,
                "timestamp": [],
                "sample_rate": sample_rate,
            }
            for uttid, (sample_rate, _) in zip(batch_uttid, batch_wav)
        ]


class ParallelComponentsTest(unittest.TestCase):
    def test_parallel_asr_preserves_input_order(self):
        model = ParallelAsrModel(FakeAsr, FakeConfig(marker="asr"), num_workers=2)

        results = model.transcribe(
            ["utt2", "utt1", "utt3"],
            [(16000, [0]), (8000, [0]), (16000, [0])],
        )

        self.assertEqual([result["uttid"] for result in results], ["utt2", "utt1", "utt3"])
        self.assertEqual([result["text"] for result in results], ["asr:utt2", "asr:utt1", "asr:utt3"])
        self.assertEqual(results[1]["sample_rate"], 8000)

    def test_parallel_timestamp_provider_preserves_input_order(self):
        provider = ParallelTimestampProvider(FakeTimestampProvider, FakeConfig(marker="ts"), num_workers=2)
        asr_results = [{"uttid": "b", "text": "b"}, {"uttid": "a", "text": "a"}]
        segments = [
            SpeechSegment("b", 0.0, 1.5, 16000, [0]),
            SpeechSegment("a", 0.0, 2.0, 16000, [0]),
        ]

        results = provider.add_timestamps(asr_results, segments)

        self.assertEqual([result["uttid"] for result in results], ["b", "a"])
        self.assertEqual(results[0]["timestamp"], [["ts", 0.0, 1.5]])
        self.assertEqual(results[1]["timestamp"], [["ts", 0.0, 2.0]])

    def test_parallel_timestamp_provider_allows_discarded_segments(self):
        provider = ParallelTimestampProvider(DiscardingTimestampProvider, FakeConfig(), num_workers=2)
        asr_results = [{"uttid": "b", "text": "b"}, {"uttid": "a", "text": "a"}]
        segments = [
            SpeechSegment("b", 0.0, 1.5, 16000, [0]),
            SpeechSegment("a", 0.0, 2.0, 16000, [0]),
        ]

        results = provider.add_timestamps(asr_results, segments)

        self.assertEqual(results, [])
        self.assertEqual(
            sorted(item["uttid"] for item in provider.last_discarded_segments),
            ["a", "b"],
        )

    def test_registry_wraps_whisper_when_num_workers_is_set(self):
        component = create_default_registry().build(
            "asr",
            "whisper_large",
            {"num_workers": 2},
        )

        self.assertIsInstance(component, ParallelAsrModel)
        self.assertIs(component.component_cls, WhisperLarge)
        self.assertEqual(component.num_workers, 2)

    def test_registry_wraps_mms_when_num_workers_is_set(self):
        component = create_default_registry().build(
            "timestamp",
            "mms_forced_aligner",
            {"num_workers": 2},
        )

        self.assertIsInstance(component, ParallelTimestampProvider)
        self.assertIs(component.component_cls, MmsForcedAlignerTimestampProvider)
        self.assertEqual(component.num_workers, 2)

    def test_pipeline_uses_parallel_asr_recommended_batch_size(self):
        asr = RecordingAsr()
        pipeline = object.__new__(SemanticAsrPipeline)
        pipeline.asr = asr
        pipeline.config = PipelineConfig(asr_batch_size=1)

        segments = [
            SpeechSegment(f"utt{i}_s0_e1000", 0.0, 1.0, 16000, [0])
            for i in range(3)
        ]

        pipeline._transcribe(segments)

        self.assertEqual(asr.batch_sizes, [2, 1])


if __name__ == "__main__":
    unittest.main()
