# Sentence ASR Pipeline

This directory contains the model-agnostic long-audio sentence ASR orchestration.
It is intentionally separate from `fireredasr2s` so new VAD, ASR, timestamp and
punctuation models can be added through adapters.

## Core Idea

The pipeline is not tied to one model family. A good sentence-level result
requires four compatible stages:

- VAD returns speech segments in seconds.
- ASR transcribes each segment and returns `uttid`, `text`, optional `confidence`.
- Timestamp provider returns token timestamps as `(token, start_s, end_s)`.
- Punctuation model adds punctuation and, with timestamps, returns sentence spans.

The core then restores long-audio global timestamps and formats the final JSON.
There is no no-VAD, no-timestamp or no-punctuation mode in this abstraction.

## Timestamp Design

Timestamp is a mandatory stage, but its source is configurable per adapter:

- ASR-native timestamp provider: for models with built-in token timestamps, such
  as CTC/alignment-branch ASR or non-autoregressive timestamp-capable ASR. The
  provider validates and normalizes `asr_result["timestamp"]`.
- Forced-aligner timestamp provider: for models without token timestamps, such
  as Whisper or Qwen3-ASR. The provider receives both ASR text and the matching
  `SpeechSegment` audio, then writes `asr_result["timestamp"]`.

This keeps the four-stage contract stable:

```text
VAD -> ASR -> TimestampProvider -> Punc -> sentence output
```

`TimestampProvider.add_timestamps(asr_results, segments)` receives filtered ASR
results and their matching VAD audio segments. A future Qwen3-ForcedAligner or
MMS adapter should implement that method.

## Expected Output

`SentenceAsrPipeline.process(wav_path, uttid)` returns:

```python
{
    "uttid": "...",
    "text": "...",
    "sentences": [{"start_ms": 0, "end_ms": 1000, "text": "..."}],
    "vad_segments_ms": [(0, 1000)],
    "dur_s": 1.0,
    "words": [{"start_ms": 0, "end_ms": 100, "text": "..."}],
    "wav_path": "...",
}
```

## FireRed Adapter

Use `sentence_asr_pipeline.adapters.build_firered_pipeline` to construct the
same model combination as the current FireRed system, while keeping the core
pipeline independent.

FireRed ASR already has token timestamp support, so its adapter forces
`return_timestamp=True` and uses `AsrTimestampProvider` to validate that
timestamps are present.
