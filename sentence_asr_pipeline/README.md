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
  provider validates and normalizes `asr_result["timestamp"]`. Whisper large is
  also handled this way when `word_timestamps=True`.
- Forced-aligner timestamp provider: for models without token timestamps, such
  as Qwen3-ASR when used without an ASR-native timestamp path. The provider
  receives both ASR text and the matching `SpeechSegment` audio, then writes
  `asr_result["timestamp"]`.

This keeps the four-stage contract stable:

```text
VAD -> ASR -> TimestampProvider -> Punc -> sentence output
```

`TimestampProvider.add_timestamps(asr_results, segments)` receives filtered ASR
results and their matching VAD audio segments. A future Qwen3-ForcedAligner or
MMS adapter should implement that method.

## Language Profiles

Different languages can use different VAD, ASR, timestamp and punctuation
combinations. `sentence_asr_pipeline.language_configs` stores explicit language
profiles, including module builder names and per-component parameters. For
example, Russian can use FireRed VAD, Whisper ASR, Qwen3 forced alignment and
Whisper text punctuation.

## VAD Merge Policy

All VAD adapters feed into a common post-processing step before audio is sliced
for ASR. The default policy merges adjacent VAD segments into longer semantic
chunks where possible:

- target at least 10 seconds per segment
- never exceed 40 seconds per merged segment
- do not merge across a silence gap greater than 3 seconds

If a segment is still shorter than 10 seconds because the surrounding gaps are
too large, it is kept as-is.

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

## Punctuation Design

The fourth stage is a punctuation strategy. It can be an external punctuation
model such as FireRedPunc, or an ASR-native punctuation splitter for models that
already emit punctuation.

When external re-punctuation is selected, the pipeline strips punctuation from
token timestamps before calling the punctuation model. When ASR-native
punctuation is selected, punctuation is preserved and sentence spans are split
from the timestamp tokens or from the ASR text, depending on the punctuation
strategy.

## FireRed Adapter

Use `sentence_asr_pipeline.adapters.build_firered_pipeline` to construct the
same model combination as the current FireRed system, while keeping the core
pipeline independent.

The FireRed ASR, VAD and Punc runtime code needed by the adapter is vendored
under `sentence_asr_pipeline.firered_runtime`, so the standalone project does
not need to import the outer `fireredasr2s` package.

FireRed ASR already has token timestamp support, so its adapter forces
`return_timestamp=True` and uses `AsrTimestampProvider` to validate that
timestamps are present.
