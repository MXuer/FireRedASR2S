# Sentence ASR Pipeline

This directory contains the model-agnostic long-audio sentence ASR orchestration.
It is intentionally separate from `fireredasr2s` so new VAD, ASR, timestamp and
punctuation models can be added through adapters.

## Core Idea

The pipeline is not tied to one model family. A good sentence-level result needs
compatible components:

- VAD returns speech segments in seconds.
- ASR transcribes each segment and returns `uttid`, `text`, optional `confidence`.
- Timestamp predictor or ASR returns token timestamps as `(token, start_s, end_s)`.
- Punctuation model adds punctuation and, with timestamps, returns sentence spans.

The core then restores long-audio global timestamps and formats the final JSON.

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
