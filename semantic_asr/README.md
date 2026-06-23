# Semantic ASR

This directory contains the model-agnostic long-audio semantic ASR orchestration.
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

## Config Profiles

Different languages can use different VAD, ASR, timestamp and punctuation
combinations. Profiles live as JSON files under top-level `configs/`, and
`semantic_asr.config` builds them through the component registry.

## External SDK Facade

External users should prefer the package-level `SemanticASR` facade:

```python
from semantic_asr import SemanticASR

asr = SemanticASR.from_config("configs/zh_cn.json")
output = asr.transcribe("audio.wav", outdir="output/audio")
```

`SemanticASR.transcribe()` reuses the loaded VAD, ASR, timestamp and
punctuation models across calls on the same instance. It returns a dictionary
with `result` for the in-memory pipeline JSON and `outputs` for any written
artifacts. `SemanticASR.transcribe_batch()` delegates to the existing
multi-process batch runner. It defaults to one pipeline worker per visible GPU;
ASR throughput should usually come from batching VAD segments inside the model
adapter.

The same external surface is available as a unified CLI:

```bash
semantic-asr transcribe --config configs/zh_cn.json --wav-path audio.wav --outdir output/audio
semantic-asr batch --config configs/hakka.json --wav-scp wav.scp --outdir output/batch --num-workers 1 --devices 4,5,6,7
semantic-asr models yue_hk --role punc
```

When the package is not installed, use `python -m semantic_asr.cli ...`.

## ASR And Timestamp VAD Policy

ASR and timestamp providers receive a postprocessed ASR VAD segment list derived
from raw VAD. Tiny speech islands shorter than `asr_vad_min_segment_s` are
merged into a nearby segment when the silence gap is no more than
`asr_vad_max_merge_silence_s`; isolated tiny islands are skipped. The remaining
speech islands are then merged into longer ASR context segments up to
`asr_vad_max_segment_s` when the intervening silence is still within
`asr_vad_max_merge_silence_s`. This gives ASR and punctuation models enough
context while still preserving long silences as hard context boundaries.

Raw VAD remains available as `raw_vad_segments_ms` and is still used for final
cut segments. The postprocessed ASR/timestamp inputs are recorded separately as
`asr_vad_segments_ms`.

For MMS forced alignment, `<star>` is used only as a probe signal by default:
the adapter first runs a star-probe alignment to identify likely real token
gaps, then splits the long context into alignment islands and performs final
timestamps with no inserted gap stars. Numeric/currency placeholders that MMS
cannot align remain separate from probe gap stars and are restored to their
original surface text.

Semantic sentence merging happens after timestamps are available, using
punctuation, token timestamps, raw VAD, frame-level VAD speech probabilities
and sentence-boundary fusion.

## Output VAD Segment Policy

Final non-speech segment output is formatted separately from ASR slicing:

- merge adjacent speech segments when the silence gap is less than 500ms
- pad final output segments by 100ms on both sides
- if adjacent padding would overlap, split the middle silence gap equally
- sentence boundaries are aligned to these final output VAD segment boundaries,
  so JSON, TextGrid, SRT and CSV outputs use the same expanded ranges

## Expected Output

`SemanticAsrPipeline.process(wav_path, uttid)` returns:

```python
{
    "uttid": "...",
    "text": "...",
    "sentences": [{"start_ms": 0, "end_ms": 1000, "text": "..."}],
    "vad_segments_ms": [(0, 1000)],
    "raw_vad_segments_ms": [(0, 900)],
    "asr_vad_segments_ms": [(0, 1000)],
    "dur_s": 1.0,
    "words": [{"start_ms": 0, "end_ms": 100, "text": "..."}],
    "timestamp_segments": [...],
    "vad_frame_speech_probs": {"frame_shift_ms": 10, "frame_length_ms": 25, "probs": [...]},
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

Use `semantic_asr.adapters.build_firered_pipeline` to construct the
same model combination as the current FireRed system, while keeping the core
pipeline independent.

The FireRed ASR, VAD and Punc runtime code needed by the adapter is vendored
under `semantic_asr.firered_runtime`, so the standalone project does
not need to import the outer `fireredasr2s` package.

FireRed ASR already has token timestamp support, so its adapter forces
`return_timestamp=True` and uses `AsrTimestampProvider` to validate that
timestamps are present.
