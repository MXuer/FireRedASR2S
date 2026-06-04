# Sentence Boundary Fusion

## Problem

Punctuation models propose semantic sentence boundaries from text, but a
predicted sentence boundary can fall inside active speech. Such a boundary is
useful as text structure but unsafe as an audio-slicing boundary.

Raw VAD segments, aligned token timestamps and the waveform provide acoustic
evidence that should be fused with punctuation instead of treating any one
signal as authoritative.

## Two Boundary Views

Keep two related views:

- `semantic_sentences`: punctuation/model-proposed text sentences. These may
  have boundaries inside continuous speech.
- `sentences`: audio-safe semantic sentences used by JSON, CSV, SRT and
  TextGrid consumers that may slice audio.

The initial implementation may expose only `sentences`, but the internal
boundary decision should retain the semantic candidate and its decision
metadata.

## Recommended Decision Policy

For every boundary between the previous sentence's last aligned token and the
next sentence's first aligned token, calculate:

- aligned-token gap duration;
- whether both tokens belong to the same raw VAD speech segment;
- whether a raw VAD non-speech interval exists between the tokens;
- optional minimum waveform energy around the candidate boundary;
- resulting merged sentence duration.

Then apply these rules:

1. Keep and snap a boundary to the midpoint or lowest-energy point of a raw
   VAD non-speech interval of at least `200ms`.
2. Merge a punctuation-proposed boundary when both neighboring tokens are in
   the same raw VAD speech segment, the token gap is below `300ms`, and no
   acoustic low-energy valley supports the boundary.
3. Treat ambiguous boundaries as soft evidence. Keep them only when needed to
   avoid exceeding a target sentence duration, initially `15s`.
4. Before exceeding the maximum sentence duration, initially `30s`, force a
   split at the strongest nearby acoustic boundary.
5. Record boundary metadata such as `punctuation`, `vad_silence`,
   `acoustic_valley`, `target_duration`, `max_duration`, `merged_active_speech`
   and a confidence score.

Thresholds must be configurable and evaluated per VAD model/language. Raw VAD
is a strong signal, but it is not perfect; waveform energy is a useful
secondary check when VAD misses a short pause or creates a false split.

## Arabic Example

For `data/test/ar_sa-short.wav`:

- The boundary `83.494s -> 83.634s` lies inside raw VAD speech segment
  `82.040s -> 85.160s`. The local waveform energy is high. This boundary
  should be merged for audio-safe output.
- The boundary `87.396s -> 87.496s` is supported by the raw VAD silence
  interval `87.180s -> 87.440s`. It should be kept and may be snapped into that
  silence interval.

This demonstrates why rejecting punctuation boundaries unsupported by acoustic
evidence is better than punctuation-only slicing. It still needs duration
limits and acoustic fallback so that long continuous speech is not merged
without bound.

## Implementation Order

The first implementation is available in `semantic_asr.sentence_boundaries`
and is enabled only by profiles that set `pipeline.sentence_boundary_fusion`.
It:

1. Keeps the punctuation-proposed candidates as `semantic_sentences`.
2. Treats active-speech safety as a hard constraint: active-speech boundaries
   are merged whenever the resulting sentence is no longer than
   `max_sentence_s`.
3. Keeps and snaps nearby boundaries supported by raw VAD silence.
4. Prevents merging beyond `max_sentence_s`.
5. Writes each candidate decision to `sentence_boundary_decisions`.

The Arabic profile is the first enabled profile. Evaluate proposed merges on
multilingual reference fixtures before enabling the policy more broadly.

`target_sentence_s` is a soft preference for ambiguous boundaries.
`max_sentence_s` is the hard limit that may prevent an active-speech merge.
