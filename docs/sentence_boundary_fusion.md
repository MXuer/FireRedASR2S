# Sentence Boundary Fusion

## Problem

Punctuation models propose semantic sentence boundaries from text, but a
predicted sentence boundary can fall inside active speech. Such a boundary is
useful as text structure but unsafe as an audio-slicing boundary.

Raw VAD segments, aligned token timestamps and the waveform provide acoustic
evidence that should be fused with punctuation instead of treating any one
signal as authoritative.

When available, frame-level VAD speech probability is the primary acoustic
signal for audio-safe sentence boundaries. It is more direct than RMS energy
and more robust across noise conditions and languages.

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
- frame-level VAD speech probability around the candidate boundary;
- optional minimum waveform energy around the candidate boundary;
- resulting merged sentence duration.

Then apply these rules:

1. Treat a raw VAD non-speech interval of at least `200ms` as evidence that a
   boundary is audio-safe, and snap a kept boundary into that interval.
2. Treat a local minimum VAD speech probability below the configured silence
   threshold as evidence that a boundary is audio-safe, and snap a kept
   boundary to that local minimum.
3. Treat an RMS acoustic valley as secondary evidence that a boundary is
   audio-safe when VAD probability is unavailable or inconclusive.
4. Keep an audio-safe boundary only when the text on the left is semantically
   complete. Continuation punctuation such as commas, colons and semicolons,
   or a short/incomplete following fragment, should merge even across a safe
   silence.
5. Merge a punctuation-proposed boundary when frame-level speech probability is
   high around the boundary, or when both neighboring tokens are in the same raw
   VAD speech segment with a short token gap and no acoustic valley.
6. Treat `target_sentence_s` and `max_sentence_s` as duration preferences, not
   permission to cut active speech. If no VAD silence, VAD probability valley or
   acoustic valley supports a boundary, continue merging and record that the
   maximum duration is waiting for silence.
7. Record boundary metadata such as `semantic_boundary`, `vad_silence`,
   `vad_prob_valley`, `acoustic_valley`, `target_duration`,
   `max_duration_wait_for_silence`, `merged_active_speech`,
   `merged_active_speech_prob` and `merged_semantic_incomplete`.

Thresholds must be configurable and evaluated per VAD model/language. Raw VAD
is a strong signal, but it is not perfect; frame-level VAD probability is the
preferred acoustic signal, and waveform energy is a secondary check when VAD
misses a short pause or creates a false split.

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
2. Treats active-speech safety as the first priority: boundaries inside high
   speech probability are merged even when the merged sentence exceeds
   `max_sentence_s`.
3. Treats raw VAD silence, local frame-level VAD probability valleys and RMS
   valleys as audio-safe evidence, not as mandatory splits.
4. Keeps an audio-safe boundary only when semantic completeness also supports
   it; comma/colon/semicolon continuations and short incomplete fragments merge.
5. Writes each candidate decision to `sentence_boundary_decisions`, including
   `audio_safe`, `audio_reason`, `semantic_complete` and `semantic_reason`.

The Arabic profile is the first enabled profile. Evaluate proposed merges on
multilingual reference fixtures before enabling the policy more broadly.

`target_sentence_s` is a soft preference for ambiguous boundaries.
`max_sentence_s` is also soft: it encourages earlier supported boundaries but
does not force a split through active speech.

## Preserving Sentence Gaps

Set top-level pipeline config `preserve_sentence_gaps: true` when final
sentence intervals should leave inter-sentence silence unassigned. In this mode:

- kept raw-VAD-silence boundaries use the raw VAD silence edges instead of a
  single midpoint;
- kept probability/acoustic-valley boundaries use the neighboring token gap
  when available;
- output VAD sentence expansion is skipped;
- final short-gap sentence merging is skipped.

This mode is intended for audio cutting. The default midpoint behavior remains
useful for continuous annotation views such as SRT/TextGrid where contiguous
sentence intervals are easier to inspect.

## Final Short-Gap Merge

After sentence-boundary fusion, output VAD alignment and overlap removal, the
pipeline applies one final configurable merge pass over the final `sentences`.
Adjacent final sentences merge when:

- the silence gap between them is below `final_sentence_merge_max_gap_s`
  (`2.0s` by default);
- the merged sentence duration would not exceed
  `final_sentence_merge_max_duration_s` (`15.0s` by default).

This pass is intentionally later than boundary fusion. It smooths short final
sentence gaps without changing ASR/MMS alignment inputs or the boundary-decision
debug trail.

## Portuguese Raw-Align Example

For `data/test/short/pt_br-short.wav`, raw-VAD ASR/MMS alignment originally
kept the following three candidates as separate final sentences because VAD
silence or an acoustic valley existed between them:

```text
47.895s-53.200s: O primeiro ponto ... negócios,
53.200s-55.408s: as empresas podem criar
55.600s-58.600s: uma conexão harmoniosa entre diferentes setores.
```

After semantic-completeness gating, the first two boundaries merge because the
left text ends in a comma and then lacks terminal punctuation. The final output
keeps one sentence from `47.895s` to `58.600s`, and then keeps the next
VAD-supported boundary after `setores.`.
