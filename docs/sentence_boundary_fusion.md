# Sentence Boundary Fusion

## Goal

The pipeline should cut long-audio ASR results into useful semantic segments
without cutting through active speech.

Current priority:

1. Do not cut active speech when VAD probability or raw VAD says the boundary is
   unsafe.
2. Keep segments near the configured target and maximum duration.
3. Preserve semantic sentence boundaries when they are compatible with audio
   safety and duration.

`max_sentence_s` is therefore a bounded pressure signal. Fusion first looks for
an audio-safe boundary. If none is available and the group is already over max
duration, a semantic-complete boundary is kept to prevent unbounded output
segments. Active incomplete boundaries still merge and record
`max_duration_wait_for_silence`.

## Timing Views

The pipeline keeps two related views:

- `semantic_sentences`: punctuation/model-proposed sentence candidates.
- `sentences`: fused, audio-aware output sentences.

Final sentence objects also contain VAD-derived cut fields:

- `cut_start_ms`
- `cut_end_ms`
- `cut_segments_ms`

CSV, SRT and TextGrid exports use `cut_start_ms` / `cut_end_ms`. JSON keeps both
annotation timing (`start_ms` / `end_ms`) and cut timing.

## Evidence Used At Each Boundary

For every boundary between the previous fused sentence and the next semantic
candidate, `fuse_sentence_boundaries()` builds one `BoundaryCandidate` with:

- aligned-token gap in milliseconds;
- whether both sides are inside the same raw VAD speech island;
- whether raw VAD provides a supported silence gap;
- frame-level VAD speech-probability stats near the boundary:
  - `speech_prob_min`
  - `speech_prob_mean`
  - `speech_prob_max`
  - `speech_prob_boundary_ms`
- semantic completeness:
  - terminal punctuation;
  - continuation punctuation such as comma/colon/semicolon;
  - short incomplete fragments;
  - target-duration fallback.

RMS/waveform valley is no longer part of the boundary policy.

## Audio-Safe Boundary Rules

A boundary is audio-safe when either condition is true:

1. Raw VAD has a silence interval of at least `min_vad_silence_s` around the
   candidate.
2. Frame-level VAD probability has a local low-probability window where:
   - `speech_prob_min <= speech_prob_silence_threshold`
   - `speech_prob_mean <= speech_prob_silence_mean_threshold`

When a probability-supported boundary is kept, it snaps to the lowest local
speech-probability window center, then clamps to the token gap when possible.

An active-speech boundary is one where:

- local probability mean is at least `speech_prob_active_threshold`; or
- both sides are in the same raw VAD speech island, token gap is below
  `merge_max_token_gap_s`, and there is no audio-safe evidence.

## Decision Table

The current greedy decision table is:

1. If the merged span would exceed `max_sentence_s`:
   - keep the boundary if it is audio-safe;
   - keep a semantic-complete active-speech boundary as
     `max_duration_terminal_punctuation` or `max_duration_semantic_boundary`;
   - merge and record `max_duration_wait_for_silence` if it is active speech
     but semantic-incomplete;
   - otherwise keep as `max_duration_forced_boundary` as a no-probability
     fallback.
2. If the boundary is active speech, merge.
3. If the boundary is audio-safe and semantic-complete, keep.
4. If the boundary is audio-safe and the merged span has reached
   `target_sentence_s`, keep.
5. If the boundary is audio-safe but semantic-incomplete, merge.
6. If terminal punctuation appears and the merged span has reached
   `target_sentence_s`, keep.
7. If semantic-complete, not in the same raw VAD island, and the token gap is
   large enough, keep.
8. If both sides are in the same raw VAD island, merge.
9. Otherwise merge.

This keeps the logic readable: audio safety is checked first, duration prevents
pathological long spans, and semantic completeness decides which non-safe
fallbacks are acceptable.

## Preserving Gaps

Set top-level `preserve_sentence_gaps: true` when kept boundaries should leave
the inter-sentence silence unassigned.

When enabled:

- raw-VAD-silence boundaries use the two silence edges instead of one midpoint;
- probability-supported boundaries use the token gap when available;
- output VAD expansion is skipped;
- sentence grouping remains entirely inside boundary fusion.

## Single Grouping Stage

Boundary fusion is the only sentence-grouping stage. There is no later
short-gap sentence merge pass, so a kept semantic/audio-safe boundary cannot be
silently undone by post-processing.

## Debug Fields

Every boundary decision records:

- `action`
- `reason`
- `audio_safe`
- `audio_reason`
- `active_speech`
- `semantic_complete`
- `semantic_reason`
- `vad_silence_ms`
- `speech_prob_min`
- `speech_prob_mean`
- `speech_prob_max`
- `speech_prob_boundary_ms`
- `speech_prob_supported_silence`
- `combined_duration_ms`

These fields are the first place to inspect when a segment is too long, too
short, or appears to cut speech.
