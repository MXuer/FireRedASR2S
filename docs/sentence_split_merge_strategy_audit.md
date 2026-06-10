# Sentence Split/Merge Strategy Audit

This note describes the current split/merge policy in
`semantic_asr/core.py`, `semantic_asr/sentence_boundaries.py`,
`semantic_asr/punctuation.py` and `semantic_asr/outputs.py`.

## Product Target

The project target is:

1. Avoid cutting active speech.
2. Keep output segments near configurable duration limits.
3. Preserve semantic sentence boundaries when they do not violate audio safety.

This means a sentence may exceed `max_sentence_s` when every nearby candidate
boundary still looks like active speech. In that case the decision reason should
make the wait explicit, instead of hiding an unsafe cut.

## Current Timing Layers

There are four timing layers:

1. Raw VAD speech islands.
2. ASR/timestamp segments.
3. Semantic sentence candidates from punctuation.
4. Final output cut ranges.

Final sentence JSON therefore has two time concepts:

- `start_ms` / `end_ms`: annotation span after semantic boundary fusion.
- `cut_start_ms` / `cut_end_ms`: actual export/cut span derived from raw VAD
  intersections.

CSV, SRT and TextGrid use `cut_start_ms` / `cut_end_ms`.

## Execution Flow

### 1. Raw VAD

`SemanticAsrPipeline.process()` calls the configured VAD adapter.

The adapter returns:

- `timestamps`: raw non-silent speech segments.
- optional `frame_speech_probs`: frame-level speech probability.

The final JSON records these as:

- `raw_vad_segments_ms`
- `vad_frame_speech_probs`

### 2. ASR Segment Selection

ASR and timestamp providers receive raw VAD speech islands directly. This is no
longer configurable in the core pipeline.

The selected ASR inputs are stored as `asr_vad_segments_ms`, which should match
`raw_vad_segments_ms`.

### 3. ASR + Timestamp Provider

ASR produces text, and the timestamp provider returns token/word timestamps.
The timestamp provider can be:

- ASR-native timestamps;
- ASR-native CTC/non-autoregressive timestamps;
- external forced alignment such as MMS or Qwen3 ForcedAligner.

For MMS, raw VAD segments are aligned immediately after ASR. Semantic merging is
done after timestamps exist.

### 4. Punctuation To Semantic Candidates

The punctuation strategy returns sentence-like text candidates.

`_format()` maps those candidates back to absolute times and creates:

- `semantic_sentences`
- global `words`

These candidates may be too short, too long, or located inside continuous
speech. They are semantic proposals, not final cut decisions.

### 5. Boundary Candidate Construction

When `sentence_boundary_fusion.enabled=true`,
`fuse_sentence_boundaries()` walks candidates left to right.

For every boundary it builds one `BoundaryCandidate`:

- `token_gap_ms`
- `same_raw_vad`
- supported raw-VAD silence, if any
- frame-probability stats near the candidate:
  - min
  - mean
  - max
  - selected low-probability boundary time
- semantic completeness reason
- active-speech flag
- combined duration

RMS/waveform valley is intentionally not used.

### 6. Boundary Decision

The current decision order is:

1. Over `max_sentence_s`:
   - keep audio-safe boundaries;
   - merge active-speech boundaries and record `max_duration_wait_for_silence`;
   - keep `max_duration_forced_boundary` only when no probability/raw-VAD
     active-speech evidence is present.
2. Merge active-speech boundaries.
3. Keep audio-safe and semantic-complete boundaries.
4. Keep audio-safe boundaries once the merged span reaches `target_sentence_s`.
5. Merge audio-safe but semantic-incomplete boundaries.
6. Keep terminal punctuation once the merged span reaches `target_sentence_s`.
7. Keep semantic-complete boundaries across raw-VAD islands when the token gap is
   large enough.
8. Merge boundaries inside the same raw VAD island.
9. Merge everything else.

The result is:

- final fused `sentences`;
- `sentence_boundary_decisions` for debugging.

### 7. Output VAD Formatting

`_format_output_vad_segments()` creates display/cut VAD ranges:

1. `merge_close_vad_segments()` merges raw VAD islands when the non-speech gap
   is below `output_vad_min_silence_merge_s`.
2. `pad_vad_segments()` extends segments left/right by `output_vad_pad_s`
   without crossing neighboring bounds.

The result is recorded as `vad_segments_ms`.

### 8. Sentence Alignment To Output VAD

When `preserve_sentence_gaps=false`, the pipeline may expand annotation spans
to the output VAD island they touch. This is for readable continuous annotation
views.

When `preserve_sentence_gaps=true`, kept gaps remain unassigned and this
expansion is skipped.

### 9. Single Sentence Grouping Stage

Boundary fusion is the only sentence grouping stage. The old final short-gap
merge pass was removed so post-processing cannot silently undo a boundary that
fusion deliberately kept.

### 10. Cut Ranges And Exports

`add_sentence_cut_segments()` intersects final sentence spans with raw VAD
islands and adds:

- `cut_segments_ms`
- `cut_start_ms`
- `cut_end_ms`

`remove_sentence_cut_overlaps()` prevents overlapping cut ranges. TextGrid no
longer contains an overlap fallback; if sentence intervals overlap in a way the
writer cannot represent, that is treated as a strategy bug.

## Current Strengths

- Raw VAD, frame-level probability, token timestamps and punctuation are all
  visible in one decision trace.
- No final short-gap merge exists after boundary fusion, so grouping decisions
  are traceable in one place.
- High speech-probability boundaries are merged even when terminal punctuation
  exists.
- `max_sentence_s` no longer forces a high-probability active-speech cut.
- The code has focused regressions for Arabic active-speech boundaries, Naqta
  terminal punctuation, German short terminal fragments, Portuguese semantic
  fragments and no-probability fallback behavior.

## Remaining Risks

### 1. Greedy Fusion Still Has No Lookback

The algorithm decides at the current boundary. It does not choose the best safe
boundary from a rolling window.

Better future behavior:

- keep a small list of recent candidate boundaries inside the current group;
- when duration pressure is high, choose the best recent audio-safe boundary;
- rank by audio safety, semantic completeness, distance to target and
  probability strength.

### 2. `max_duration_forced_boundary` Is A Fallback, Not A Guarantee

If frame probabilities are missing and raw VAD does not mark the candidate as
active speech, the pipeline can still keep a max-duration fallback boundary.

This is useful for old JSON or nonstandard VADs, but it is weaker than the main
policy. Production VAD adapters should expose `frame_speech_probs`.

### 3. `preserve_sentence_gaps` Controls Several Behaviors

The flag currently affects:

- boundary placement;
- output VAD expansion;
- output VAD expansion.

Cleaner future config names would be:

- `preserve_annotation_gaps`
- `align_annotations_to_output_vad`

### 4. Annotation Times And Cut Times Are Easy To Confuse

Exports use cut times, while many debugging conversations inspect annotation
times. This needs schema documentation and maybe clearer field names later:

- `annotation_start_ms` / `annotation_end_ms`
- `cut_start_ms` / `cut_end_ms`

### 5. Punctuation-To-Token Mapping Is Still Fragile

`split_text_by_punctuation()` maps text back to timestamps using token counting.
This can be fragile for:

- no-space scripts;
- Arabic punctuation/diacritics;
- punctuation-only fragments;
- numeric and ITN forms;
- tokenizer differences between ASR and punctuation models.

### 6. First/Last Candidate Bounds Can Absorb Silence

`_format()` still forces the first candidate in an ASR segment to segment start
and the last candidate to segment end. Raw-VAD ASR reduces the damage, but this
can still attach leading/trailing silence to text.

## Simplification Direction

The current design is already closer to one grouping stage. The next useful
simplification is not another merge pass; it is a better boundary selector
inside boundary fusion.

Recommended next version:

1. Build all `BoundaryCandidate` objects for a current group.
2. Maintain a rolling buffer of recent candidates.
3. If the group is below target, cut only on very strong semantic + audio-safe
   evidence.
4. If the group is near target, cut on audio-safe + semantic-complete evidence.
5. If the group is over max, choose the best recent audio-safe boundary.
6. If no safe boundary exists, keep waiting and emit
   `max_duration_wait_for_silence`.

This preserves the project priority while making long-segment behavior easier
to reason about.
