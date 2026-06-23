# Debugging A Boundary Case

Use this checklist when a user reports that a segment cuts speech, merges too
much, splits too aggressively, or produces overlapping exports.

## Required Inputs

Ask for or locate:

- config path
- wav path or `job_id`
- output JSON path
- optional TextGrid/CSV/SRT path
- exact sentence text and time range
- expected behavior

Good bug report shape:

```text
config: configs/de_de.json
job_id: ...
json: output/.../<uttid>.json
bad segment: 308.062s - 312.563s
problem: right edge cuts speech
expected: extend to next VAD-safe boundary or merge with following segment
```

## Inspect JSON First

Fields to inspect:

- `raw_vad_segments_ms`
- `asr_vad_segments_ms`
- `vad_segments_ms`
- `vad_frame_speech_probs`
- `words`
- `sentences`
- `sentence_boundary_decisions`

For the bad sentence, compare:

- `start_ms` / `end_ms`
- `cut_start_ms` / `cut_end_ms`
- `cut_segments_ms`

Writers use cut fields. If TextGrid is wrong, first check whether JSON cut
fields are already wrong.

## Classify The Failure

| Symptom | Likely area |
| --- | --- |
| Token times touch with no gaps | Timestamp provider or preserve-gap mode. |
| CJK sentence consumes wrong number of timestamps | Char-level matching bug. |
| Boundary falls inside one raw VAD island | Boundary fusion or VAD probability logic. |
| TextGrid overlap | Upstream sentence/cut interval bug. |
| Very long segment | Duration pressure / rolling selector / long silence threshold. |
| Very short fragment before complete sentence | Short incomplete fragment merge rule. |
| MMS CTC target too long | VAD island too short, ASR hallucination, or token normalization. |
| MMS empty target | Text normalization removed all alignable tokens. |

## Boundary Decision Reasons

Common `sentence_boundary_decisions.reason` values:

- `long_vad_silence`: raw VAD found a hard pause.
- `vad_silence`: raw VAD supports a safe boundary.
- `vad_prob_silence`: frame probability supports a safe boundary.
- `merged_active_speech_prob`: boundary was active speech by probability.
- `merged_active_speech`: boundary was active speech by VAD/timestamp logic.
- `previous_short_incomplete_fragment`: short prefix should merge forward.
- `terminal_punctuation`: punctuation and target duration allowed a split.
- `max_duration_rolling_boundary`: max duration pressure selected a recent low-probability boundary.
- `max_duration_wait_for_silence`: over cap but still active speech; keep merging.
- `merged_non_monotonic_boundary`: token timestamps overlap or go backward.

## Fix Order

1. Reproduce or inspect the exact JSON.
2. Identify which module produced the bad evidence:
   VAD, ASR, timestamp, punctuation, fusion, cut adjustment or writer.
3. Add or update the narrowest test possible.
4. Fix the earliest responsible stage.
5. Re-run the representative case.
6. Update `tasks/progress.md` with config, wav/job, output path, root cause and command.

Avoid writer-layer fallbacks that hide upstream errors.
