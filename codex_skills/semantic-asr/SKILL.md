---
name: semantic-asr
description: Use when working in the Semantic ASR repository on multilingual long-audio ASR, sentence boundary fusion, VAD/ASR/timestamp/punctuation adapters, service workers, WebUI review, translation sidecars, or model onboarding.
---

# Semantic ASR

## Start Of Work

Before changing code, read:

1. `AGENT.md`
2. `PROGRESS.md`
3. `DECISIONS.md`
4. `docs/codex_context.md`
5. `tasks/todo.md`

Then add the current task to `tasks/todo.md`.

## Core Rules

- Every pipeline profile has four mandatory roles: `vad`, `asr`, `timestamp`, `punc`.
- Use canonical language-region ids such as `zh_cn`, `de_de`, `ar_sa`, `ko_kr`.
- Model adapters convert canonical ids to model-native codes internally.
- Sentence cutting priority: avoid cutting speech, then keep duration usable, then preserve semantic completeness.
- Do not hide TextGrid/CSV/SRT overlap in writers; overlap is an upstream strategy bug.
- CJK MMS timestamps are character-level. Do not consume them by whitespace token counts.
- GPU/service changes must record device, worker count, port and log path in `PROGRESS.md`.

## Common Workflows

### Boundary Bug

Read `docs/debugging/debug_boundary_case.md`.

Inspect:

- `raw_vad_segments_ms`
- `asr_vad_segments_ms`
- token/word timestamps
- `sentence_boundary_decisions`
- `cut_segments_ms`
- `cut_start_ms` / `cut_end_ms`

Analyze the exact case before patching strategy code.

### New Model

For every new model:

- add adapter
- register it in `semantic_asr/registry.py`
- update `semantic_asr/language_support.py`
- add `docs/models/<model>.md`
- add unit test or smoke example
- update `docs/model_matrix.md`

### Service / WebUI

Read `docs/architecture/service_runtime.md`.

Local convention:

- WebUI/API: `0.0.0.0:10086`
- demo token: `demo-local`
- ASR workers: GPUs `6,7`
- translation service: GPU `5`

Keep ASR and translation worker pools separate.

## Architecture References

- `docs/architecture/current_pipeline.md`
- `docs/architecture/sentence_boundary_strategy.md`
- `docs/architecture/service_runtime.md`
- `docs/model_matrix.md`
- `docs/codex_context.md`

## Finish Work

Before final response:

1. Run the narrowest useful tests.
2. Update `tasks/todo.md`.
3. Update `PROGRESS.md` for non-trivial decisions, commands or service changes.
4. Commit completed work when requested or when ending a migration/documentation task.
