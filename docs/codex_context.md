# Codex Context Handoff

This document preserves the working context behind the standalone Semantic ASR
repository migration. It is meant for future Codex sessions and maintainers; it
is not a replacement for the git history.

## Project Goal

Semantic ASR turns long audio into language-aware, semantically coherent
sentence segments with stable timestamps and export artifacts.

The current target is multilingual long-audio ASR where the first priority is
not cutting through speech, the second priority is semantic sentence quality,
and the third priority is configurable segment duration.

## Core Pipeline

The pipeline requires four roles for every profile:

```text
VAD -> ASR -> Timestamp Provider -> Punctuation / Boundary Strategy -> Outputs
```

- `vad`: finds speech islands and may provide frame-level speech probabilities.
- `asr`: transcribes longer VAD-derived context segments.
- `timestamp`: either validates ASR-native token timestamps or runs forced
  alignment.
- `punc`: restores punctuation or applies a sentence-boundary strategy when
  punctuation is weak or unavailable.

Every profile lives under `configs/` and should use canonical language-region
ids such as `zh_cn`, `de_de`, `ar_sa`, `ko_kr`, `ja_jp`, `vi_vn`.
Adapters convert these canonical ids to model-native language names/codes
internally.

## Current Model Families

VAD adapters:

- FireRed VAD
- Silero VAD
- Ten-VAD

ASR adapters:

- Whisper large
- Fun-ASR-Nano
- Qwen3-ASR
- Dolphin
- Seamless M4T
- FireRedASR2

Timestamp providers:

- ASR-native timestamps
- MMS forced alignment
- Qwen3 ForcedAligner
- FireRedASR2 native timestamps

Punctuation / boundary adapters:

- ASR-native punctuation
- FireRedPunc
- FunASR `ct-punc`
- XLM-R punctuation
- Naqta Arabic punctuation
- Yue punctuation restoration
- Qwen semantic boundary strategy

Translation:

- Hunyuan-MT service is optional and sits outside the core ASR pipeline.
- Chinese-source to `zh_cn` translation is skipped with an identity sidecar.

## Timestamp And Boundary Strategy

Important current behavior:

- Raw VAD speech islands are preserved for debugging and final cut decisions.
- ASR may receive merged VAD context segments so recognition has enough context.
- MMS alignment now uses `<star>` as an optional probe signal to detect real
  gaps, then uses no-star alignment for final token timestamps.
- Numeric, currency and symbol tokens that MMS cannot align are replaced by
  placeholder star tokens during alignment and restored afterward.
- For Chinese, Japanese and Korean MMS timestamps are character-level. Do not
  consume token timestamps by whitespace splitting for these languages.
- Frame-level VAD probability is preferred over RMS valleys. RMS valley logic
  was intentionally removed/sidelined as unreliable in noisy multilingual data.
- Sentence splitting uses punctuation and aligned token timestamps, but should
  avoid cutting in active speech.
- Raw VAD silence above the configured hard gap threshold blocks merging.
- When a segment grows beyond the target duration, boundary fusion should use a
  rolling selector: choose the best recent low-speech-probability semantic
  boundary instead of cutting at the first boundary after the limit.

Current priority order:

```text
avoid cutting speech > duration cap / usability > semantic completeness
```

In practice, very long segments are also bad output. The current target is to
keep around configurable 30s caps while still snapping to safe acoustic
boundaries whenever possible.

## Outputs

The pipeline writes:

- JSON
- CSV
- SRT
- TextGrid
- resolved config

`cut_start_ms` and `cut_end_ms` are the final export-time boundaries. Writers
should use those final cut fields when present. Internal `start_ms` and
`end_ms` can represent semantic/timestamp boundaries before final VAD-safe
adjustment.

TextGrid overlap must be treated as a real strategy bug, not silently fixed in
the writer. If intervals overlap, inspect the JSON and boundary decisions.

## Service Architecture

`semantic_asr_service/` exposes an async HTTP service:

- `POST /v1/jobs`
- `GET /v1/jobs/{job_id}`
- `GET /v1/jobs/{job_id}/result`
- `GET /v1/jobs/{job_id}/artifacts/{format}`
- `GET /v1/models`
- `GET /demo`

The API process should not run heavy models. Worker processes claim queued jobs
from SQLite and bind to assigned GPUs.

Recent local convention:

- WebUI/API: `0.0.0.0:10086`
- Demo token: `demo-local`
- ASR workers: GPUs `6,7`
- Translation service: GPU `5`
- Translation endpoints may run on `10096,10097,10098`

The demo page supports upload, user/profile filtering, waveform review,
segment playback, zoom/drag, artifact download and bilingual display.

## Operational Lessons

- Always write or update `tasks/todo.md` before starting a work item.
- Every new model needs install/download documentation under `docs/models/`
  and a standalone smoke/test example.
- GPU commands often need sandbox escalation.
- Do not put model weights, audio data, outputs, logs or SQLite service data in
  the standalone repository.
- Keep `uroman/data/*.txt` and FireRed runtime `data/*.py` tracked even though
  top-level `data/` is ignored.
- Batch `--num-workers` means workers per selected GPU.
- For long service jobs, ASR success should not be converted into ASR failure
  only because optional translation failed.
- Translation cache files must be per job to avoid imported batch jobs sharing
  one output directory and mixing translations.

## Current Migration Intent

The new standalone repository should contain only maintainable project assets:

- `semantic_asr/`
- `semantic_asr_service/`
- `configs/`
- `docs/`
- `examples/`
- `scripts/`
- `tests/`
- `tasks/`
- `uroman/` runtime files
- root docs and packaging files

It should exclude:

- `.git/` history from FireRedASR2S
- `data/`
- `output/`
- `logs/`
- `service_data/`
- `pretrained_models/`
- `*.egg-info`
- Python bytecode caches

## Quick Validation

After migration, run lightweight checks first:

```bash
python -m py_compile semantic_asr/config.py semantic_asr/core.py semantic_asr_service/app.py
python -m unittest tests.test_language_mapping tests.test_config_runner
```

Full model tests require local model installations and GPUs.
