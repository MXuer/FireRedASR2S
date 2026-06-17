# Progress

Current state:

- German ASR-native text punctuation now protects `n. Chr.` / `ca.` historical
  abbreviations. The bug in job `ede8b27b09de431ca03cb56d96d3f6db` happened in
  `split_text_by_punctuation()` before sentence-boundary fusion: `n. Chr.` was
  split into `n.` and `Chr.` fragments, and `Chr., der ...` was split before the
  comma continuation. The new narrow rule keeps `n.` before `Chr`, keeps `Chr.`
  before comma/lowercase continuation, keeps `ca.` before year-like numbers,
  and still allows `n. Chr. Der ...` to start a new sentence.
- MMS waveform handling now matches the l2s/torchaudio input convention. The
  current pipeline reads audio with `soundfile.read(dtype="int16")`; before this
  fix, `semantic_asr.mms_runtime.MmsAligner` converted those int16 PCM values
  directly to float and fed large-amplitude audio to MMS. `generate_emissions()`
  now normalizes integer PCM waveforms to float `[-1, 1]` before inference and
  preserves already-float waveforms as-is. The Vietnamese `/home/duhu/vi_sub.wav`
  alignment with the provided text now matches l2s for the inspected tokens:
  `có.` at `5.304s-5.384s`, `Trên` at `6.565s-6.665s`, with an inferred
  use-star gap of `1.181s`.
- `scripts/debug_mms_alignment.py` now accepts explicit `--text` or
  `--text-file`, writes real `first_pass_use_star_token_alignment`, and derives
  `first_pass_inferred_star_gaps` from adjacent real-token gaps. The previous
  low-level expanded-star spans remain only as
  `first_pass_expanded_star_span_debug`.
- Added `scripts/debug_mms_alignment.py` for focused MMS alignment inspection.
  It clips a region from an existing job wav and writes both the first-pass
  inserted-`<star>` alignment and the second-pass no-star alignment into one
  JSON file. The Vietnamese case
  `dbac1886ef0b438f98cfda640bb59e18`, clip `18.680s-34.331s`, was written to
  `output/vi_mms_alignment_debug/dbac1886ef0b438f98cfda640bb59e18_clip_18680_34331.json`.
  In that standalone clip, the first-pass `<star>` between `có.` and `Trên` is
  only `24.484s-24.504s` (20ms), while the second-pass no-star alignment places
  `có.` at `24.364s-24.504s` and `Trên` at `24.524s-25.044s`. This is not a
  meaningful silence boundary.
- The previous demo/API/worker process group running from the main repository
  path `/data/duhu/FireRedASR2S` has been stopped. The unrelated
  `/data/duhu/semantic-asr-trans-api` service on port 8000 was left untouched.
- The active demo/API/worker service now runs from this derived worktree:
  `/home/duhu/.codex/worktrees/0d7a/FireRedASR2S`. It listens on port `10086`,
  uses `service_data/demo`, and has ASR workers on GPUs `6,7` with two workers
  per GPU. Translation still reuses `10087,10088,10089`. Health check returned
  `{"ok":true}`, and `/demo` served the web UI.
- Added `configs/zh_cn_mms_starprobe.json` as a real Chinese long-audio smoke
  profile for the new MMS star-probe path: Silero VAD, FunASR-Nano ASR, MMS
  forced alignment with `star_probe_enabled=true`, and FireRedPunc.
- The generic repository `data/` ignore rule also hid `uroman/data`, which MMS
  needs for romanization. The missing uroman tables caused an earlier derived
  worktree run to produce empty MMS targets and mostly discard the recognized
  text. `.gitignore` now explicitly unignores `uroman/data/*.txt`, and the
  required runtime tables are tracked.
- Real GPU smoke passed on GPU 2 for
  `/data/duhu/FireRedASR2S/data/zh-cn-long.wav` with
  `configs/zh_cn_mms_starprobe.json`. Outputs are under
  `output/experiments/zh_cn_long_mms_starprobe`; the final JSON has 35
  sentences, 3418 words, 35 timestamp segments, 0 discarded ASR segments, 4
  timestamp segments with selected MMS star-probe gaps, and 0 invalid or
  overlapping sentence/cut intervals. JSON/CSV/SRT/TextGrid were generated.
- FireRed runtime package data directories are now explicitly unignored and
  tracked despite the repository-wide `data/` ignore rule. This restores
  `semantic_asr.firered_runtime.fireredasr2.data` and
  `semantic_asr.firered_runtime.fireredpunc.data` in derived worktrees and
  clean clones. Full test discovery currently passes with 187 tests.
- ASR VAD postprocessing now builds longer ASR context segments: tiny raw VAD
  islands are still merged/skipped first, then adjacent speech islands are
  merged up to `asr_vad_max_segment_s` when the silence gap is within
  `asr_vad_max_merge_silence_s`. Raw VAD remains preserved for final cuts and
  JSON debugging.
- MMS forced alignment now uses `<star>` as an optional probe signal rather
  than as the final timestamp strategy. The adapter runs a star-probe pass to
  identify likely real token gaps, filters them with duration plus raw
  VAD/frame-probability evidence when available, then splits the long context
  into no-star alignment islands for final token timestamps. Numeric/currency
  placeholder stars remain distinct from temporary gap-probe stars.
- The web demo upload form now shows browser upload progress for multi-file
  submissions. Job creation requests use `XMLHttpRequest` so the UI can display
  current file index, per-file percentage, overall progress, and approximate
  aggregate upload speed while still freezing the selected profile/formats for
  the whole batch.
- Translation skips Hunyuan-MT for Chinese-source to Chinese-target jobs
  (`zh_cn`, `zh`, `hakka`, etc. -> `zh_cn`). The service writes an identity
  translation cache with `model="identity"` and `skipped=true` so the web UI
  can still load the expected bilingual sidecar without using GPU.
- The web demo freezes the selected Language/Profile and output formats at the
  start of a multi-file upload. Changing the dropdown while files are still
  being submitted no longer affects later files in that same upload batch; the
  profile and format controls are disabled until submission finishes.
- FireRedASR2 is now available as composable components:
  `firered_asr` for ASR and `firered_asr_native` for native timestamp
  validation/pass-through. The ASR component defaults to the AED Chinese model
  path `pretrained_models/FireRedASR2-AED` with `return_timestamp=true`.
  Language support metadata marks both as Chinese (`zh`) components, and
  `docs/models/firered_asr.md` documents the config fragment.
- Translation now uses `tencent/HY-MT1.5-1.8B-FP8` by default. The Hunyuan
  startup script discovers the newest local HuggingFace snapshot unless
  `HUNYUAN_MT_MODEL_PATH` is set, supports `HUNYUAN_MT_PORTS` for multiple
  same-GPU replicas, and runs with `conda run --no-capture-output` so service
  logs are visible. Current live translation replicas are on GPU `5`, ports
  `10087,10088,10089`, each with `--max-concurrent 8`.
- Translation clients now accept comma-separated
  `SEMANTIC_ASR_TRANSLATION_BASE_URL` values and round-robin requests across
  replicas. Defaults are `SEMANTIC_ASR_TRANSLATION_BATCH_SIZE=64`,
  `SEMANTIC_ASR_TRANSLATION_MAX_CONCURRENCY=24`, and
  `SEMANTIC_ASR_TRANSLATION_TIMEOUT_S=300`.
- Translation backfill exists at
  `python -m semantic_asr_service.backfill_translations --target zh_cn`. The
  demo service data backfill completed successfully: 21 succeeded jobs with
  JSON outputs now have `zh_cn` translation caches, with 0 missing.
- Current live services are background `setsid` processes: demo/API on
  `0.0.0.0:10086`, ASR workers on GPUs `6,7` with two workers per GPU, and
  Hunyuan-MT replicas on ports `10087,10088,10089`. Health checks passed for
  all four ports.
- Failed/canceled demo jobs can be retried through
  `POST /v1/jobs/{job_id}/retry`; failed rows show compact Error/Retry buttons
  instead of dumping long traceback text into the Jobs table. Auto translation
  now runs in a daemon background thread after the ASR job is marked succeeded,
  so ASR workers do not wait for Hunyuan-MT before claiming the next ASR job.
  Demo ASR worker defaults are GPUs `6,7` with four workers per GPU, and
  `scripts/start_hunyuan_mt_service.sh` starts Hunyuan-MT on GPU `5`.
- The web demo Jobs table has been simplified for cleanup work: language/profile
  is now a separate column, file labels are truncated in-table with full text in
  the cell title, the Stage column was removed, and jobs can be multi-selected
  and deleted. Backend `DELETE /v1/jobs/{job_id}` enforces ownership/admin
  access, rejects running jobs with 409, removes the DB row, and only deletes
  upload/output directories under the configured service roots. The running
  `10086` demo service has been restarted with the current code.
- The web demo Review panel no longer shows Text/Target/Translate controls or
  the Zoom slider. It defaults to bilingual display with `zh_cn` translation,
  loads cached `translations/zh_cn.json` when a job is opened, and keeps
  mouse-wheel zoom plus drag-to-pan. Workers can now auto-translate completed
  ASR jobs via `SEMANTIC_ASR_AUTO_TRANSLATE_TARGETS`; the demo startup defaults
  this to `zh_cn`. Translation failures are logged but do not fail the ASR job.
- Earlier Hunyuan-MT slowness was traced to a single local transformers model
  server underusing GPU 5. A single HY-MT1.5 instance stayed around 30% GPU
  utilization; three same-GPU replicas raised utilization to 100% during
  backfill and completed the pending translation cache generation.
- The web demo job panel now matches the upload panel height, scrolls the table
  inside the panel, and provides `Language / Profile` plus `PM / Username`
  filters. `GET /v1/jobs` supports optional `config` and `user_id` filters
  while preserving normal-user scoping; admin tokens can filter across PMs.
- Internal smoke-test jobs whose ids start with `translation_smoke_` are hidden
  from job lists and counts. The visible `translation_smoke_hunyuan_mt` entry
  was a hand-created Hunyuan-MT smoke validation row in `service_data/demo`.
- `scripts/start_demo_service.sh` now uses a short timeout for the translation
  service health check so an unhealthy translation endpoint cannot block ASR
  demo API/worker startup.
- MMS runtime now normalizes uroman token whitespace before both CTC alignment
  and span reconstruction, then drops alignment items whose normalized uroman
  token is empty. This prevents repeated/edge spaces or empty uroman outputs
  from creating empty expected characters in `get_spans()`.
- Whisper large now exposes normal decode options plus short-audio overrides.
  Short clips default to stricter deterministic beam decoding with
  `short_temperature=0.0`, `short_beam_size=5` and
  `short_length_penalty=0.0`; this is a mitigation only, not the main
  hallucination guard.
- MMS forced alignment now performs adapter-level ASR feasibility checks before
  calling alignment. It estimates frame count, uroman/dictionary target count,
  consecutive repeats and required frame count. Empty text/target,
  CTC-impossible spans and dense tiny-segment hallucinations are skipped by
  default and recorded in top-level `discarded_asr_segments`.
- Approximate MMS timestamp fallback is now opt-in with
  `fallback_on_feasibility_error=true`; the default path avoids producing
  fake monotonic timestamps for unalignable ASR text.
- ASR VAD postprocessing now treats `duration <= asr_vad_min_segment_s` as tiny,
  so exact-threshold 500ms islands no longer slip through.
- MMS CTC-length failure analysis for Arabic `batch_4_output` is complete. The 30 `targets length is too long for CTC` errors are mostly raw-VAD microsegment failures: 27/30 have only 2-16 MMS emission frames, and a VAD-only scan found matching 50-340ms FireRed VAD speech islands. ASR/MMS input VAD is now postprocessed before transcription/alignment: tiny islands below `asr_vad_min_segment_s` are merged into nearby speech when possible, or skipped when isolated. The remaining historical failures are text-density/repeat cases, but their stored `error.json` files lack ASR text, and current reruns did not reproduce the same dense ASR output.
- The project is being reshaped from the original FireRedASR2S repository into a standalone multilingual semantic ASR pipeline project.
- Current Arabic batch TextGrid boundary audit found that the three output-time TextGrid errors are caused by already-reversed JSON sentence intervals, not by the TextGrid writer itself. The common root cause is punctuation splitting inside code/URL/decimal tokens (`console.write`, `getlink.io`, `1.3400`) combined with whole-segment timestamp fallback and a boundary-fusion keep path that accepts negative-gap candidates.
- Fixed the TextGrid reversed-boundary chain by protecting periods inside ASCII token-like strings and decimals, removing whole-segment timestamp fallback for empty punctuation slices, merging negative-gap boundary candidates, and validating final sentence/cut intervals before JSON reuse or output writing. The three Arabic failing audios reran successfully under `output/textgrid_boundary_fix/` with zero invalid intervals and TextGrid files written.
- Repository layout now keeps runtime package code under `semantic_asr/` and
  places configuration, documentation, tests and examples at top-level
  `configs/`, `docs/`, `tests/` and `examples/`.
- Removed the obsolete top-level `l2s/` source tree; MMS forced alignment uses
  only the extracted `semantic_asr.mms_runtime`.
- The original top-level FireRedASR2S package, original README, assets, inference examples and runtime directories were removed by the user as part of this cleanup.
- The reusable long-audio ASR abstraction now lives in `semantic_asr`.
- The core pipeline requires four explicit stages: VAD, ASR, timestamp provider and punctuation.
- Timestamp handling supports both ASR-native timestamps and external forced alignment.
- Punctuation is mandatory as a strategy stage, but it may preserve ASR-native punctuation or strip existing punctuation before external re-punctuation.
- FireRed runtime code needed by the standalone project is vendored inside the pipeline package boundary.
- Existing adapters include Silero VAD, Fun-ASR-Nano-2512, Whisper large, Qwen3-ForcedAligner and FireRed VAD/Punc runtime bridges.
- Added language-specific profiles for `zh`, `en` and `ru`.
- ASR and timestamp providers now receive postprocessed ASR VAD segments derived from raw VAD; raw VAD remains preserved for `raw_vad_segments_ms` and final cut segments.
- Final output VAD formatting is separate from ASR slicing: short non-speech gaps can be merged, segments can be padded, and sentence boundaries are aligned to output VAD ranges.
- Output writers support JSON, JSONL, CSV, SRT and TextGrid.
- Added a registry/config composition layer:
  - component factories are registered by role: `vad`, `asr`, `timestamp`, and `punc`;
  - JSON/YAML pipeline profiles are validated and used to build `SemanticAsrPipeline`;
  - `run_pipeline.py` provides one config-driven CLI;
  - each run writes `resolved_config.json` by default.
- Checked-in configs are language/scenario-named profiles under `configs/`.
  The current set is `zh_cn`, `ar_sa`, `de_de`, `en_us`, `hakka`, `hi_in`,
  `ja_jp`, `ko_kr`, `pt_br`, `ru_ru`, `th_th` and `vi_vn`.
- Combination-specific example scripts and builder adapters were removed; use
  `semantic_asr/run_pipeline.py` or `semantic_asr/run_batch.py`.
- Fake registry/config smoke tests validate config parsing, component resolution and runner output writing.
- Architecture diagram exists at `docs/architecture.drawio`.
- Added model/language support metadata in `semantic_asr.language_support`.
- Added `semantic_asr/query_models.py` for language-to-model and model-to-language queries.
- Added ASR adapters, docs and standalone test scripts for:
  - Qwen3-ASR-1.7B;
  - Dolphin;
  - Seamless M4T v2 large.
- Fixed Dolphin ASR timestamp config to use `word_timestamp` for word-level timestamps; `predict_time` is sentence-level timing and is not used for the pipeline word timestamp contract.
- Added XLM-R punctuation/fullstop/truecase model as a `punc` component:
  - adapter: `semantic_asr.adapters.xlm_roberta_punctuation`;
  - registry name: `xlm_roberta_punctuation`;
  - docs and standalone skip-load test were added.
- Added Naqta Arabic punctuation model as a `punc` component:
  - adapter: `semantic_asr.adapters.naqta_punctuation`;
  - registry name: `naqta`;
  - language support: Arabic profiles such as `ar_sa`;
  - docs, standalone skip-load test and `configs/ar_sa_naqta.json` were added.
- Added nizzzo Cantonese punctuation restoration model as a `punc` component:
  - adapter: `semantic_asr.adapters.yue_punctuation`;
  - registry name: `yue_punctuation`;
  - language support: `yue_hk` / Cantonese aliases;
  - docs and standalone skip-load test were added.
- Added Qwen semantic-boundary prompting as a `punc` component:
  - adapter: `semantic_asr.adapters.qwen_semantic_boundary`;
  - registry name: `qwen_semantic_boundary`;
  - the component returns index-only sentence boundary candidates and never
    rewrites ASR text;
  - docs and `configs/th_th_qwen_boundary.json` were added.
- Added MMS forced aligner as a `timestamp` component:
  - adapter: `semantic_asr.adapters.mms_forced_aligner`;
  - registry name: `mms_forced_aligner`;
  - language support is based on vendored `semantic_asr.mms_runtime.model_registry.MMS_CODE_MAP`;
  - Chinese text is split into character tokens before alignment.
- Refactored MMS forced aligner to use vendored `semantic_asr.mms_runtime` and align in-memory `SpeechSegment.wav` audio directly, without writing temporary segment wav files or importing an external `l2s` package.
- Added `docs/test_audio_matrix.md` with the minimum multilingual speech data needed to validate current module combinations:
  - must-have: `zh_cn`, `en_us`, `ru_ru`, `ja_jp`, `th_th`, `hi_in`;
  - optional expansion: `ar_sa`, `vi_in`, `ko_kr`, `bn_bd`, `pt_br`.
- Added the reusable multilingual real-model smoke runner at `examples/run_multilingual_smoke_tests.py`.
- Completed the available-model multilingual smoke run documented in `docs/experiments/multilingual_smoke_20260604.md`.
- Real smoke tests passed for Silero VAD, FireRed VAD, Fun-ASR-Nano, Whisper large, Qwen3-ASR, Qwen3 ForcedAligner, MMS Forced Aligner and XLM-R punctuation.
- Testing fixed adapter/runtime issues for FunASR batch fallback, Qwen3-ASR float waveform input, MMS forced alignment CUDA stability, XLM-R local ONNX loading and Dolphin output normalization.
- Dolphin official GitHub package version `20260513` passed grouped multilingual word-timestamp smoke tests after result normalization was fixed.
- Seamless M4T v2 large passed all-nine-language local-model smoke testing after adding 16 kHz resampling and defaulting `tgt_lang` to `src_lang`.
- FireRedPunc real inference passes again. Installing Qwen3-ASR upgraded Transformers, whose newer safe loader rejected the trusted legacy BERT `.bin` under torch 2.1; the vendored FireRedPunc runtime now loads and validates that local checkpoint directly.
- Corrected the XLM-R punctuation smoke test and report: Thai is not supported and is excluded.
- Added `docs/thai_sentence_boundary.md`: Thai should preserve ASR text and derive semantic sentence boundaries from aligned-token pauses and duration limits instead of inventing punctuation.
- Replaced Thai audio retesting passed interface smoke tests for Silero VAD, FireRed VAD, Whisper large-v3, Qwen3-ASR, Qwen3 ForcedAligner, MMS Forced Aligner, Dolphin, Seamless M4T and FunASR; Dolphin returned 26 Thai word timestamps, while FunASR incorrectly transcribed the Thai sample as Chinese.
- Corrected Qwen3-ASR language support to its specified 30 languages. Catalog queries use short codes, but the adapter converts configured codes such as `th_th` to the full model API value `Thai`.
- Corrected Fun-ASR-Nano and `funasr_native` timestamp support to Chinese, English and Japanese only.
- Added centralized canonical language mapping in `semantic_asr.language_mapping`.
- Pipeline profiles now configure language once at the top level using ids such as `zh_cn`; config construction injects it into language-aware ASR and timestamp components.
- MMS forced aligner now defaults to `zh_cn` and maps it to native `cmn` before calling MMSAlign, fixing the previous bare-`zh` mismatch.
- MMS forced aligner splits Chinese, Korean and Japanese no-space scripts into
  character tokens before alignment.
- VAD adapters can expose optional frame-level speech probability via
  `frame_speech_probs`; current support covers FireRed VAD, Silero VAD and TEN
  VAD.
- Sentence-boundary fusion uses frame-level VAD speech probability as the
  preferred speech-safety signal. High-probability active-speech boundaries
  merge while the span is below `max_sentence_s`; when an over-max continuous
  speech run needs a semantic cap, a rolling selector chooses the recent
  semantic-complete boundary with the lowest local speech probability.
  Semantic-incomplete active boundaries still record
  `max_duration_wait_for_silence`.
- ASR and MMS forced alignment now use postprocessed ASR VAD segments derived from raw VAD.
  Semantic sentence merging happens after token timestamps are available.
- MMS forced aligner forces `use_star=False`; configs cannot override it back
  to true.
- Sentence-boundary fusion now separates audio safety from semantic
  completeness. Raw VAD silence and low frame-level VAD probability windows
  mark a boundary as safe to cut, but comma/colon/semicolon continuations and
  short incomplete fragments still merge.
- Raw VAD silence at or above `max_merge_vad_silence_s` is a hard no-merge
  boundary in sentence fusion. The default threshold is 1.0s.
- Final cut timestamps now use padded output VAD segments, so
  `output_vad_pad_s` affects `cut_segments_ms`, `cut_start_ms`,
  `cut_end_ms` and therefore TextGrid/CSV/SRT export times. Raw VAD remains
  preserved unchanged in `raw_vad_segments_ms`.
- External users can now call the package through the `SemanticASR` SDK facade
  instead of invoking CLI modules directly. The facade exposes reusable
  single-file transcription, batch transcription, and model/language query
  helpers from the package root.
- External users can also use the unified CLI `semantic-asr` after editable
  install. It wraps the SDK with `transcribe`, `batch`, `models` and
  `model-languages` subcommands.
- External remote users can now call an asynchronous FastAPI service. The API
  accepts uploads, records SQLite jobs, and separate worker processes claim jobs
  and run `SemanticASR` on assigned GPUs.
  The service default port is `10086`.
- The HTTP service now includes a built-in browser demo at `/demo`. The page
  lets users enter an API token, choose an allowed language/profile, upload
  multiple local audio files, submit jobs, poll status and download generated
  JSON/SRT/CSV/TextGrid artifacts.

Recent validation:

- The demo jobs panel now matches the upload/control panel height. The table
  scrolls inside the jobs panel and pagination remains fixed at the bottom, so
  the right side no longer grows taller than the left side. Internal translation
  smoke-test jobs are hidden from `GET /v1/jobs` and its `total` count using
  the `translation_smoke_` prefix. The previously visible
  `translation_smoke_hunyuan_mt` record was a hand-created Hunyuan-MT smoke job
  in `service_data/demo` under `user_id=dev`, not a user upload. Validation
  passed: `tests.test_service`, `compileall semantic_asr_service
  tests/test_service.py`, and `git diff --check`.

- Demo user switching now uses the `User Name` field as the effective per-PM
  job namespace for normal tokens. The API token remains the access gate, and
  the browser sends `X-Semantic-ASR-User`; admin tokens ignore that header and
  can still see all jobs. Added `GET /v1/me` and paginated
  `GET /v1/jobs?limit=&offset=`, and the web demo now shows 8 jobs per page
  with Prev/Next controls so the waveform review area is no longer pushed far
  down by a long job table. Switching user/token clears the current review
  panel. Segment-click playback now stops at the clicked segment's end instead
  of continuing into the next segment. The demo startup script now defaults
  translation concurrency to `SEMANTIC_ASR_TRANSLATION_BATCH_SIZE=32` while
  keeping `concurrent_single` mode. Validation passed: `tests.test_service`,
  `compileall semantic_asr_service tests/test_service.py`, `bash -n
  scripts/start_demo_service.sh`, and `git diff --check`.

- Demo translation now defaults to the one-sentence-per-request concurrent mode
  in `scripts/start_demo_service.sh`
  (`SEMANTIC_ASR_TRANSLATION_REQUEST_MODE=concurrent_single`). Added
  `POST /v1/jobs/{job_id}/translations/stream`, which returns NDJSON events so
  the browser can display each sentence translation as soon as it completes and
  then writes the normal sidecar cache on the final `done` event. Uploads now
  include a best-effort `local_path` hint for display: browsers cannot expose a
  true absolute local path, so the demo records `webkitRelativePath` when
  available, otherwise the file name. The job list and review title prioritize
  that source hint over the job id. Current-session waveform review still uses
  the browser `File` object first; after refresh the page falls back to the
  server-saved upload from `GET /v1/jobs/{job_id}/audio`. Validation passed:
  `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`,
  `bash -n scripts/start_demo_service.sh`, `git diff --check`, and an HTTP
  smoke of `/v1/jobs/translation_smoke_hunyuan_mt/translations/stream`.

- The HTTP demo now has a unified startup script:
  `scripts/start_demo_service.sh`. It starts the API and the configured number
  of workers together against the same `service_data/demo` SQLite queue, so the
  demo no longer comes up with a green `/health` but no job consumers. Workers
  now lazy-import `SemanticASR` after claiming a job, update the job to
  `running` before model loading, and print startup/claim/loading/running/
  success/failure logs. The web demo now persists user name, token and selected
  profile, reloads historical jobs from `GET /v1/jobs`, and can fetch the
  original uploaded audio through `GET /v1/jobs/{job_id}/audio` for waveform
  review after page refresh or service restart. Translation button state is
  tracked per `job_id:target`, so switching jobs while a translation is running
  no longer disables translation globally. Added
  `SEMANTIC_ASR_TRANSLATION_REQUEST_MODE=concurrent_single` for the
  one-sentence-per-request concurrent translation style; `json_batch` remains
  supported. Validation passed: `tests.test_service`, `compileall
  semantic_asr_service tests/test_service.py`, `bash -n
  scripts/start_demo_service.sh`, and `git diff --check`.

- Hunyuan-MT translation now supports batch translation after ASR completion.
  `SEMANTIC_ASR_TRANSLATION_BATCH_SIZE` defaults to `16`; each batch sends a
  JSON array of sentence `index/text` records to the OpenAI-compatible Hunyuan
  service, validates that the returned `index` values match the source
  sentences, and falls back to per-sentence translation only for malformed
  batches. Tests now cover structured batch success and malformed batch
  fallback. Validation passed: `tests.test_service`, `compileall
  semantic_asr_service tests/test_service.py`, `git diff --check`, and a real
  Hunyuan-MT smoke on `translation_smoke_hunyuan_mt` translating two Korean
  sentences to Chinese. The demo/API service was restarted on
  `0.0.0.0:10086` with `SEMANTIC_ASR_TRANSLATION_BATCH_SIZE=16`; Hunyuan-MT
  remains on `127.0.0.1:10087`.

- Hunyuan-MT-7B-fp8 real local serving is working. The downloaded model lives at
  `/home/duhu/.cache/huggingface/hub/models--tencent--Hunyuan-MT-7B-fp8/snapshots/81e5a3f7199524570ba75e61360e990ba88665e4`.
  Since `fireredasr2s` has no vLLM and its transformers import is broken, the
  separate `llm` conda environment was updated to `transformers==4.56.0` and
  `compressed-tensors==0.11.0`. Added
  `semantic_asr_service.hunyuan_mt_server`, a minimal transformers-based
  OpenAI-compatible wrapper that patches the FP8 config into
  `service_data/hunyuan_mt_fp8_patched` and serves `/v1/chat/completions` on
  `127.0.0.1:10087` using GPU 6. Direct smoke passed:
  `It is on the house.` -> `这顿饭由我们公司来买单。` Semantic ASR API smoke
  passed on short job `translation_smoke_hunyuan_mt`: Korean sentences
  translated to Chinese and were readable via
  `/v1/jobs/{job_id}/translations/zh_cn`. Translating a full long Korean job
  synchronously was too slow, so the next translation iteration should make
  translation asynchronous or add batching/progress for long recordings.

- Added Hunyuan-MT translation scaffolding for the HTTP service and web demo.
  Translation is a post-ASR sidecar: completed jobs can be translated through
  `POST /v1/jobs/{job_id}/translations`, cached under
  `outputs/translations/{target}.json`, and shown in the Review panel as
  original, translated or bilingual text without changing waveform intervals or
  cut timestamps. The v1 client targets an OpenAI-compatible Hunyuan-MT service
  through `SEMANTIC_ASR_TRANSLATION_BASE_URL`, with default model
  `hunyuan-mt` and target allowlist from `SEMANTIC_ASR_TRANSLATION_TARGETS`.
  Fake translator tests cover cache reuse, target allowlist rejection, missing
  translation-service config and route behavior. Validation passed:
  `tests.test_service`, `compileall semantic_asr_service
  tests/test_service.py`, and `git diff --check`. The demo server was started
  on `0.0.0.0:10086` with translation base URL `http://127.0.0.1:10087`; two
  GPU 7 ASR workers were also started. `/health`, `/v1/translation-targets`
  and `/demo` responded correctly.

- Added Hunyuan-MT translation integration design in
  `docs/hunyuan_mt_translation_design.md`. The recommended v1 approach is to
  deploy `Hunyuan-MT-7B-fp8` or a local quantized `Hunyuan-MT-7B` as a separate
  OpenAI-compatible service, keep ASR workers independent, and produce
  per-job translation sidecar JSON files keyed by sentence index/timestamps.
  `Hunyuan-MT-Chimera` is reserved for later high-quality/offline refinement
  because the interactive review UI needs predictable one-sentence-in,
  one-translation-out behavior. The design includes API routes, cache layout,
  web UI display modes and implementation TODOs.

- Improved long-audio waveform review in the web demo. The waveform now has a
  `1x-48x` zoom slider, mouse-wheel zoom around the cursor, horizontal
  drag-to-pan, a visible-window readout and auto-scroll while playback moves
  outside the current viewport. Canvas waveform peaks are cached per zoom width
  to avoid rescanning all audio samples on every playback cursor update. The
  sentence/time-text list now sits below the waveform so the audio timeline can
  use the full review width. Validation passed: `tests.test_service`,
  `compileall semantic_asr_service tests/test_service.py` and `git diff
  --check`. The demo server was restarted on `0.0.0.0:10086`; `/demo` contains
  the zoom/drag logic and `/health` returned `{"ok": true}`. GPU 7 workers
  remained running.

- Added browser-side waveform review to the web demo. Completed jobs now expose
  a `View` button that fetches the result JSON with bearer auth, decodes the
  still-available local uploaded audio file in the browser, draws a canvas
  waveform, overlays sentence `cut_start_ms/cut_end_ms` intervals and lets the
  user click a segment to seek/play the corresponding audio. The waveform also
  shows a playback cursor and highlights the active segment. Validation passed:
  `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`
  and `git diff --check`. The demo server was restarted on
  `0.0.0.0:10086`; `/demo` contains the review logic and `/health` returned
  `{"ok": true}`. Existing GPU 7 workers remained running.

- Fixed `asr_text` sentence-to-timestamp mapping for CJK character-level MMS
  timestamps. `split_text_by_punctuation()` now first consumes timestamp tokens
  by normalized text/token matching, so Korean/Japanese/Chinese character
  tokens are not counted with whitespace words. If normalized matching fails,
  the previous whitespace token-count heuristic remains as fallback. Regression
  tests cover Korean `감회가 새롭습니다.`, Japanese and Chinese character-level
  timestamps. The affected Korean demo segment now maps
  `감회가 새롭습니다.` to `23660-24961ms`, covering `감` through `다`.
  Validation passed: `tests.test_punctuation_strategies`, full `unittest
  discover tests` (154 tests), `compileall semantic_asr
  tests/test_punctuation_strategies.py`, and `git diff --check`. The demo
  server and both GPU 7 workers were restarted; `/health` returned
  `{"ok": true}` on `127.0.0.1:10086`.

- Korean demo job `2e472ebc81214ede9818f2a2e7c09a29` exposed a timestamp/text
  drift in `asr_text` punctuation mapping. The reported sentence
  `감회가 새롭습니다.` has text that should span `감` through `다`
  (`23660-24961ms`), but `split_text_by_punctuation()` uses
  `sentence_text.split()` to estimate token count. Because MMS Korean
  timestamps are character-level, the sentence consumed only two timestamp
  tokens (`감`, `회`) and was assigned `23660-24040ms`. Boundary fusion then
  kept the VAD silence boundary at `23435ms`, and `add_sentence_cut_segments()`
  intersected it with padded output VAD `23460-43270ms`, producing
  `cut_start_ms=23460` and `cut_end_ms=24040`. The main fix should replace
  whitespace token-count mapping in `asr_text` with normalized timestamp-token
  matching for CJK/no-space scripts.

- Fixed authenticated artifact downloads in the web demo. The page no longer
  renders raw artifact `<a href>` links, because browser link clicks do not
  include `Authorization: Bearer ...` headers. Artifact controls now fetch the
  file with the current API token through `apiFetch()`, convert the response to
  a blob and trigger the browser download. Validation passed:
  `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`
  and `git diff --check`. The running demo server on `0.0.0.0:10086` was
  restarted so the MacBook page receives the new JavaScript; the two GPU 7
  workers remained running.

- The web demo is currently reachable from the server at
  `http://127.0.0.1:10086/demo` and is listening on `0.0.0.0:10086`. The
  server's LAN IP is `10.10.23.8`, so a machine on the same reachable network
  should open `http://10.10.23.8:10086/demo`. The running setup uses
  `SEMANTIC_ASR_SERVICE_DATA_DIR=service_data/demo`, token `dev-token`, and two
  workers started with `semantic_asr_service.worker --device 7`.

- Added the web demo route and `GET /v1/configs` for allowlist-backed profile
  selection. Documented a GPU 7 demo deployment with the API server on port
  `10086` and two `semantic-asr-worker --device 7` processes. Validation
  passed: `tests.test_service`, `compileall semantic_asr_service
  tests/test_service.py`, `git diff --check`, and a real HTTP smoke where
  `/demo` returned the page and authenticated `/v1/configs` returned
  `{"configs":["vi_vn","zh_cn"]}` on `127.0.0.1:10086`.

- Real asynchronous HTTP service smoke passed for Vietnamese `vi_vn`.
  The service ran on port `10086` with `SEMANTIC_ASR_ALLOWED_CONFIGS=vi_vn`,
  a dedicated data dir `service_data/vi_vn_smoke`, and one worker bound to GPU
  4. Submitted
  `/data/duhu/FireRedASR2S/data/vi_vn/98e27c7d-561c-4a4f-a607-e54cd6d06c9d.wav`
  through `POST /v1/jobs`; job
  `c818c05a1f434c959a952addb68120d6` completed with status `succeeded`.
  Outputs were generated and downloadable through the API: JSON, SRT, CSV and
  TextGrid. The JSON result reports `300.299s` duration, 57 sentences and 775
  words. The smoke service and worker processes were stopped after validation.

- Changed the HTTP service default port to `10086`. `semantic-asr-server` now
  reads `SEMANTIC_ASR_SERVICE_HOST` and `SEMANTIC_ASR_SERVICE_PORT`, defaulting
  to `0.0.0.0:10086`. README examples were updated from port 8000 to 10086.
  Real smoke test passed outside the sandbox:
  `GET http://127.0.0.1:10086/health` returned `{"ok": true}`. Validation
  passed: focused service tests, full unit discover (149 tests), `compileall
  semantic_asr_service tests/test_service.py`, and `git diff --check`.

- Added `semantic_asr_service`:
  - FastAPI routes for `POST /v1/jobs`, job status, JSON result download,
    artifact download, model queries and health checks;
  - API-key bearer auth with owner/admin job access, config allowlist, upload
    extension/size checks and optional audio-duration limit;
  - SQLite-backed queue and worker loop with `semantic-asr-server` and
    `semantic-asr-worker` console scripts.
  Validation passed: focused service/CLI/API tests, worker help/import smoke,
  full unit discover (148 tests), `compileall semantic_asr semantic_asr_service
  tests examples`, and `git diff --check`.

- Added unified external CLI:
  - `semantic-asr transcribe` wraps `SemanticASR.transcribe()` and exposes
    `--devices` before model loading plus output formats/cache controls;
  - `semantic-asr batch` wraps `SemanticASR.transcribe_batch()` and exposes
    `--num-workers`, `--devices` and `--max-seconds`;
  - `semantic-asr models` and `semantic-asr model-languages` expose the model
    support query helpers.
  Added `pyproject.toml` console script metadata. Validation passed: focused
  CLI/API tests, real `python -m semantic_asr.cli models yue_hk --role punc`,
  full unit discover (139 tests), `compileall semantic_asr tests examples`,
  and `git diff --check`.

- Added external Python SDK facade `SemanticASR`:
  - `SemanticASR.from_config()` loads a profile and initializes models once;
  - `transcribe()` reuses the loaded pipeline, writes selected artifacts, and
    returns `{"result": ..., "outputs": ...}`;
  - `transcribe_batch()` delegates to the existing multi-process batch runner
    and supports temporary `CUDA_VISIBLE_DEVICES` selection through `devices`;
  - package root now exports `list_models()`, `list_model_languages()` and
    `suggest_components()`.
  Validation passed: focused API/config/batch tests, full unit discover
  (134 tests), `compileall semantic_asr tests examples`, and `git diff --check`.

- Fixed `output_vad_pad_s` propagation to final exports by feeding padded
  `output_vad_segments_ms` into `add_sentence_cut_segments()`. Regression
  coverage confirms raw `1200-1350ms` becomes non-overlapping padded
  `1100-1475ms` and final `cut_end_ms` uses the padded boundary. Validation
  passed: focused ASR-VAD/punctuation/output tests, full unit discover
  (125 tests), `compileall semantic_asr tests`, and `git diff --check`.

- Added `yue_punctuation` for `nizzzo/zh-yue-punctuation-restore-v3`.
  The adapter uses Hugging Face token classification, maps labels directly to
  timestamp token indices for no-space Cantonese text, and exposes configurable
  label-to-punctuation mapping. Validation passed: focused punctuation and
  language-support tests, standalone skip-load example, full unit discover
  (128 tests), `compileall semantic_asr tests examples`, and `git diff --check`.
  At initial integration time, a real model load was not run because the
  checkpoint was not yet available locally and Hugging Face raw config access
  failed with SSL/network errors.

- Real `yue_punctuation` smoke test passed after the checkpoint was downloaded
  locally. The local snapshot
  `~/.cache/huggingface/hub/models--nizzzo--zh-yue-punctuation-restore-v3/snapshots/015afa4682164b92e38d69271d6700f59d443098`
  loaded on GPU with `CUDA_VISIBLE_DEVICES=4` and produced timestamp-mapped
  output: `我今日返工，你去邊係唔係一齊食飯？`. The real model labels use
  sequence prefixes such as `S-。` and `S-，`; `yue_punctuation` now strips
  `B/I/E/S/U` prefixes before mapping labels to punctuation. Validation passed:
  focused punctuation/language-support tests, full unit discover (128 tests),
  `compileall semantic_asr tests examples`, and `git diff --check`.
- Real `yue_punctuation` Cantonese-English mixed-input smoke tests passed on
  GPU with the same local snapshot. Example outputs:
  `我今日開meeting，你send email畀我，好唔好？`,
  `呢個project deadline係Friday。` / `你confirm咗未？`, and
  `如果client approve咗，我哋就start development下個sprint，再review`.
  English tokens remained in the text and timestamp-mapped `punc_sentences`
  were returned. The third sample did not receive final punctuation, so
  code-switch input is supported technically but still needs domain quality
  checks before production use.
- Real `yue_punctuation` simplified Cantonese and simplified Cantonese-English
  smoke tests also passed with timestamp-mapped output. Example outputs:
  `我今日返工，你去边系唔系一齐食饭？`,
  `我今日开meeting，你send email畀我好唔好？`, and
  `如果client approve咗，我哋就start development下个sprint，再review`.
  Simplified input is supported technically; longer mixed samples show the same
  quality caveat as traditional code-switch input, where final punctuation is
  not guaranteed.

- Fixed the remaining German `batch_2_output` MMS span errors by collapsing
  whitespace in uroman tokens before `get_alignments()` and `get_spans()`, then
  dropping empty normalized uroman tokens with their matching alignment items.
  Validation passed: `tests.test_mms_forced_aligner`, full unit discover
  (124 tests), `compileall semantic_asr tests`, `git diff --check`, and GPU
  batch verification on the four affected German wavs under
  `output/de_de_batch2_remaining4_verify`.

- Implemented the four-part short-segment anti-hallucination and MMS
  feasibility strategy:
  - Whisper short-audio decode parameters are configurable and tested.
  - MMS checks target/frame feasibility before forced alignment.
  - Unalignable tiny hallucinations and empty targets are recorded as
    `discarded_asr_segments` instead of flowing into approximate timestamps by
    default.
  - Exact `asr_vad_min_segment_s` duration now counts as tiny.
  Validation passed: focused adapter/ASR-VAD/MMS tests, full unit discover
  before the final doc-only edits, `compileall semantic_asr tests`, config parse
  for 13 JSON profiles and `git diff --check`. A real GPU rerun of
  `838db147...` was requested but rejected by the automatic approval reviewer.

- Whisper length-penalty demo on the known hallucinated Arabic `838db147` 500ms
  segment (`296770-297270ms`) showed that decode-time length penalty is only a
  partial mitigation. Current-like decoding reproduced
  `إذا حصلت على محاولة تحقيق المنطقة، فإنها تتحقق بمعرفة المنطقة.`;
  `beam_size=5` without length penalty produced an even longer repetitive
  hallucination; `beam_size=5` with `length_penalty=0.0`, `0.2` or `1.0`
  shortened the output to `اشتركوا في القناة` but still emitted hallucinated
  text. The shortened result had `no_speech_prob≈0.296` and
  `avg_logprob≈-0.56`, so Whisper's default no-speech/logprob filters would not
  reject it.

- Detailed MMS failure analysis for the three remaining Arabic `batch_5_output_fireredvad` errors:
  - `838db147-31d3-412a-ae62-edd30ef8a554` is an end-of-audio 500ms VAD island at
    `296770-297270ms`. Whisper produced
    `إذا حصلت على محاولة تحقيق المنطقة، فإنها تتحقق بمعرفة المنطقة.`;
    MMS uroman/dictionary mapping produced 52 target symbols with 3 repeats, but
    only 24 emission frames were available. CTC needs at least 55 frames, so the
    upstream issue is a dense ASR hallucination on a tiny isolated VAD island.
  - `8606f16f-8713-43d8-8491-6236446b61ac` failed historically because
    `forced_align()` received an empty target tensor after uroman/dictionary
    filtering. The stored `error.json` does not include the ASR segment text, and
    a current diagnostic rerun no longer reproduces an empty-target segment.
  - `afdb46d5-b788-4d16-bc36-edac8632ae6a` failed historically with 30 MMS
    emission frames versus 34 target symbols plus 1 repeat, so CTC needed at
    least 35 frames. The stored error lacks ASR text and the GPU diagnostic rerun
    request was rejected, so the exact text cannot be recovered from available
    artifacts.

- MMS forced alignment now has preflight checks and fallback for the remaining
  Arabic batch failures:
  - `empty_target` is detected before torchaudio sees an empty `targets`
    tensor;
  - `ctc_target_too_long` is detected when
    `frames < target_chars + repeats`;
  - those feasibility errors fall back to monotonic approximate token
    timestamps for that ASR segment, and JSON records
    `timestamp_segments[].timestamp_fallback`;
  - unexpected MMS/runtime errors still propagate normally.
  Validation passed for focused MMS/ASR-VAD tests, full unit discover
  (115 tests), compileall and `git diff --check`. Real GPU rerun of the three
  remaining Arabic files was not executed because the sandbox escalation
  request was rejected by the approval reviewer.

- Arabic `batch_5_output_fireredvad` error audit found 5 errors:
  - 3 MMS CTC feasibility failures: `838db147...` is a micro/dense case
    likely helped by ASR VAD tiny-island merging; `afdb46d5...` is a 600ms
    near-threshold dense case; `66166626...` is a 2.54s ASR-text-density case.
  - 1 MMS empty-target failure: `8606f16f...` reaches
    `forced_align()` with zero `token_indices` after uroman/dictionary
    filtering, so MMS needs an explicit preflight guard before torchaudio.
  - 1 stale bad JSON resume failure: `fd41e9e3...` contains a 0-length
    `worldbank.` sentence from old negative-gap boundary logic. Replaying its
    `semantic_sentences` through the current fusion code merges
    `live. worldbank. org.` and validates successfully.

- ASR VAD microsegment fix validation:
  - Added `prepare_asr_vad_segments()` with defaults
    `asr_vad_min_segment_s=0.5` and `asr_vad_max_merge_silence_s=1.0`.
  - Focused unit tests cover merging a tiny island to the nearest neighbor,
    dropping an isolated tiny island, merging a tiny chain without overlap, and
    preserving raw VAD while using postprocessed ASR VAD in the pipeline.
  - Real Arabic rerun passed for
    `/data_151/duhu/DBC/ASR/22424_微软ITN5语种混合模型测试/ar_sa/batch_4/5e2e123b-f598-4e7d-973d-3dd8877ccadf.wav`
    under `output/mms_ctc_fix/5e2e/`: raw VAD still contains
    `204220-204280ms`, while `asr_vad_segments_ms` has no segment shorter than
    500ms and MMS CTC alignment completed.

- Arabic batch_4 error audit for
  `/data_151/duhu/DBC/ASR/22424_微软ITN5语种混合模型测试/ar_sa/batch_4_output`:
  47 `*.error.json` files were present. 44 failed in MMS forced alignment and
  3 failed while writing TextGrid. MMS failures were mostly CTC alignment
  feasibility errors: 30 `targets length is too long for CTC`, including 27
  cases with only 14-16 emission frames; 6 empty-target `torch.max()` failures;
  7 `get_spans` label/token assertion mismatches; and 1 short-emission shape
  error. The TextGrid failures already had JSON outputs but each contained one
  reversed sentence interval, leaving overlapping neighboring intervals after
  TextGrid sorting.

- Qwen German punctuation/boundary experiment on
  `output/de_de/848c4ebd-419a-4c42-a85a-2bb1ab52b121.json`: full-file
  punctuation over 799 stripped tokens timed out, and 90-second window tests
  also timed out. A small `289s-328s` sentence-end-only test returned valid
  JSON for 89 tokens, but made a worse semantic split than the existing German
  candidates around "die Status Quo Situation ... hin zum gewünschten
  Ergebnis". A 25-token comma/full-punctuation test returned inconsistent
  sentence-end markup and missed natural German commas. Current recommendation:
  do not replace existing German ASR punctuation with Qwen; keep Qwen as an
  optional semantic-boundary fallback only.

- Qwen semantic-boundary requests now include the OpenAI-compatible
  `response_format={"type":"json_object"}` field by default, with
  `response_format_json=false` available for incompatible backends. The local
  Qwen endpoint accepted the field in a real Thai adapter smoke test.
- Validation passed: `tests.test_qwen_semantic_boundary`, full unit discover
  (101 tests), `compileall semantic_asr tests`, config parse for all 13 JSON
  profiles, `git diff --check`, and a real local-Qwen adapter smoke with
  `response_format` enabled.

- Implemented the Qwen index-only semantic-boundary component with strict JSON
  validation, bounded retry and duration-based fallback candidate boundaries.
  The component sends `chat_template_kwargs={"enable_thinking": false}`, asks
  for `{"token_count": N, "end_indices": [...]}`, and reconstructs text only
  from original timestamp tokens.
- Validation passed: `tests.test_qwen_semantic_boundary`, full unit discover
  (100 tests), `compileall semantic_asr tests`, config parse for all 13 JSON
  profiles, `query_models.py language th_th --role punc`, `git diff --check`,
  and a real local-Qwen adapter smoke on the existing Thai timestamp sample.

- Qwen3.6-27B local API smoke-tested for Thai semantic sentence-boundary use:
  `/v1/models` is reachable at `http://10.10.23.16:18000/v1` when run outside
  the sandbox. Plain prompts and `/no_think` still emitted visible thinking
  text, so programmatic use must send
  `chat_template_kwargs={"enable_thinking": false}`. With that option, a
  no-punctuation Thai chunk sample returned 3 monotonic JSON spans, and the
  existing real Thai ASR token sample returned one valid span covering tokens
  `0-102` and mapping to `460-15660ms`.

- Implemented rolling over-max boundary selection. Instead of cutting at the
  first terminal-punctuation boundary after `max_sentence_s`, fusion keeps
  recent semantic-complete candidates inside the current group and selects the
  lowest local speech-probability boundary.
- Offline recomputation for
  `output/de_de/848c4ebd-419a-4c42-a85a-2bb1ab52b121.json` moves the reported
  tail split from `318.745s` to `305.081s`; the selected boundary has
  `speech_prob_mean=0.4548` versus `0.7208` at the previous trigger point.
- Boundary decision JSON now records `rolling_selected_candidate_index`,
  `rolling_selected_boundary_ms` and `rolling_selected_speech_prob_mean`.
- Validation passed: `tests.test_sentence_boundaries` (21 tests), full unit
  discover (94 tests), `compileall semantic_asr tests`, config parse for all 12
  JSON profiles, `git diff --check` and offline German JSON recomputation.

- Added `SentenceBoundaryFusionConfig.max_merge_vad_silence_s`, defaulting to
  1.0s. Boundary fusion now keeps `long_vad_silence` boundaries before
  semantic-incomplete merge logic, so candidates are not merged across raw VAD
  pauses of at least 1s.
- Boundary decision JSON now includes `long_vad_silence`.
- Validation passed: `tests.test_sentence_boundaries` (20 tests), full unit
  discover (93 tests), `compileall semantic_asr tests`, config parse for all 12
  JSON profiles and `git diff --check`.

- Fixed the Arabic one-sentence collapse reported for
  `configs/ar_sa.json` on
  `05162293-323c-4c4c-8e2e-5c3d0b568b3d.wav`.
  Punctuation was not the problem: the existing JSON had 21
  `semantic_sentences`, but boundary fusion merged every boundary because all
  candidates were high speech probability and recorded
  `max_duration_wait_for_silence`.
- Over-max active-speech fusion now keeps semantic-complete boundaries as a
  bounded fallback (`max_duration_terminal_punctuation` or
  `max_duration_semantic_boundary`) while semantic-incomplete active boundaries
  still wait for silence.
- Offline recomputation of the reported JSON changes the final fusion from 1
  sentence to 13 sentences, with max duration 29.1s.
- Validation passed: `tests.test_sentence_boundaries` (18 tests), full unit
  discover (91 tests), `compileall semantic_asr tests` and `git diff --check`.

- Simplified the public pipeline/config surface after pushing checkpoint
  `04283fd` to `origin/red-asr`.
- Removed legacy combination-specific configs, runner scripts and builder
  adapters. The checked-in config set is now language/scenario named only:
  `zh_cn`, `ar_sa`, `de_de`, `en_us`, `hakka`, `hi_in`, `ja_jp`, `ko_kr`,
  `pt_br`, `ru_ru`, `th_th` and `vi_vn`.
- Removed pre-ASR VAD merging from `SemanticAsrPipeline`; raw VAD speech
  islands now feed ASR and timestamp providers directly.
- Removed the final short-gap sentence merge pass. Boundary fusion is the
  single sentence grouping stage.
- Config files now rely on defaults for sentence-boundary fusion and output
  flags. FireRed VAD/Punc config supports direct params such as `use_gpu` and
  `extend_speech_frame`.
- `semantic_asr.registry` and `semantic_asr.adapters` now import concrete model
  adapters lazily, so parsing configs and querying the registry no longer loads
  heavy model runtimes.
- Cleanup validation passed: 90 unit tests, compileall, config parse for all 12
  profiles, `query_models.py language ar_sa --role punc`, draw.io XML parse,
  obsolete-field scan and `git diff --check`.

- Removed RMS/waveform valley from the active sentence-boundary policy.
  Boundary fusion now uses raw VAD silence and frame-level speech probability
  as audio-safety evidence, with probability silence requiring both low local
  minimum and low local mean.
- High speech-probability active-speech boundaries now merge below
  `max_sentence_s`. Once the combined span is over max, semantic-complete
  boundaries cap the run with `max_duration_terminal_punctuation` or
  `max_duration_semantic_boundary`, while incomplete boundaries still record
  `max_duration_wait_for_silence`.
- Sentence-boundary fusion is now the single grouping stage for fusion-enabled
  profiles. The legacy final short-gap merge runs only when
  `sentence_boundary_fusion.enabled=false`.
- Updated current strategy docs and diagram:
  `docs/sentence_boundary_fusion.md`,
  `docs/sentence_split_merge_strategy_audit.md` and
  `docs/sentence_split_merge_strategy.drawio`.
- Validation passed: `tests.test_sentence_boundaries` (17 tests), full unit
  discover (98 tests), `compileall`, draw.io XML parse, checked-in config scan
  for obsolete `acoustic_*` keys and `git diff --check`.

- Fixed the `pt_br` raw-align semantic fragmentation around `47.895s-58.600s`.
  The reported three candidates:
  `... tecnologia e negócios,` / `as empresas podem criar` /
  `uma conexão harmoniosa ... setores.` now merge into one final sentence.
- The semantic-completeness retest passed under
  `output/experiments/tenvad_whisper_mms_nativepunc_pt_br_semantic`: 37 final
  sentences, 574 words, 18,750 VAD probability frames and zero sentence/word
  overlaps. Boundary decisions now record `audio_safe`, `audio_reason`,
  `semantic_complete` and `semantic_reason`.

- Switched the default pipeline strategy from pre-ASR VAD merging to raw-VAD
  ASR/timestamp processing by setting `PipelineConfig.merge_vad_segments=False`.
- Kept the legacy ASR context merge available through explicit
  `pipeline.merge_vad_segments=true`.
- Real `pt_br` raw-align smoke passed with TEN VAD + Whisper + MMS +
  ASR-native punctuation under
  `output/experiments/tenvad_whisper_mms_nativepunc_pt_br_rawalign`: 37 raw VAD
  segments, 37 ASR/MMS segments, 36 timestamp segments, 574 words, 39 final
  sentences and zero sentence/word overlaps.
- The raw-align run confirmed `raw_vad_segments_ms == asr_vad_segments_ms`.
- Added regression coverage for default raw-VAD ASR/timestamp processing,
  explicit legacy VAD merge and forced `use_star=False`.

- Added frame-level VAD speech probability to the pipeline JSON as
  `vad_frame_speech_probs` and passed it into sentence-boundary fusion.
- FireRed VAD writes 10ms-shift / 25ms-frame probabilities, TEN VAD writes
  16ms probabilities and Silero VAD writes 32ms window probabilities.
- Boundary decisions now include `speech_prob_min`, `speech_prob_mean`,
  `speech_prob_max`, `speech_prob_boundary_ms` and
  `speech_prob_supported_silence`.
- Boundary-fusion reasons now include `vad_prob_silence`,
  `merged_active_speech_prob` and `max_duration_wait_for_silence`.
- Real `pt_br` smoke passed with TEN VAD + Whisper + MMS + ASR-native
  punctuation under
  `output/experiments/tenvad_whisper_mms_nativepunc_pt_br_probs`: 18,750 VAD
  probability frames, 562 words, 24 final sentences, 7 merges and zero
  sentence/word overlaps.
- Real `ar_sa` 60-second smoke passed with FireRed VAD + Seamless + MMS +
  XLM-R punctuation under `output/experiments/ar_sa_short_probs60`: 5,998 VAD
  probability frames, probability-backed boundary decisions and zero
  sentence/word overlaps.
- Silero VAD 30-second smoke on `en_us-short.wav` returned 938 probability
  frames with 32ms metadata.
- Validation after frame-level VAD probability integration passed: 45 tests,
  package/tests/examples compile and all configs parse.

- Naqta Arabic punctuation integration passed lightweight validation without
  loading the still-downloading checkpoint: targeted punctuation/language/config
  tests, `examples/test_naqta_punctuation.py --skip_model_load 1`,
  `query_models.py language ar_sa --role punc`, and the full unit suite
  passed with 93 tests.
- Fixed the Naqta Arabic semantic-cut regression from `output/ar_sa_error-6`.
  Naqta had produced 21 semantic candidates, but boundary fusion did not treat
  Arabic question mark `؟` as terminal punctuation, word-gap lookup crossed
  candidate sentence ranges, and the final short-gap merge re-merged terminal
  punctuation boundaries. Boundary fusion now recognizes Arabic terminal
  punctuation, uses sentence-local word lookup and preserves terminal
  punctuation token boundaries when word timestamps do not overlap. Final
  short-gap merge skips previous sentences that already end with terminal
  punctuation. The real rerun under `output/ar_sa_error-6_fix2` emits 19 final
  sentences, keeps the `وما أدراك ما الحق؟` split, keeps the final
  `المسكين.` split, and has a maximum sentence duration of 29.37s.
- Refined the terminal-punctuation boundary rule after the German TEN-VAD case
  under `output/de_de-tenvad-3`. Terminal punctuation is no longer an absolute
  keep signal when the boundary has no audio-safe evidence. If adjacent
  terminal-punctuation candidates are short, inside the same raw VAD speech
  island, and separated by 0-20ms/high speech probability, they merge until the
  span approaches `target_sentence_s` or the hard max-duration policy applies.
  Merged text now preserves the original punctuation. Offline recomputation of
  the reported `308s-328s` region reduces the five short segments to three
  spans: `300.099-312.563`, `312.563-322.546`, and `322.546-327.688`.
- Added a strategy audit document and draw.io diagram for the current
  sentence split/merge logic:
  - `docs/sentence_split_merge_strategy_audit.md`
  - `docs/sentence_split_merge_strategy.drawio`
  The audit explains current timing layers, boundary-fusion decision order,
  output cut timing, known logical holes and the next simplification: keep one
  grouping stage but add a rolling recent-boundary selector.

- Added a TEN VAD adapter:
  - adapter: `semantic_asr.adapters.ten_vad.TenVadAdapter`;
  - registry name: `ten_vad`;
  - language support: language-agnostic VAD entry;
  - docs: `docs/models/ten_vad.md`;
  - standalone test: `examples/test_ten_vad.py`.
- Documented TEN VAD installation:
  `git clone git@github.com:TEN-framework/ten-vad.git && cd ten-vad && python setup.py install`.
- Added `configs/tenvad_whisper_mms_nativepunc_pt_br.json` for
  TEN VAD + Whisper large + MMS Forced Aligner + ASR-native punctuation.
- The full `pt_br` run passed on `data/test/short/pt_br-short.wav` and wrote
  JSON, JSONL, CSV, SRT, TextGrid and resolved config outputs under
  `output/experiments/tenvad_whisper_mms_nativepunc_pt_br`.
- The TEN VAD Portuguese run produced 37 raw VAD segments, 12 ASR VAD
  segments, 12 timestamp segments, 562 words and 30 final sentences with zero
  sentence/word timestamp overlaps.
- Validation after TEN VAD integration passed: 40 tests, package/tests/examples
  compile, `pt_br` VAD query includes `ten_vad`, config parsing passed and the
  standalone Ten-VAD 30-second sample produced timestamps.

- Enabled `sentence_boundary_fusion` for every checked-in `configs/*.json`
  profile, including language-named configs and legacy combination configs.
- Ran all available short fixtures from `data/test/short` through their
  language-named profiles and wrote outputs under
  `output/experiments/multilingual_short_boundary_fusion/<language>/`.
- The multilingual short run produced JSON, JSONL, CSV, SRT, TextGrid and
  resolved-config outputs for `ar_sa`, `en_us`, `hi_in`, `ja_jp`, `ko_kr`,
  `pt_br`, `ru_ru`, `th_th` and `vi_vn`.
- Every multilingual short output includes `timestamp_segments`,
  `semantic_sentences` and `sentence_boundary_decisions` when applicable.
  Final sentence and word overlap counts are zero for all nine fixtures.
- `th_th-short.wav` is currently 16.17 seconds, while the other short fixtures
  are approximately 300 seconds.
- Validation after enabling all-language boundary fusion passed: all configs
  confirm fusion enabled, 37 tests pass and package/tests/examples compile.

- Completed a 5-minute multilingual config run for all available `data/test`
  language fixtures: `ar_sa`, `en_us`, `hi_in`, `ja_jp`, `ko_kr`, `pt_br`,
  `ru_ru`, `th_th` and `vi_vn`.
- Added language-named profiles under `configs/` for those nine fixtures and
  wrote JSON outputs under `output/experiments/multilingual_5min/<language>/`.
- Every multilingual 5-minute output includes `timestamp_segments`; sentence
  and word overlap counts are zero for all nine languages.
- `ja_jp` initially failed with Seamless + MMS because MMS CTC alignment
  reported targets too long; the coverage profile now uses Whisper native
  timestamps.
- Multilingual 5-minute validation passed: all nine JSON/CSV/SRT/TextGrid
  outputs exist, all six legacy configs plus the nine language-named configs
  parse, 37 tests pass, package/tests/examples compile and `git diff --check`
  passes.
- Config profiles moved to top-level `configs/`. All six profiles parse, and
  example wrappers load their configs from the new location.
- Final JSON now includes segment-grouped `timestamp_segments`, preserving the
  timestamp provider output with segment text/confidence plus relative seconds
  and absolute millisecond token timestamps.
- Validation after config relocation and timestamp JSON recording passed: 37
  tests, package/tests/examples compile, `git diff --check`, wrapper `--help`,
  and a fake runner JSON check for `timestamp_segments`.
- Top-level layout validation passed after moving docs/tests/examples and
  deleting `l2s`: 37 tests, package/tests/examples compile, `git diff --check`,
  all example `--help` commands, compatibility wrapper config lookup, and five
  standalone `--skip_model_load` model examples.
- The validated layout restructure remains uncommitted because the platform
  rejected the external Git write request after reaching its current usage
  limit.
- Corrected sentence-boundary fusion priority after Arabic listening review:
  active-speech safety now overrides the soft target duration whenever the
  merged sentence remains below the hard maximum duration.
- The complete Arabic V2 retest passed under
  `output/experiments/ar_seamless_mms_xlm_ar_sa_short_boundary_fusion_v2`.
  It emits 41 audio-safe sentences from 54 semantic candidates, merges 13
  active-speech boundaries, has no remaining boundary that satisfies the merge
  criteria, and keeps the maximum sentence duration at 27.87 seconds.
- Reported active-speech cuts at `202.797-203.037` and `350.485-350.525` are
  now merged into `186.430-211.500` and `337.200-364.789`.
- Full validation after the active-speech priority correction passed: 37
  tests, package compile and `git diff --check`.
- Implemented configurable sentence-boundary fusion in
  `semantic_asr.sentence_boundaries`. Enabled profiles retain
  `semantic_sentences`, emit audio-safe `sentences`, and record
  `sentence_boundary_decisions`.
- The complete Arabic fusion retest passed and wrote all outputs under
  `output/experiments/ar_seamless_mms_xlm_ar_sa_short_boundary_fusion`.
  It reduced 54 semantic candidates to 47 audio-safe sentences by merging
  seven active-speech boundaries; six VAD-silence boundaries were kept and
  snapped. Final sentence/word overlaps are zero and the longest sentence is
  27.87 seconds.
- The reported Arabic target region is now split at `73.770-87.418` and
  `87.418-97.920`: the active-speech cut was merged and the VAD-supported
  boundary was retained.
- Full validation after sentence-boundary fusion passed: 36 tests, package
  compile and `git diff --check`.
- Investigated punctuation-proposed Arabic sentence boundaries against raw
  VAD, aligned-token gaps and local waveform energy. The reported
  `83.494s -> 83.634s` boundary cuts active speech and should be merged, while
  `87.396s -> 87.496s` is supported by a raw VAD silence interval.
- Added `docs/sentence_boundary_fusion.md` with a proposed
  punctuation + raw VAD + token pause + waveform energy + duration-limit
  policy for audio-safe sentence boundaries. Default behavior remains
  unchanged until multilingual evaluation.
- The full ten-minute Arabic profile passed on `data/test/ar_sa-short.wav`:
  FireRed VAD + Seamless M4T v2 large + MMS Forced Aligner + XLM-R
  punctuation.
- Arabic output artifacts were written under
  `output/experiments/ar_seamless_mms_xlm_ar_sa_short_final`: JSON, JSONL,
  CSV, SRT, TextGrid and resolved config.
- The Arabic run produced 54 sentences and 699 aligned words with no remaining
  sentence or word overlaps. It also exposed 14 `<UNK>` tokens and awkward
  punctuation that require later transcript-quality evaluation.
- Fixed output-VAD alignment creating overlapping adjacent sentence intervals;
  sentence overlaps are now removed at the shared pipeline-result level before
  every output writer.
- Full validation after the Arabic end-to-end fix passed: 30 tests, package
  compile and `git diff --check`.
- Unified-language mapping validation passed: all profiles parsed with canonical ids, MMS `zh_cn -> cmn`, Qwen `th_th -> Thai`, and Seamless `ar_sa -> arb`.
- Full validation after unified language configuration passed: 29 tests, package compile and `git diff --check`.
- Unified language configuration committed as `51e0480`.
- Focused Qwen/FunASR language support and adapter tests passed: 13 tests.
- `semantic_asr/query_models.py language th_th` no longer returns Fun-ASR-Nano.
- Qwen language conversion returned `Chinese`, `Thai` and `Cantonese` for `zh_cn`, `th_th` and `yue`.
- A focused real Qwen Thai GPU retest was requested but could not run because the platform rejected the external GPU execution request due to its current usage limit.
- The validated Qwen/FunASR language-support correction remains uncommitted because external `.git` write approval was rejected by the platform's current usage limit.
- `conda run -n fireredasr2s python -m compileall semantic_asr` passed after the rename.
- `conda run -n fireredasr2s python -m unittest tests` passed after the rename.
- `--help` passed for `semantic_asr/run_pipeline.py` and all three compatibility wrapper scripts after the rename.
- All config profiles under `configs` parsed successfully after the rename.
- Top-level `README.md` and `requirements.txt` were refreshed for the standalone project.
- `conda run -n fireredasr2s python -m unittest discover -s tests -p 'test_*.py'` passed after adding language support metadata.
- `semantic_asr/query_models.py language en_us` and `semantic_asr/query_models.py model qwen3_asr_1_7b --role asr` returned expected JSON.
- New standalone scripts for Qwen3-ASR, Dolphin and Seamless M4T passed `--skip_model_load 1`.
- `conda run -n fireredasr2s python -m compileall semantic_asr` passed after the Dolphin/XLM-R punctuation changes.
- `conda run -n fireredasr2s python -m unittest discover -s tests -p 'test_*.py'` passed after the Dolphin/XLM-R punctuation changes.
- `examples/test_xlm_roberta_punctuation.py --skip_model_load 1` passed.
- `semantic_asr/query_models.py language en_us --role punc` returns `xlm_roberta_punctuation`.
- Corrected `xlm_roberta_punctuation` support to the full 47-language list provided by the user, including Chinese.
- `semantic_asr/query_models.py language zh_cn --role punc` now returns `xlm_roberta_punctuation`.
- `conda run -n fireredasr2s python -m unittest discover -s tests -p 'test_*.py' tests` passed after adding MMS forced aligner.
- `examples/test_mms_forced_aligner.py --skip_model_load 1 --language zh_cn --text "你好 世界"` passed and produced Chinese character tokens.
- `semantic_asr/query_models.py language zh_cn --role timestamp` returns `mms_forced_aligner`.
- `conda run -n fireredasr2s python -m compileall semantic_asr` passed after MMS in-memory runtime extraction.
- `16` unit tests passed after multilingual smoke-test fixes.
- Real Qwen3-ASR -> Qwen3 ForcedAligner English alignment passed with timestamps.
- Real Qwen3-ASR -> MMS ForcedAligner Hindi alignment passed with timestamps.
- XLM-R punctuation real ONNX inference passed for its supported English, Russian, Japanese, Hindi and Arabic samples; Thai was removed because it is unsupported.
- Dolphin GitHub-version grouped tests passed for Arabic, Hindi, Japanese, Korean, Russian and Vietnamese with word timestamps; the selected Thai prefix was empty.
- Seamless M4T v2 large local-model tests passed all nine languages while preserving source-language output.

Next step:

- Review the Arabic sentence-boundary fusion output by listening, then tune
  thresholds or enable the strategy for additional language profiles.
- Implement and evaluate the Thai timestamp-and-pause sentence-boundary strategy.
- Build speech-bearing quality fixtures with reference transcripts instead of testing fixed file prefixes.
- Evaluate the Arabic end-to-end output against a reference transcript, with
  emphasis on `<UNK>` tokens and punctuation quality.
- Add nested CLI override support if ad-hoc experiment overrides become common.
- Run real smoke tests for Qwen3-ASR, Dolphin and Seamless once local model paths/checkpoints are confirmed.
