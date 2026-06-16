# Todo

- [x] Current work: skip Hunyuan translation calls for Chinese-source to Chinese-target jobs.
- [x] Current work: write identity translation cache for skipped Chinese jobs so the UI can still load bilingual data.
- [x] Current work: validate sync and streaming translation paths.

Review:
- `translate_job_result()` and `stream_translate_job_result()` now skip Hunyuan calls when the source profile is Chinese (`zh_cn`, `zh`, `hakka`, etc.) and the target is Chinese.
- Skipped translations write an identity cache with `model="identity"` and `skipped=true`, preserving the UI cache/display path without spending GPU.
- Added tests for both synchronous and streaming Chinese-to-Chinese translation skip behavior.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service/translation.py tests/test_service.py`, and `git diff --check`.

- [x] Current work: freeze the selected Language/Profile at the start of a multi-file upload.
- [x] Current work: disable profile/format controls while the upload batch is being submitted.
- [x] Current work: add demo smoke assertions and validate service tests.

Review:
- The web demo now captures `uploadConfig = configSelect.value` and `uploadFormats = selectedFormats()` once at the start of `submitJobs()`.
- Every file in the same multi-file upload batch uses that frozen profile and format list, even if the user changes the dropdown before all requests finish.
- The Language/Profile select and output-format checkboxes are disabled while the batch is being submitted.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service/demo.py tests/test_service.py`, and `git diff --check`.

- [x] Current work: register FireRedASR as an ASR component named `firered_asr`.
- [x] Current work: register FireRedASR native timestamps as a timestamp component named `firered_asr_native`.
- [x] Current work: add language-support metadata and unit tests for FireRedASR ASR/timestamp discovery.
- [x] Current work: validate registry/config parsing without loading model checkpoints.

Review:
- Added `FireRedAsrAdapter` and `FireRedAsrAdapterConfig` in `semantic_asr/adapters/firered.py`.
- Registered `firered_asr` under ASR and `firered_asr_native` under timestamp in the default component registry.
- `firered_asr` defaults to `asr_type=aed`, `model_dir=pretrained_models/FireRedASR2-AED`, and `return_timestamp=true`.
- Added language metadata marking FireRedASR2 and its native timestamps as Chinese (`zh`) components.
- Added `docs/models/firered_asr.md` with config fragments and dependency notes.
- Validation passed: targeted language/config tests, query-model CLI checks, `compileall`, `git diff --check`, and full unittest discovery with 181 tests.

- [x] Current work: add a reusable translation backfill command for succeeded jobs missing cached translations.
- [x] Current work: document the HY-MT1.5 backfill flow and keep the service docs aligned with the current startup scripts.
- [x] Current work: validate the backfill code with unit tests and syntax checks.
- [x] Current work: restart/check the HY-MT1.5 translation service on port 10087.
- [x] Current work: raise HY-MT1.5 translation concurrency after observing low GPU memory use.
- [x] Current work: dry-run and execute zh_cn translation backfill for the demo service data.

Review:
- Added `semantic_asr_service.backfill_translations`, which scans succeeded jobs with existing ASR JSON and missing target translation cache, then writes the cache without rerunning ASR.
- HY-MT1.5 now uses the local HF repo id `tencent/HY-MT1.5-1.8B-FP8`; the startup script handles lowercase HF cache paths and uses `conda run --no-capture-output` so logs are visible.
- Translation clients accept comma-separated base URLs and round-robin requests across replicas.
- Defaults are now `SEMANTIC_ASR_TRANSLATION_BATCH_SIZE=64`, `SEMANTIC_ASR_TRANSLATION_MAX_CONCURRENCY=24`, and `SEMANTIC_ASR_TRANSLATION_TIMEOUT_S=300`.
- `scripts/start_hunyuan_mt_service.sh` supports `HUNYUAN_MT_PORTS=10087,10088,10089` for multiple same-GPU HY-MT1.5 replicas.
- Real backfill completed for `service_data/demo`: 21 succeeded jobs have `zh_cn` translation cache, 0 missing.
- Live services are background `setsid` processes: demo/API on `10086`, ASR workers on GPUs `6,7`, and Hunyuan-MT replicas on `10087,10088,10089`.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`, `bash -n` for both startup scripts, and `git diff --check`.

- [x] Current work: switch translation defaults from Hunyuan-MT-7B-fp8 to HY-MT1.5-1.8B-FP8.
- [x] Current work: increase translation client/server concurrency defaults for the smaller model.
- [x] Current work: make the Hunyuan startup script discover the downloaded HY-MT1.5 snapshot path.
- [x] Current work: make job success wait for configured auto translation to finish.
- [ ] Current work: update docs/tests and validate without requiring the model download to be complete.

Review:
- `scripts/start_hunyuan_mt_service.sh` now defaults to `Tencent-Hunyuan/HY-MT1.5-1.8B-FP8` and discovers the latest HuggingFace snapshot under the local cache if `HUNYUAN_MT_MODEL_PATH` is not explicitly set.
- Hunyuan-MT server/client translation concurrency defaults are raised to `4`; demo translation batch size defaults to `16`.
- Worker success semantics now include configured auto translation: when `SEMANTIC_ASR_AUTO_TRANSLATE_TARGETS` is set, a job is marked `succeeded` only after those translations finish and the cache is written. Translation failures make the job `failed`, so Retry can rerun it.
- This reintroduces waiting inside the ASR worker during translation by design, matching the UI requirement that `succeeded` means ASR plus translation are ready.

- [ ] Current work: reduce demo ASR worker default to GPUs 6/7 with two workers per GPU.
- [ ] Current work: restart 10086 demo/API workers without touching Hunyuan-MT on GPU 5.
- [ ] Current work: verify service health and queue state after restart.

- [x] Current work: split job language/profile into its own Jobs table column.
- [x] Current work: truncate long file labels in the Jobs table while preserving the full value as hover text.
- [x] Current work: remove the Stage column from the Jobs table.
- [x] Current work: add multi-select deletion for job records and outputs/uploads.
- [ ] Current work: restart the demo service after host-command approval is available.
- [x] Current work: add retry for failed/canceled jobs.
- [x] Current work: make failed error display compact in the Jobs table.
- [x] Current work: make auto translation asynchronous so ASR workers do not wait for translation.
- [x] Current work: change demo ASR worker defaults to GPUs 6 and 7 with two workers per GPU, and document translation on GPU 5.

Review:
- Failed/canceled jobs can now be retried through `POST /v1/jobs/{job_id}/retry`, which requeues the job and clears previous error/start/finish metadata. Running jobs cannot be retried.
- The Jobs table no longer dumps traceback text into the cell. Failed jobs show compact `Error` and `Retry` buttons; `Error` opens the full text on demand.
- Automatic translation is now fire-and-forget after `mark_succeeded()`: ASR workers do not wait for Hunyuan-MT translation before claiming more ASR jobs.
- Demo ASR worker defaults are now `SEMANTIC_ASR_DEMO_WORKER_DEVICES=6,7` and `SEMANTIC_ASR_DEMO_WORKERS_PER_DEVICE=2`.
- Added `scripts/start_hunyuan_mt_service.sh`, defaulting Hunyuan-MT to GPU 5 and port 10087.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`, `bash -n scripts/start_demo_service.sh`, `bash -n scripts/start_hunyuan_mt_service.sh`, and `git diff --check`.
- Demo service restart still needs to be done when host-command approval is available; the previous attempt to inspect/restart host processes was blocked by the approval usage limit.

Review:
- Jobs table now has separate selection, File, Language, Job ID, Status and Artifacts columns.
- File labels are truncated to 42 characters in the table and keep the full value in the cell title/hover text.
- Removed the Stage column.
- Added page-level multi-select plus a Delete button. Deleting calls `DELETE /v1/jobs/{job_id}` for each selected job.
- Backend delete is authorized: normal users can delete only their own jobs, admins can delete visible jobs. Running jobs return 409 and are not deleted.
- Delete removes the DB record plus upload/output directories only when those paths are under the configured service upload/jobs roots.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`, `bash -n scripts/start_demo_service.sh`, and `git diff --check`.
- Demo restart was not performed because host process/port commands were blocked by the current approval usage limit; restart 10086 later to load the new page/API.

- [x] Current work: remove visible Text/Target translation controls from the demo review panel.
- [x] Current work: remove the visible Zoom slider from the demo review panel while keeping mouse-wheel zoom.
- [x] Current work: default review display to bilingual Chinese translation and auto-load cached translation.
- [x] Current work: run translation automatically after ASR job success and store the sidecar cache.
- [x] Current work: validate, restart affected services when safe, and commit.

Review:
- Removed visible `Text`, `Target`, `Translate` and `Zoom` slider controls from the Review panel.
- Review now defaults internally to bilingual display with `zh_cn` translation and tries to load cached `translations/zh_cn.json` when a completed job is opened.
- Mouse-wheel zoom and drag-to-pan remain available even without the slider.
- Added `SEMANTIC_ASR_AUTO_TRANSLATE_TARGETS`; the demo startup defaults it to `zh_cn`.
- Workers now try automatic translation after ASR output is written and before marking the job succeeded. Translation failures are logged but do not fail the ASR job.
- Restarted the 10086 demo API/workers after confirming there were no queued/running demo jobs; 10087 Hunyuan-MT stayed running.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`, `bash -n scripts/start_demo_service.sh`, `git diff --check`, `/health`, and `/demo` control-removal smoke.

- [x] Current work: diagnose why Hunyuan-MT translation streams only produce one sentence every few minutes.
- [x] Current work: check whether old/in-flight translation requests or server-side serialization are blocking new requests.
- [x] Current work: propose or implement the minimal fix after identifying the bottleneck.

Review:
- Root cause: the demo used `concurrent_single` with `SEMANTIC_ASR_TRANSLATION_BATCH_SIZE=32`, which meant up to 32 one-sentence requests were sent to one local transformers Hunyuan-MT model server at once.
- The Hunyuan-MT server previously had no generation semaphore, so concurrent `model.generate()` calls could occupy/queue the single model for a long time. A direct tiny request to `10087` timed out after 30s while the old server was wedged.
- Added `SEMANTIC_ASR_TRANSLATION_MAX_CONCURRENCY` and changed the client to use it as the true concurrent request window, separate from batch size.
- Added a Hunyuan-MT server-side generation semaphore (`--max-concurrent`, default/env `HUNYUAN_MT_MAX_CONCURRENT=1`) so the model is not hammered by parallel generate calls.
- Demo startup now defaults to `SEMANTIC_ASR_TRANSLATION_BATCH_SIZE=8` and `SEMANTIC_ASR_TRANSLATION_MAX_CONCURRENCY=1`.
- Restarted only the Hunyuan-MT service on `127.0.0.1:10087` with `--max-concurrent 1`; the same tiny request then returned in about 3s.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`, `bash -n scripts/start_demo_service.sh`, `git diff --check`, and direct Hunyuan-MT latency smoke.

- [x] Current work: add language/profile and PM username filters to the demo job list.
- [x] Current work: keep hidden internal smoke jobs out of filtered job results.
- [x] Current work: validate service tests, restart/check the demo, and commit.

Review:
- The demo job list now has `Language / Profile` and `PM / Username` filters above the table.
- `GET /v1/jobs` now accepts optional `config` and `user_id` filters; normal users remain scoped to their effective User Name, while admin tokens can filter across PMs.
- Internal jobs with prefix `translation_smoke_` remain hidden even when filters are applied.
- `translation_smoke_hunyuan_mt` was a hand-created Hunyuan-MT smoke-test job in `service_data/demo` (`user_id=dev`, `config=ko_kr`, dummy upload), so it appeared before we hid internal smoke jobs.
- Fixed `scripts/start_demo_service.sh` so translation health checks use a short timeout and cannot block API/worker startup.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`, `bash -n scripts/start_demo_service.sh`, `git diff --check`, and HTTP smoke on `/demo` plus filtered `/v1/jobs`.

- [x] Current work: align the jobs panel height with the upload/control panel height.
- [x] Current work: hide internal translation smoke jobs from the demo job list.
- [x] Current work: explain why `translation_smoke_hunyuan_mt` appeared and validate the fix.

Review:
- The top jobs panel now uses the same fixed responsive height as the upload/control panel, with the job table scrolling inside that panel and pagination fixed at the bottom.
- `translation_smoke_hunyuan_mt` appeared because it was a hand-created Hunyuan-MT smoke-test job stored in `service_data/demo` with `user_id=dev`.
- Internal jobs with prefix `translation_smoke_` are now hidden from `GET /v1/jobs` and its total count, including admin lists, so smoke artifacts no longer pollute the demo file list.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`, and `git diff --check`.

- [x] Current work: make demo User Name the effective per-PM job namespace while keeping API token as access control.
- [x] Current work: add `/v1/me` and paginated `/v1/jobs` results.
- [x] Current work: update the web demo to switch users by name and paginate jobs.
- [x] Current work: increase default translation concurrency for the demo startup script.
- [x] Current work: stop playback at the clicked segment end instead of continuing into the next segment.
- [x] Current work: update tests/docs, validate, restart demo and commit.

Review:
- Normal demo requests now send `X-Semantic-ASR-User` from the User Name field; the API token remains the access gate, while User Name is the PM/job namespace. Admin tokens ignore the header and can see all jobs.
- Added `GET /v1/me` and paginated `GET /v1/jobs?limit=&offset=`.
- The demo job table now shows 8 jobs per page with Prev/Next controls, and clears the review panel when switching user/token.
- Segment-click playback now stops at the clicked segment end instead of continuing into the next segment.
- `scripts/start_demo_service.sh` now defaults translation concurrency to `SEMANTIC_ASR_TRANSLATION_BATCH_SIZE=32`.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`, `bash -n scripts/start_demo_service.sh`, and `git diff --check`.

- [x] Current work: change demo startup default translation mode to `concurrent_single`.
- [x] Current work: add streaming translation so each completed sentence can appear in the page immediately.
- [x] Current work: store an upload-time local path hint and show it in the jobs/review UI.
- [x] Current work: keep browser `File` loading as the first choice, then fall back to server-uploaded audio after refresh.
- [x] Current work: update tests/docs, validate, restart demo and commit.

Review:
- `scripts/start_demo_service.sh` now defaults to `SEMANTIC_ASR_TRANSLATION_REQUEST_MODE=concurrent_single`.
- Added `POST /v1/jobs/{job_id}/translations/stream`, returning NDJSON sentence events as translations complete, followed by a final `done` event and cache write.
- The demo reads the translation stream and updates the segment list sentence-by-sentence instead of waiting for the whole job.
- Uploads now send a best-effort `local_path` hint. Browsers cannot expose absolute local paths, so this is `webkitRelativePath` when available, otherwise the file name.
- Job list/review titles prioritize `local_path`, then file name, then job id. Current-session review still uses the browser `File` object first; after refresh it fetches the saved upload from `/audio`.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`, `bash -n scripts/start_demo_service.sh`, `git diff --check`, and an HTTP smoke on `/translations/stream`.

- [x] Current work: add a single demo startup script that starts API plus worker processes together.
- [x] Current work: make service workers claim jobs before model loading and print startup/claim/completion logs.
- [x] Current work: fix web demo translation state so switching jobs does not leave the translate button disabled or silent.
- [x] Current work: add a user-history flow so the demo can reload previous jobs after page refresh or service restart.
- [x] Current work: document systemd/supervisor tradeoffs, validate with tests and commit.

Review:
- Added `scripts/start_demo_service.sh` to start the demo API plus two default GPU 7 workers under the same `service_data/demo` queue.
- Worker startup now logs its DB/device, claims jobs before loading `SemanticASR`, and logs claim/loading/running/success/failure events.
- Added `GET /v1/jobs` and `GET /v1/jobs/{job_id}/audio`; the demo reloads the authenticated user's historical jobs and can review prior uploaded audio without re-uploading.
- The demo stores user name, token and selected profile in browser local storage.
- Translation UI state is tracked per `job_id:target`, so switching jobs no longer leaves the translate button globally disabled or silent.
- Added `SEMANTIC_ASR_TRANSLATION_REQUEST_MODE=concurrent_single` for one-sentence-per-request concurrent translation; `json_batch` remains available.
- Documented the startup script, translation modes, and what `systemd`/`supervisor` are.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`, `bash -n scripts/start_demo_service.sh`, and `git diff --check`.

- [x] Current work: add batched Hunyuan-MT translation for completed ASR sentence lists.
- [x] Current work: make the batch prompt return structured JSON and validate sentence indexes before accepting it.
- [x] Current work: keep per-sentence fallback when batch output is malformed.
- [x] Current work: add unit tests for batch success and fallback behavior, validate and commit.

Review:
- Added `SEMANTIC_ASR_TRANSLATION_BATCH_SIZE`, defaulting to `16`.
- `translate_job_result()` now translates completed ASR sentences in batches, validates returned sentence indexes, and only falls back to per-sentence translation for the malformed batch.
- Added tests for structured batch success and malformed batch fallback.
- Updated Hunyuan-MT service docs to describe batch prompting and fallback.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`, `git diff --check`, and a real Hunyuan-MT smoke on `translation_smoke_hunyuan_mt`.
- Restarted the `10086` demo/API service with `SEMANTIC_ASR_TRANSLATION_BATCH_SIZE=16`; Hunyuan-MT remains on `127.0.0.1:10087`.

- [x] Current work: start a real Hunyuan-MT-7B-fp8 OpenAI-compatible translation service on port `10087`.
- [x] Current work: smoke test direct Hunyuan-MT chat completion with the downloaded model.
- [x] Current work: smoke test the Semantic ASR translation API against an existing completed job.
- [x] Current work: update docs/progress with the verified model path/start command and commit.

Review:
- `fireredasr2s` has no vLLM and its transformers import is broken, so Hunyuan-MT was deployed from the separate `llm` conda environment.
- Installed `transformers==4.56.0` and `compressed-tensors==0.11.0` into `llm`, matching the Hunyuan-MT FP8 model card. This conflicts with LLaMAFactory constraints in that env, so it should be treated as the translation-serving env for now.
- Added `semantic_asr_service.hunyuan_mt_server`, a minimal OpenAI-compatible `/v1/chat/completions` wrapper around transformers loading.
- Started Hunyuan-MT-7B-fp8 on GPU 6 at `127.0.0.1:10087`, using runtime-patched config under `service_data/hunyuan_mt_fp8_patched`.
- Direct smoke passed: `It is on the house.` -> `这顿饭由我们公司来买单。`
- Semantic ASR translation API smoke passed on short job `translation_smoke_hunyuan_mt`: Korean sentences translated to Chinese and were readable through `GET /v1/jobs/{job_id}/translations/zh_cn`.
- Full long Korean job synchronous translation was too slow, so the next iteration should make translation asynchronous or add batching/progress before translating entire long recordings.

- [x] Current work: implement Hunyuan-MT translation settings and OpenAI-compatible client scaffolding.
- [x] Current work: add translation cache/API endpoints for completed ASR jobs.
- [x] Current work: add web demo controls for target language and original/translation/bilingual display.
- [x] Current work: add fake translation tests and validate without requiring downloaded model files.
- [x] Current work: document Hunyuan-MT-7B-fp8 startup and commit the integration scaffolding.

Review:
- Added `semantic_asr_service.translation` with an OpenAI-compatible Hunyuan-MT client, prompt builder, language-name mapping and per-job translation cache under `outputs/translations/{target}.json`.
- Added translation settings: `SEMANTIC_ASR_TRANSLATION_BASE_URL`, `SEMANTIC_ASR_TRANSLATION_MODEL`, `SEMANTIC_ASR_TRANSLATION_API_KEY` and `SEMANTIC_ASR_TRANSLATION_TARGETS`.
- Added service endpoints: `GET /v1/translation-targets`, `POST /v1/jobs/{job_id}/translations` and `GET /v1/jobs/{job_id}/translations/{target_language}`.
- Added web demo review controls for target language, `Translate`, and display mode `Original / Translation / Bilingual`.
- Added fake translator tests for cache creation/reuse, target allowlist rejection, missing service config and route behavior.
- Documented Hunyuan-MT-7B-fp8 vLLM startup in `semantic_asr_service/README.md`.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`, and `git diff --check`.
- Restarted demo server on `0.0.0.0:10086` with translation base URL `http://127.0.0.1:10087`, and restarted two GPU 7 ASR workers. `/health`, `/v1/translation-targets` and `/demo` all responded correctly.

- [x] Current work: analyze Hunyuan-MT model choices, deployment options and constraints for translation.
- [x] Current work: inspect the current service/web demo structure and identify the clean translation integration points.
- [x] Current work: design how users can toggle ASR original text vs translated text in the web review UI.
- [x] Current work: document recommended deployment and implementation TODOs without requiring downloaded model files.

Review:
- Added `docs/hunyuan_mt_translation_design.md`.
- Recommended `Hunyuan-MT-7B-fp8` or another local quantized `Hunyuan-MT-7B` variant for the first interactive deployment; keep `Hunyuan-MT-Chimera` for later high-quality/offline mode.
- Recommended serving Hunyuan-MT as a separate OpenAI-compatible service, preferably on a GPU separate from ASR workers, instead of loading it inside ASR workers.
- Designed translation as a post-ASR sidecar artifact keyed by sentence index and timestamps, not as part of the core VAD/ASR/timestamp/punctuation pipeline.
- Proposed async translation APIs, translation cache files under each job output directory, and web review display modes: original, translation and bilingual.

- [x] Current work: add waveform zoom controls for long-audio review in the web demo.
- [x] Current work: make waveform canvas horizontally scrollable and redraw segments/playback cursor at the selected zoom scale.
- [x] Current work: keep click-to-seek accurate after zoom/scroll changes.
- [x] Current work: support mouse-wheel zoom around the cursor and drag-to-pan on the waveform.
- [x] Current work: move the segment/time-text list below the waveform to maximize waveform width.
- [x] Current work: add focused demo tests, validate, restart the demo server and commit.

Review:
- Added waveform zoom for long-audio review with a `1x-48x` slider and mouse-wheel zoom around the cursor position.
- The waveform canvas now expands horizontally with zoom inside a scrollable timeline. Min/max waveform peaks are cached per canvas width so playback cursor redraws do not rescan all audio samples.
- Added drag-to-pan on the waveform and kept click-to-seek accurate by suppressing seek after a drag gesture.
- Segment/time-text list now sits below the waveform instead of in a right column, giving the waveform the full review width.
- Playback cursor and active segment update at the zoomed scale; playback auto-scrolls when the cursor leaves the visible window.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`, and `git diff --check`.
- Restarted the demo server on `0.0.0.0:10086`; `/demo` contains zoom/drag logic and `/health` returned `{"ok": true}`. GPU 7 workers were left running.

- [x] Current work: add a browser-side waveform review panel to the web demo.
- [x] Current work: overlay completed ASR sentence/cut intervals on the waveform and show corresponding text.
- [x] Current work: support audio playback, seek-by-segment, and current playback cursor in the review panel.
- [x] Current work: add focused demo tests and restart the demo server after validation.

Review:
- Added a web demo review panel with an audio player, waveform canvas and clickable sentence list.
- Completed jobs now show a `View` button. It fetches the job JSON with bearer auth, decodes the local uploaded audio file in the browser and overlays `cut_start_ms/cut_end_ms` intervals on the waveform.
- Clicking a segment seeks playback to that sentence/cut interval. The waveform also shows a playback cursor and the active segment is highlighted while audio plays.
- The browser can only review audio files still available from the current upload session; after a page refresh, users need to reselect and resubmit the local audio file.
- Added focused service-demo assertions for waveform/review JS and documented the feature in `semantic_asr_service/README.md`.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`, and `git diff --check`.
- Restarted the demo server on `0.0.0.0:10086`; `/demo` contains the waveform review logic and `/health` returned `{"ok": true}`. GPU 7 workers were left running.

- [x] Current work: fix `asr_text` punctuation mapping for CJK character-level MMS timestamps.
- [x] Current work: add regression tests for Korean/Japanese/Chinese character-level timestamp consumption.
- [x] Current work: rerun the affected Korean job or targeted reproduction to confirm `감회가 새롭습니다.` spans through `다`.
- [x] Current work: restart the demo server on port `10086` after validation.
- [x] Current work: update progress/TODO and commit the fix.

Review:
- `split_text_by_punctuation()` now first aligns each punctuated sentence to timestamp tokens by normalized token-string matching, which works for Korean/Japanese/Chinese character-level MMS timestamps.
- If normalized matching fails, it falls back to the previous whitespace token-count heuristic to preserve behavior on mismatched or unusual text.
- Added regression tests for Korean, Japanese and Chinese character-level timestamp consumption.
- Targeted reproduction on job `2e472ebc81214ede9818f2a2e7c09a29` now maps `감회가 새롭습니다.` from `23660ms` through `24961ms`, covering `감` through `다`.
- Validation passed: punctuation strategy tests, full `unittest discover tests` with 154 tests, `compileall semantic_asr tests/test_punctuation_strategies.py`, and `git diff --check`.
- Restarted the demo server on `0.0.0.0:10086` and restarted two `semantic_asr_service.worker --device 7` workers. `/health` returned `{"ok": true}` after restart.

- [x] Current work: analyze why job `2e472ebc81214ede9818f2a2e7c09a29` has `cut_start_ms` later than sentence `start_ms` for Korean segment `감회가 새롭습니다.`
- [x] Current work: inspect the job JSON, raw/output VAD segments, timestamp words and config around `23435-24040ms`.
- [x] Current work: trace the code path that assigns `cut_segments_ms` and explain the root cause before changing code.

Review:
- Job `2e472ebc81214ede9818f2a2e7c09a29` used profile `ko_kr` with `whisper_large + mms_forced_aligner + asr_text` punctuation.
- Around the reported segment, raw VAD is `20700-23210ms` and `23660-43070ms`; padded output VAD is `0-23410ms` and `23460-43270ms`.
- The sentence-boundary fusion kept the VAD silence boundary between the previous sentence and the Korean sentence at `23435ms`, derived from previous token end `23190ms` and current token start `23660ms`.
- `cut_segments_ms` is then intersected with output VAD in `add_sentence_cut_segments()`, so the reported sentence gets `cut_start_ms=23460`, the start of the padded second output VAD island.
- The larger root cause is earlier in `semantic_asr.punctuation.split_text_by_punctuation()`: `asr_text` computes sentence token counts with `sentence_text.split()`, but Korean MMS timestamps are character-level. For `감회가 새롭습니다.`, this consumes only two timestamp tokens (`감`, `회`) and assigns the sentence `23660-24040ms` even though the text continues through `다` at `24961ms`.
- Because the text/time mapping is already drifted, the exported `cut_end_ms=24040` cuts inside the spoken phrase. The fix should make `asr_text` align sentence text to timestamp tokens by normalized character/token matching for CJK/no-space scripts instead of whitespace token count.

- [x] Current work: fix web demo artifact downloads so they include the bearer token.
- [x] Current work: add regression coverage that the demo uses authenticated JS downloads instead of raw links.
- [x] Current work: validate service tests and record the fix.

Review:
- Fixed the web demo artifact download path: artifact controls are now buttons that call `apiFetch()` with the current bearer token, fetch the artifact as a blob, and trigger a browser download.
- Removed raw artifact `<a href>` links from the generated job table so downloads no longer hit authenticated endpoints without headers.
- Added service test assertions that the demo uses authenticated JS downloads and does not include the old raw target link pattern.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`, and `git diff --check`.
- Restarted only the demo server on `0.0.0.0:10086` so the MacBook page receives the new JavaScript. The two GPU 7 workers stayed running.

- [x] Current work: diagnose why the web demo is not reachable from a MacBook.
- [x] Current work: check whether the service is running and listening on `0.0.0.0:10086`.
- [x] Current work: identify the server IP/URL the MacBook should use and document access options.
- [x] Current work: update progress/TODO if any repo documentation changes are needed.

Review:
- The demo service had not been running on port `10086`; only an unrelated `uvicorn app.main:app` process was listening on port `8000`.
- Started `semantic_asr_service.app` with `SEMANTIC_ASR_SERVICE_HOST=0.0.0.0` and `SEMANTIC_ASR_SERVICE_PORT=10086`.
- Started two `semantic_asr_service.worker --device 7` processes using the same `service_data/demo` queue.
- Server IP is `10.10.23.8`; MacBook should open `http://10.10.23.8:10086/demo`.
- Local HTTP validation passed: `/demo` returned the page, `/v1/configs` returned the allowed config list with `Authorization: Bearer dev-token`, and `ss` showed `0.0.0.0:10086` listening.

- [x] Current work: add a built-in web demo page served by the existing FastAPI service.
- [x] Current work: support API-token entry, language/profile selection, multiple local audio uploads, job polling and artifact downloads in the demo.
- [x] Current work: document how to start the server on port 10086 with two workers bound to GPU 7.
- [x] Current work: add focused tests for the demo route/static UI and run service validation.
- [x] Current work: update progress and commit the web demo change set.

Review:
- Added a built-in `/demo` page served by the existing FastAPI app. The page uses the existing API service directly; there is no second backend.
- Added `GET /v1/configs` so the demo can populate the language/profile selector from the service allowlist.
- The demo supports API-token entry, language/profile selection, multiple local audio file uploads, selectable JSON/SRT/CSV/TextGrid outputs, status polling and artifact download links.
- Documented demo startup in `semantic_asr_service/README.md`, including port `10086` and two workers bound to GPU 7 with `semantic-asr-worker --device 7` in two shells.
- Validation passed: `tests.test_service`, `compileall semantic_asr_service tests/test_service.py`, `git diff --check`, and real HTTP smoke for `/demo` plus authenticated `/v1/configs` on `127.0.0.1:10086`.

- [x] Current work: start the HTTP service on port 10086 with `vi_vn` allowed and a dedicated smoke-test data dir.
- [x] Current work: start one worker on GPU 4 and submit `/data/duhu/FireRedASR2S/data/vi_vn/98e27c7d-561c-4a4f-a607-e54cd6d06c9d.wav`.
- [x] Current work: poll the job until completion/failure and inspect generated JSON/SRT/CSV/TextGrid artifacts.
- [x] Current work: record the service smoke result in progress and commit the test record if files change.

Review:
- Real HTTP service smoke passed on port `10086` with `SEMANTIC_ASR_ALLOWED_CONFIGS=vi_vn`, API key auth and `SEMANTIC_ASR_SERVICE_DATA_DIR=service_data/vi_vn_smoke`.
- Started one worker on GPU 4 and submitted `/data/duhu/FireRedASR2S/data/vi_vn/98e27c7d-561c-4a4f-a607-e54cd6d06c9d.wav` through `POST /v1/jobs`.
- Job `c818c05a1f434c959a952addb68120d6` reached `succeeded`; status API returned JSON/SRT/CSV/TextGrid artifact URLs.
- Generated artifacts were present under `service_data/vi_vn_smoke/jobs/c818c05a1f434c959a952addb68120d6/outputs/`: JSON `1120173` bytes, SRT `6035` bytes, CSV `7062` bytes, TextGrid `9613` bytes.
- JSON inspection found duration `300.299s`, `57` sentences and `775` words. Artifact download endpoints for `json`, `srt`, `csv` and `textgrid` all returned valid files.
- The service and worker processes were stopped after the smoke test.

- [x] Current work: change HTTP service default port from 8000 to 10086 and make it configurable.
- [x] Current work: update service/README examples from 8000 to 10086.
- [x] Current work: smoke test the service on port 10086 with `/health`.
- [x] Current work: run validation, update progress, and commit the port change.

Review:
- `semantic-asr-server` now defaults to `0.0.0.0:10086` through `ServiceSettings.port`.
- Added `SEMANTIC_ASR_SERVICE_HOST` and `SEMANTIC_ASR_SERVICE_PORT`; default port is `10086`, and tests cover override to a custom port.
- README and service README examples now use `server:10086`.
- Real smoke test passed outside the sandbox: temporary server started on `0.0.0.0:10086`, and `GET http://127.0.0.1:10086/health` returned `{"ok": true}`.
- Validation passed: focused service tests, full unit discover with 149 tests, `compileall semantic_asr_service tests/test_service.py`, and `git diff --check`.

- [x] Current work: implement FastAPI async job service with API-key auth and SQLite-backed job queue.
- [x] Current work: implement service worker that claims queued jobs and runs `SemanticASR` into per-job output dirs.
- [x] Current work: expose `semantic-asr-server` and `semantic-asr-worker` commands plus service docs and requirements.
- [x] Current work: add unit/API tests for auth, submit/status/download, allowlist, worker success/failure, and artifact 404.
- [x] Current work: run validation, update progress, and commit only service-related files.

Review:
- Added `semantic_asr_service` with FastAPI app, SQLite job store, artifact helpers, settings, schemas, and worker loop.
- API supports async job submission, status lookup, JSON result download, artifact download, model query, and health check.
- Service uses API-key bearer auth, config allowlist, upload extension/size limits, optional audio-duration limit, and owner/admin job access checks.
- Worker claims queued jobs, marks running/succeeded/failed, and runs `SemanticASR.from_config(...).transcribe(...)` into per-job output dirs.
- Added console scripts `semantic-asr-server` and `semantic-asr-worker`; documented API/worker deployment and curl examples.
- Added service tests for auth, allowlist, queued jobs, owner/admin access, artifact response shape, upload limits, worker success/failure, and model endpoint core behavior.
- Validation passed: focused service/CLI/API tests, worker help/import smoke, full unit discover with 148 tests, `compileall semantic_asr semantic_asr_service tests examples`, and `git diff --check`.

- [x] Current work: add unified external CLI on top of the `SemanticASR` SDK.
- [x] Current work: expose `--num-workers` and `--devices` in batch CLI, and expose `--devices` in single-file CLI before model loading.
- [x] Current work: add console-script packaging metadata plus CLI tests/docs.
- [x] Current work: run validation, update progress, and commit CLI-related files.

Review:
- Added unified CLI module `semantic_asr.cli` with subcommands `transcribe`, `batch`, `models`, and `model-languages`.
- Added `pyproject.toml` console script so editable installs expose `semantic-asr`; without install, users can run `python -m semantic_asr.cli`.
- `semantic-asr batch` exposes `--num-workers` and `--devices`; `--devices` is forwarded to SDK batch as temporary `CUDA_VISIBLE_DEVICES`.
- `semantic-asr transcribe` exposes `--devices` before model loading, plus `--formats`, `--max-seconds`, `--uttid`, `--outdir`, and `--no-cache`.
- Added CLI tests for transcribe, batch worker/device forwarding, model queries, and format parsing.
- Validation passed: focused CLI/API tests, real `python -m semantic_asr.cli models yue_hk --role punc`, full unit discover with 139 tests, `compileall semantic_asr tests examples`, and `git diff --check`.

- [x] Current work: add external Python SDK facade `SemanticASR` for single-file transcription with reusable loaded models.
- [x] Current work: expose batch transcription, model query helpers, and package-level imports for external users.
- [x] Current work: add focused SDK tests and README/API docs.
- [x] Current work: run validation, update progress, and commit only SDK-related files.

Review:
- Added package-level external SDK facade `SemanticASR` with `from_config()`, `from_profile()`, `transcribe()` and `transcribe_batch()`.
- Single-file `transcribe()` reuses the loaded pipeline on the same SDK instance, supports JSON/SRT/CSV/TextGrid output writing, validates cached JSON, and returns `{"result": ..., "outputs": ...}`.
- Batch `transcribe_batch()` delegates to the existing multi-process batch runner and accepts an optional `devices` value that temporarily sets `CUDA_VISIBLE_DEVICES`.
- Exposed `list_models()`, `list_model_languages()` and `suggest_components()` from the package root.
- Added SDK docs in README files and focused tests for output writing, cache reuse, no-output mode, format validation, batch delegation/device handling, and model query exports.
- Validation passed: focused API/config/batch tests, full unit discover with 134 tests, `compileall semantic_asr tests examples`, and `git diff --check`.

- [x] Current work: run real `yue_punctuation` smoke tests on simplified Cantonese and simplified Cantonese-English input.
- [x] Current work: inspect whether simplified text gets punctuation and valid timestamp-mapped `punc_sentences`.
- [x] Current work: record the result in progress and commit the test record if files change.

Review:
- Real GPU smoke tests passed for simplified Cantonese and simplified Cantonese-English inputs using the local `nizzzo/zh-yue-punctuation-restore-v3` snapshot on `CUDA_VISIBLE_DEVICES=4`.
- Simplified Cantonese output: `我今日返工，你去边系唔系一齐食饭？`.
- Simplified Cantonese-English output: `我今日开meeting，你send email畀我好唔好？`.
- A longer simplified mixed sample returned `如果client approve咗，我哋就start development下个sprint，再review`, matching the previous traditional mixed sample behavior: internal commas work, final punctuation is not guaranteed.
- Timestamp-mapped `punc_sentences` remained valid for all samples.

- [x] Current work: run real `yue_punctuation` smoke tests on Cantonese-English mixed input.
- [x] Current work: inspect whether punctuation and timestamp-mapped sentence spans remain valid for code-switch tokens.
- [x] Current work: record the result in progress and commit the test record if files change.

Review:
- Real GPU smoke tests passed for three Cantonese-English mixed inputs using the local `nizzzo/zh-yue-punctuation-restore-v3` snapshot on `CUDA_VISIBLE_DEVICES=4`.
- The adapter handled English tokens such as `meeting`, `send email`, `project deadline`, `Friday`, `confirm`, `client approve`, `start development`, `sprint`, and `review` without crashing or losing timestamp spans.
- Outputs remained timestamp-mapped `punc_sentences`. Two examples produced sentence-final punctuation and one produced only internal commas, so code-switch input is technically supported but punctuation quality still needs domain smoke tests.

- [x] Current work: run a real-model smoke test for downloaded `nizzzo/zh-yue-punctuation-restore-v3`.
- [x] Current work: verify the output shape includes timestamp-mapped `punc_sentences`.
- [x] Current work: record the result in progress and commit the test record.

Review:
- Real model load passed from local snapshot `~/.cache/huggingface/hub/models--nizzzo--zh-yue-punctuation-restore-v3/snapshots/015afa4682164b92e38d69271d6700f59d443098` on `CUDA_VISIBLE_DEVICES=4`.
- Smoke text `我 今日 返工 你 去 邊 係 唔 係 一齊 食 飯` produced one timestamp-mapped sentence: `我今日返工，你去邊係唔係一齊食飯？`.
- The checkpoint's real labels are `O`, `S-。`, `S-，`, `S-、`, `S-？`, `S-！`, `S-；`, `S-︰`; fixed `yue_punctuation` to strip sequence-label prefixes such as `S-`.
- Validation passed: focused punctuation/language-support tests, full unit discover with 128 tests, `compileall semantic_asr tests examples`, and `git diff --check`.

- [x] Current work: inspect the Hugging Face model metadata for `nizzzo/zh-yue-punctuation-restore-v3` and decide the adapter interface.
- [x] Current work: add a Cantonese punctuation adapter, registry entry, and language-support metadata.
- [x] Current work: add standalone example/docs and unit tests for model loading skip mode plus punctuation output mapping.
- [x] Current work: run focused/full validation, update progress, and commit only this change set.

Review:
- Added `yue_punctuation` as a Cantonese token-classification punctuation component for `nizzzo/zh-yue-punctuation-restore-v3`.
- The adapter maps predictions directly back to timestamp token indices instead of re-tokenizing no-space Cantonese text, so sentence spans remain aligned to the original ASR/forced-alignment timeline.
- Registered the model in the component registry and language-support catalog for `yue_hk` / `Cantonese`.
- Added `examples/test_yue_punctuation.py` and `docs/models/yue_punctuation.md`.
- Validation passed: focused punctuation/language-support tests, skip-load standalone example, full unit discover with 128 tests, `compileall semantic_asr tests examples`, and `git diff --check`.

- [x] Current work: make final `cut_segments_ms` / `cut_start_ms` / `cut_end_ms` use padded `output_vad_segments` instead of raw VAD.
- [x] Current work: add regression coverage proving `output_vad_pad_s` affects TextGrid/CSV/SRT output times through `cut_*`.
- [x] Current work: run focused/full validation and update progress/commit.

Review:
- Pipeline now computes `output_vad_segments_ms` once from merged/padded output VAD and uses it for both sentence-to-output-VAD alignment and final `add_sentence_cut_segments()`.
- Final `cut_segments_ms`, `cut_start_ms` and `cut_end_ms` now reflect `output_vad_pad_s`; TextGrid/CSV/SRT already consume those `cut_*` fields.
- Raw VAD is still preserved unchanged under `raw_vad_segments_ms` for debugging and boundary-fusion support.
- Added regression coverage showing a raw segment `1200-1350ms` becomes padded/clipped output VAD `1100-1475ms`, and the final sentence `cut_end_ms` follows that padded value.
- Validation passed: focused ASR-VAD/punctuation/output tests, full `unittest discover tests` with 125 tests, `compileall semantic_asr tests`, and `git diff --check`.


- [x] Current work: inspect the four German `batch_2_output` errors again after MMS whitespace normalization and determine whether they are stale or newly reproduced.
- [x] Current work: if newly reproduced, trace the exact MMS token/span mismatch and patch the smallest upstream normalization gap.
- [x] Current work: run focused/unit validation plus GPU verification on the affected German samples.

Review:
- The four German errors were newly reproduced after the first whitespace-collapse fix. Their mtimes were `2026-06-12 02:16-02:17`, and traceback line numbers pointed to the normalized-token code path.
- The remaining gap was empty uroman tokens, not repeated spaces inside non-empty tokens. Target construction ignored empty pieces, but `get_spans()` still saw an empty token and tried to match labels such as `o`, `<star>`, `a` and `d` against `""`.
- Added `_drop_empty_alignment_tokens()` after `_normalize_token_spaces()` in MMS runtime, keeping `items` and `tokens` aligned and dropping only tokens whose normalized uroman string is empty.
- Added a regression test that simulates uroman returning `["", " o "]`; the empty raw token is dropped, `get_alignments()`/`get_spans()` receive only `["o"]`, and the returned timestamp item is the non-empty token.
- Validation passed: `tests.test_mms_forced_aligner`, full `unittest discover tests` with 124 tests, `compileall semantic_asr tests`, and `git diff --check`.
- GPU verification passed on the four affected German wavs with `CUDA_VISIBLE_DEVICES=4,5,6,7` and `semantic_asr/run_batch.py`; all four returned `ok: true` and generated JSON/TextGrid/SRT/CSV under `output/de_de_batch2_remaining4_verify`.

- [x] Current work: normalize MMS uroman token whitespace before both `get_alignments()` and `get_spans()`.
- [x] Current work: add regression coverage for a uroman token containing repeated/edge spaces.
- [x] Current work: run focused MMS tests and update progress/TODO.

Review:
- Added MMS runtime whitespace normalization immediately after uroman token generation: each non-`<star>` token is collapsed with `" ".join(token.split())`.
- The normalized token list is now the single source used by both `get_alignments()` and `get_spans()`, preventing empty split pieces from creating assertions such as `===> o <=> `.
- Added a runtime regression test that simulates uroman returning `" a  b "` and verifies both alignment and span reconstruction receive `"a b"`.
- Validation passed: `tests.test_mms_forced_aligner`, full `unittest discover tests` with 123 tests, `compileall semantic_asr tests`, and `git diff --check`.


- [x] Current work: inspect the 4 remaining German error JSON files in `de_de/batch_2_output`.
- [x] Current work: identify the exact exception class, failing MMS stage and whether normal JSON/result output exists for those uttids.
- [x] Current work: explain why the remaining errors differ from the already staged cut-segment overlap fix.

Review:
- Remaining error files: `1be9ef3a-4984-4b66-aea8-fc1759c775c4`, `54b7866f-a300-4145-98d6-9cec3dce698c`, `5ac7a083-cdcf-46ab-9af3-dd75b89c5eee`, `60aaac37-88f4-4dd8-a437-6c9d469b3e38`.
- All 4 fail in `semantic_asr.mms_runtime.align_utils.get_spans()` with `AssertionError('===> <label> <=> ')`, where `<label>` is `o`, `<star>`, `a` or `d` and the expected token character is empty.
- This is an MMS token/span reconstruction bug: uromanized token strings are split with `token.split(" ")`, so multiple spaces or empty pieces create an empty expected character. The forced-align path has labels for real symbols, but `get_spans()` tries to match them against `""`.
- No ordinary JSON exists for these 4 uttids and `result.jsonl` has no successful records for them, so the exact ASR text/segment cannot be recovered from current artifacts without rerunning with diagnostic logging.
- This is separate from the staged Ten-VAD padded-cut overlap fix, which addressed the earlier `Overlapping cut segment` errors.

- [x] Current work: inspect all error files in `/data_151/duhu/DBC/ASR/22424_微软ITN5语种混合模型测试/de_de/batch_2_output` and group by concrete exception.
- [x] Current work: trace the failing code path against the current uncommitted MMS/Whisper changes.
- [x] Current work: fix the root cause with minimal code changes and add regression coverage.
- [x] Current work: rerun focused/unit validation and a representative German batch sample after the fix.
- [x] Current work: update progress/TODO and commit if git approval allows it.

Review:
- The original 30 German `batch_2_output` failures were cut-segment validation errors from overlapping padded Ten-VAD islands. The staged fix coalesces overlapping raw VAD cut islands before final `cut_segments_ms` generation.
- A real rerun of `02503fdb-e58c-42dd-b0d1-ad4bc7faed21.wav` passed and generated JSON/TextGrid/SRT/CSV under `output/de_de_batch2_verify_one`.
- The current directory now has 4 remaining errors, all in MMS span reconstruction: `get_spans()` asserts because `seg.label` is `o`, `<star>`, `a` or `d`, while the expected `ltr` is an empty string.
- These 4 remaining errors are not the cut-overlap bug. They happen after MMS forced alignment has produced segments, when uromanized token strings are split with `token.split(" ")`; multiple spaces or empty pieces in the token string create an empty expected character that cannot match any CTC segment label.
- The existing error JSON files do not contain the ASR text or segment id, and `result.jsonl` has no success records for those 4 uttids, so exact offending text cannot be recovered without a diagnostic rerun.


- [x] Current work: expose Whisper decode parameters and allow stricter short-audio decode settings.
- [x] Current work: add a pre-MMS ASR alignment quality check that computes target/frame feasibility and detects empty targets before MMS.
- [x] Current work: discard or record unalignable tiny hallucination segments before forced alignment instead of relying on approximate fallback.
- [x] Current work: treat `duration <= asr_vad_min_segment_s` as tiny in ASR VAD preprocessing and add focused regression tests.
- [x] Current work: run focused tests, update progress and commit.

Review:
- `WhisperLargeConfig` now exposes normal decode options and short-audio overrides. For short clips, the adapter defaults to `temperature=0.0`, `beam_size=5`, `length_penalty=0.0`.
- `prepare_asr_vad_segments()` now treats `duration <= asr_vad_min_segment_s` as tiny, so exact-threshold 500ms islands are merged or dropped instead of slipping through.
- `MmsForcedAlignerTimestampProvider` now computes alignment feasibility before calling MMS alignment: estimated frame count, uroman/dictionary target count, repeat count and required frame count. Empty text/target, CTC-impossible spans and dense tiny-segment hallucinations are skipped by default.
- Skipped ASR segments are recorded on the final JSON as `discarded_asr_segments`; approximate monotonic timestamp fallback is now opt-in with `fallback_on_feasibility_error=true`.
- Updated Whisper and MMS docs for the new parameters and default behavior.
- Validation passed: focused adapter/ASR-VAD/MMS tests, full unit discover before docs, compileall, config parse for 13 JSON profiles and `git diff --check`.
- Real GPU rerun of `838db147` was requested for end-to-end verification but rejected by the automatic approval reviewer, so this turn could not produce a real sample JSON.


- [x] Current work: run a focused Whisper length-penalty demo on the known hallucinated `838db147` 500ms segment before changing pipeline logic.
- [x] Current work: compare default/current-like decoding against beam-search length-penalty variants and record whether hallucination is reduced.

Review:
- Demo segment: `/data_151/duhu/DBC/ASR/22424_微软ITN5语种混合模型测试/ar_sa/batch_5/838db147-31d3-412a-ae62-edd30ef8a554.wav`, `296770-297270ms`, duration `0.5s`.
- Current-like Whisper large-v3 decoding reproduced the old hallucination: `إذا حصلت على محاولة تحقيق المنطقة، فإنها تتحقق بمعرفة المنطقة.`
- `beam_size=5` without length penalty produced an even longer repetitive hallucination.
- `beam_size=5` with `length_penalty=0.0`, `0.2` or `1.0` shortened the hallucination to `اشتركوا في القناة`, but still did not suppress text.
- The segment metadata was not enough for Whisper's default filters to reject it: `no_speech_prob≈0.296`, `avg_logprob≈-0.56` for the shortened hallucination. Conclusion: length penalty can reduce output length but cannot be the main anti-hallucination mechanism for tiny VAD islands.


- [x] Current work: inspect available artifacts for `838db147`, `8606f16f`, and `afdb46d5` MMS errors and locate exact segment/text where recoverable.
- [x] Current work: compute or recover MMS CTC target length, repeat count and frame count for each failed segment where artifacts allow it.
- [x] Current work: explain why each failure happens and which upstream stage should be fixed before any fallback.

Review:
- `838db147-31d3-412a-ae62-edd30ef8a554` is fully recoverable from the diagnostic rerun: the failing segment is `296770-297270ms`, text `إذا حصلت على محاولة تحقيق المنطقة، فإنها تتحقق بمعرفة المنطقة.`, prepared as 10 Arabic word tokens. Uroman/dictionary mapping produces 52 target symbols with 3 consecutive repeats, but the 500ms audio island produces only 24 MMS emission frames, so CTC needs at least 55 frames and fails.
- `8606f16f-8713-43d8-8491-6236446b61ac` failed historically because MMS reached `forced_align()` with an empty target tensor after uroman/dictionary filtering. The old `error.json` stores only the traceback, not the ASR segment text or token list. A current diagnostic rerun completes and has 103 timestamp segments with zero empty-target segments, so the exact old text cannot be recovered from checked artifacts.
- `afdb46d5-b788-4d16-bc36-edac8632ae6a` failed historically with the same CTC feasibility pattern as `838db147`: old torchaudio message reports `targets length: 30, log_probs length: 34, repeats: 1`, which our preflight interpretation maps to 30 emission frames, 34 target symbols and 1 repeat. CTC needs at least 35 frames, so the segment is about one frame short. The old `error.json` does not store the ASR text; a GPU diagnostic rerun request was rejected by the approval reviewer, so the exact segment text was not recoverable in this turn.

- [x] Current work: add an MMS preflight error for empty target indices before calling torchaudio forced alignment.
- [x] Current work: add an MMS preflight error for CTC-impossible `frames < target_chars + repeats` inputs before calling torchaudio forced alignment.
- [x] Current work: make the MMS adapter fall back to monotonic approximate token timestamps for preflight-impossible segments instead of failing the whole audio.
- [x] Current work: add unit tests for empty-target fallback, CTC-impossible fallback and normal MMS error propagation.
- [x] Current work: run focused tests and update progress/commit.

Review:
- Added `MmsAlignmentFeasibilityError` for two preflight failures before `torchaudio.functional.forced_align()`: `empty_target` and `ctc_target_too_long`.
- `MmsForcedAlignerTimestampProvider` catches only that feasibility error and falls back to monotonic approximate token timestamps for the current ASR segment. Unexpected MMS/runtime errors still propagate.
- Fallback metadata is preserved under `timestamp_segments[].timestamp_fallback` in the output JSON.
- Added unit tests for empty-target preflight, CTC-impossible preflight, adapter fallback, unexpected-error propagation and JSON fallback metadata.
- Validation passed: `tests.test_mms_forced_aligner tests.test_asr_vad_postprocess`, full `unittest discover tests` with 115 tests, `compileall`, and `git diff --check`.
- Real GPU rerun of the three remaining Arabic files could not be executed in this turn because the sandbox escalation request was rejected by the approval reviewer.

- [x] Current work: inspect the 5 Arabic `batch_5_output_fireredvad` error JSON files and group them by failing stage.
- [x] Current work: compare each error with any matching normal JSON output to see whether failure is from pipeline inference or output writing.
- [x] Current work: trace MMS failures to the exact CTC/target/token pattern and check whether the tiny-VAD fix should address them.
- [x] Current work: summarize root causes and next fixes without changing code unless the cause is obvious and narrow.

Review:
- Found 5 errors in `/data_151/duhu/DBC/ASR/22424_微软ITN5语种混合模型测试/ar_sa/batch_5_output_fireredvad`.
- Four fail during `timestamp:mms_forced_aligner`; none of these have completed result JSON files, so the ASR text printed before MMS was not recoverable from stored artifacts.
- Three MMS failures are CTC feasibility errors:
  - `838db147-31d3-412a-ae62-edd30ef8a554`: about 24 frames / 480ms versus 52 target chars plus 3 repeats; this is a micro/dense segment and should be helped by the new ASR VAD tiny-island merge.
  - `afdb46d5-b788-4d16-bc36-edac8632ae6a`: about 30 frames / 600ms versus 34 target chars plus 1 repeat; this is just above the 0.5s tiny threshold and may still need MMS feasibility guarding or a slightly larger ASR VAD minimum.
  - `66166626-13c6-4cca-898d-82f1f92fd98c`: about 127 frames / 2.54s versus 140 target chars plus 2 repeats; this is not a tiny island, but an ASR-text-density/CTC-capacity failure.
- `8606f16f-8713-43d8-8491-6236446b61ac` fails because MMS `token_indices` is empty after uroman/dictionary filtering, causing torchaudio `forced_align()` to call `torch.max()` on an empty target tensor. This needs an explicit MMS guard before calling `forced_align()`.
- `fd41e9e3-7722-49c8-bcb7-b4d8d88f634f` has an existing stale bad JSON and fails in resume validation with `Invalid sentence interval at index 15: 77120-77120`. The bad interval is caused by old negative-gap boundary decisions splitting `live. worldbank. org.` into three candidates with the same start time.
- Replaying the old `fd41...` `semantic_sentences` through the current `fuse_sentence_boundaries()` no longer creates the invalid interval: the two negative-gap candidates are merged with reason `merged_non_monotonic_boundary`, producing one valid `في live. worldbank. org.` sentence.
- Next fixes: add an MMS preflight/guard for empty and CTC-impossible targets, and decide how `run_pipeline` should handle an existing invalid JSON cache instead of failing before regeneration.

- [x] Current work: add an ASR/MMS input VAD postprocess that merges tiny VAD islands into a nearby speech segment before transcription/alignment.
- [x] Current work: keep `raw_vad_segments_ms` and final output VAD formatting unchanged while recording the postprocessed ASR VAD segments separately.
- [x] Current work: add focused unit tests for tiny-island merge, isolated tiny-island drop and no-overlap monotonic behavior.
- [x] Current work: rerun a previously failing Arabic microsegment sample to verify MMS CTC no longer fails.
- [x] Current work: update progress and commit the completed fix.

Review:
- Added `PipelineConfig.asr_vad_min_segment_s` and `PipelineConfig.asr_vad_max_merge_silence_s`; defaults are 0.5s and 1.0s.
- Added `prepare_asr_vad_segments()`: tiny raw VAD islands merge to the nearest neighbor within the configured silence gap, chained tiny islands collapse safely, and isolated tiny islands are skipped.
- Pipeline ASR/timestamp stages now use the postprocessed ASR VAD segments, while `raw_vad_segments_ms` and final cut generation still use the original raw VAD detector output.
- Added focused tests in `tests/test_asr_vad_postprocess.py` and updated config runner tests for the new default ASR VAD behavior.
- Real Arabic MMS CTC rerun passed for `5e2e123b-f598-4e7d-973d-3dd8877ccadf.wav`; raw VAD still contains `204220-204280ms`, but `asr_vad_segments_ms` contains no sub-500ms segment.

- [x] Current work: identify all Arabic batch errors whose MMS traceback contains `targets length is too long for CTC`.
- [x] Current work: extract per-error CTC frame/target/repeat counts and recover ASR text where the stored logs allow it.
- [x] Current work: group the failures by short-segment, token-density, numeric/symbol token and language/text-normalization patterns.
- [x] Current work: inspect MMS adapter/runtime code paths to explain exactly where the CTC length constraint is triggered.
- [x] Current work: summarize concrete root causes and candidate fixes before changing code.

Review:
- Found 30 Arabic batch errors with `targets length is too long for CTC` in `batch_4_output`.
- The torchaudio error message names are misleading for our diagnosis: a small local forced-align check confirmed the first reported length corresponds to available emission frames and the second to target characters. CTC feasibility is effectively `frames >= target_chars + repeats`.
- 27/30 failures are micro-segment cases. FireRed VAD produced 50-340ms raw speech islands, and those raw VAD islands were sent directly to ASR and then MMS; Whisper can hallucinate several Arabic words on such tiny audio snippets, leaving MMS with only 2-16 frames for roughly 14-16 uroman target characters.
- The remaining stored CTC failures are text-density/repeat cases: e.g. original stale errors reported 66 frames vs 197 target chars and 263 frames vs 214 target chars plus 84 repeats. Current reruns did not reproduce those exact ASR texts, which indicates ASR output variability, but the CTC math still explains why those historical inputs failed.
- The stored `*.error.json` files contain only traceback metadata, not the ASR text/tokens printed immediately before MMS alignment, so full per-error ASR text cannot be reconstructed offline from the old failures.
- MMS code path: `semantic_asr/core.py` builds one ASR/MMS segment per raw VAD timestamp; `semantic_asr/adapters/mms_forced_aligner.py` tokenizes ASR text and forces `use_star=False`; `semantic_asr/mms_runtime/aligner.py` uromanizes alignment tokens and calls `torchaudio.functional.forced_align()`.
- Numeric/currency `<star>` replacement is not the primary cause for these 30 CTC-length errors; it generally reduces target length. The dominant issue is segment duration versus ASR hallucinated/dense target text.
- Candidate fixes should focus on pre-MMS feasibility: avoid sending raw VAD microsegments directly to ASR/MMS, attach/drop/merge tiny VAD islands for ASR slicing, and add an MMS guard that skips or merges impossible `frames < target_chars + repeats` alignments before calling `forced_align()`.

- [x] Current work: add regression tests for punctuation-split token-like strings that previously created non-monotonic timestamp candidates.
- [x] Current work: prevent `split_text_by_punctuation()` from falling back empty timestamp slices to the whole segment.
- [x] Current work: make sentence-boundary fusion merge/reject invalid negative-gap keep boundaries before they can create reversed intervals.
- [x] Current work: add a final sentence interval invariant before output writing so invalid JSON cannot silently reach TextGrid.
- [x] Current work: run unit validation and rerun the three Arabic problem audios.

Review:
- `split_text_by_punctuation()` now keeps periods inside ASCII token-like strings such as `console.write`, `getlink.io` and decimal numbers such as `1.3400`, and empty timestamp slices are appended to the previous mapped sentence instead of falling back to the whole segment.
- Boundary fusion now merges candidates with negative token gaps using reason `merged_non_monotonic_boundary`, preventing audio-safe keep paths from creating reversed sentence intervals.
- Added `validate_sentence_intervals()` and call it before returning new pipeline JSON and before regenerating outputs from an existing JSON.
- Added regression coverage for token-like period handling, empty timestamp slices, negative-gap fusion and reversed/overlapping final intervals.
- Validation passed: focused punctuation/boundary/output tests, full unit discover, package compile, `git diff --check`, and real reruns of the three Arabic TextGrid-failing audios.
- Real rerun outputs:
  - `output/textgrid_boundary_fix/32147e84-8904-42fd-9274-c7bef22af88e`: 78 sentences, 0 invalid intervals, TextGrid written.
  - `output/textgrid_boundary_fix/53f25da4-c944-43ef-9ab6-980f1c6779d4`: 34 sentences, 0 invalid intervals, TextGrid written.
  - `output/textgrid_boundary_fix/aa17261e-760f-439b-a343-1ebb3ec3488a`: 58 sentences, 0 invalid intervals, TextGrid written.

- [x] Current work: analyze the three Arabic TextGrid error JSON files with reversed sentence intervals.
- [x] Current work: trace each reversed final sentence back to nearby `semantic_sentences` and `sentence_boundary_decisions`.
- [x] Current work: inspect nearby word/timestamp ordering to explain why start/end became non-monotonic.
- [x] Current work: summarize concrete root cause per file and the minimal guard needed.

Review:
- The TextGrid failures are downstream symptoms. The JSON files already contain one reversed sentence interval each (`end_ms < start_ms`), and TextGrid raises once the reversed interval is skipped and the neighboring intervals overlap.
- All three reversed cases share the same upstream pattern: punctuation splitting breaks inside token-like strings (`console.write`, `getlink.io`, decimals such as `1.3400`), then `split_text_by_punctuation()` creates a non-monotonic semantic candidate by falling back to the whole timestamp segment.
- Boundary fusion then sees a negative token gap but can still keep the boundary as audio-safe (`vad_prob_silence`). In the non-preserved-gap path it writes `previous["end_ms"] = boundary_ms` without proving the boundary is after `previous["start_ms"]`, creating the reversed interval.
- Concrete cases:
  - `32147e84-8904-42fd-9274-c7bef22af88e`: `console.write` / `C sharp` split creates `sharp.` with segment-start fallback; candidate 74 keeps boundary `241745ms`, reversing previous `242606-245110ms` into `242606-241745ms`.
  - `53f25da4-c944-43ef-9ab6-980f1c6779d4`: decimal tokens `1.33`, `3.330`, `1.3400` are split as sentence punctuation; candidate 37 keeps boundary `156135ms`, reversing previous `156876-163220ms`.
  - `aa17261e-760f-439b-a343-1ebb3ec3488a`: URL token `getlink.io` is split at the period; candidate 5 keeps boundary `21220ms`, reversing previous `22982-29650ms`.
- Minimal guard needed before code changes: punctuation mapping should never fall back an empty slice to the whole timestamp segment, and boundary fusion should never keep a boundary with negative token gap or a boundary outside `[previous.start_ms, current.end_ms]`.

- [x] Current work: inspect `/data_151/duhu/DBC/ASR/22424_微软ITN5语种混合模型测试/ar_sa/batch_4_output` error files.
- [x] Current work: group Arabic batch errors by exception type, stage and root cause.
- [x] Current work: inspect representative JSON outputs where needed to distinguish output-writer errors from pipeline strategy errors.
- [x] Current work: summarize the error causes and recommended fixes.

Review:
- Found 47 `*.error.json` files in `batch_4_output`: 44 failed in `timestamp:mms_forced_aligner`, 3 failed while writing TextGrid.
- MMS failures break down into 30 `targets length is too long for CTC`, 6 empty-target `torch.max()` failures, 7 `get_spans` label/token assertion mismatches and 1 short-emission shape failure.
- 27 of the CTC-length errors had only 14-16 MMS emission frames, which points to very short raw VAD/ASR segments being sent into MMS with more alignment characters than the CTC path can support.
- The remaining larger CTC errors indicate ASR text/character count is too dense for the available audio frames, especially with repeated characters or code-switched/unsupported text.
- The 3 TextGrid errors already had JSON files, but each JSON contains one reversed sentence interval (`end_ms < start_ms`), which then leaves overlapping neighboring output intervals after TextGrid sorting.
- The reversed TextGrid cases are tied to non-monotonic punctuation/timestamp candidates around mixed Arabic/English/numeric fragments such as `write console`, `330`, and `getlink.io`.

- [x] Current work: inspect `output/de_de/848c4ebd-419a-4c42-a85a-2bb1ab52b121.json` for current German sentence/punctuation output.
- [x] Current work: run a Qwen JSON-only experiment on the German timestamp tokens with boundary and punctuation suggestions.
- [x] Current work: compare Qwen suggested spans/punctuation against the current output and manually judge German readability.
- [x] Current work: record whether Qwen is useful for German punctuation/boundary repair.

Review:
- The existing German output has 799 words, 59 punctuation-derived semantic candidates and 21 final fused sentences.
- Full-file Qwen punctuation over 799 stripped tokens timed out; 90-second window tests also timed out. This is too slow for regular whole-file punctuation.
- A smaller `289s-328s` sentence-end-only test returned valid JSON for 89 tokens, but split the phrase "die Status Quo Situation ... hin zum gewünschten Ergebnis" incorrectly; current semantic candidates are better there.
- A 25-token tail full-punctuation test returned, but produced an invalid/inconsistent sentence-end mark (`tricky,` as a sentence end) and missed useful German commas such as `natürlich, wenn ...`.
- The retry prompt for that invalid punctuation output timed out. Recommendation: use Qwen for German only as an optional semantic-boundary fallback, not as a replacement for existing ASR punctuation or German comma restoration.

- [x] Current work: add OpenAI-compatible `response_format={"type":"json_object"}` to Qwen semantic-boundary requests.
- [x] Current work: keep the setting configurable in case a backend does not support structured JSON mode.
- [x] Current work: update tests/docs and verify the local Qwen endpoint accepts the request.
- [x] Current work: run validation, update progress and commit.

Review:
- Qwen semantic-boundary requests now include `response_format={"type":"json_object"}` by default.
- Added `response_format_json` config switch so incompatible OpenAI-compatible backends can disable the field.
- Updated tests to assert the default request body and the opt-out path.
- Updated `configs/th_th_qwen_boundary.json` and `docs/models/qwen_semantic_boundary.md`.
- Validation passed: `tests.test_qwen_semantic_boundary`, full unit discover (101 tests), compileall, all config parse, `git diff --check`, and a real local-Qwen adapter smoke with `response_format` enabled.

- [x] Current work: design Qwen semantic-boundary support as an index-only component that never rewrites ASR text.
- [x] Current work: implement JSON/end-index parsing, validation, bounded retries and local-rule fallback behavior.
- [x] Current work: wire the Qwen boundary result into existing sentence-boundary fusion without changing final text generation.
- [x] Current work: add tests covering valid output, retry repair, malformed output fallback and multilingual text input.
- [x] Current work: document the Qwen API contract, required thinking-disable option and language expectations.
- [x] Current work: run focused/full validation and update progress.

Review:
- Added `qwen_semantic_boundary` as an index-only `punc` component backed by the local Qwen3.6 chat API.
- The component asks for `{"token_count": N, "end_indices": [...]}` and reconstructs sentence text only from original timestamp tokens.
- Added strict validation for JSON parsing, token count, monotonicity, no overlap, complete coverage and index bounds.
- Added bounded retry with the validation error included in the next prompt; repeated failure falls back to duration-based candidate boundaries.
- Registered the component, added language catalog support as `*`, added `configs/th_th_qwen_boundary.json`, and documented the API contract in `docs/models/qwen_semantic_boundary.md`.
- Validation passed: `tests.test_qwen_semantic_boundary`, full unit discover, compileall, all config parse, `query_models.py language th_th --role punc`, `git diff --check`, and a real Thai Qwen adapter smoke on the existing timestamp sample.

- [x] Current work: read `/data/duhu/FireRedASR2S/docs/qwen_api.md` and identify the local Qwen API contract.
- [x] Current work: prepare a Thai token/text sample suitable for sentence-boundary testing.
- [x] Current work: call Qwen with a JSON-only prompt for Thai semantic sentence spans.
- [x] Current work: validate whether the output is monotonic, non-overlapping, and usable for timestamp mapping.
- [x] Current work: record the result and recommendation.

Review:
- Qwen API model listing works at `http://10.10.23.16:18000/v1/models`, but sandboxed network cannot reach it; API smoke calls need unrestricted execution.
- Plain chat requests and `/no_think` prompts produced visible thinking text and are not safe for direct JSON parsing.
- Adding `chat_template_kwargs={"enable_thinking": false}` produced clean parseable JSON.
- Synthetic no-punctuation Thai chunks were split into 3 semantic spans covering chunks `0-5` continuously.
- Existing real Thai ASR token sample from `output/experiments/multilingual_short_boundary_fusion/th_th/th_th.json` returned one span covering tokens `0-102`, mapping cleanly to `460-15660ms`.
- Recommendation: Qwen is viable as a Thai semantic-boundary backend if the client disables thinking, validates monotonic coverage, ignores/freezes `reason` as debug-only, and falls back to acoustic/duration rules on malformed responses.

- [x] Current work: implement rolling selector for over-`max_sentence_s` boundary choice.
- [x] Current work: keep recent boundary candidates with speech-probability stats inside the current fused group.
- [x] Current work: choose the lowest-probability semantic candidate instead of the first over-max candidate.
- [x] Current work: add regression coverage for the German tail case shape.
- [x] Current work: update docs/progress, validate, commit and push.

Review:
- Added `rolling_boundary_max_candidates` to `SentenceBoundaryFusionConfig`, defaulting to 8 recent candidates.
- Boundary fusion now keeps hidden current-group candidate history and, on over-max semantic caps, selects the recent semantic-complete boundary with the lowest local speech probability.
- The reported German tail recomputation moves the split from `318.745s` to `305.081s`, choosing `speech_prob_mean=0.4548` instead of the over-max trigger at `0.7208`.
- Boundary decisions now record `rolling_selected_candidate_index`, `rolling_selected_boundary_ms` and `rolling_selected_speech_prob_mean`.
- Validation passed: `tests.test_sentence_boundaries` (21 tests), full unit discover (94 tests), `compileall semantic_asr tests`, config parse for all 12 JSON profiles and `git diff --check`.

- [x] Current work: add a sentence-boundary rule that raw VAD silence gaps over 1s are not merged.
- [x] Current work: expose the 1s threshold as boundary-fusion config with a conservative default.
- [x] Current work: add regression tests for long raw-VAD silence overriding semantic-incomplete merge.
- [x] Current work: update strategy docs/progress and run validation.
- [x] Current work: commit and push the completed fix.

Review:
- Added `SentenceBoundaryFusionConfig.max_merge_vad_silence_s`, defaulting to 1.0s.
- Boundary fusion now keeps `long_vad_silence` boundaries before semantic-incomplete merge logic, so raw VAD pauses of at least the configured threshold are not merged across.
- Boundary decisions now include `long_vad_silence` for JSON/debug inspection.
- Added regression tests for long-silence splitting and configurable threshold behavior.
- Validation passed: `tests.test_sentence_boundaries` (20 tests), full unit discover (93 tests), `compileall semantic_asr tests`, config parse for all 12 JSON profiles, and `git diff --check`.

- [x] Current work: investigate why `configs/ar_sa.json` outputs a single sentence for `05162293-323c-4c4c-8e2e-5c3d0b568b3d`.
- [x] Current work: identify whether the collapse comes from punctuation, boundary fusion, timestamp gaps, or cleanup-time config changes.
- [x] Current work: fix the Arabic config/logic so semantic candidates are preserved when audio-safe.
- [x] Current work: validate with focused tests and existing output inspection.
- [x] Current work: update progress/TODO and commit/push if code changes are needed.

Review:
- Root cause: Naqta/punctuation had produced 21 `semantic_sentences`, but boundary fusion treated every boundary as active speech and kept recording `max_duration_wait_for_silence`, so the 287.935s file collapsed into one final sentence.
- Fixed over-max active-speech handling: audio-safe boundaries still win; semantic-complete active boundaries now cap an over-`max_sentence_s` run with `max_duration_terminal_punctuation` or `max_duration_semantic_boundary`; semantic-incomplete active boundaries still wait for silence.
- Offline recomputation on `output/ar_sa_error-01/05162293-323c-4c4c-8e2e-5c3d0b568b3d.json` changes final fusion from 1 sentence to 13 sentences, with max duration 29.1s.
- Validation passed: `tests.test_sentence_boundaries` (18 tests), full unit discover (91 tests), `compileall semantic_asr tests`, and `git diff --check`.

- [x] Current work: checkpoint and push current `red-asr` state to GitHub before further simplification.
- [x] Current work: inspect configs and strategy code for historical compatibility clutter.
- [x] Current work: remove obsolete configs/options and simplify the public configuration surface.
- [x] Current work: run validation after simplification.
- [x] Current work: update docs/progress/TODO and commit the cleanup.

Review:
- First pushed checkpoint commit `04283fd` to `origin/red-asr`.
- Removed legacy combination configs/scripts/builders and kept only language or
  scenario named configs: `zh_cn`, `ar_sa`, `de_de`, `en_us`, `hakka`,
  `hi_in`, `ja_jp`, `ko_kr`, `pt_br`, `ru_ru`, `th_th`, `vi_vn`.
- Removed pre-ASR VAD merge and final short-gap sentence merge from the core
  pipeline; raw VAD now feeds ASR/timestamp directly, and boundary fusion is
  the single sentence grouping stage.
- Simplified config files by relying on defaults and changed FireRed VAD/Punc
  params to support direct fields such as `use_gpu` and
  `extend_speech_frame`.
- Made the registry and adapter package lightweight by lazy-importing concrete
  model adapters only when components are built.
- Validation passed: 90 full unit tests, compileall, config parse for all 12
  profiles, `query_models.py language ar_sa --role punc`, draw.io XML parse,
  obsolete-field scan and `git diff --check`.

- [x] Current work: record implementation plan for speech-probability-first single-pass sentence grouping.
- [x] Current work: remove RMS valley from boundary config, docs and config profiles.
- [x] Current work: refactor boundary fusion to build explicit boundary candidates and use one decision table.
- [x] Current work: disable final short-gap merge when sentence boundary fusion is enabled.
- [x] Current work: add regression tests for Arabic, German and no-probability fallback behavior.
- [x] Current work: run targeted and full validation.
- [x] Current work: update strategy docs and progress notes.

Review:
- Removed RMS/waveform valley from the active sentence-boundary policy and
  cleaned checked-in configs of obsolete `acoustic_*` fields.
- Boundary fusion now builds explicit `BoundaryCandidate` objects and uses a
  single decision table based on raw VAD silence, frame-level speech
  probability, semantic completeness and duration pressure.
- High speech-probability active boundaries now merge even when the span is
  over `max_sentence_s`; the decision records `max_duration_wait_for_silence`.
- `merge_final_sentences_by_gap()` is disabled whenever
  `sentence_boundary_fusion.enabled=true`, leaving fusion as the single
  grouping stage.
- Updated the current strategy docs and draw.io diagram. Validation passed:
  17 targeted boundary tests, 98 full unit tests, compileall, draw.io XML parse,
  config obsolete-key scan and `git diff --check`.

- [x] Current work: record strategy redesign discussion: remove RMS valley and center on speech probability.
- [x] Current work: restate the project objective as an explicit optimization problem.
- [x] Current work: propose a simpler speech-probability-first segmentation strategy.
- [x] Current work: compare the proposed strategy against current logic and recent Arabic/German cases.
- [x] Current work: list implementation steps if we choose to migrate.

Review:
- Agreed conceptually that RMS valley should be removed from the core boundary policy because it is not robust under noise.
- Reframed the target as one segmentation optimization problem: never cut active speech first, then prefer semantic completeness, then fit configurable duration.
- Proposed a speech-probability-first boundary candidate strategy with a single grouping pass instead of boundary fusion plus final merge.

- [x] Current work: record sentence split/merge strategy audit before reading code.
- [x] Current work: inspect current strategy code paths in `core.py`, `sentence_boundaries.py`, punctuation and output timing helpers.
- [x] Current work: summarize the current split/merge pipeline in a detailed but readable form.
- [x] Current work: create a draw.io diagram for the current strategy.
- [x] Current work: identify logical holes, priority conflicts and risky edge cases.
- [x] Current work: propose a simpler strategy that preserves the current product goals.
- [x] Current work: write the analysis into docs and report the key points.

Review:
- Added `docs/sentence_split_merge_strategy_audit.md` with the current split/merge execution flow, decision order, known logical holes, and simplification proposal.
- Added `docs/sentence_split_merge_strategy.drawio` with a draw.io diagram of candidate generation, boundary fusion, output VAD alignment, final merge and cut-range export.
- Verified the draw.io XML parses and `git diff --check` passes.

- [x] Current work: record German TEN-VAD short-segment active-speech boundary regression before investigation.
- [x] Current work: inspect `/data/duhu/FireRedASR2S/output/de_de-tenvad-3/848c4ebd-419a-4c42-a85a-2bb1ab52b121.json` around 308s-328s.
- [x] Current work: identify whether boundary fusion, final merge, VAD alignment or cut segment generation preserved active-speech cuts.
- [x] Current work: adjust strategy so short adjacent terminal-punctuation sentences merge when the acoustic boundary is not safe.
- [x] Current work: add regression tests for short terminal sentences with no silence/high speech probability.
- [x] Current work: run targeted and full validation.

Review:
- Root cause: the previous Naqta fix made terminal punctuation boundaries too strong. In the German case, all reported boundaries lived inside one raw TEN-VAD speech island with 0-20ms token gaps and high frame-level speech probability, but `terminal_punctuation` kept each 2-5s sentence.
- Updated boundary fusion so terminal punctuation without audio-safe evidence is kept only once the merged span reaches `target_sentence_s`; otherwise adjacent short terminal sentences merge through active speech until the target duration or hard max policy intervenes.
- Preserved sentence punctuation when merging active-speech terminal sentences, so merged German text keeps `Schienen. Und...` rather than losing periods.
- Offline recomputation for the reported 308s-328s region now merges the five short segments into three spans: `300.099-312.563`, `312.563-322.546`, and `322.546-327.688`.
- Validation passed: targeted boundary/punctuation tests, full unit discover, and `git diff --check`.

- [x] Current work: record Naqta Arabic semantic-cut regression before investigation.
- [x] Current work: inspect `output/ar_sa_error-6` JSON for semantic candidates, final sentences and boundary decisions.
- [x] Current work: identify whether Naqta punctuation, timestamp mapping or boundary fusion caused the over-merged intervals.
- [x] Current work: adjust sentence-boundary strategy so Arabic/Naqta semantic punctuation is respected under the 30s hard-duration policy.
- [x] Current work: add a focused regression test for semantic punctuation boundaries surviving fusion.
- [x] Current work: run targeted validation.

Review:
- Root cause: Naqta produced semantic candidates, but sentence-boundary fusion did not treat Arabic question mark `؟` as terminal punctuation; sentence-local word lookup also crossed candidate boundaries and produced negative token gaps; final short-gap merge then merged terminal punctuation boundaries again.
- Fixed Arabic terminal punctuation recognition in punctuation splitting and boundary fusion.
- Changed boundary fusion word lookup to use words inside each candidate sentence instead of global before/after boundary searches.
- Kept terminal-punctuation token boundaries when word timestamps do not overlap, even if frame-level VAD probability is high.
- Made final short-gap merge skip previous sentences that already end with terminal punctuation.
- Real rerun `output/ar_sa_error-6_fix2` now has 19 final sentences from 21 semantic candidates; max duration is 29.37s and the reported `وما أدراك ما الحق؟` split is preserved.
- Validation passed: targeted tests, full unit discover, `git diff --check`, and real GPU rerun with `configs/ar_sa_naqta.json`.

- [x] Current work: record Naqta Arabic punctuation integration plan before coding.
- [x] Current work: inspect existing punctuation component contracts and language catalog.
- [x] Current work: add a Naqta punctuation adapter with configurable HF token-label mapping.
- [x] Current work: register Naqta and expose Arabic language support/query results.
- [x] Current work: add docs and focused tests without requiring the model download.
- [x] Current work: run targeted validation.

Review:
- Added the Arabic-only `naqta` punctuation component with a configurable Hugging Face token-classification label-to-punctuation mapping.
- Registered `naqta` in the default component registry and language-support catalog.
- Added `docs/models/naqta.md`, `examples/test_naqta_punctuation.py`, and `configs/ar_sa_naqta.json`.
- Extended punctuation sentence splitting to recognize Arabic question mark `؟`.
- Validation passed: targeted unit tests, skip-load example, Arabic punc query, and full unit discover.

- [x] Current work: record plan for single-GPU multi-process acceleration for non-batch Whisper and MMS alignment.
- [x] Current work: inspect current ASR and timestamp provider batch contracts.
- [x] Current work: design configurable process worker wrappers without changing model adapter semantics.
- [x] Current work: wire parallel workers for Whisper ASR and MMS forced aligner.
- [x] Current work: add focused tests and run validation.

- [x] Current work: record priority change: audio-safe boundary first, 30s length second, semantic completeness third.
- [x] Current work: inspect `output/ar_sa_error-2` for long merged Arabic sentences.
- [x] Current work: update boundary fusion so over-30s spans choose the best audio-safe boundary before semantic completeness.
- [x] Current work: add regression tests for long semantic-incomplete spans that have audio-safe candidates.
- [x] Current work: run targeted and full validation.

- [x] Current work: record investigation for Arabic long sentence segment `206.77s-287.935s`.
- [x] Current work: inspect final sentence, semantic sentences, VAD segments and boundary decisions around the long span.
- [x] Current work: identify which merge/keep decision created the long segment.
- [x] Current work: summarize root cause and recommend a targeted fix.

- [x] Current work: record decision to remove TextGrid overlap fallback.
- [x] Current work: make TextGrid export fail on overlapping intervals instead of silently rewriting times.
- [x] Current work: keep upstream pure-punctuation sentence fix as the real prevention.
- [x] Current work: adjust output tests and run validation.

- [x] Current work: record TextGrid overlap failure when rebuilding exports from existing JSON.
- [x] Current work: inspect the Arabic overlap example to identify why sentence intervals overlap.
- [x] Current work: make TextGrid sentence intervals non-overlapping even for old JSON files.
- [x] Current work: add regression coverage for overlapping JSON-derived TextGrid intervals.
- [x] Current work: run targeted and full validation.

- [x] Current work: record rerun-skip/export-rebuild and batch-error handling plan before coding.
- [x] Current work: inspect single-run and batch output naming rules.
- [x] Current work: skip model inference when per-utt JSON already exists and rebuild missing CSV/SRT/TextGrid from JSON.
- [x] Current work: skip an item entirely when JSON/CSV/SRT/TextGrid are all present.
- [x] Current work: write per-item batch `error.json` with wav path and traceback when inference fails, then continue.
- [x] Current work: add regression tests and run validation.

- [x] Current work: record Russian TextGrid overlap failure before coding.
- [x] Current work: inspect output interval generation and cut-time overlap handling.
- [x] Current work: make final writer intervals non-overlapping while preserving sentence text.
- [x] Current work: add regression coverage for overlapping cut times in TextGrid/exports.
- [x] Current work: run targeted validation.

- [x] Current work: record MMS numeric-like span placeholder expansion before coding.
- [x] Current work: group currency, percent, units and numeric range markers with adjacent digits for MMS alignment.
- [x] Current work: preserve restored surface tokens while aligning grouped numeric spans as `<star>`.
- [x] Current work: add regression tests and update MMS docs.
- [x] Current work: run targeted MMS validation.

- [x] Current work: fix overlapping sentence cut ranges when adjacent sentences share one raw VAD segment.
- [x] Current work: clip `cut_segments_ms` to sentence annotation bounds before SRT/CSV/TextGrid writing.
- [x] Current work: add regression coverage for TextGrid output with shared raw VAD island.

- [x] Current work: add `wav.scp` batch runner with configurable worker count.
- [x] Current work: distribute workers across `CUDA_VISIBLE_DEVICES`, defaulting to 8 device slots when unset.
- [x] Current work: write one JSON/CSV/SRT/TextGrid set per audio basename.
- [x] Current work: add unit tests for wav.scp parsing and GPU assignment.
- [x] Current work: run targeted validation.

- [x] Current work: make SRT, CSV and TextGrid sentence outputs use `cut_start_ms`/`cut_end_ms`.
- [x] Current work: add output writer regression tests with fallback to `start_ms`/`end_ms`.
- [x] Current work: run targeted output validation.

- [x] Current work: add VAD-derived `cut_start_ms`, `cut_end_ms` and `cut_segments_ms` to final sentences.
- [x] Current work: make hakka use continuous sentence annotations while preserving VAD-based cut ranges.
- [x] Current work: add regression tests for sentences spanning multiple raw VAD speech islands.
- [x] Current work: run targeted validation.

- [x] Current work: investigate hakka preserved-gap sentence cutting across multiple raw VAD segments.

- [x] Current work: add `preserve_sentence_gaps` config for final sentence timing.
- [x] Current work: preserve silence gaps when boundary fusion keeps an audio-safe boundary.
- [x] Current work: skip output VAD sentence expansion and final short-gap merge when preserving gaps.
- [x] Current work: add regression tests and run targeted validation.

- [x] Current work: design MMS numeric-token placeholder strategy without conflicting with inserted `<star>`.
- [x] Current work: implement numeric token placeholder alignment and restore original numeric tokens in timestamps.
- [x] Current work: add unit coverage for Korean numeric token preparation/restoration.
- [x] Current work: run MMS smoke on `data/test/0a41b2ee-4947-4192-9f9e-cf796d5dc955.wav` if a matching config is available.

- [x] Current work: add configurable final sentence merge by short silence gap and max merged duration.
- [x] Current work: apply final merge after boundary fusion, output VAD alignment and overlap removal.
- [x] Current work: add unit tests for gap/duration final sentence merge behavior.
- [x] Current work: run targeted validation.

- [x] Current work: summarize recent Codex-assisted complex work into a Markdown report.

- [x] Current work: record semantic-completeness boundary strategy before coding.
- [x] Current work: make VAD silence/probability/acoustic valley an audio-safe signal, not a mandatory split.
- [x] Current work: add semantic completeness checks for continuation punctuation and short raw-VAD fragments.
- [x] Current work: add pt_br regression for the reported `O primeiro ponto...` fragment sequence.
- [x] Current work: rerun pt_br raw-align smoke and inspect the corrected sentence merge.
- [x] Current work: update docs/PROGRESS/TODO, validate and commit.

- [x] Current work: record raw-VAD ASR/MMS alignment strategy before coding.
- [x] Current work: make raw VAD segments feed ASR and MMS by default.
- [x] Current work: keep `use_star=False` fixed for MMS forced alignment.
- [x] Current work: add regression tests that ASR/MMS segments use raw VAD and `asr_vad_segments_ms` equals raw VAD by default.
- [x] Current work: run pt_br Ten-VAD + Whisper + MMS smoke and inspect timestamp/VAD boundaries.
- [x] Current work: update docs/PROGRESS/TODO, validate and commit.

- [x] Current work: record frame-level VAD probability implementation plan before coding.
- [x] Current work: add optional `frame_speech_probs` output for FireRed VAD, Ten-VAD and Silero VAD.
- [x] Current work: pass VAD frame probabilities through pipeline JSON and sentence-boundary fusion.
- [x] Current work: update boundary fusion to prefer VAD probability valleys and make `max_sentence_s` soft.
- [x] Current work: add MMS Korean/Japanese tokenization and probability-boundary unit tests.
- [x] Current work: run targeted real smoke tests for `pt_br`, `ar_sa` and Silero VAD, then validate and update docs/progress.

- [x] Current work: inspect `ten_vad_utils.py` and current VAD adapter contracts.
- [x] Current work: add a Ten-VAD adapter and registry/language-support entry.
- [x] Current work: document Ten-VAD GitHub installation and add a standalone output test.
- [x] Current work: add a `ten-vad + whisper + mms + asr_native` Portuguese config.
- [x] Current work: run `data/test/short/pt_br-short.wav` through the new config and inspect JSON/exports.
- [x] Current work: run validation and update PROGRESS/TODO.
- [ ] Current work: commit the Ten-VAD integration after external Git write approval is available.

- [x] Current work: inspect `data/test/short` 300s fixtures and match them to language configs.
- [x] Current work: enable sentence-boundary fusion for all checked-in configs.
- [x] Current work: run every short fixture and write JSON outputs.
- [x] Current work: inspect sentence-boundary decisions, timestamp fields and overlap counts.
- [x] Current work: update docs/PROGRESS/TODO and commit the all-language boundary-fusion run.

- [x] Current work: inspect `data/test` language fixtures and existing top-level configs.
- [x] Current work: create or select one config per available test language.
- [x] Current work: run each language for five minutes and write JSON outputs.
- [x] Current work: inspect output shape, timestamp fields and sentence-boundary validity.
- [x] Current work: update PROGRESS/TODO and commit the multilingual 5-minute config run.

- [x] Current work: move config profiles to top-level `configs` and update all config references.
- [x] Current work: add timestamp-provider outputs to the final JSON in a segment-grouped form.
- [x] Current work: update docs/examples/tests for the top-level config layout and timestamp JSON field.
- [x] Current work: run validation and update PROGRESS/TODO.

- [x] Current work: inspect `l2s` references and top-level/docs/tests/examples path conflicts.
- [x] Current work: delete the obsolete top-level `l2s` directory.
- [x] Current work: move `semantic_asr/docs`, `semantic_asr/tests` and `semantic_asr/examples` to top-level directories.
- [x] Current work: update imports, commands, links and documentation for the new layout.
- [x] Current work: run full validation and update PROGRESS/TODO for the restructure.
- [ ] Current work: commit the validated restructure after external Git write approval is available.

- [x] Current work: inspect the reported `350.485s -> 350.525s` and `202.797s -> 203.037s` Arabic boundaries and their fusion decisions.
- [x] Current work: make active-speech safety override target sentence duration when under the maximum duration.
- [x] Current work: add regression coverage and regenerate the Arabic boundary-fusion outputs.
- [x] Current work: update docs/PROGRESS/TODO and commit the correction.

- [x] Current work: design configurable audio-safe sentence-boundary fusion contracts and inspect config/test patterns.
- [x] Current work: implement punctuation + raw VAD + aligned-token-gap sentence fusion with boundary metadata.
- [x] Current work: add focused regression tests and enable the strategy for `ar.json`.
- [x] Current work: rerun `ar_sa-short.wav` on GPU and inspect the new output artifacts.
- [x] Current work: update experiment docs/PROGRESS/TODO and commit the validated implementation.

- [x] Current work: inspect the three reported Arabic sentence boundaries against raw VAD, semantic VAD, output VAD and word timestamps.
- [x] Current work: determine a robust policy for sentence boundaries that fall inside active speech.
- [x] Current work: document the recommendation and defer changing default output behavior until multilingual evaluation.

- [x] Current work: inspect `configs/ar.json` and `data/test/ar_sa-short.wav`.
- [x] Current work: run the Arabic canonical-language pipeline on a real GPU.
- [x] Current work: inspect JSON/CSV/SRT/TextGrid outputs and record the result.
- [x] Current work: fix overlapping sentence intervals that block TextGrid export.
- [x] Current work: update PROGRESS/TODO and commit the experiment record.

- [x] Current work: audit model adapters for inconsistent language input formats, including MMS default `zh`.
- [x] Current work: define canonical project language ids such as `zh_cn` and centralized model-native mappings.
- [x] Current work: make adapters/config profiles accept canonical ids and convert internally.
- [x] Current work: add mapping/query/config regression tests and update docs/decisions/progress.
- [x] Current work: run full validation.
- [x] Current work: commit unified language configuration.

- [x] Current work: correct Qwen3-ASR supported-language list and full-name inference parameters.
- [x] Current work: restrict Fun-ASR-Nano language support to Chinese, English and Japanese.
- [x] Current work: update language-query tests, model docs, lessons and progress.
- [ ] Current work: commit the validated language-support correction after external git-write approval is available.

- [x] Current work: investigate why FireRedPunc previously worked but now fails to load.
- [x] Current work: correct Thai punctuation support assumptions and multilingual smoke report.
- [x] Current work: design a Thai semantic sentence-boundary strategy without XLM-R punctuation.
- [x] Current work: rerun Thai audio smoke tests after the user fixed `th_th.wav`.
- [x] Current work: update docs/catalog/tests/PROGRESS/TODO and run final validation.

- [x] Current work: record Dolphin GitHub-version and Seamless local-model retest task.
- [x] Current work: verify Dolphin installation source/version and locate Seamless checkpoint.
- [x] Current work: rerun Dolphin multilingual word-timestamp tests with the GitHub package.
- [x] Current work: run Seamless M4T multilingual ASR smoke tests with the local checkpoint.
- [x] Current work: update multilingual test report, PROGRESS and TODO with final results.
- [x] Current work: run final compile/unit/diff validation.

- [x] Current work: record complete multilingual model test task before running tests.
- [x] Current work: inspect available language audio durations and current model test CLIs.
- [x] Current work: fix language-region lookup so `ja_jp` can match `ja` model entries.
- [x] Current work: build a coverage matrix for VAD, ASR, timestamp and punctuation modules.
- [x] Current work: run lightweight code/unit validation before real-model tests.
- [x] Current work: run real smoke tests for currently available VAD/ASR/timestamp/punctuation models.
- [x] Current work: fix FunASR batch fallback for installed versions without batch decoding.
- [x] Current work: normalize Qwen3-ASR integer waveform input to float32.
- [x] Current work: replace Dolphin PyPI dependency/docs with GitHub installation and mark Dolphin results for retest.
- [x] Current work: collect outputs/logs into dated experiment directories.
- [x] Current work: summarize pass/fail issues in the multilingual smoke-test report.
- [x] Follow-up: rerun Dolphin tests after the user installs the official GitHub package.
- [x] Follow-up: run Seamless M4T tests after the user copies the local model.
- [x] Follow-up: resolve FireRedPunc torch/transformers checkpoint-loading compatibility.
- [x] Current work: update PROGRESS/TODO for completed tests and pending follow-ups.

- [x] Current work: record test-audio data matrix task before writing docs.
- [x] Current work: skip Dolphin/Whisper path changes because the user will handle those paths.
- [x] Current work: derive minimal language/audio test matrix from current model catalog.
- [x] Current work: document the test-audio language matrix under `docs`.
- [x] Current work: run a lightweight status/diff review after the test-audio matrix.
- [x] Current work: update PROGRESS/TODO review for test data needs.

- [x] Current work: record MMS adapter correction: no temp segment wav and no external `l2s` package dependency.
- [x] Current work: extract minimal MMS alignment runtime into `semantic_asr`.
- [x] Current work: change MMS forced aligner adapter to align in-memory `SpeechSegment.wav`.
- [x] Current work: update MMS docs/tests after internal runtime extraction.
- [x] Current work: validate compile/tests after MMS in-memory alignment change.
- [x] Current work: update PROGRESS/DECISIONS/TODO review for MMS in-memory alignment.

- [x] Current work: record MMS forced aligner timestamp provider plan before coding.
- [x] Current work: inspect local `l2s` ALIGNER API and existing timestamp provider contract.
- [x] Current work: add MMS forced aligner timestamp provider adapter.
- [x] Current work: register MMS forced aligner and add language support metadata.
- [x] Current work: add MMS forced aligner docs and standalone skip-load test.
- [x] Current work: validate compile/tests after MMS forced aligner changes.
- [x] Current work: update PROGRESS/DECISIONS/TODO review for MMS forced aligner.

- [x] Current work: record XLM-R punctuation language support correction.
- [x] Current work: update XLM-R punctuation support to the full 47-language list including Chinese.
- [x] Current work: update XLM-R punctuation docs and query tests.
- [x] Current work: validate compile/tests/query after language correction.
- [x] Current work: update PROGRESS/TODO review after correction.

- [x] Current work: record Dolphin word timestamp fix and XLM-R punctuation model plan before coding.
- [x] Current work: verify `1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase` usage and language support.
- [x] Current work: replace Dolphin timestamp param with `word_timestamp`.
- [x] Current work: add XLM-R punctuation/truecase adapter.
- [x] Current work: register XLM-R punctuation model and language support metadata.
- [x] Current work: add docs and standalone test for XLM-R punctuation model.
- [x] Current work: validate compile/tests/query after punctuation adapter changes.
- [x] Current work: update `PROGRESS.md`, `DECISIONS.md` and TODO review.

- [x] Current work: record language/model mapping and new ASR model plan before coding.
- [x] Current work: verify official language/capability notes for Qwen3-ASR, Dolphin and Seamless M4T v2 large.
- [x] Current work: add model-to-language support metadata.
- [x] Current work: add language-to-model query API/CLI.
- [x] Current work: add Qwen3-ASR-1.7B ASR adapter docs and standalone output test skeleton.
- [x] Current work: add Dolphin ASR adapter docs and standalone output test skeleton.
- [x] Current work: add Seamless M4T v2 large ASR adapter docs and standalone output test skeleton.
- [x] Current work: validate compile/tests after language catalog changes.
- [x] Current work: update `PROGRESS.md`, `DECISIONS.md` and TODO review.

- [x] Current work: record restructure plan before coding.
- [x] Current work: restore deleted `PROGRESS.md` with latest progress.
- [x] Current work: rename the standalone package to `semantic_asr`.
- [x] Current work: update imports, configs, examples, docs and architecture references after rename.
- [x] Current work: refresh top-level `README.md` for the standalone semantic ASR project.
- [x] Current work: refresh top-level `requirements.txt` for the current environment.
- [x] Current work: validate compile/tests/help after structure cleanup.
- [x] Current work: update `PROGRESS.md`, `DECISIONS.md` and TODO review.

- [x] Read `AGENT.md` and align with its workflow.
- [x] Keep the original `fireredasr2s` package unchanged for this abstraction.
- [x] Create a separate `semantic_asr` project directory.
- [x] Extract model-agnostic long-audio semantic ASR orchestration.
- [x] Add a FireRed adapter as the first concrete model-family bridge.
- [x] Verify the standalone pipeline with fake components.
- [x] Move generated abstraction work from `main` to `red-asr`.
- [x] Remove optional no-VAD, no-timestamp and no-punctuation paths.
- [x] Make VAD, ASR, timestamp provider and punctuation mandatory.
- [x] Redesign timestamp as a provider stage supporting ASR-native timestamps and forced alignment.
- [x] Add Silero VAD adapter with install/download documentation and standalone output test.
- [x] Add Fun-ASR-Nano-2512 adapter with install/download documentation and standalone test script.
- [x] Verify Fun-ASR-Nano-2512 real model output after the model download completes.
- [x] Build and run Silero VAD + Fun-ASR-Nano timestamp + FireRedPunc experiment on `data/test/short.wav`.
- [x] Add CSV, SRT and TextGrid output support inside `semantic_asr`.
- [x] Copy the FireRed runtime code needed by `semantic_asr` into the standalone project boundary.
- [x] Redesign punctuation handling for ASR models that can emit punctuation natively.
- [x] Normalize optional punctuation stripping before re-punctuation.
- [x] Add Whisper large ASR adapter documentation and standalone output-shape test.
- [x] Check batch inference support for newly added models and update Fun-ASR-Nano batching if available.
- [x] Verify the updated pipeline on the 60-second `data/test/short.wav`.
- [x] Add language-based pipeline/module configuration support.
- [x] Add VAD segment post-processing merge policy with min 10s, max 40s and max merge gap 3s.
- [x] Add FireRed VAD + Whisper + Qwen3-ForcedAligner + ASR-native punctuation Russian experiment config/script.
- [x] Check whether Qwen3-ForcedAligner-0.6B is available locally and document its environment/model setup.
- [x] Run the Russian `data/test/ru_ru.wav` experiment and write JSON/CSV/SRT/TextGrid outputs.
- [x] Add configurable micro-silence VAD merge for gaps below 500ms.
- [x] Add configurable 100ms VAD segment padding with half-gap allocation when adjacent padding would overlap.
- [x] Verify VAD post-processing helper behavior and compile the pipeline.
- [x] Move micro-silence merge and padding from ASR VAD slicing to final non-speech segment output only.
- [x] Keep ASR VAD slicing on semantic 10s/30s/3s merge policy.
- [x] Verify output VAD formatting helpers and compile the pipeline after the separation.
- [x] Align sentence boundaries to final output VAD segment ranges for JSON/TextGrid/SRT/CSV consistency.
- [x] Draft registry/config-driven architecture diagram for pipeline composition.
- [x] Draft TODO for moving from one-script-per-combination to registry plus config runner.
- [x] Generate draw.io architecture diagram at `docs/architecture.drawio`.

## Next Architecture TODO

- [x] Current work: add executable registry/config runner plan before coding.
- [x] Current work: implement component registry contracts and config loader.
- [x] Current work: implement a single config-driven `run_pipeline.py` entrypoint.
- [x] Current work: add config profiles for the three existing model combinations.
- [x] Current work: add fake registry/config smoke tests.
- [x] Current work: update progress/decision notes after validation.
- [x] Define component registry contracts for `vad`, `asr`, `timestamp`, and `punc`.
- [x] Split current adapters into role-oriented modules or registration entries.
- [x] Add config schema for pipeline profiles, including model name, device, language, batching, VAD merge policy, output VAD policy, and output writers.
- [x] Implement a config loader that supports YAML and JSON.
- [x] Implement a single `run_pipeline.py` entrypoint that builds a pipeline from config.
- [x] Convert existing combination scripts into config files:
  - [x] Silero VAD + Fun-ASR-Nano + FireRedPunc
  - [x] Silero VAD + Whisper large + ASR-native punctuation
  - [x] FireRed VAD + Whisper large + Qwen3-ForcedAligner + Whisper text punctuation
- [x] Copy the resolved config into each output directory for experiment provenance.
- [x] Add standalone tests for registry resolution and config validation.
- [x] Add smoke tests that run fake registered components from config.
- [x] Keep existing example scripts temporarily as compatibility wrappers around `run_pipeline.py`.

## Review

- Changed the default ASR/timestamp input strategy to raw VAD segments:
  `PipelineConfig.merge_vad_segments=False`.
- Kept explicit legacy merge support with `pipeline.merge_vad_segments=true`.
- Fixed MMS forced alignment to keep `use_star=False` even if a config passes
  `use_star=true`.
- Added regression tests for default raw-VAD ASR/timestamp segments, explicit
  merged VAD behavior and forced MMS `use_star=False`.
- Real `pt_br` raw-align smoke passed:
  `raw_vad_segments_ms == asr_vad_segments_ms`, 37 ASR/MMS segments, 36
  timestamp segments, 574 words, 39 final sentences and zero sentence/word
  overlaps.
- Updated MMS docs and the Portuguese TEN VAD experiment report for the new
  raw-align strategy.

- Added frame-level VAD speech probability plumbing:
  `frame_speech_probs` from VAD adapters and `vad_frame_speech_probs` in final
  JSON.
- FireRed VAD now exposes existing 10ms/25ms model probabilities, TEN VAD
  exposes 16ms frame probabilities and Silero VAD collects 32ms window
  probabilities.
- Sentence-boundary fusion now uses VAD probability valleys as primary
  acoustic support, records probability diagnostics in
  `sentence_boundary_decisions`, and treats `max_sentence_s` as a soft
  preference instead of a hard active-speech cut.
- MMS forced aligner token preparation now covers Korean and Japanese
  character splitting in addition to Chinese.
- Real smoke checks passed:
  `pt_br` TEN VAD + Whisper + MMS produced 18,750 probability frames, 24 final
  sentences and zero overlaps; `ar_sa` 60s FireRed VAD + MMS produced 5,998
  probability frames and zero overlaps; Silero VAD returned 938 probability
  frames on a 30s `en_us` sample.
- Validation passed: 45 tests, compile, config parsing and `git diff --check`.

- Added `semantic_asr.adapters.ten_vad.TenVadAdapter` and registered it as
  `vad.ten_vad`.
- Added TEN VAD language-support metadata, model docs with the GitHub install
  command, a standalone `examples/test_ten_vad.py` output test and focused
  post-processing unit tests.
- Added `configs/tenvad_whisper_mms_nativepunc_pt_br.json` for
  TEN VAD + Whisper large + MMS Forced Aligner + ASR-native punctuation.
- Ran the full Portuguese 300s fixture:
  `data/test/short/pt_br-short.wav` -> `output/experiments/tenvad_whisper_mms_nativepunc_pt_br`.
- Output metrics: 37 raw TEN VAD segments, 12 ASR VAD segments, 12 timestamp
  segments, 562 words, 31 semantic candidates, 30 final sentences, one
  active-speech merge, and zero sentence/word overlaps.
- Validation passed: Ten-VAD standalone 30s test, config parsing, `pt_br` VAD
  query includes `ten_vad`, 40 tests and package/tests/examples compile.
- Quality caveat: the SRT has a few fused Portuguese tokens such as `poisé`,
  likely from the Whisper + MMS tokenization path; the structural pipeline
  output is valid.

- Enabled `sentence_boundary_fusion` for every `configs/*.json` profile,
  including the language-named profiles and older combination profiles.
- Ran the nine available `data/test/short/*-short.wav` fixtures through
  `configs/<language>.json` and wrote JSON/JSONL/CSV/SRT/TextGrid outputs under
  `output/experiments/multilingual_short_boundary_fusion/<language>/`.
- Output summary: `ar_sa` 20 sentences from 29 semantic candidates, `en_us` 36
  from 51, `hi_in` 14 from 14, `ja_jp` 23 from 27, `ko_kr` 27 from 30,
  `pt_br` 28 from 32, `ru_ru` 43 from 45, `th_th` 1 from 1, and `vi_vn` 20
  from 20.
- All nine outputs include `timestamp_segments`; final sentence and word
  overlap counts are zero for every fixture.
- Noted fixture caveat: `th_th-short.wav` is 16.17s in the current test set,
  while the other short clips are about 300s.
- Validation passed: all configs confirm fusion enabled, 37 tests, package
  compile.

- Added language-named configs for all nine available `data/test` language
  fixtures and ran each one with `--max_seconds 300`.
- Wrote JSON outputs under `output/experiments/multilingual_5min/<language>/`;
  all include `timestamp_segments`, and every language has zero sentence/word
  timestamp overlaps.
- Switched `ja_jp`, `pt_br` and `vi_vn` coverage configs to Whisper native
  timestamps. The initial Japanese Seamless + MMS attempt failed with a CTC
  target-length error.
- Added `docs/experiments/multilingual_5min_20260605.md` with config choices,
  output paths and per-language counts.
- Validation passed: all language-named configs parse, 37 tests, compile and
  `git diff --check`.
- Moved config profiles from `semantic_asr/configs` to top-level `configs/`
  and updated README, experiment docs and example wrapper paths.
- Final JSON now includes `timestamp_segments`, preserving timestamp-provider
  output by ASR segment with segment text, confidence, relative seconds and
  absolute millisecond token timestamps.
- Validation passed: all six top-level config profiles parse, 37 tests,
  package/tests/examples compile, `git diff --check`, wrapper `--help`, and a
  fake runner JSON check for `timestamp_segments`.
- Removed the obsolete top-level `l2s/`; MMS forced alignment continues to use
  the extracted `semantic_asr.mms_runtime`.
- Moved package-adjacent resources to top-level `docs/`, `tests/` and
  `examples/`, leaving `semantic_asr/` focused on runtime code and configs.
- Updated README layout, documentation commands, example import/config paths
  and current progress references.
- Validation passed: 37 tests, package/tests/examples compile,
  `git diff --check`, every example `--help`, all compatibility wrapper config
  paths, and five standalone `--skip_model_load` model examples.
- The validated restructure remains uncommitted because the platform rejected
  the external Git write request after reaching its current usage limit.
- Corrected fusion priority so active-speech safety overrides the soft target
  duration whenever the merged sentence stays below the hard maximum.
- Added regression coverage for the reported `202.797-203.037` and
  `350.485-350.525` Arabic active-speech boundaries.
- The complete V2 Arabic retest emits 41 audio-safe sentences from 54 semantic
  candidates, merges 13 active-speech boundaries, has no remaining boundary
  satisfying the merge criteria, and keeps the longest sentence at 27.87s.
- Full validation passed: 37 tests, package compile and `git diff --check`.
- Implemented configurable sentence-boundary fusion using punctuation
  candidates, raw VAD silence, aligned-token gaps, waveform valley ratio and
  target/maximum sentence durations.
- Enabled the strategy only for `configs/ar.json`; other language
  profiles retain their current behavior.
- The full Arabic retest preserved 54 `semantic_sentences` and produced 47
  audio-safe `sentences`: seven active-speech boundaries merged, six nearby
  VAD-silence boundaries snapped, no sentence/word overlaps, and a 27.87s
  maximum sentence duration.
- The reported region is now `73.770-87.418` plus `87.418-97.920`, matching
  the intended active-speech merge and VAD-supported split.
- Full validation passed: 36 tests, package compile and `git diff --check`.
- Investigated the reported Arabic boundaries using raw VAD, aligned-word gaps
  and local waveform energy.
- Confirmed that `83.494s -> 83.634s` cuts active speech inside one raw VAD
  segment, while `87.396s -> 87.496s` is supported by a raw VAD silence gap.
- Added `docs/sentence_boundary_fusion.md`, recommending that
  punctuation propose semantic boundaries while raw VAD, token pauses,
  waveform energy and duration limits determine audio-safe boundaries.
- Deferred changing default sentence merging until the policy is evaluated on
  multilingual reference fixtures; a simple 200ms same-VAD rule would merge
  12 boundaries in the current Arabic result.
- The full ten-minute Arabic profile passed with FireRed VAD, Seamless M4T v2
  large, MMS Forced Aligner and XLM-R punctuation.
- Final outputs contain 54 non-overlapping sentences and 699 non-overlapping
  aligned words; JSON, JSONL, CSV, SRT, TextGrid and resolved config were
  written successfully.
- Fixed overlapping adjacent sentence intervals introduced by final output VAD
  alignment at the shared result layer, with a regression test.
- The Arabic transcript contains 14 `<UNK>` tokens and awkward punctuation, so
  integration passed but quality approval still requires a reference
  transcript.
- Full validation passed: 30 tests, package compile and `git diff --check`.
- Added centralized `semantic_asr.language_mapping` for canonical profile ids and model-native values.
- Pipeline profiles now require one canonical top-level language such as `zh_cn`; language-aware ASR/timestamp components receive it automatically.
- MMS forced aligner defaults to `zh_cn`, and a regression test verifies that the MMS runtime receives native `cmn`.
- Migrated all existing profiles and standalone model examples away from model-native language fields.
- Added unified-language documentation and model-specific documentation updates.
- All existing profiles parsed with canonical ids; model mapping checks passed.
- Full validation passed: 29 tests, package compile and `git diff --check`.
- Corrected Qwen3-ASR catalog support to the specified 30 languages.
- Qwen3-ASR converts short/region codes to the full language names required by the official model API; `zh_cn`, `th_th` and `yue` map to `Chinese`, `Thai` and `Cantonese`.
- Restricted Fun-ASR-Nano and `funasr_native` timestamp support to Chinese, English and Japanese.
- Multilingual smoke tests now run Fun-ASR-Nano only for supported fixtures and explicitly set Qwen language per fixture.
- Focused language/adapter tests passed: 13 tests.
- Full validation passed: 21 tests, package compile and `git diff --check`.
- A focused real Qwen Thai GPU retest could not run because the platform rejected the external GPU execution request due to its current usage limit.
- The final git commit could not be created because the workspace `.git` directory requires external write approval and the platform rejected that approval due to its current usage limit.
- FireRedPunc real inference passed after adding a scoped trusted-local legacy BERT checkpoint loader for the current `torch==2.1.0` and `transformers==4.57.6` environment.
- XLM-R punctuation smoke tests now cover only supported languages; Thai was removed.
- Added `docs/thai_sentence_boundary.md` with the recommended pause-and-duration Thai sentence-boundary strategy.
- Corrected `th_th.wav` passed interface smoke tests for Silero VAD, FireRed VAD, FunASR, Whisper, Qwen3-ASR, Qwen3 ForcedAligner, MMS Forced Aligner, Dolphin and Seamless M4T.
- FunASR returned timestamps but transcribed the corrected Thai sample as Chinese; it should not be treated as a quality-approved Thai ASR choice.
- `conda run -n fireredasr2s python -m compileall -q semantic_asr` passed.
- `conda run -n fireredasr2s python -m unittest discover -s tests -p 'test_*.py'` passed: 18 tests.
- `git diff --check` passed.
- `conda run -n fireredasr2s python -m compileall semantic_asr` passed.
- Fake VAD/ASR/Punc smoke test passed with `CUDA_VISIBLE_DEVICES=4,5,6,7`.
- Fake mandatory VAD/ASR/Timestamp/Punc smoke test passed with `CUDA_VISIBLE_DEVICES=4,5,6,7`.
- Fake ASR-native timestamp provider and forced-aligner timestamp provider smoke tests passed.
- `silero-vad` and `tiktoken` were installed in the `fireredasr2s` env.
- Silero VAD standalone test passed on the first 30 seconds of `data/test/conf_0002_0002_001003.wav`.
- Fun-ASR-Nano standalone test passed on `data/test/short.wav`.
- Silero VAD + Fun-ASR-Nano timestamp + FireRedPunc experiment passed on `data/test/short.wav`.
- Output artifacts were written under `output/experiments/silero_funasr_fireredpunc`: `short.json`, `result.jsonl`, `asr_csv/short.csv`, `asr_srt/short.srt`, and `asr_tg/short.TextGrid`.
- Added vendored FireRed ASR/VAD/Punc runtime code under `semantic_asr/firered_runtime`.
- Added ASR-native punctuation strategy and punctuation stripping before external re-punctuation.
- Updated Fun-ASR-Nano to batch temporary wav-path inputs in one `AutoModel.generate` call when the pipeline ASR batch size is greater than one.
- Whisper large standalone test passed on `data/test/short.wav` first 30 seconds.
- Silero VAD + Fun-ASR-Nano timestamp + FireRedPunc passed on 60-second `data/test/short.wav` with `asr_batch_size=8`.
- Silero VAD + Whisper large timestamp + ASR-native punctuation passed on 60-second `data/test/short.wav`.
- Added language profiles for `zh`, `en` and `ru`.
- Added common VAD merge post-processing: max 40s, merge gaps no larger than 3s, target at least 10s when possible.
- Installed `qwen-asr==0.0.6` from the official Qwen3-ASR GitHub repository; this also installed `transformers==4.57.6`.
- Added a torch pytree compatibility shim for the current `torch==2.1.0+cu118` plus `transformers==4.57.6` environment.
- Full `data/test/ru_ru.wav` Russian experiment passed with FireRed VAD + Whisper large + Qwen3-ForcedAligner + Whisper text punctuation.
- Russian output artifacts were written under `output/experiments/fireredvad_whisper_qwenaligner_textpunc_ru_full`: `ru_ru.json`, `result.jsonl`, `asr_csv/ru_ru.csv`, `asr_srt/ru_ru.srt`, and `asr_tg/ru_ru.TextGrid`.
- VAD helper checks passed for micro-silence merge, semantic merge and 100ms padding with half-gap allocation.
- Verified that ASR VAD slicing remains unpadded while final output `vad_segments_ms` applies micro-silence merge and 100ms padding.
- Verified sentence boundary alignment expands only the first and last sentence in each final output VAD segment.
- `conda run -n fireredasr2s python -m compileall semantic_asr` passed after adding the config-driven runner.
- `conda run -n fireredasr2s python -m unittest tests` passed.
- Parsed all config profiles under `configs`: Silero + Fun-ASR + FireRedPunc, Silero + Whisper + ASR-native punctuation, and FireRed VAD + Whisper + Qwen3 forced aligner + ASR text punctuation.
- Existing combination example scripts now load config profiles and call the shared `run_profile()` runner.
- `--help` passed for `semantic_asr/run_pipeline.py` and all three compatibility wrapper scripts.
- Restored deleted `PROGRESS.md`.
- Renamed the standalone package to `semantic_asr`.
- Added a top-level standalone project `README.md` and refreshed `requirements.txt` for the current `fireredasr2s` environment.
- `conda run -n fireredasr2s python -m compileall semantic_asr` passed after the rename.
- `conda run -n fireredasr2s python -m unittest tests` passed after the rename.
- `--help` passed for `semantic_asr/run_pipeline.py` and all three compatibility wrapper scripts after the rename.
- Parsed all config profiles under `configs` after the rename.
- Added `semantic_asr/language_support.py` with model-to-language metadata and language-to-model query helpers.
- Added `semantic_asr/query_models.py` CLI; `language en_us` and `model qwen3_asr_1_7b --role asr` returned expected JSON.
- Added Qwen3-ASR-1.7B, Dolphin and Seamless M4T v2 large ASR adapters.
- Added docs and standalone `--skip_model_load` tests for Qwen3-ASR-1.7B, Dolphin and Seamless M4T v2 large.
- `conda run -n fireredasr2s python -m compileall semantic_asr` passed after adding language support and new ASR adapters.
- `conda run -n fireredasr2s python -m unittest discover -s tests -p 'test_*.py'` passed.
- Corrected `xlm_roberta_punctuation` language support to the full 47-language list provided by the user, including Chinese.
- Added a lesson to prefer explicit supported-language lists over incomplete model page tags.
- `conda run -n fireredasr2s python -m unittest tests` passed after the correction.
- `semantic_asr/query_models.py language zh_cn --role punc` now returns `xlm_roberta_punctuation`.
- Added MMS forced aligner timestamp provider `mms_forced_aligner`.
- Added MMS docs and standalone skip-load test that verifies Chinese character token splitting.
- `conda run -n fireredasr2s python -m unittest discover -s tests -p 'test_*.py' tests` passed.
- `semantic_asr/query_models.py language zh_cn --role timestamp` returns `mms_forced_aligner`.
- Refactored MMS forced aligner to use vendored `semantic_asr.mms_runtime` and align in-memory `SpeechSegment.wav` audio without writing temporary segment wav files.
