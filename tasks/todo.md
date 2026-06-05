# Todo

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
