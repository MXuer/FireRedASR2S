# Progress

Current state:

- The project is being reshaped from the original FireRedASR2S repository into a standalone multilingual semantic ASR pipeline project.
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
- ASR VAD slicing uses semantic segment merge defaults: target at least 10s where possible, max 30s, and no merge across gaps above 3s.
- Final output VAD formatting is separate from ASR slicing: short non-speech gaps can be merged, segments can be padded, and sentence boundaries are aligned to output VAD ranges.
- Output writers support JSON, JSONL, CSV, SRT and TextGrid.
- Added a registry/config composition layer:
  - component factories are registered by role: `vad`, `asr`, `timestamp`, and `punc`;
  - JSON/YAML pipeline profiles are validated and used to build `SemanticAsrPipeline`;
  - `run_pipeline.py` provides one config-driven CLI;
  - each run writes `resolved_config.json` by default.
- Added config profiles for:
  - Silero VAD + Fun-ASR-Nano native timestamps + FireRedPunc;
  - Silero VAD + Whisper large native timestamps + ASR-native punctuation;
  - FireRed VAD + Whisper large + Qwen3-ForcedAligner + ASR text punctuation for Russian.
- Existing combination example scripts are compatibility wrappers around the config-driven runner.
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
  preferred acoustic signal. `max_sentence_s` is a soft preference and no
  longer forces a split through high-probability active speech.

Recent validation:

- Added frame-level VAD speech probability to the pipeline JSON as
  `vad_frame_speech_probs` and passed it into sentence-boundary fusion.
- FireRed VAD writes 10ms-shift / 25ms-frame probabilities, TEN VAD writes
  16ms probabilities and Silero VAD writes 32ms window probabilities.
- Boundary decisions now include `speech_prob_min`, `speech_prob_mean`,
  `speech_prob_max`, `speech_prob_boundary_ms` and
  `speech_prob_supported_silence`.
- Boundary-fusion reasons now include `vad_prob_valley`,
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
