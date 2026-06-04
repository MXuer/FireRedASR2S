# Progress

Current state:

- The project is being reshaped from the original FireRedASR2S repository into a standalone multilingual semantic ASR pipeline project.
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
- Added `semantic_asr/docs/test_audio_matrix.md` with the minimum multilingual speech data needed to validate current module combinations:
  - must-have: `zh_cn`, `en_us`, `ru_ru`, `ja_jp`, `th_th`, `hi_in`;
  - optional expansion: `ar_sa`, `vi_in`, `ko_kr`, `bn_bd`, `pt_br`.
- Added the reusable multilingual real-model smoke runner at `semantic_asr/examples/run_multilingual_smoke_tests.py`.
- Completed the available-model multilingual smoke run documented in `semantic_asr/docs/experiments/multilingual_smoke_20260604.md`.
- Real smoke tests passed for Silero VAD, FireRed VAD, Fun-ASR-Nano, Whisper large, Qwen3-ASR, Qwen3 ForcedAligner, MMS Forced Aligner and XLM-R punctuation.
- Testing fixed adapter/runtime issues for FunASR batch fallback, Qwen3-ASR float waveform input, MMS forced alignment CUDA stability, XLM-R local ONNX loading and Dolphin output normalization.
- Dolphin official GitHub package version `20260513` passed grouped multilingual word-timestamp smoke tests after result normalization was fixed.
- Seamless M4T v2 large passed all-nine-language local-model smoke testing after adding 16 kHz resampling and defaulting `tgt_lang` to `src_lang`.
- FireRedPunc real inference passes again. Installing Qwen3-ASR upgraded Transformers, whose newer safe loader rejected the trusted legacy BERT `.bin` under torch 2.1; the vendored FireRedPunc runtime now loads and validates that local checkpoint directly.
- Corrected the XLM-R punctuation smoke test and report: Thai is not supported and is excluded.
- Added `semantic_asr/docs/thai_sentence_boundary.md`: Thai should preserve ASR text and derive semantic sentence boundaries from aligned-token pauses and duration limits instead of inventing punctuation.
- Replaced Thai audio retesting passed interface smoke tests for Silero VAD, FireRed VAD, Whisper large-v3, Qwen3-ASR, Qwen3 ForcedAligner, MMS Forced Aligner, Dolphin, Seamless M4T and FunASR; Dolphin returned 26 Thai word timestamps, while FunASR incorrectly transcribed the Thai sample as Chinese.

Recent validation:

- `conda run -n fireredasr2s python -m compileall semantic_asr` passed after the rename.
- `conda run -n fireredasr2s python -m unittest semantic_asr.tests.test_config_runner` passed after the rename.
- `--help` passed for `semantic_asr/run_pipeline.py` and all three compatibility wrapper scripts after the rename.
- All config profiles under `semantic_asr/configs` parsed successfully after the rename.
- Top-level `README.md` and `requirements.txt` were refreshed for the standalone project.
- `conda run -n fireredasr2s python -m unittest semantic_asr.tests.test_config_runner semantic_asr.tests.test_language_support` passed after adding language support metadata.
- `semantic_asr/query_models.py language en_us` and `semantic_asr/query_models.py model qwen3_asr_1_7b --role asr` returned expected JSON.
- New standalone scripts for Qwen3-ASR, Dolphin and Seamless M4T passed `--skip_model_load 1`.
- `conda run -n fireredasr2s python -m compileall semantic_asr` passed after the Dolphin/XLM-R punctuation changes.
- `conda run -n fireredasr2s python -m unittest semantic_asr.tests.test_config_runner semantic_asr.tests.test_language_support` passed after the Dolphin/XLM-R punctuation changes.
- `semantic_asr/examples/test_xlm_roberta_punctuation.py --skip_model_load 1` passed.
- `semantic_asr/query_models.py language en_us --role punc` returns `xlm_roberta_punctuation`.
- Corrected `xlm_roberta_punctuation` support to the full 47-language list provided by the user, including Chinese.
- `semantic_asr/query_models.py language zh_cn --role punc` now returns `xlm_roberta_punctuation`.
- `conda run -n fireredasr2s python -m unittest semantic_asr.tests.test_config_runner semantic_asr.tests.test_language_support semantic_asr.tests.test_mms_forced_aligner` passed after adding MMS forced aligner.
- `semantic_asr/examples/test_mms_forced_aligner.py --skip_model_load 1 --language zh_cn --text "你好 世界"` passed and produced Chinese character tokens.
- `semantic_asr/query_models.py language zh_cn --role timestamp` returns `mms_forced_aligner`.
- `conda run -n fireredasr2s python -m compileall semantic_asr` passed after MMS in-memory runtime extraction.
- `16` unit tests passed after multilingual smoke-test fixes.
- Real Qwen3-ASR -> Qwen3 ForcedAligner English alignment passed with timestamps.
- Real Qwen3-ASR -> MMS ForcedAligner Hindi alignment passed with timestamps.
- XLM-R punctuation real ONNX inference passed for its supported English, Russian, Japanese, Hindi and Arabic samples; Thai was removed because it is unsupported.
- Dolphin GitHub-version grouped tests passed for Arabic, Hindi, Japanese, Korean, Russian and Vietnamese with word timestamps; the selected Thai prefix was empty.
- Seamless M4T v2 large local-model tests passed all nine languages while preserving source-language output.

Next step:

- Implement and evaluate the Thai timestamp-and-pause sentence-boundary strategy.
- Build speech-bearing quality fixtures with reference transcripts instead of testing fixed file prefixes.
- Run one short real-model config through `semantic_asr/run_pipeline.py` and compare output shape with the older wrapper output.
- Add nested CLI override support if ad-hoc experiment overrides become common.
- Run real smoke tests for Qwen3-ASR, Dolphin and Seamless once local model paths/checkpoints are confirmed.
