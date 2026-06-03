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

Recent validation:

- `conda run -n fireredasr2s python -m compileall semantic_asr` passed after the rename.
- `conda run -n fireredasr2s python -m unittest semantic_asr.tests.test_config_runner` passed after the rename.
- `--help` passed for `semantic_asr/run_pipeline.py` and all three compatibility wrapper scripts after the rename.
- All config profiles under `semantic_asr/configs` parsed successfully after the rename.
- Top-level `README.md` and `requirements.txt` were refreshed for the standalone project.

Next step:

- Run one short real-model config through `semantic_asr/run_pipeline.py` and compare output shape with the older wrapper output.
- Add nested CLI override support if ad-hoc experiment overrides become common.
