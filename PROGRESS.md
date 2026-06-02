# Progress

Current state:

- Created `sentence_asr_pipeline` as a separate abstraction directory.
- Restored the original `fireredasr2s` package from the earlier in-package extraction attempt.
- Added model-agnostic core pipeline and FireRed adapter.
- Verified the standalone pipeline with fake components.
- Moved the abstraction work from `main` to `red-asr`; local `main` points back to `origin/main`.
- Tightened the abstraction so VAD, ASR, timestamp predictor and punctuation are mandatory.

Next step:

- Add adapters for the next target VAD, ASR, timestamp predictor or punctuation model.
