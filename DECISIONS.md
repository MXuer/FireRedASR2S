# Decisions

- The sentence ASR abstraction lives in `sentence_asr_pipeline`, not inside the existing `fireredasr2s` package.
- The core pipeline depends only on normalized component interfaces.
- Existing FireRed models are connected through `sentence_asr_pipeline.adapters.firered`.
- GPU assignment is left to each concrete model config; available GPUs are 4, 5, 6 and 7.
- The abstraction requires all four modules: VAD, ASR, timestamp predictor and punctuation.
- LID and no-VAD/no-timestamp/no-punctuation paths are outside the current abstraction.
