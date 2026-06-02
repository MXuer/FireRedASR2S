# Decisions

- The sentence ASR abstraction lives in `sentence_asr_pipeline`, not inside the existing `fireredasr2s` package.
- The core pipeline depends only on normalized component interfaces.
- Existing FireRed models are connected through `sentence_asr_pipeline.adapters.firered`.
- GPU assignment is left to each concrete model config; available GPUs are 4, 5, 6 and 7.
- The abstraction requires all four stages: VAD, ASR, timestamp provider and punctuation.
- LID and no-VAD/no-timestamp/no-punctuation paths are outside the current abstraction.
- The mandatory timestamp stage is represented by `TimestampProvider`, not only by a standalone predictor model.
- `TimestampProvider` can either validate ASR-native timestamps or run a forced aligner using ASR text plus the matching VAD audio segment.
