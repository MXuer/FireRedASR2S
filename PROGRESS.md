# Progress

Current state:

- Created `sentence_asr_pipeline` as a separate abstraction directory.
- Restored the original `fireredasr2s` package from the earlier in-package extraction attempt.
- Added model-agnostic core pipeline and FireRed adapter.
- Verified the standalone pipeline with fake components.
- Moved the abstraction work from `main` to `red-asr`; local `main` points back to `origin/main`.
- Tightened the abstraction so VAD, ASR, timestamp provider and punctuation are mandatory.
- Redesigned the timestamp stage as `TimestampProvider` to support both ASR-native timestamps and external forced alignment.
- Added Silero VAD and Fun-ASR-Nano-2512 adapters, docs and runnable test examples.
- Silero VAD test passed on `data/test/conf_0002_0002_001003.wav` first 30 seconds.
- Fun-ASR-Nano standalone test passed on `data/test/short.wav`.
- Silero VAD + Fun-ASR-Nano timestamp + FireRedPunc experiment passed on `data/test/short.wav`.
- Experiment outputs were written as JSON, JSONL, CSV, SRT and TextGrid under `output/experiments/silero_funasr_fireredpunc`.
- Vendored the FireRed ASR/VAD/Punc runtime code needed by `sentence_asr_pipeline` under `sentence_asr_pipeline/firered_runtime`.
- Redesigned punctuation as a mandatory strategy stage: external punctuation can strip existing ASR punctuation, while ASR-native punctuation can preserve and split on native punctuation.
- Updated Fun-ASR-Nano to use batch `AutoModel.generate(input=[...])` when the pipeline ASR batch size is greater than one.
- Added Whisper large ASR adapter, docs and runnable tests.
- Verified 60-second `data/test/short.wav` with Silero + FunASR + FireRedPunc and Silero + Whisper large + ASR-native punctuation.

Next step:

- Review the 60-second FunASR and Whisper outputs and decide the next model adapter or forced-aligner integration.
