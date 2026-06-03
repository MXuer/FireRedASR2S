# Todo

- [x] Read `AGENT.md` and align with its workflow.
- [x] Keep the original `fireredasr2s` package unchanged for this abstraction.
- [x] Create a separate `sentence_asr_pipeline` project directory.
- [x] Extract model-agnostic long-audio sentence ASR orchestration.
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
- [x] Add CSV, SRT and TextGrid output support inside `sentence_asr_pipeline`.
- [x] Copy the FireRed runtime code needed by `sentence_asr_pipeline` into the standalone project boundary.
- [x] Redesign punctuation handling for ASR models that can emit punctuation natively.
- [x] Normalize optional punctuation stripping before re-punctuation.
- [x] Add Whisper large ASR adapter documentation and standalone output-shape test.
- [x] Check batch inference support for newly added models and update Fun-ASR-Nano batching if available.
- [x] Verify the updated pipeline on the 60-second `data/test/short.wav`.

## Review

- `conda run -n fireredasr2s python -m compileall sentence_asr_pipeline` passed.
- Fake VAD/ASR/Punc smoke test passed with `CUDA_VISIBLE_DEVICES=4,5,6,7`.
- Fake mandatory VAD/ASR/Timestamp/Punc smoke test passed with `CUDA_VISIBLE_DEVICES=4,5,6,7`.
- Fake ASR-native timestamp provider and forced-aligner timestamp provider smoke tests passed.
- `silero-vad` and `tiktoken` were installed in the `fireredasr2s` env.
- Silero VAD standalone test passed on the first 30 seconds of `data/test/conf_0002_0002_001003.wav`.
- Fun-ASR-Nano standalone test passed on `data/test/short.wav`.
- Silero VAD + Fun-ASR-Nano timestamp + FireRedPunc experiment passed on `data/test/short.wav`.
- Output artifacts were written under `output/experiments/silero_funasr_fireredpunc`: `short.json`, `result.jsonl`, `asr_csv/short.csv`, `asr_srt/short.srt`, and `asr_tg/short.TextGrid`.
- Added vendored FireRed ASR/VAD/Punc runtime code under `sentence_asr_pipeline/firered_runtime`.
- Added ASR-native punctuation strategy and punctuation stripping before external re-punctuation.
- Updated Fun-ASR-Nano to batch temporary wav-path inputs in one `AutoModel.generate` call when the pipeline ASR batch size is greater than one.
- Whisper large standalone test passed on `data/test/short.wav` first 30 seconds.
- Silero VAD + Fun-ASR-Nano timestamp + FireRedPunc passed on 60-second `data/test/short.wav` with `asr_batch_size=8`.
- Silero VAD + Whisper large timestamp + ASR-native punctuation passed on 60-second `data/test/short.wav`.
