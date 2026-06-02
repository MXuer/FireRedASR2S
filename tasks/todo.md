# Todo

- [x] Read `AGENT.md` and align with its workflow.
- [x] Keep the original `fireredasr2s` package unchanged for this abstraction.
- [x] Create a separate `sentence_asr_pipeline` project directory.
- [x] Extract model-agnostic long-audio sentence ASR orchestration.
- [x] Add a FireRed adapter as the first concrete model-family bridge.
- [x] Verify the standalone pipeline with fake components.

## Review

- `conda run -n fireredasr2s python -m compileall sentence_asr_pipeline` passed.
- Fake VAD/ASR/Punc smoke test passed with `CUDA_VISIBLE_DEVICES=4,5,6,7`.
