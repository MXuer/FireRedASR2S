# Multilingual Model Smoke Test - 2026-06-04

## Scope

Test audio under `data/test`:

`ar_sa`, `en_us`, `hi_in`, `ja_jp`, `ko_kr`, `pt_br`, `ru_ru`, `th_th`,
and `vi_vn`.

Most audio files are long recordings, so model smoke tests used the first
8-20 seconds. These tests validate loading, input compatibility and normalized
output shape. They are not accuracy benchmarks.

Structured outputs are under `output/model_smoke_tests/`.

## Results

| Module | Status | Coverage / notes |
| --- | --- | --- |
| Silero VAD | Pass | Ran all nine languages. Some first-five-second clips contained no detected speech. |
| FireRed VAD | Pass | Representative `en_us`, `ru_ru`, `th_th` clips. |
| Fun-ASR-Nano-2512 | Pass after adapter fix | Ran all nine languages with native timestamps. Installed FunASR reports batch decoding is not implemented, so the adapter now falls back to sequential inference. |
| Whisper large-v3 | Pass | Ran all nine languages with native word timestamps. |
| Qwen3-ASR-1.7B | Pass after adapter fix | Ran all nine languages. Integer waveforms are now normalized to float32 before Qwen inference. |
| Qwen3 ForcedAligner | Pass | Real `en_us` Qwen3-ASR -> Qwen3 ForcedAligner chain returned 28 timestamps. |
| MMS Forced Aligner | Pass after runtime fix | Real `hi_in` Qwen3-ASR -> MMS chain returned 28 timestamps. Forced-alignment DP now runs on CPU to avoid torchaudio CUDA illegal-memory-access failures. |
| XLM-R 47-language punctuation | Pass after adapter/environment fix | Installed `punctuators`; adapter now loads the local `pcs_47lang` ONNX snapshot. Tested English, Russian, Japanese, Hindi, Thai and Arabic samples. |
| ASR-native / ASR-text punctuation strategies | Pass | Covered by unit tests. |
| Dolphin | Pass after GitHub-version retest | Official GitHub package version `20260513` passed grouped tests for Arabic, Hindi, Japanese, Korean, Russian and Vietnamese with normalized word timestamps. The selected Thai prefix returned empty text/timestamps and needs a speech-bearing quality fixture. |
| Seamless M4T v2 large | Pass after adapter/environment fixes | Local 18GB checkpoint passed all nine languages. Adapter now resamples input to 16kHz and defaults `tgt_lang=src_lang`; current environment uses `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`. |
| FireRedPunc | Blocked by environment | Current `transformers` refuses the local PyTorch `.bin` checkpoint because `torch==2.1.0` is below the required safe-load version 2.6. |

## Quality Warnings

Passing a smoke test does not mean the model transcribed the selected clip
correctly:

- fixed file-start clips sometimes contained silence or speech in another
  language;
- Qwen3-ASR returned empty text for the selected `pt_br` and `th_th` prefixes;
- Whisper and FunASR produced incorrect-language text for several prefixes;
- Dolphin returned empty text for the selected Thai prefix;
- XLM-R produced `<UNK>` tokens for the hand-written Thai punctuation sample.

Future quality tests should select speech-bearing intervals from TextGrid/VAD
instead of always using the beginning of each file, and should compare against
reference transcripts.

## Bugs Fixed During Testing

- Region language lookup now falls back from codes such as `ja_jp` to `ja`.
- `vi_vn` is aliased to the MMS `vi_in` mapping.
- FunASR batch inference automatically falls back when the installed model
  reports that batch decoding is not implemented.
- Qwen3-ASR receives normalized float32 waveform arrays.
- MMS forced alignment runs its dynamic-programming alignment on CPU.
- XLM-R punctuation loads the correct local 47-language ONNX checkpoint.
- Dolphin result normalization reads `text_nospecial` and `word_timestamps`.
- Seamless M4T resamples input to 16 kHz and defaults transcription output to
  the source language.

## Validation

```text
python -m compileall semantic_asr: passed
17 unit tests: passed
git diff --check: passed
```

## Remaining Work

1. Resolve the FireRedPunc torch/transformers checkpoint-loading compatibility
   issue.
2. Build quality fixtures from speech-bearing intervals plus reference
   transcripts.
