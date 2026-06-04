# Seamless M4T v2 Large

Source: https://huggingface.co/facebook/seamless-m4t-v2-large

## Install

The adapter uses Hugging Face Transformers:

```text
transformers==4.57.6
torch==2.1.0+cu118
torchaudio==2.1.0+cu118
```

Model weights are loaded from `facebook/seamless-m4t-v2-large` unless
overridden in config.

The adapter converts multi-channel input to mono and resamples audio to the
model-required 16 kHz before feature extraction.

With `sentencepiece==0.1.99` and newer protobuf releases, run with:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

## Language Support

Seamless M4T v2 large supports ASR for many languages using Seamless/FLORES
language codes such as `eng`, `cmn`, and `rus`. The pipeline catalog records
the model as broadly multilingual and maps common aliases like `en_us -> eng`,
`zh_cn -> cmn`, and `ru_ru -> rus`.

For real runs, configure a canonical pipeline language such as `en_us`. The
adapter derives the model's Seamless/FLORES source-language code.

## Timestamps And Punctuation

The current adapter produces ASR text only and does not provide word
timestamps. Pair it with a forced aligner before using it in the mandatory
four-stage pipeline.

For transcription, the adapter defaults the target language to the canonical
source language so output stays in the source language. Set
`target_language` only when translation is intended.

## Standalone Test

Configuration/import shape only:

```bash
conda run -n fireredasr2s python examples/test_seamless_m4t.py --skip_model_load 1
```

Real model smoke test:

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n fireredasr2s python examples/test_seamless_m4t.py \
  --wav_path data/test/short.wav \
  --language en_us \
  --max_seconds 30
```
