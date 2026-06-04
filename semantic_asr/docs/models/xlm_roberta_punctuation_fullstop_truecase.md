# XLM-R Punctuation Fullstop Truecase

Source: https://huggingface.co/1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase

## Install

The model card recommends the `punctuators` package:

```bash
pip install punctuators
```

`requirements.txt` records this dependency. The adapter imports:

```python
from punctuators.models import PunctCapSegModelONNX
```

The `punctuators` package exposes the 47-language ONNX checkpoint as
`pcs_47lang`. The adapter first looks for the local snapshot under
`~/.cache/huggingface/hub/models--1-800-BAD-CODE--punct_cap_seg_47_language`
and falls back to the package's `pcs_47lang` downloader when it is absent.

## Language Support

The model supports 47 languages:

Afrikaans, Amharic, Arabic, Bulgarian, Bengali, German, Greek, English,
Spanish, Estonian, Persian, Finnish, French, Gujarati, Hindi, Croatian,
Hungarian, Indonesian, Icelandic, Italian, Japanese, Kazakh, Kannada, Korean,
Kyrgyz, Lithuanian, Latvian, Macedonian, Malayalam, Marathi, Dutch, Oriya,
Panjabi, Polish, Pashto, Portuguese, Romanian, Russian, Kinyarwanda, Somali,
Serbian, Swahili, Tamil, Telugu, Turkish, Ukrainian, and Chinese.

The pipeline catalog records this as the punctuation component
`xlm_roberta_punctuation`.

Before production use, run a standalone smoke test for the exact language
because punctuation and truecasing quality differs by language and domain.

## Pipeline Role

This is a `punc` component. It receives token timestamps from the pipeline,
restores punctuation/truecase text, and maps generated sentence text back onto
the input token timeline.

It does not provide ASR or timestamp prediction.

## Standalone Test

Configuration/import shape only:

```bash
conda run -n fireredasr2s python semantic_asr/examples/test_xlm_roberta_punctuation.py --skip_model_load 1
```

Real model smoke test:

```bash
conda run -n fireredasr2s python semantic_asr/examples/test_xlm_roberta_punctuation.py \
  --text "hello world how are you"
```
