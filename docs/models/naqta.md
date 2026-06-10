# Naqta Arabic Punctuation

Source: https://huggingface.co/MostafaMaroof/Naqta

## Install And Download

The adapter uses the standard Hugging Face Transformers token-classification
API:

```bash
pip install transformers torch
```

Download the model through Hugging Face cache or point the adapter to a local
snapshot:

```json
{
  "components": {
    "punc": {
      "name": "naqta",
      "params": {
        "model_name_or_path": "MostafaMaroof/Naqta",
        "local_model_dir": "~/.cache/huggingface/hub/models--MostafaMaroof--Naqta/snapshots/<snapshot>",
        "device": "cuda:0"
      }
    }
  }
}
```

`local_model_dir` is optional. Paths are expanded with `os.path.expanduser`.

## Language Support

The pipeline catalog records this as an Arabic-only punctuation component:
`naqta`.

It is intended for canonical Arabic profile ids such as `ar_sa`. Use a real
domain smoke test before replacing `xlm_roberta_punctuation` or `asr_text` in a
production profile.

## Pipeline Role

This is a `punc` component. It receives token timestamps from ASR or forced
alignment, restores Arabic punctuation, and maps the punctuated text back onto
the input token timeline.

The adapter assumes the model is a Hugging Face token-classification checkpoint.
Predicted labels are mapped to punctuation with a configurable
`label_to_punctuation` dictionary. The default mapping covers common label names
such as `COMMA`, `PERIOD`, `QUESTION_MARK`, `SEMICOLON` and their Arabic
punctuation-symbol variants.

If the downloaded checkpoint exposes different labels, keep the adapter and
override only the mapping:

```json
{
  "label_to_punctuation": {
    "LABEL_0": "",
    "LABEL_1": "،",
    "LABEL_2": ".",
    "LABEL_3": "؟"
  }
}
```

## Standalone Test

Configuration/import shape only:

```bash
conda run -n fireredasr2s python examples/test_naqta_punctuation.py --skip_model_load 1
```

Real model smoke test:

```bash
conda run -n fireredasr2s python examples/test_naqta_punctuation.py \
  --text "هذا اختبار هل تسمعني" \
  --device cuda:0
```
