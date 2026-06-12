# Cantonese Punctuation Restoration

Source: https://huggingface.co/nizzzo/zh-yue-punctuation-restore-v3

## Install And Download

The adapter uses the standard Hugging Face Transformers token-classification
API:

```bash
pip install transformers torch
```

`requirements.txt` already includes both packages. Download the model through
the Hugging Face cache or point the adapter to a local snapshot:

```json
{
  "components": {
    "punc": {
      "name": "yue_punctuation",
      "params": {
        "model_name_or_path": "nizzzo/zh-yue-punctuation-restore-v3",
        "local_model_dir": "~/.cache/huggingface/hub/models--nizzzo--zh-yue-punctuation-restore-v3/snapshots/<snapshot>",
        "device": "cuda:0"
      }
    }
  }
}
```

`local_model_dir` is optional. Paths are expanded with `os.path.expanduser`.

## Language Support

The pipeline catalog records this as a Cantonese punctuation component:
`yue_punctuation`.

It is intended for canonical Cantonese profile ids such as `yue_hk`. The model
card describes a BERT/NER-style punctuation restoration model and lists support
for these punctuation marks: `。` `，` `、` `？` `！` `；` `︰`.

## Pipeline Role

This is a `punc` component. It receives token timestamps from ASR or forced
alignment, predicts a punctuation label per token, appends the mapped
punctuation to that token, and returns sentence spans on the original timestamp
timeline.

Unlike the generic text-splitting path, this adapter maps predictions by token
index instead of converting Cantonese text back through whitespace token counts.
That keeps no-space Chinese/Cantonese timestamps stable.

Predicted labels are mapped to punctuation with a configurable
`label_to_punctuation` dictionary. The downloaded checkpoint exposes labels
such as `O`, `S-。`, `S-，`, `S-、`, `S-？`, `S-！`, `S-；` and `S-︰`.
The adapter strips common sequence-label prefixes such as `B-`, `I-`, `E-`,
`S-` and `U-`, so those labels map through the literal punctuation entries.

If a future checkpoint exposes different labels, keep the adapter and override
only the mapping:

```json
{
  "label_to_punctuation": {
    "LABEL_0": "",
    "LABEL_1": "，",
    "LABEL_2": "。",
    "LABEL_3": "？",
    "LABEL_4": "！",
    "LABEL_5": "；",
    "LABEL_6": "︰"
  }
}
```

## Standalone Test

Configuration/import shape only:

```bash
conda run -n fireredasr2s python examples/test_yue_punctuation.py --skip_model_load 1
```

Real model smoke test:

```bash
conda run -n fireredasr2s python examples/test_yue_punctuation.py \
  --text "我 今日 返工 你 去 邊" \
  --device cuda:0
```
