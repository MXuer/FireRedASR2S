# Cadence-Fast

Source: https://huggingface.co/ai4bharat/Cadence-Fast

Registry name: `cadence_fast`

Role: `punc`

## Environment

Install the official helper package in the model environment if it is not
already present:

```bash
pip install cadence-punctuation
```

Verified package:

```text
cadence-punctuation==1.1.0
import name: cadence
```

Download the model outside Codex with `hf download`:

```bash
hf download ai4bharat/Cadence-Fast
```

## Capabilities

- Languages: English plus 22 official Indian languages:
  `as_in`, `bn_in`, `brx_in`, `doi_in`, `gu_in`, `hi_in`, `kn_in`, `kok_in`,
  `ks_in`, `mai_in`, `ml_in`, `mni_in`, `mr_in`, `ne_in`, `or_in`, `pa_in`,
  `sa_in`, `sat_in`, `sd_in`, `ta_in`, `te_in`, `ur_in`
- Batch inference: yes, `punctuate([...], batch_size=N)`
- Native timestamps: no
- ITN / digit normalization: no

This adapter receives timestamp tokens, sends the plain text batch to
Cadence-Fast, then maps punctuation back with the shared
`split_text_by_punctuation()` helper.

## Config

Profile: [../../configs/hi_in_cadence.json](../../configs/hi_in_cadence.json)

`model_path` is passed to the official package as a Hugging Face `cache_dir`.
For the normal shared HF cache, leave it unset. Do not pass a snapshot directory
such as `.../snapshots/<sha>` to the official wrapper; it will try to resolve
`ai4bharat/Cadence-Fast` inside that directory and fail.

```json
{
  "name": "cadence_fast",
    "params": {
      "model": "Cadence-Fast",
      "model_path": null,
      "batch_size": 32
    }
  }
```

## Smoke

Adapter smoke passed with default HF cache and `HF_HUB_OFFLINE=1` on GPU4. It
accepted a two-item English/Hindi batch and returned punctuation-mapped
sentence spans.

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n fireredasr2s \
  python semantic_asr/run_pipeline.py \
  --config configs/hi_in_cadence.json \
  --wav_path data/test/short/hi_in-short.wav \
  --uttid hi_in_cadence \
  --outdir output/experiments/hi_in_cadence
```
