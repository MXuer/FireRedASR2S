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

Use `model_path` when loading from a pre-downloaded local directory:

```json
{
  "name": "cadence_fast",
  "params": {
    "model": "Cadence-Fast",
    "model_path": "/path/to/Cadence-Fast",
    "batch_size": 32
  }
}
```

## Smoke

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n fireredasr2s \
  python semantic_asr/run_pipeline.py \
  --config configs/hi_in_cadence.json \
  --wav_path data/test/short/hi_in-short.wav \
  --uttid hi_in_cadence \
  --outdir output/experiments/hi_in_cadence
```
