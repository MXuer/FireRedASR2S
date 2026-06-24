# PhoWhisper Large

Source: https://huggingface.co/vinai/PhoWhisper-large

Registry name: `phowhisper_large`

Role: `asr`

## Environment

Use the existing `fireredasr2s` environment first. This adapter uses
Transformers, so it calls the project pytree compatibility patch before
importing Transformers.

The model should be downloaded outside Codex with `hf download`:

```bash
hf download vinai/PhoWhisper-large
```

## Capabilities

- Language: Vietnamese, `vi_vn`
- Batch inference: yes, batched audio arrays through `AutoProcessor` and
  `model.generate()`
- Native timestamps: no
- Native punctuation: yes, from Whisper text generation
- ITN / digit normalization control: not documented; verify with domain data

Use `mms_forced_aligner` for timestamps.

## Config

```json
{
  "components": {
    "asr": {
      "name": "phowhisper_large",
      "params": {
        "batch_size": 64
      }
    },
    "timestamp": {
      "name": "mms_forced_aligner"
    },
    "punc": {
      "name": "asr_text"
    }
  }
}
```

Profile: [../../configs/vi_vn_phowhisper.json](../../configs/vi_vn_phowhisper.json)

## Smoke

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n fireredasr2s \
  python semantic_asr/run_pipeline.py \
  --config configs/vi_vn_phowhisper.json \
  --wav_path data/test/short/vi_vn-short.wav \
  --uttid vi_vn_phowhisper \
  --outdir output/experiments/vi_vn_phowhisper
```
