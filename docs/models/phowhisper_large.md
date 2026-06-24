# PhoWhisper Large

Source: https://huggingface.co/vinai/PhoWhisper-large

Registry name: `phowhisper_large`

Role: `asr`

## Environment

Do not use the current `fireredasr2s` environment for this checkpoint unless
the weight file is converted to safetensors. The downloaded model contains
`pytorch_model.bin`; with `transformers==4.57.6`, loading `.bin` weights requires
`torch>=2.6`.

Verified environment:

```text
conda env: qwen3-asr
torch: 2.6.0+cu124
transformers: 4.57.6
torchaudio: 2.6.0+cu124
```

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

ASR-only smoke passed on the first 15 seconds of
`data/test/short/vi_vn-short.wav` in `qwen3-asr`, producing Vietnamese text.

The full pipeline profile below still needs either a torch>=2.6 runtime or a
safetensors conversion before it can run in `fireredasr2s`.

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n fireredasr2s \
  python semantic_asr/run_pipeline.py \
  --config configs/vi_vn_phowhisper.json \
  --wav_path data/test/short/vi_vn-short.wav \
  --uttid vi_vn_phowhisper \
  --outdir output/experiments/vi_vn_phowhisper
```
