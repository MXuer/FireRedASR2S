# Thai Whisper Large V3 Combined

Source: https://huggingface.co/biodatlab/whisper-th-large-v3-combined

Registry name: `whisper_th_large_v3_combined`

Role: `asr`

## Environment

Use the existing `fireredasr2s` environment first. The adapter is the same thin
Transformers Whisper adapter used by `phowhisper_large`.

Verified in `fireredasr2s`:

```text
torch: 2.1.0+cu118
transformers: 4.57.6
weight format: safetensors
```

The model should be downloaded outside Codex with `hf download`:

```bash
hf download biodatlab/whisper-th-large-v3-combined
```

## Capabilities

- Language: Thai, `th_th`
- Batch inference: yes, batched audio arrays through `AutoProcessor` and
  `model.generate()`
- Native timestamps: no
- Native punctuation: generated text may contain punctuation, but Thai sentence
  boundaries still need domain smoke tests
- ITN / digit normalization control: not documented; verify with domain data

Use `mms_forced_aligner` for timestamps. If Thai punctuation is weak, use the
existing `qwen_semantic_boundary` strategy instead of trusting punctuation.

## Config

Profile: [../../configs/th_th_whisper_th.json](../../configs/th_th_whisper_th.json)

## Smoke

ASR-only smoke passed on the first 15 seconds of
`data/test/short/th_th-short.wav` on GPU4.

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n fireredasr2s \
  python semantic_asr/run_pipeline.py \
  --config configs/th_th_whisper_th.json \
  --wav_path data/test/short/th_th-short.wav \
  --uttid th_th_whisper \
  --outdir output/experiments/th_th_whisper
```
