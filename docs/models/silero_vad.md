# Silero VAD

Reference: https://github.com/snakers4/silero-vad

## Environment

Use the project conda environment:

```bash
conda activate fireredasr2s
pip install silero-vad
```

Silero VAD requires PyTorch and audio I/O support. This repository already has
`torch`, `torchaudio` and `soundfile`; `silero-vad` was installed as the only
new dependency for this adapter.

## Adapter

`semantic_asr.adapters.silero.SileroVad` normalizes Silero output to:

```python
{"timestamps": [(start_s, end_s), ...]}
```

## Test

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n fireredasr2s \
  python examples/test_silero_vad.py \
  --wav_path data/test/conf_0002_0002_001003.wav \
  --max_seconds 30
```
