# NVIDIA Arabic FastConformer

Adapter:

```text
semantic_asr.adapters.nvidia_fastconformer.NvidiaFastConformerAsr
```

Model:

```text
nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0
```

## Environment

This adapter uses NVIDIA NeMo:

```bash
conda activate fireredasr2s
python -m pip install "nemo_toolkit[asr]"
```

Do not run the install through the proxy. Use `no-proxy` for installs on this
machine.

Current local status on 2026-06-24:

- HF cache exists under
  `/home/duhu/.cache/huggingface/hub/models--nvidia--stt_ar_fastconformer_hybrid_large_pcd_v1.0`
- `nemo` is installed in the `qwen3-asr` environment and was used for the real
  probe.
- `nemo` is not installed in `fireredasr2s`; run this adapter in `qwen3-asr` or
  install NeMo in the main runtime before using the profile there.

## Capabilities

- ASR text: yes, Arabic
- Native punctuation: yes, including periods, Arabic commas and Arabic question
  marks
- Diacritics: yes
- Native timestamps: yes through NeMo `transcribe(..., timestamps=True)`;
  `Hypothesis.timestamp["word"]` contains word/start/end dictionaries
- Batch inference: yes through NeMo `transcribe([...], batch_size=N, timestamps=True)`
- ITN: model card says output may need inverse text normalization

Use `nvidia_ar_fastconformer_native` for timestamps and `asr_text` for
punctuation splitting.

Sources:

- [Hugging Face model card](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0)

## Config Fragment

```json
{
  "components": {
    "asr": {
      "name": "nvidia_ar_fastconformer",
      "params": {
        "model_name": "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0",
        "device": "cuda:0",
        "batch_size": 8
      }
    },
    "timestamp": {
      "name": "nvidia_ar_fastconformer_native"
    },
    "punc": {
      "name": "asr_text"
    }
  }
}
```

## Smoke Test

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n fireredasr2s \
  python semantic_asr/run_pipeline.py \
  --config configs/ar_sa_nvidia_fastconformer.json \
  --wav_path data/test/short/ar_sa-short.wav \
  --uttid ar_sa_nvidia_fastconformer \
  --outdir output/experiments/ar_sa_nvidia_fastconformer
```

## Batch Benchmark

Measured on A40 GPU4 in `qwen3-asr` with a repeated 254.488s Arabic wav and
`timestamps=True`:

| Batch size | Result | Peak allocated MB |
| ---: | --- | ---: |
| 1 | ok | 2828.8 |
| 2 | ok | 5074.4 |
| 4 | ok | 9590.2 |
| 8 | ok | 18605.6 |
| 16 | ok | 36620.0 |
| 24 | OOM | 39936.0 before failure |

Use `batch_size=8` as the safe default for unknown segment durations. For
pipeline VAD segments capped around 30s, rerun a segment-length-specific
benchmark before raising the default.
