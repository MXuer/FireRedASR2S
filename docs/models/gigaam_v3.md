# GigaAM-v3

Adapter:

```text
semantic_asr.adapters.gigaam_v3.GigaAmV3Asr
```

## Environment

The adapter uses the official `gigaam` PyTorch package. Install the official
repository package in the active environment, or set `repo_dir` to a local
clone path so the adapter can import `gigaam`.

```bash
conda activate fireredasr2s
git clone git@github.com:salute-developers/GigaAM.git /path/to/GigaAM
python -m pip install -e /path/to/GigaAM
```

The local model files are expected under:

```text
pretrained_models/gigaam_v3/
```

Required files:

```text
v3_e2e_rnnt.ckpt
v3_e2e_rnnt_tokenizer.model
```

## Capabilities

- ASR text: yes, Russian
- Native punctuation: yes for `v3_e2e_ctc` / `v3_e2e_rnnt`
- Text normalization: yes for `v3_e2e_ctc` / `v3_e2e_rnnt`
- Batch inference: yes through the official PyTorch model forward path
- Native word timestamp: yes with `word_timestamps=True`

The ONNX helper only returns text, so this project uses the PyTorch model to
keep word timestamps.

Sources:

- [GigaAM README](https://github.com/salute-developers/GigaAM)
- [official `model.py`](https://github.com/salute-developers/GigaAM/blob/main/gigaam/model.py)

## Config Fragment

```json
{
  "components": {
    "asr": {
      "name": "gigaam_v3",
      "params": {
        "model_name": "v3_e2e_rnnt",
        "model_dir": "pretrained_models/gigaam_v3",
        "device": "cuda:0",
        "batch_size": 16
      }
    },
    "timestamp": {
      "name": "gigaam_v3_native"
    },
    "punc": {
      "name": "asr_native"
    }
  }
}
```

`num_workers` controls CPU DataLoader workers. `model_workers` controls how many
separate GigaAM model processes are loaded through the existing ASR parallel
wrapper.

## Full Pipeline Example

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n fireredasr2s \
  python semantic_asr/run_pipeline.py \
  --config configs/ru_ru_gigaam_v3.json \
  --wav_path data/test/short/ru_ru-short.wav \
  --uttid ru_ru_gigaam \
  --outdir output/experiments/ru_ru_gigaam_v3
```
