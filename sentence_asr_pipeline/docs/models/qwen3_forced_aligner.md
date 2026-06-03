# Qwen3 ForcedAligner 0.6B

Adapter:

```text
sentence_asr_pipeline.adapters.qwen3_forced_aligner.Qwen3ForcedAlignerTimestampProvider
```

## Environment

The model files are cached locally:

```text
/home/duhu/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B
```

The adapter expects the official Qwen ASR Python package:

```bash
conda activate fireredasr2s
pip install git+https://github.com/QwenLM/Qwen3-ASR.git
```

Source:

```text
https://github.com/QwenLM/Qwen3-ASR
```

Installed in the current environment:

```text
qwen-asr==0.0.6
transformers==4.57.6
```

The current environment uses `torch==2.1.0+cu118`; the adapter patches the
older `torch.utils._pytree` API before importing Qwen/Transformers.

## Capabilities

- Timestamp provider: yes
- Input: ASR text plus the matching VAD audio segment
- Batch alignment: yes, through `Qwen3ForcedAligner.align(audio=[...], text=[...], language=[...])`
- Russian support: yes, use `language="Russian"`

## Standalone/Experiment Test

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n fireredasr2s \
  python sentence_asr_pipeline/examples/run_fireredvad_whisper_qwenaligner_textpunc.py \
  --wav_path data/test/ru_ru.wav \
  --uttid ru_ru \
  --device cuda:0 \
  --max_seconds 60
```
