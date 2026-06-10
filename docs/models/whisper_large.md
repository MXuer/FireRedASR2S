# Whisper Large

Adapter:

```text
semantic_asr.adapters.whisper_large.WhisperLarge
```

## Environment

The current `fireredasr2s` conda environment already contains `openai-whisper`.

```bash
conda activate fireredasr2s
python -c "import whisper; print(whisper.available_models())"
```

The local model file is available at:

```text
~/.cache/whisper/large-v3.pt
```

## Capabilities

- ASR text: yes
- Word-level timestamp: yes, with `word_timestamps=True`
- Native punctuation: yes
- Batch inference: no native batch API in `openai-whisper`; set `num_workers`
  to run multiple segment-level worker processes on the same visible GPU.

## Parallel Segment Workers

For long audio with many VAD segments, configure:

```json
{
  "components": {
    "asr": {
      "name": "whisper_large",
      "params": {
        "device": "cuda:0",
        "num_workers": 2
      }
    }
  },
  "pipeline": {
    "asr_batch_size": 8
  }
}
```

Each worker process loads one Whisper model instance. This can improve
throughput on a large GPU, but GPU memory usage increases roughly with
`num_workers`.

## Standalone Output Test

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n fireredasr2s \
  python examples/test_whisper_large.py \
  --wav_path data/test/short.wav \
  --max_seconds 30 \
  --device cuda:0
```

## Full Pipeline Example

This uses Silero VAD, Whisper large native word timestamps and ASR-native
punctuation splitting.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n fireredasr2s \
  python semantic_asr/run_pipeline.py \
  --config configs/en_us.json \
  --wav_path data/test/short.wav \
  --uttid short \
  --outdir output/experiments/en_us
```
