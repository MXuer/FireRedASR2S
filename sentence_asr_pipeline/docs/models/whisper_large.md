# Whisper Large

Adapter:

```text
sentence_asr_pipeline.adapters.whisper_large.WhisperLarge
```

## Environment

The current `fireredasr2s` conda environment already contains `openai-whisper`.

```bash
conda activate fireredasr2s
python -c "import whisper; print(whisper.available_models())"
```

The local model file is available at:

```text
/home/duhu/.cache/whisper/large-v3.pt
```

## Capabilities

- ASR text: yes
- Word-level timestamp: yes, with `word_timestamps=True`
- Native punctuation: yes
- Batch inference: no native batch API in `openai-whisper`; the adapter accepts
  a batch from the pipeline but runs items sequentially.

## Standalone Output Test

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n fireredasr2s \
  python sentence_asr_pipeline/examples/test_whisper_large.py \
  --wav_path data/test/short.wav \
  --max_seconds 30 \
  --device cuda:0
```

## Full Pipeline Example

This uses Silero VAD, Whisper large native word timestamps and ASR-native
punctuation splitting.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n fireredasr2s \
  python sentence_asr_pipeline/examples/run_silero_whisper_nativepunc.py \
  --wav_path data/test/short.wav \
  --uttid short \
  --device cuda:0
```

