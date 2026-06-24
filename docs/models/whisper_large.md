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
- Word-level timestamp: not used in this project
- Native punctuation: yes
- Batch inference: yes, via `model.decode()` on batched mel features

Whisper's `transcribe(..., word_timestamps=True)` path is intentionally not
used. Whisper profiles should use a forced aligner such as `mms_forced_aligner`
or `qwen3_forced_aligner` for timestamps.

## VAD Segment Batch Decode

For long audio with many VAD segments, configure:

```json
{
  "components": {
    "asr": {
      "name": "whisper_large",
      "params": {
        "device": "cuda:0",
        "batch_size": 24
      }
    }
  }
}
```

The main speed path is batching VAD segments from the same long audio inside
one Whisper model. `num_workers` loads extra model processes and should stay at
the default `1` unless a single model instance leaves the GPU underused.

## Short Segment Decode Settings

The adapter exposes Whisper decode options such as `temperature`, `beam_size`,
`best_of`, `patience` and `length_penalty`. For short VAD segments it can
override the normal settings with:

```json
{
  "components": {
    "asr": {
      "name": "whisper_large",
      "params": {
        "short_audio_threshold_s": 1.0,
        "short_temperature": 0.0,
        "short_beam_size": 5,
        "short_length_penalty": 0.0
      }
    }
  }
}
```

A "short" segment means `duration <= short_audio_threshold_s`; the default is
`1.0` second. The default short path uses `short_beam_size=5` to make very short
clips less greedy and usually more stable, but beam search is slower. Because
OpenAI Whisper's batched beam decode is unsafe, the adapter keeps normal
non-beam decode batched and decodes short beam-search items one by one.
Disable this path with `"short_beam_size": null` if throughput matters more
than short-segment stability.

A focused Arabic 500ms hallucination test showed that length penalty can shorten
hallucinated output, but it does not eliminate hallucination. Keep the MMS
pre-alignment feasibility check enabled as the main guard for tiny VAD islands.

## Standalone Output Test

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n fireredasr2s \
  python examples/test_whisper_large.py \
  --wav_path data/test/short.wav \
  --max_seconds 30 \
  --device cuda:0
```

## Full Pipeline Example

This uses Silero VAD, Whisper large batch ASR, MMS forced alignment and
ASR-native punctuation splitting.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n fireredasr2s \
  python semantic_asr/run_pipeline.py \
  --config configs/en_us.json \
  --wav_path data/test/short.wav \
  --uttid short \
  --outdir output/experiments/en_us
```
