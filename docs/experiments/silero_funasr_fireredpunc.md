# Silero VAD + Fun-ASR-Nano + FireRedPunc

Test data:

```text
data/test/short.wav
```

## Components

- VAD: `SileroVad`
- ASR: `FunAsrNano`
- Timestamp provider: `FunAsrNanoTimestampProvider`
- Punctuation: `FireRedPunc`

## Command

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n fireredasr2s \
  python examples/run_silero_funasr_fireredpunc.py \
  --wav_path data/test/short.wav \
  --device cuda:0 \
  --hub ms
```

## Current Status

- `silero-vad` and `tiktoken` were installed in the `fireredasr2s` conda env.
- `test_silero_vad.py` passed on the first 30 seconds of the test wav.
- Fun-ASR-Nano model is cached locally and the standalone test passed.
- FunASR adapter uses temporary wav-path input to match the official
  `AutoModel.generate(input="audio.wav")` path and preserve timestamps for 8k
  source audio.
- FunASR adapter now sends multiple segment wav paths in one `generate` call
  when the pipeline ASR batch size is greater than one.
- The pipeline writes `.json`, `result.jsonl`, CSV, SRT and TextGrid outputs.
- The 60-second `data/test/short.wav` experiment passed with `asr_batch_size=8`.

## Latest Output

```text
output/experiments/silero_funasr_fireredpunc_short60/short.json
output/experiments/silero_funasr_fireredpunc_short60/result.jsonl
output/experiments/silero_funasr_fireredpunc_short60/asr_csv/short.csv
output/experiments/silero_funasr_fireredpunc_short60/asr_srt/short.srt
output/experiments/silero_funasr_fireredpunc_short60/asr_tg/short.TextGrid
```

Summary:

- Duration: 60.0 seconds
- VAD segments: 7
- Sentences: 7
- Word/token timestamps: 29

To rerun the standalone FunASR output-shape check:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n fireredasr2s \
  python examples/test_funasr_nano.py \
  --wav_path data/test/short.wav \
  --max_seconds 30 \
  --device cuda:0 \
  --hub ms
```

Then rerun the full pipeline command above.
