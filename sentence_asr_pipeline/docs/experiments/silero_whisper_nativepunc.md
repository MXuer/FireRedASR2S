# Silero VAD + Whisper Large + ASR-Native Punctuation

Test data:

```text
data/test/short.wav
```

## Components

- VAD: `SileroVad`
- ASR: `WhisperLarge`
- Timestamp provider: `WhisperLargeTimestampProvider`
- Punctuation: `AsrNativePunc`

## Command

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n fireredasr2s \
  python sentence_asr_pipeline/examples/run_silero_whisper_nativepunc.py \
  --wav_path data/test/short.wav \
  --uttid short \
  --device cuda:0
```

## Notes

- Whisper large emits punctuation in the ASR text and word timestamp tokens.
- This experiment keeps those punctuation marks and uses `AsrNativePunc` to
  split sentences.
- To force external re-punctuation instead, configure the pipeline with
  `strip_punctuation_before_punc=True` and use a concrete external punc model
  such as FireRedPunc.

## Latest Output

```text
output/experiments/silero_whisper_nativepunc_short60/short.json
output/experiments/silero_whisper_nativepunc_short60/result.jsonl
output/experiments/silero_whisper_nativepunc_short60/asr_csv/short.csv
output/experiments/silero_whisper_nativepunc_short60/asr_srt/short.srt
output/experiments/silero_whisper_nativepunc_short60/asr_tg/short.TextGrid
```

Summary:

- Duration: 60.0 seconds
- VAD segments: 7
- Sentences: 7
- Word/token timestamps: 26
