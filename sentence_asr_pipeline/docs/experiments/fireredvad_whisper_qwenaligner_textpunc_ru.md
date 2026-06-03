# FireRed VAD + Whisper + Qwen3 ForcedAligner + Whisper Text Punctuation

Test data:

```text
data/test/ru_ru.wav
```

## Components

- VAD: `FireRedVad`
- ASR: `WhisperLarge`
- Timestamp provider: `Qwen3ForcedAlignerTimestampProvider`
- Punctuation: `AsrTextPunc`

## Command

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n fireredasr2s \
  python sentence_asr_pipeline/examples/run_fireredvad_whisper_qwenaligner_textpunc.py \
  --wav_path data/test/ru_ru.wav \
  --uttid ru_ru \
  --device cuda:0 \
  --max_seconds 60
```

## Notes

- Whisper provides Russian ASR text and punctuation.
- Qwen3-ForcedAligner provides token timestamps from Whisper text plus the VAD
  audio segment.
- `AsrTextPunc` splits sentence text by Whisper punctuation and assigns sentence
  spans from aligned token timestamps.
- The common VAD merge policy runs before ASR slicing.

## Latest Output

```text
output/experiments/fireredvad_whisper_qwenaligner_textpunc_ru_full/ru_ru.json
output/experiments/fireredvad_whisper_qwenaligner_textpunc_ru_full/result.jsonl
output/experiments/fireredvad_whisper_qwenaligner_textpunc_ru_full/asr_csv/ru_ru.csv
output/experiments/fireredvad_whisper_qwenaligner_textpunc_ru_full/asr_srt/ru_ru.srt
output/experiments/fireredvad_whisper_qwenaligner_textpunc_ru_full/asr_tg/ru_ru.TextGrid
```

Summary:

- Duration: 299.931 seconds
- Raw VAD segments: 74
- Merged VAD segments: 8
- Sentences: 48
- Word/token timestamps: 530
