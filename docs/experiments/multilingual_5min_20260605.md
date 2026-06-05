# Multilingual 5-Minute Config Run - 2026-06-05

## Scope

Ran one language-named config per available `data/test` language fixture, with
`--max_seconds 300`. The Thai fixture is only 16.17 seconds, so the run used the
full available audio.

Outputs are under `output/experiments/multilingual_5min/<language>/`.

## Command Shape

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=4 \
conda run -n fireredasr2s python semantic_asr/run_pipeline.py \
  --config configs/<language>.json \
  --wav_path data/test/<language>.wav \
  --uttid <language> \
  --outdir output/experiments/multilingual_5min/<language> \
  --max_seconds 300
```

## Results

| Language | Config | Duration | Sentences | Words | Timestamp segments | Output JSON |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `ar_sa` | `configs/ar_sa.json` | 300.00s | 20 | 368 | 17 | `output/experiments/multilingual_5min/ar_sa/ar_sa.json` |
| `en_us` | `configs/en_us.json` | 300.00s | 51 | 815 | 11 | `output/experiments/multilingual_5min/en_us/en_us.json` |
| `hi_in` | `configs/hi_in.json` | 300.00s | 14 | 570 | 11 | `output/experiments/multilingual_5min/hi_in/hi_in.json` |
| `ja_jp` | `configs/ja_jp.json` | 300.00s | 23 | 164 | 15 | `output/experiments/multilingual_5min/ja_jp/ja_jp.json` |
| `ko_kr` | `configs/ko_kr.json` | 300.00s | 30 | 225 | 24 | `output/experiments/multilingual_5min/ko_kr/ko_kr.json` |
| `pt_br` | `configs/pt_br.json` | 300.00s | 32 | 566 | 12 | `output/experiments/multilingual_5min/pt_br/pt_br.json` |
| `ru_ru` | `configs/ru_ru.json` | 299.93s | 45 | 532 | 11 | `output/experiments/multilingual_5min/ru_ru/ru_ru.json` |
| `th_th` | `configs/th_th.json` | 16.17s | 1 | 103 | 1 | `output/experiments/multilingual_5min/th_th/th_th.json` |
| `vi_vn` | `configs/vi_vn.json` | 300.00s | 20 | 286 | 19 | `output/experiments/multilingual_5min/vi_vn/vi_vn.json` |

All JSON files include `timestamp_segments`, preserving the timestamp provider
output by ASR segment. Sentence and word overlap counts are zero for every
language.

Each run also wrote `result.jsonl`, CSV, SRT, TextGrid and
`resolved_config.json`.

## Config Choices

- `ar_sa`: FireRed VAD + Seamless M4T + MMS Forced Aligner + XLM-R punctuation.
- `hi_in`: FireRed VAD + Seamless M4T + MMS Forced Aligner + XLM-R punctuation.
- `en_us`, `ja_jp`, `pt_br`, `th_th`, `vi_vn`: FireRed/Silero VAD +
  Whisper native word timestamps + ASR-native/text punctuation.
- `ko_kr`: FireRed VAD + Qwen3-ASR + Qwen3 ForcedAligner + XLM-R punctuation.
- `ru_ru`: FireRed VAD + Whisper + Qwen3 ForcedAligner + ASR text punctuation.

## Notes

An initial `ja_jp` attempt using Seamless M4T + MMS Forced Aligner failed in
MMS CTC alignment:

```text
targets length is too long for CTC
```

The Japanese config was switched to Whisper native timestamps for this coverage
run. This gives a complete JSON output and leaves Japanese MMS alignment as a
separate follow-up.

## Validation

```text
All 9 language JSON outputs exist.
All 9 outputs include timestamp_segments.
Sentence overlaps: 0 for every language.
Word overlaps: 0 for every language.
37 unit tests: passed.
compileall semantic_asr/tests/examples: passed.
git diff --check: passed.
```
