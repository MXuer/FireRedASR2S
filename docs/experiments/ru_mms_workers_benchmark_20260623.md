# Russian MMS Worker Benchmark - 2026-06-23

## Goal

Measure whether increasing `mms_forced_aligner.params.num_workers` speeds up
MMS forced alignment on a real Russian `Whisper + MMS` long-audio case.

## Test Data

- Wav: `data/ru_ru/ru-p2/5d1e1db2-e006-40e3-8964-f3ecb1c24e12.wav`
- Duration: `300.01s`
- Device: `CUDA_VISIBLE_DEVICES=2`
- ASR/VAD shape:
  - Raw VAD segments: 68
  - ASR VAD segments: 13
  - ASR results passed to MMS: 13

Temporary profile:

- VAD: `firered_vad`
- ASR: `whisper_large`, `batch_size=24`
- Timestamp: `mms_forced_aligner`, `star_probe_enabled=true`
- Punctuation: `asr_text`

## Full Pipeline-Style Measurement

This run built the pipeline, ran VAD + Whisper once, then tested MMS
`num_workers` values inside the same Python process.

Preparation:

| Stage | Time |
| --- | ---: |
| Pipeline/model build | 29.2334s |
| VAD | 0.6136s |
| Whisper ASR | 11.4177s |

MMS timing:

| MMS workers | Time | Speedup vs 1 worker | Timestamped | Discarded |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 27.9309s | 1.000x | 13 | 0 |
| 2 | 55.9003s | 0.500x | 13 | 0 |
| 4 | 39.0839s | 0.715x | 13 | 0 |
| 8 | 30.4784s | 0.916x | 13 | 0 |
| 12 | 35.2956s | 0.791x | 13 | 0 |

## Isolated MMS Measurement

To reduce Whisper interference, ASR/VAD output was saved as a fixture and each
worker setting was measured in a fresh Python process.

Fixture:

- `output/experiments/ru_mms_workers_benchmark_20260623/asr_fixture.pkl`
- ASR results: 13
- ASR VAD segments: 13

MMS timing:

| MMS workers | Time | Speedup vs 1 worker | Timestamped | Discarded |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 27.2866s | 1.000x | 13 | 0 |
| 2 | 58.1912s | 0.469x | 13 | 0 |
| 4 | 39.3908s | 0.693x | 13 | 0 |
| 8 | 31.6612s | 0.862x | 13 | 0 |
| 12 | 36.7765s | 0.742x | 13 | 0 |

## Interpretation

The current multi-worker implementation does not speed up this case. It is
slower than single-worker MMS for all tested worker counts.

Main reason: `ParallelTimestampProvider.add_timestamps()` creates a new
`ProcessPoolExecutor` for each call. Each benchmark run pays process spawn,
Python import, MMS model load, and task serialization overhead. For one 300s
audio producing only 13 ASR/VAD segments, that overhead dominates the useful
parallel work.

Observed GPU memory during the 12-worker run was modest per worker, so VRAM is
not the limiting factor. The limiting factor is worker lifecycle overhead and
small task count.

## Recommendation

Do not increase `mms_forced_aligner.params.num_workers` in the current pipeline
for this workload. Keep `num_workers=1` unless a different workload has many
more segments and has been measured.

If we want worker parallelism to help, the implementation should change first:

1. Make timestamp workers persistent across audio files or across the process
   lifetime instead of creating a pool inside every `add_timestamps()` call.
2. Load one MMS model per persistent worker once.
3. Feed many audio files or many segment tasks through the same pool.
4. Only then benchmark `2/4/8/12` workers again.

For the current code, the better near-term optimization remains reducing
duplicated MMS work inside a single worker, or batching/reusing emissions after
that path is simplified.

