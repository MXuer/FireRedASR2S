# ASR Batch Size Benchmark - 2026-06-23

## Goal

Measure single-GPU ASR batch-size scaling for locally runnable ASR models.
Model loading time is excluded from the measured `elapsed_s`; each batch-size
run loads the model, performs one single-item warmup, resets CUDA peak memory
stats, and then times one `transcribe()` call.

Device:

- `CUDA_VISIBLE_DEVICES=2`
- GPU: NVIDIA A40, 46GB

Raw JSONL outputs:

- `output/experiments/asr_batch_benchmark_20260623/whisper_large.jsonl`
- `output/experiments/asr_batch_benchmark_20260623/gigaam_v3.jsonl`
- `output/experiments/asr_batch_benchmark_20260623/qwen3_asr_1_7b.jsonl`
- `output/experiments/asr_batch_benchmark_20260623/firered_asr.jsonl`
- `output/experiments/asr_batch_benchmark_20260623/probes.jsonl`

## Important Caveats

- These are throughput benchmarks, not transcript-quality evaluations.
- Whisper and GigaAM used a repeated 30s Russian segment.
- Qwen3-ASR used a repeated 10s English segment.
- FireRedASR used a repeated 10s Chinese test segment.
- Different audio lengths and languages mean cross-model absolute throughput
  is not directly comparable. The useful signal is each model's scaling curve.

## Whisper Large

Input:

- Wav: `data/ru_ru/ru-p2/5d1e1db2-e006-40e3-8964-f3ecb1c24e12.wav`
- Segment duration: 30s
- Adapter: `whisper_large`

| Batch size | Time | Items/s | Audio seconds/s | Peak allocated MB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6.1279s | 0.1632 | 4.8957 | 6419.4 |
| 2 | 6.9424s | 0.2881 | 8.6426 | 6686.5 |
| 4 | 7.4354s | 0.5380 | 16.1390 | 7220.0 |
| 8 | 8.1534s | 0.9812 | 29.4357 | 8321.6 |
| 16 | 9.3769s | 1.7063 | 51.1897 | 10416.1 |
| 24 | 11.1938s | 2.1440 | 64.3211 | 12552.5 |
| 32 | 12.5591s | 2.5480 | 76.4388 | 14711.4 |
| 48 | 15.9242s | 3.0143 | 90.4282 | 18945.5 |
| 64 | 19.7134s | 3.2465 | 97.3957 | 23194.0 |

Result:

- Largest tested batch size: `64`
- No OOM up to `64`
- Throughput still increases at `64`, but marginal gain from `48 -> 64` is
  small compared with memory growth.
- Practical default: `batch_size=24` or `32`
- High-throughput single-worker setting: `batch_size=48`
- Aggressive setting when GPU is dedicated to Whisper: `batch_size=64`

## GigaAM-v3

Input:

- Wav: `data/ru_ru/ru-p2/5d1e1db2-e006-40e3-8964-f3ecb1c24e12.wav`
- Segment duration: 30s
- Adapter: `gigaam_v3`

| Batch size | Time | Items/s | Audio seconds/s | Peak allocated MB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5495s | 1.8198 | 54.5952 | 544.8 |
| 2 | 0.5956s | 3.3579 | 100.7357 | 696.1 |
| 4 | 0.6726s | 5.9466 | 178.3993 | 937.3 |
| 8 | 0.9842s | 8.1285 | 243.8554 | 1415.9 |
| 16 | 1.3745s | 11.6405 | 349.2138 | 2370.2 |
| 24 | 1.7111s | 14.0262 | 420.7861 | 3325.9 |
| 32 | 2.0662s | 15.4874 | 464.6205 | 4281.9 |
| 48 | 2.7154s | 17.6770 | 530.3100 | 6192.0 |
| 64 | 3.4170s | 18.7300 | 561.9001 | 8105.7 |

Result:

- Largest tested batch size: `64`
- No OOM up to `64`
- Throughput continues increasing at `64` with modest memory use.
- Practical default: `batch_size=32`
- High-throughput setting: `batch_size=48` or `64`

## Qwen3-ASR-1.7B

Input:

- Wav: `data/test/short/en_us-short.wav`
- Segment duration: 10s
- Adapter: `qwen3_asr_1_7b`

| Batch size | Time | Items/s | Audio seconds/s | Peak allocated MB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.4127s | 0.7079 | 7.0785 | 3995.3 |
| 2 | 1.4743s | 1.3566 | 13.5658 | 4023.7 |
| 4 | 1.1878s | 3.3677 | 33.6770 | 4145.7 |
| 8 | 1.2865s | 6.2186 | 62.1861 | 4389.4 |
| 16 | 1.4877s | 10.7547 | 107.5467 | 4877.2 |
| 24 | 1.6969s | 14.1432 | 141.4324 | 5366.1 |
| 32 | 1.9206s | 16.6615 | 166.6153 | 5854.6 |

Result:

- Largest tested batch size: `32`
- No OOM up to `32`
- Throughput increases strongly through `32` for this 10s sample.
- Practical default: `batch_size=16`
- High-throughput setting: `batch_size=24` or `32`
- Needs a separate 30s multilingual benchmark before using these values for
  long VAD segments.

## FireRedASR

Input:

- Wav: `data/test/short.wav`
- Segment duration: 10s
- Adapter: `firered_asr`

| Batch size | Time | Items/s | Audio seconds/s | Peak allocated MB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2749s | 3.6376 | 36.3759 | 4748.3 |
| 2 | 0.4215s | 4.7451 | 47.4508 | 4720.9 |
| 4 | 0.4684s | 8.5394 | 85.3938 | 4844.1 |
| 8 | 0.8335s | 9.5983 | 95.9827 | 5087.4 |

Result:

- Largest tested batch size: `8`
- No OOM up to `8`
- Throughput improves up to `8`, but batch `4` is already close in throughput
  with lower latency.
- Practical default: `batch_size=4`
- Test larger batch sizes before changing production defaults.

## Unmeasured / Failed

Fun-ASR-Nano:

- Probe failed in the current environment with
  `AssertionError: FunAudioLLM/Fun-ASR-Nano-2512 is not registered`.
- The installed `funasr==1.3.1` attempted to locate files on the Hub and did
  not find them in the local cache.
- This benchmark was stopped to avoid uncontrolled downloads.

Dolphin / Seamless:

- Current adapters are not marked as batch-capable in `language_support.py`.
- They were not included in this single-GPU batch-size sweep.

## Recommendations

For ASR-heavy workers on A40-class GPUs:

| Model | Conservative | Throughput | Aggressive |
| --- | ---: | ---: | ---: |
| Whisper large | 24 | 32-48 | 64 |
| GigaAM-v3 | 32 | 48 | 64 |
| Qwen3-ASR-1.7B | 16 | 24 | 32 |
| FireRedASR | 4 | 8 | needs larger sweep |

Operational guidance:

- Prefer one warm ASR model per GPU with a larger internal batch size.
- Avoid increasing outer pipeline workers until model-internal batch is tuned.
- Keep separate benchmarks for 10s and 30s segments; Whisper pads to 30s, but
  other models may scale differently with duration.
- Re-run these benchmarks after changing model versions, precision, device, or
  adapter batching code.

