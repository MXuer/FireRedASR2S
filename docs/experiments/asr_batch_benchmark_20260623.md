# ASR Batch Size Benchmark - 2026-06-23

## Goal

Measure single-GPU ASR batch-size scaling for locally runnable ASR models.
Model loading time is excluded from the measured `elapsed_s`; each batch-size
run loads the model, performs one single-item warmup, resets CUDA peak memory
stats, and then times one batched inference call.

Device:

- `CUDA_VISIBLE_DEVICES=2`
- GPU: NVIDIA A40, 46GB

Raw JSONL outputs:

- `output/experiments/asr_batch_benchmark_20260623/whisper_large.jsonl`
- `output/experiments/asr_batch_benchmark_20260623/gigaam_v3.jsonl`
- `output/experiments/asr_batch_benchmark_20260623/qwen3_asr_1_7b_en.jsonl`
- `output/experiments/asr_batch_benchmark_20260623/firered_asr.jsonl`
- `output/experiments/asr_batch_benchmark_20260623/dolphin_native.jsonl`
- `output/experiments/asr_batch_benchmark_20260623/seamless_native.jsonl`
- `output/experiments/asr_batch_benchmark_20260623/probes.jsonl`

## Important Caveats

- These are throughput benchmarks, not transcript-quality evaluations.
- Whisper and GigaAM used a repeated 30s Russian segment.
- Qwen3-ASR used a repeated 10s English segment with `ASR_LANG=en_us`.
- FireRedASR used a repeated 10s Chinese test segment.
- Dolphin and Seamless used a repeated 10s Russian segment and native batch
  probes, not the current repository adapters, because the current adapters
  still loop item-by-item.
- Different audio lengths and languages mean cross-model absolute throughput is
  not directly comparable. The useful signal is each model's scaling curve.

## Whisper Large

Input:

- Wav: `data/ru_ru/ru-p2/5d1e1db2-e006-40e3-8964-f3ecb1c24e12.wav`
- Segment duration: 30s
- Adapter: `whisper_large`

| Batch size | Time | Audio seconds/s | Peak allocated MB |
| ---: | ---: | ---: | ---: |
| 1 | 6.1279s | 4.8957 | 6419.4 |
| 2 | 6.9424s | 8.6426 | 6686.5 |
| 4 | 7.4354s | 16.1390 | 7220.0 |
| 8 | 8.1534s | 29.4357 | 8321.6 |
| 16 | 9.3769s | 51.1897 | 10416.1 |
| 24 | 11.1938s | 64.3211 | 12552.5 |
| 32 | 12.5591s | 76.4388 | 14711.4 |
| 48 | 15.9242s | 90.4282 | 18945.5 |
| 64 | 19.7134s | 97.3957 | 23194.0 |
| 96 | 28.9500s | 99.4819 | 31735.6 |
| 128 | 37.8224s | 101.5270 | 40237.0 |

Result:

- Largest successful batch size: `128`
- Failed batch size: `160`, CUDA OOM while decoding.
- Practical default: `batch_size=32`
- High-throughput setting: `batch_size=48` or `64`
- Aggressive dedicated-GPU setting: `batch_size=96`
- Batch `128` works on this A40, but the marginal throughput gain is tiny and
  memory is close to the card limit.

## GigaAM-v3

Input:

- Wav: `data/ru_ru/ru-p2/5d1e1db2-e006-40e3-8964-f3ecb1c24e12.wav`
- Segment duration: 30s
- Adapter: `gigaam_v3`

| Batch size | Time | Audio seconds/s | Peak allocated MB |
| ---: | ---: | ---: | ---: |
| 1 | 0.5495s | 54.5952 | 544.8 |
| 2 | 0.5956s | 100.7357 | 696.1 |
| 4 | 0.6726s | 178.3993 | 937.3 |
| 8 | 0.9842s | 243.8554 | 1415.9 |
| 16 | 1.3745s | 349.2138 | 2370.2 |
| 24 | 1.7111s | 420.7861 | 3325.9 |
| 32 | 2.0662s | 464.6205 | 4281.9 |
| 48 | 2.7154s | 530.3100 | 6192.0 |
| 64 | 3.4170s | 561.9001 | 8105.7 |
| 96 | 4.8016s | 599.8010 | 11926.2 |
| 128 | 6.1464s | 624.7607 | 15751.4 |
| 192 | 8.8636s | 649.8518 | 23395.3 |
| 256 | 11.5813s | 663.1388 | 31043.7 |
| 320 | 14.2870s | 671.9401 | 38688.0 |

Result:

- Largest successful batch size: `320`
- Failed batch size: `384`, CUDA OOM in encoder attention.
- Practical default: `batch_size=32` or `64`
- High-throughput setting: `batch_size=128`
- Aggressive dedicated-GPU setting: `batch_size=192` or `256`
- Batch `320` is the measured upper success point but leaves little memory
  headroom.

## Qwen3-ASR-1.7B

Input:

- Wav: `data/test/short/en_us-short.wav`
- Segment duration: 10s
- Adapter: `qwen3_asr_1_7b`
- Environment: `ASR_LANG=en_us`

| Batch size | Time | Audio seconds/s | Peak allocated MB |
| ---: | ---: | ---: | ---: |
| 1 | 1.0681s | 9.3623 | 3995.3 |
| 2 | 1.1443s | 17.4785 | 4023.7 |
| 4 | 1.1762s | 34.0087 | 4145.7 |
| 8 | 1.2797s | 62.5160 | 4389.4 |
| 16 | 1.4869s | 107.6064 | 4877.2 |
| 24 | 1.6922s | 141.8239 | 5366.1 |
| 32 | 1.9159s | 167.0224 | 5854.6 |
| 48 | 2.3442s | 204.7614 | 6832.7 |
| 64 | 3.3343s | 191.9459 | 7812.8 |
| 96 | 3.9952s | 240.2883 | 9757.2 |
| 128 | 5.1803s | 247.0885 | 11708.5 |
| 192 | 7.4980s | 256.0687 | 15619.0 |
| 256 | 9.7284s | 263.1475 | 19515.8 |
| 320 | 12.1670s | 263.0066 | 23428.0 |
| 384 | 14.4991s | 264.8446 | 27324.7 |
| 512 | 19.3590s | 264.4769 | 35130.5 |
| 640 | 24.2533s | 263.8813 | 42938.6 |

Result:

- Largest successful batch size: `640`
- Failed batch size: `704`, CUDA OOM in `lm_head` during generation.
- Practical default: `batch_size=32` or `48`
- High-throughput setting: `batch_size=96` or `128`
- Aggressive dedicated-GPU setting: `batch_size=256`
- Batch sizes above `256` mostly burn memory without improving throughput.

## FireRedASR

Input:

- Wav: `data/test/short.wav`
- Segment duration: 10s
- Adapter: `firered_asr`

| Batch size | Time | Audio seconds/s | Peak allocated MB |
| ---: | ---: | ---: | ---: |
| 1 | 0.2749s | 36.3759 | 4748.3 |
| 2 | 0.4215s | 47.4508 | 4720.9 |
| 4 | 0.4684s | 85.3938 | 4844.1 |
| 8 | 0.8335s | 95.9827 | 5087.4 |
| 16 | 1.5821s | 101.1333 | 5570.0 |
| 32 | 3.0607s | 104.5495 | 6544.9 |
| 64 | 5.8902s | 108.6549 | 8490.6 |
| 128 | 11.7249s | 109.1693 | 12383.9 |
| 256 | 23.5351s | 108.7735 | 20172.2 |
| 384 | 35.4597s | 108.2921 | 27961.0 |
| 512 | 48.6101s | 105.3279 | 35750.5 |

Result:

- Largest successful batch size: `512`
- Failed batch size: `640`, CUDA OOM. The adapter returned zero results after
  logging the OOM, so it must be treated as failure despite the process
  printing a JSON summary.
- Practical default: `batch_size=8` or `16`
- High-throughput setting: `batch_size=32` or `64`
- Aggressive dedicated-GPU setting: `batch_size=128`
- Larger values do not improve throughput meaningfully.

## Dolphin Native Batch

Input:

- Wav: `data/ru_ru/ru-p2/5d1e1db2-e006-40e3-8964-f3ecb1c24e12.wav`
- Segment duration: 10s
- Native path: `/data/duhu/Dolphin/benchmark.py` style
  `extract_feats(...) -> model.decode(...)`
- Model: `~/.cache/dolphin/small`

| Batch size | Time | Audio seconds/s | Peak allocated MB |
| ---: | ---: | ---: | ---: |
| 1 | 0.3296s | 30.3397 | 2134.1 |
| 2 | 0.6322s | 31.6340 | 2764.2 |
| 4 | 1.1800s | 33.8980 | 4005.8 |
| 8 | 2.3245s | 34.4164 | 6508.3 |
| 16 | 4.5985s | 34.7939 | 11553.5 |
| 24 | 7.0631s | 33.9794 | 16565.5 |
| 32 | 10.4005s | 30.7679 | 21514.5 |
| 48 | 15.4507s | 31.0665 | 31525.5 |
| 64 | 18.5917s | 34.4240 | 41517.8 |

Result:

- Largest successful batch size: `64`
- Failed batch size: `96`, CUDA OOM in encoder attention.
- Practical default: `batch_size=4` or `8`
- High-throughput setting: `batch_size=16`
- Aggressive dedicated-GPU setting: `batch_size=24`
- Native Dolphin batching works, but this measured model is throughput-flat:
  larger batches mostly increase latency and memory.

## Seamless M4T v2 Large Native Batch

Input:

- Wav: `data/ru_ru/ru-p2/5d1e1db2-e006-40e3-8964-f3ecb1c24e12.wav`
- Segment duration: 10s
- Native path: `processor(audios=list) -> model.generate(...)`
- Model snapshot:
  `~/.cache/huggingface/hub/models--facebook--seamless-m4t-v2-large/snapshots/5f8cc790b19fc3f67a61c105133b20b34e3dcb76`

| Batch size | Time | Audio seconds/s | Peak allocated MB |
| ---: | ---: | ---: | ---: |
| 1 | 1.5724s | 6.3597 | 5894.9 |
| 2 | 1.7294s | 11.5645 | 5968.0 |
| 4 | 1.9534s | 20.4772 | 6120.0 |
| 8 | 2.2260s | 35.9387 | 6424.3 |
| 16 | 2.8870s | 55.4205 | 7032.8 |
| 24 | 3.4477s | 69.6110 | 7634.0 |
| 32 | 4.1238s | 77.5979 | 8238.6 |
| 48 | 5.5388s | 86.6615 | 9452.9 |
| 64 | 6.9931s | 91.5194 | 10661.7 |
| 96 | 9.6552s | 99.4282 | 13086.6 |
| 128 | 12.3992s | 103.2323 | 15509.1 |
| 192 | 19.0058s | 101.0218 | 20349.9 |
| 256 | 23.9663s | 106.8169 | 25197.0 |
| 384 | 36.1394s | 106.2551 | 34881.8 |
| 512 | 47.8580s | 106.9831 | 44573.2 |

Result:

- Largest successful batch size: `512`
- Failed batch size: `640`, CUDA OOM in encoder attention.
- Practical default: `batch_size=32` or `48`
- High-throughput setting: `batch_size=96` or `128`
- Aggressive dedicated-GPU setting: `batch_size=256`
- Batch `512` works but is effectively at the A40 memory ceiling.

## Failed Probe

Fun-ASR-Nano:

- Probe failed in the current environment with
  `AssertionError: FunAudioLLM/Fun-ASR-Nano-2512 is not registered`.
- The installed `funasr==1.3.1` attempted to locate files on the Hub and did
  not find them in the local cache.
- This benchmark was stopped to avoid uncontrolled downloads.

## Recommendations

For ASR-heavy workers on A40-class GPUs:

| Model | Conservative | Throughput | Aggressive |
| --- | ---: | ---: | ---: |
| Whisper large | 32 | 48-64 | 96 |
| GigaAM-v3 | 64 | 128 | 192-256 |
| Qwen3-ASR-1.7B | 48 | 96-128 | 256 |
| FireRedASR | 16 | 32-64 | 128 |
| Dolphin native | 8 | 16 | 24 |
| Seamless native | 48 | 96-128 | 256 |

Operational guidance:

- Prefer one warm ASR model per GPU with a larger internal batch size.
- Avoid increasing outer pipeline workers until model-internal batch is tuned.
- For Dolphin and Seamless, update the repository adapters before relying on
  these native batch numbers in production.
- Keep separate benchmarks for 10s and 30s segments; Whisper pads to 30s, but
  other models may scale differently with duration.
- Re-run these benchmarks after changing model versions, precision, device, or
  adapter batching code.
