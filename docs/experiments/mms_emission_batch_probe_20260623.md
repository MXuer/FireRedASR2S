# MMS Emission Batch Probe - 2026-06-23

## Question

Can `MmsAligner.generate_emissions()` batch multiple VAD segments after ASR, so
one MMS model forward produces emissions for many short segments?

## Test Data

- Wav: `data/ru_ru/ru-p2/5d1e1db2-e006-40e3-8964-f3ecb1c24e12.wav`
- Model: `pretrained_models/mmsalign/model.pt`
- Device: `CUDA_VISIBLE_DEVICES=2`
- All test segments were <= 30s.

## Correctness Probe

Compared current single-segment `generate_emissions()` with a temporary batch
implementation that pads waveforms and passes real sample lengths into the
torchaudio wav2vec2 model.

Batch with `lengths` is numerically close to current single inference:

| Segment duration | Frames | Max abs diff | Mean abs diff | Frame delta |
| ---: | ---: | ---: | ---: | ---: |
| 6.0s | 299 | 0.00258 | 0.000054 | 0 |
| 9.5s | 474 | 0.00073 | 0.000038 | 0 |
| 12.0s | 599 | 0.00217 | 0.000066 | 0 |
| 16.0s | 799 | 0.00151 | 0.000038 | 0 |
| 20.0s | 999 | 0.00059 | 0.000031 | 0 |
| 24.0s | 1199 | 0.00341 | 0.000033 | 0 |
| 28.0s | 1399 | 0.00114 | 0.000035 | 0 |
| 29.5s | 1474 | 0.00080 | 0.000031 | 0 |

Batch without `lengths` is not usable: shorter samples are padded to the longest
sample, frame counts become 1474 for every item, and max absolute differences
reach about 8-12.

## Speed Probe

Each timing uses the fastest of three runs after warmup.

| Case | Durations | Single loop | One batch | Two buckets | One-batch speedup | Two-bucket speedup | Pad ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| varied_6_to_29s | 6, 9.5, 12, 16, 20, 24, 28, 29.5 | 0.5748s | 0.9971s | 0.7224s | 0.577x | 0.796x | 1.628 |
| short_equal_8s | 8 x 8s | 0.2753s | 0.1865s | 0.1987s | 1.476x | 1.385x | 1.000 |
| long_equal_28s | 8 x 28s | 0.9157s | 0.9114s | 0.9295s | 1.005x | 0.985x | 1.000 |
| bucketed_close_lengths | 20, 21, 22, 23, 26, 27, 28, 29 | 0.7821s | 0.9646s | 0.8353s | 0.811x | 0.936x | 1.184 |

## Conclusion

MMS emission batch inference is feasible, but only when real lengths are passed
to the model and emissions are cropped by returned output lengths.

Naively batching all segments is not always faster. If segment durations differ
widely, padding waste can dominate and make one large batch slower than the
current single loop. Equal short segments show a clear speedup, while equal
long segments are roughly tied.

Recommended implementation:

- Add `generate_emissions_batch()` that pads waveforms, passes `lengths`, and
  returns one cropped emission tensor per input segment.
- Sort or bucket segments by duration before emission batching.
- Keep per-segment CPU `forced_align`, because every text target is different.
- Keep `num_workers` as a fallback for cases where one model batch does not
  saturate the GPU, but do not replace it blindly with one huge mixed-duration
  batch.

## Forced Alignment CPU vs GPU Probe

The original Meta MMS code keeps emissions and targets on GPU for
`torchaudio.functional.forced_align`, then only moves the final path to CPU.
The current local runtime moves emissions to CPU before `forced_align` because
CUDA forced alignment has been unstable for some scripts in previous testing.

To isolate cost, a real Russian alignment segment was tested:

- Wav: `data/test/short/ru_ru-short.wav`
- Source JSON: `output/experiments/multilingual_short_boundary_fusion/ru_ru/ru_ru.json`
- Segment: `ru_ru_s0_e29870`, `0ms-29870ms`
- Raw tokens: 55
- MMS target indices: 305
- Emission frames: 1493 x 31

Timing, fastest of repeated runs:

| Step | Min | Avg | Notes |
| --- | ---: | ---: | --- |
| `generate_emissions` | 116.547ms | 119.326ms | GPU model forward |
| CPU `forced_align` | 5.202ms | 5.272ms | Includes emission GPU->CPU copy |
| GPU `forced_align` | 5.843ms | 5.867ms | Meta-style GPU path |

CPU and GPU forced alignment produced identical paths on this sample:

- `gpu_forced_align_supported`: `true`
- `cpu_gpu_path_equal`: `true`
- CPU/GPU segment count: `589 / 589`

Conclusion for this sample:

- Emission generation is the bottleneck: about 22x slower than CPU
  `forced_align`.
- Moving `forced_align` back to GPU does not improve speed on this sample; it
  is slightly slower than the CPU path.
- The best optimization target remains emission batching/reuse, not the DP
  alignment device.
- Keeping CPU `forced_align` is reasonable unless a broader multilingual test
  proves GPU alignment is consistently faster and stable.
