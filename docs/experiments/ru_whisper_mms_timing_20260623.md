# Russian Whisper + MMS Timing Probe - 2026-06-23

## Goal

Measure a real long-audio `Whisper large + MMS forced aligner` run and decide
whether MMS should group segments into emission batches.

## Setup

- Wav: `data/ru_ru/ru-p2/5d1e1db2-e006-40e3-8964-f3ecb1c24e12.wav`
- Duration: `300.01s`
- Temporary profile:
  - VAD: `firered_vad`
  - ASR: `whisper_large`, `batch_size=24`
  - Timestamp: `mms_forced_aligner`, `num_workers=1`, `star_probe_enabled=true`
  - Punctuation: `asr_text`
- Device: `CUDA_VISIBLE_DEVICES=2`
- Raw timing JSON:
  `output/experiments/ru_whisper_mms_timing_20260623/timing.json`

## Output Shape

| Metric | Value |
| --- | ---: |
| Raw VAD segments | 68 |
| ASR VAD segments | 13 |
| Timestamp segments | 13 |
| Sentences | 58 |
| Words | 756 |
| Discarded ASR segments | 0 |

## Pipeline Timing

| Stage | Time |
| --- | ---: |
| Pipeline build / model load | 30.0743s |
| Process total | 59.9911s |
| VAD | 0.5588s |
| Whisper ASR | 10.7777s |
| MMS timestamp provider | 29.1763s |
| Punctuation (`asr_text`) | 0.0049s |
| Formatting | 0.0033s |

Model loading is separated from `process total`; the full command wall time is
roughly model load plus process time.

## MMS Two-Round Breakdown

MMS ran exactly two `get_alignments()` calls per ASR/VAD segment:

- `13` star-probe calls
- `13` final-align calls
- `26` total emission generations

| MMS phase | `get_alignments` calls | Total | Emission | Forced-align estimate | Emission ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `star_probe` | 13 | 4.2401s | 4.0538s | 0.1863s | 95.61% |
| `final_align` | 13 | 6.2545s | 6.1788s | 0.0757s | 98.79% |
| Combined | 26 | 10.4946s | 10.2326s | 0.2620s | 97.50% |

Within `get_alignments()`, emission generation dominates. CPU
`forced_align()` is tiny on this run.

## Hidden MMS Overhead

The timestamp provider took `29.1763s`, while the raw `get_alignments()` calls
took only `10.4946s`.

Additional measured overhead:

| Area | Time |
| --- | ---: |
| `star_probe` method total | 8.7949s |
| `star_probe` `get_alignments` only | 4.2401s |
| `star_probe` overhead outside `get_alignments` | 4.5549s |
| `final_align` method total | 10.7668s |
| `final_align` `get_alignments` only | 6.2545s |
| `final_align` overhead outside `get_alignments` | 4.5123s |
| Timestamp-provider time outside those two methods | 9.6146s |

This strongly suggests the current MMS path is spending substantial time in
repeated text preparation, uromanization, alignment checks, span mapping, and
Python orchestration. Emission batching alone cannot remove this overhead.

## Batch Decision

For this real run, grouping emissions into batches is not the first optimization
to implement.

Reasons:

- The whole MMS timestamp stage is `29.1763s`.
- Raw emission generation is `10.2326s`, about 35% of MMS timestamp time.
- CPU forced alignment is only `0.2620s`.
- The two-round design recomputes emissions for the same segment: star-probe
  and final-align each run model forward once.
- Earlier emission-batch probing showed mixed-duration batches can be slower
  unless they are length-aware and bucketed.

Better optimization order:

1. Cache per-segment uroman/alignment-token preparation so `_alignment_check`,
   star-probe and final-align do not repeat the same expensive work.
2. Reuse emissions between star-probe and final-align when final alignment uses
   the same full segment.
3. For selected star gaps that create alignment islands, slice frame ranges from
   the already generated full-segment emission when possible instead of running
   the model again on sliced audio.
4. After reuse/caching, add length-aware emission batching by duration bucket.

Practical conclusion: MMS should eventually support batched emissions, but the
current real bottleneck is duplicated two-round MMS work and text/uroman
overhead. A naive segment batch would not fix most of this run's timestamp
time.

## Emission Reuse Probe

After the baseline run, three temporary monkey-patch experiments were run on the
same wav and profile.

| Variant | Process total | MMS timestamp | `get_alignments` total | `generate_emissions` total | Cache hits / misses |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 59.9911s | 29.1763s | 10.4946s | 10.2326s | n/a |
| Object-id cache | 62.6079s | 30.9042s | 11.3527s | 10.9899s | 0 / 26 |
| Data-pointer cache | 57.3542s | 25.5212s | 6.4844s | 6.1366s | 10 / 16 |
| Full-slice + data-pointer cache | 53.0399s | 21.9861s | 3.2651s | 2.9178s | 13 / 13 |

The object-id cache did not work because final alignment calls
`_slice_speech_segment()` even for a full-span segment, producing a different
numpy object. Pointer/range-style caching catches most repeated full-segment
emissions. Adding a full-slice fast path, where `_slice_speech_segment()` returns
the original `SpeechSegment` when the requested span is the whole segment, gives
one cache hit per ASR/VAD segment.

For the full-slice cache run:

| MMS phase | `get_alignments` calls | Total | Emission | Forced-align estimate |
| --- | ---: | ---: | ---: | ---: |
| `star_probe` | 13 | 2.9991s | 2.9178s | 0.0813s |
| `final_align` | 13 | 0.2660s | 0.0001s | 0.2659s |

The final alignment round becomes almost pure CPU forced alignment because it
reuses star-probe emissions. MMS timestamp time improves from `29.1763s` to
`21.9861s`, a 24.65% reduction for this run.

Updated recommendation:

1. Implement a full-slice fast path in `_slice_speech_segment()`.
2. Add an explicit per-segment emission cache inside `MmsForcedAligner` or
   `MmsAligner`, keyed by stable audio range rather than Python object id.
3. Reuse the full-segment emission between star-probe and final-align when the
   final alignment is over the full segment.
4. Only after this reuse lands, revisit duration-bucketed emission batching.
