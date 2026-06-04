# Arabic End-to-End Pipeline Test

## Scope

- Config: `semantic_asr/configs/ar.json`
- Audio: `data/test/ar_sa-short.wav`
- Audio duration: 600 seconds
- Pipeline: FireRed VAD + Seamless M4T v2 large + MMS Forced Aligner + XLM-R punctuation
- Device: GPU 4

## Command

The local models were already cached. Offline Hugging Face mode avoided an
unreliable `hf-mirror.com` metadata request during model loading.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=4 \
conda run -n fireredasr2s python semantic_asr/run_pipeline.py \
  --config semantic_asr/configs/ar.json \
  --wav_path data/test/ar_sa-short.wav \
  --uttid ar_sa_short \
  --outdir output/experiments/ar_seamless_mms_xlm_ar_sa_short_final
```

## Result

The complete ten-minute pipeline passed and wrote JSON, JSONL, CSV, SRT,
TextGrid and resolved-config outputs.

| Metric | Result |
| --- | ---: |
| Raw VAD segments | 129 |
| Semantic ASR segments | 34 |
| Final output VAD segments | 116 |
| Sentences | 54 |
| Aligned words | 699 |
| Sentence overlaps | 0 |
| Word overlaps | 0 |
| Non-positive word intervals | 4 |
| `<UNK>` tokens | 14 |

The four non-positive word intervals are skipped by the TextGrid writer. The
final TextGrid spans the full 600-second recording.

## Bug Found And Fixed

The first inference run completed, but TextGrid export failed because output
VAD alignment expanded two adjacent sentence intervals into an overlap:

```text
181.440-182.035  أ.
181.910-182.359  أأأ.
```

The pipeline now removes adjacent sentence overlaps at the shared result level
after output VAD alignment. The overlap boundary above becomes `181.972`
seconds, so JSON, JSONL, CSV, SRT and TextGrid all receive the same
non-overlapping sentence intervals.

## Quality Notes

This run validates component integration and output contracts, not recognition
accuracy. The transcript is mostly Arabic/Darija-looking text, but it contains
14 `<UNK>` tokens, some English text, and awkward punctuation sequences such
as `؟.` and `..`. These should be evaluated with an Arabic reference
transcript before this combination is treated as quality-approved.

## Sentence-Boundary Fusion Retest

The configurable sentence-boundary fusion stage was enabled for the Arabic
profile and the complete ten-minute audio was rerun:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=4 \
conda run -n fireredasr2s python semantic_asr/run_pipeline.py \
  --config semantic_asr/configs/ar.json \
  --wav_path data/test/ar_sa-short.wav \
  --uttid ar_sa_short \
  --outdir output/experiments/ar_seamless_mms_xlm_ar_sa_short_boundary_fusion
```

The result preserves 54 punctuation-proposed `semantic_sentences` and emits 47
audio-safe `sentences`. Seven active-speech boundaries were merged and six
nearby VAD-silence boundaries were kept and snapped. The longest final
sentence is 27.87 seconds; sentence and word overlaps are both zero.

The reported target region changed from three sentences to:

```text
73.770-87.418  كان يتحدث ... قال لي هذا الشيء الذي كان يفعله،.
87.418-97.920  و بعد ذلك عندما هددني ... هذا الأسبوع..
```

The incorrect `83.494-83.634` active-speech boundary was merged. The
`87.396-87.496` candidate was supported by raw VAD silence
`87.180-87.440` and snapped to `87.418`.

The punctuation sequences `،.` and `..` remain punctuation-model quality
issues and are intentionally not changed by sentence-boundary fusion.
