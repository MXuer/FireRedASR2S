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

