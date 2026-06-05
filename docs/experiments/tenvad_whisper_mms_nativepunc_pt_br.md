# TEN VAD + Whisper + MMS + ASR-Native Punctuation - pt_br

## Goal

Validate the new TEN VAD adapter in a full pipeline using:

- VAD: `ten_vad`
- ASR: `whisper_large`
- Timestamp provider: `mms_forced_aligner`
- Punctuation: `asr_native`

## Input

```text
data/test/short/pt_br-short.wav
```

The current fixture is mono 16 kHz and 300 seconds long.

## Config

```text
configs/tenvad_whisper_mms_nativepunc_pt_br.json
```

Whisper is configured with `word_timestamps=false`; MMS provides the required
word timestamps. `asr_native` keeps Whisper's own punctuation in the timestamp
tokens for sentence splitting.

## Command

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
CUDA_VISIBLE_DEVICES=4 \
conda run -n fireredasr2s python semantic_asr/run_pipeline.py \
  --config configs/tenvad_whisper_mms_nativepunc_pt_br.json \
  --wav_path data/test/short/pt_br-short.wav \
  --uttid pt_br_tenvad \
  --outdir output/experiments/tenvad_whisper_mms_nativepunc_pt_br
```

## Outputs

```text
output/experiments/tenvad_whisper_mms_nativepunc_pt_br/pt_br_tenvad.json
output/experiments/tenvad_whisper_mms_nativepunc_pt_br/result.jsonl
output/experiments/tenvad_whisper_mms_nativepunc_pt_br/asr_csv/pt_br_tenvad.csv
output/experiments/tenvad_whisper_mms_nativepunc_pt_br/asr_srt/pt_br_tenvad.srt
output/experiments/tenvad_whisper_mms_nativepunc_pt_br/asr_tg/pt_br_tenvad.TextGrid
output/experiments/tenvad_whisper_mms_nativepunc_pt_br/resolved_config.json
```

## Result

| Metric | Value |
| --- | ---: |
| Duration | 300.00s |
| Raw TEN VAD segments | 37 |
| ASR VAD segments | 12 |
| Output VAD segments | 35 |
| Timestamp segments | 12 |
| Words | 562 |
| Semantic sentence candidates | 31 |
| Final sentences | 30 |
| Boundary decisions | 30 |
| Merge decisions | 1 |
| Sentence overlaps | 0 |
| Word overlaps | 0 |

Boundary-fusion reasons observed: `vad_silence`, `punctuation`,
`target_duration`, `acoustic_valley` and `merged_active_speech`.

The structural pipeline test passed. The transcript should still be reviewed
for model quality; a few Portuguese words in the SRT appear fused by the
Whisper/MMS tokenization path, for example `poisé`.

## Validation

Passed:

```bash
conda run -n fireredasr2s python -m unittest discover -s tests -p 'test_*.py'
conda run -n fireredasr2s python -m compileall -q semantic_asr tests examples
conda run -n fireredasr2s python semantic_asr/query_models.py language pt_br --role vad
conda run -n fireredasr2s python examples/test_ten_vad.py \
  --wav_path data/test/short/pt_br-short.wav \
  --max_seconds 30
```

