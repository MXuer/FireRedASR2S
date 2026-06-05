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

After frame-level VAD probability support was added, the same command was
rerun with `--uttid pt_br_tenvad_probs` and
`--outdir output/experiments/tenvad_whisper_mms_nativepunc_pt_br_probs`.

After the pipeline switched MMS alignment to raw VAD segments, the same config
was rerun with `--uttid pt_br_tenvad_rawalign` and
`--outdir output/experiments/tenvad_whisper_mms_nativepunc_pt_br_rawalign`.

After semantic-completeness gating was added to sentence-boundary fusion, the
same config was rerun with `--uttid pt_br_tenvad_semantic` and
`--outdir output/experiments/tenvad_whisper_mms_nativepunc_pt_br_semantic`.

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
| ASR VAD segments before raw-align strategy | 12 |
| ASR VAD segments after raw-align strategy | 37 |
| Output VAD segments | 35 |
| Timestamp segments before raw-align strategy | 12 |
| Timestamp segments after raw-align strategy | 36 |
| Words after raw-align strategy | 574 |
| Semantic sentence candidates after raw-align strategy | 46 |
| Final sentences before VAD probability support | 30 |
| Final sentences after VAD probability support | 24 |
| Final sentences after raw-align strategy | 39 |
| Final sentences after semantic-completeness gating | 37 |
| Boundary decisions after raw-align strategy | 45 |
| Boundary decisions after semantic-completeness gating | 45 |
| VAD probability frames | 18,750 |
| Sentence overlaps | 0 |
| Word overlaps | 0 |

After raw-align strategy, `raw_vad_segments_ms` and `asr_vad_segments_ms` are
identical. Boundary-fusion reasons observed: `vad_prob_valley`, `vad_silence`,
`acoustic_valley`, `target_duration`, `punctuation`,
`merged_active_speech_prob` and `merged_active_speech`.

The structural pipeline test passed. The transcript should still be reviewed
for model quality; a few Portuguese words in the SRT appear fused by the
Whisper/MMS tokenization path, for example `poisé`.

The reported raw-align fragment around `47.895s-58.600s` is fixed by treating
VAD silence and acoustic valleys as audio-safe evidence rather than mandatory
sentence splits. The final sentence is now:

```text
47.895s-58.600s
O primeiro ponto que gostaria de destacaréque, ao alinhar estratégia de tecnologia e negócios, as empresas podem criar uma conexão harmoniosa entre diferentes setores.
```

The next sentence, `resultando em maior sinergia organizacional.`, remains
separate because the preceding text has terminal punctuation and the boundary
is supported by VAD silence.

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
