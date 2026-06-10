# Multilingual Short Boundary-Fusion Run - 2026-06-05

## Goal

Enable sentence-boundary fusion for every checked-in pipeline config and run the
available multilingual short fixtures from `data/test/short`.

## Inputs

The run used the manually clipped short fixtures:

- `data/test/short/ar_sa-short.wav`
- `data/test/short/en_us-short.wav`
- `data/test/short/hi_in-short.wav`
- `data/test/short/ja_jp-short.wav`
- `data/test/short/ko_kr-short.wav`
- `data/test/short/pt_br-short.wav`
- `data/test/short/ru_ru-short.wav`
- `data/test/short/th_th-short.wav`
- `data/test/short/vi_vn-short.wav`

Most clips are 300 seconds. `ru_ru-short.wav` is 299.93 seconds and
`th_th-short.wav` is 16.17 seconds in the current fixture set.

## Config Change

Sentence-boundary fusion is now the default pipeline policy, so language
profiles do not need to repeat the default threshold block. Current checked-in
profiles are language or scenario named; legacy combination profiles were
removed after this experiment.

## Command

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
CUDA_VISIBLE_DEVICES=4 bash -lc '
set -o pipefail
for lang in ar_sa en_us hi_in ja_jp ko_kr pt_br ru_ru th_th vi_vn; do
  conda run -n fireredasr2s python semantic_asr/run_pipeline.py \
    --config "configs/${lang}.json" \
    --wav_path "data/test/short/${lang}-short.wav" \
    --uttid "${lang}" \
    --outdir "output/experiments/multilingual_short_boundary_fusion/${lang}" || exit $?
done
'
```

## Outputs

Each language wrote:

- `<lang>.json`
- `result.jsonl`
- `resolved_config.json`
- `asr_csv/<lang>.csv`
- `asr_srt/<lang>.srt`
- `asr_tg/<lang>.TextGrid`

Base directory:

```text
output/experiments/multilingual_short_boundary_fusion
```

## Summary

| Language | Duration | Sentences | Semantic Candidates | Words | Timestamp Segments | Boundary Decisions | Merge Decisions | Sentence Overlaps | Word Overlaps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ar_sa | 300.00s | 20 | 29 | 368 | 17 | 28 | 9 | 0 | 0 |
| en_us | 300.00s | 36 | 51 | 815 | 11 | 50 | 15 | 0 | 0 |
| hi_in | 300.00s | 14 | 14 | 570 | 11 | 13 | 0 | 0 | 0 |
| ja_jp | 300.00s | 23 | 27 | 206 | 16 | 26 | 4 | 0 | 0 |
| ko_kr | 300.00s | 27 | 30 | 225 | 24 | 29 | 3 | 0 | 0 |
| pt_br | 300.00s | 28 | 32 | 566 | 12 | 31 | 4 | 0 | 0 |
| ru_ru | 299.93s | 43 | 45 | 532 | 11 | 44 | 2 | 0 | 0 |
| th_th | 16.17s | 1 | 1 | 103 | 1 | 0 | 0 | 0 | 0 |
| vi_vn | 300.00s | 20 | 20 | 293 | 19 | 19 | 0 | 0 | 0 |

Boundary decision reasons observed across the run include punctuation,
VAD-supported silence, active-speech merge, target duration, max duration and
acoustic valley cuts. All final sentence and word timestamp overlap counts are
zero.

## Validation

Passed:

```bash
conda run -n fireredasr2s python -c '... verify all configs enable sentence_boundary_fusion ...'
conda run -n fireredasr2s python -m unittest discover -s tests -p 'test_*.py'
conda run -n fireredasr2s python -m compileall -q semantic_asr tests examples
```
