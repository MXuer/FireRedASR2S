# Semantic ASR

Multilingual long-audio semantic ASR pipeline.

`semantic_asr` is a config-driven orchestration layer for turning long speech
audio into sentence-like semantic segments with word/token timestamps and
standard subtitle/export artifacts.

## Why This Project

Long-audio ASR quality is not only about transcribing words. For practical
review, subtitle, annotation and meeting workflows, the output also needs
stable semantic sentence boundaries. This project standardizes that process
around four mandatory modules:

```text
VAD -> ASR -> Timestamp Provider -> Punctuation -> Semantic sentence output
```

The modules are intentionally pluggable. Some ASR models provide token
timestamps themselves, while others require a forced aligner. Some ASR models
emit usable punctuation, while others need a separate punctuation model.

## Current Package Layout

```text
semantic_asr/
  adapters/          model adapters and old combination wrappers
  configs/           JSON/YAML pipeline profiles
  docs/              model and experiment notes
  examples/          runnable examples and compatibility wrappers
  firered_runtime/   vendored FireRed runtime used by FireRed adapters
  tests/             smoke tests for config/registry orchestration
  core.py            semantic ASR pipeline core
  config.py          profile loader and pipeline builder
  registry.py        component registry
  run_pipeline.py    unified config-driven CLI
  outputs.py         JSONL, CSV, SRT and TextGrid writers
```

Top-level support files:

```text
docs/architecture.drawio
tasks/todo.md
PROGRESS.md
DECISIONS.md
requirements.txt
```

## Environment

The working conda environment is:

```bash
conda activate fireredasr2s
```

Available GPUs in the current machine are `4,5,6,7`. Device selection is kept
inside component config, for example `cuda:0` after setting
`CUDA_VISIBLE_DEVICES=4`.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Large model weights are not stored in this repository. Keep model setup notes
under `semantic_asr/docs/models/` whenever adding a new adapter.

Multilingual test-audio planning lives in
`semantic_asr/docs/test_audio_matrix.md`.

## Run A Config

```bash
python semantic_asr/run_pipeline.py \
  --config semantic_asr/configs/silero_funasr_fireredpunc.json \
  --wav_path data/test/short.wav \
  --uttid short \
  --outdir output/experiments/silero_funasr_fireredpunc
```

The runner writes:

- `<uttid>.json`
- `result.jsonl`
- `asr_csv/<uttid>.csv`
- `asr_srt/<uttid>.srt`
- `asr_tg/<uttid>.TextGrid`
- `resolved_config.json`

## Existing Profiles

- `semantic_asr/configs/silero_funasr_fireredpunc.json`
- `semantic_asr/configs/silero_whisper_nativepunc.json`
- `semantic_asr/configs/fireredvad_whisper_qwenaligner_textpunc_ru.json`

The old example scripts in `semantic_asr/examples/` are compatibility wrappers
around the same config runner.

## Query Model Language Support

Model-to-language and language-to-model metadata lives in
`semantic_asr/language_support.py`.

Query available modules for a language:

```bash
python semantic_asr/query_models.py language en_us
```

Query supported languages for a model:

```bash
python semantic_asr/query_models.py model qwen3_asr_1_7b --role asr
```

The query output is grouped by role: `vad`, `asr`, `timestamp`, and `punc`.

Pipeline profiles configure language once using a canonical language-region id
such as `zh_cn`, `en_us`, or `th_th`. Model-native language values are derived
internally. See `semantic_asr/docs/language_configuration.md`.

## Development Contract

Every new model adapter should include:

- installation/download/environment notes under `semantic_asr/docs/models/`;
- a standalone example or test that verifies the model output shape;
- registry/config coverage if the model is used in a pipeline profile;
- batching notes when the model supports batch inference.

Before marking work complete, run:

```bash
conda run -n fireredasr2s python -m compileall semantic_asr
conda run -n fireredasr2s python -m unittest semantic_asr.tests.test_config_runner
```
