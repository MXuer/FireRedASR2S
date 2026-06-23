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
  adapters/          model adapters
  firered_runtime/   vendored FireRed runtime used by FireRed adapters
  mms_runtime/       vendored MMS forced-alignment runtime
  core.py            semantic ASR pipeline core
  config.py          profile loader and pipeline builder
  registry.py        component registry
  run_pipeline.py    unified config-driven CLI
  outputs.py         JSONL, CSV, SRT and TextGrid writers
```

Top-level repository resources:

```text
docs/               architecture, model and experiment notes
configs/            JSON/YAML pipeline profiles
examples/           runnable examples and compatibility wrappers
tests/              unit and orchestration tests
tasks/todo.md
tasks/progress.md
tasks/decisions.md
requirements.txt
```

## Documentation Map

Start here when entering the project:

- [AGENTS.md](AGENTS.md): collaboration rules for Codex and maintainers.
- [tasks/progress.md](tasks/progress.md): current status, recent fixes, validation commands and service notes.
- [tasks/decisions.md](tasks/decisions.md): durable architecture and product decisions.
- [docs/codex_context.md](docs/codex_context.md): compact handoff summary of the long Codex build history.
- [docs/maintenance_index.md](docs/maintenance_index.md): reading order for architecture, debugging, models and service docs.

Architecture and strategy:

- [docs/architecture/current_pipeline.md](docs/architecture/current_pipeline.md): end-to-end VAD, ASR, timestamp, punctuation, fusion and export flow.
- [docs/architecture/current_pipeline.drawio](docs/architecture/current_pipeline.drawio): draw.io diagram for the current pipeline.
- [docs/architecture/sentence_boundary_strategy.md](docs/architecture/sentence_boundary_strategy.md): current split/merge strategy and decision rules.
- [docs/architecture/service_runtime.md](docs/architecture/service_runtime.md): HTTP API, worker, WebUI and translation runtime layout.
- [docs/architecture/service_runtime.drawio](docs/architecture/service_runtime.drawio): draw.io diagram for service deployment.

Debugging and operations:

- [docs/debugging/debug_boundary_case.md](docs/debugging/debug_boundary_case.md): checklist for investigating bad cuts, over-merge, overlap and timestamp issues.
- [docs/sentence_boundary_fusion.md](docs/sentence_boundary_fusion.md): lower-level notes on boundary fusion.
- [docs/sentence_split_merge_strategy_audit.md](docs/sentence_split_merge_strategy_audit.md): historical strategy audit and simplification notes.
- [docs/hunyuan_mt_translation_design.md](docs/hunyuan_mt_translation_design.md): translation sidecar design.

Models and language support:

- [docs/model_matrix.md](docs/model_matrix.md): current VAD/ASR/timestamp/punctuation/translation capability matrix.
- [docs/models/](docs/models/): per-model install, configuration and smoke-test notes.
- [docs/language_configuration.md](docs/language_configuration.md): canonical language id and adapter mapping rules.
- [docs/test_audio_matrix.md](docs/test_audio_matrix.md): multilingual test audio planning.

Codex workflow:

- [codex_skills/semantic-asr/SKILL.md](codex_skills/semantic-asr/SKILL.md): versioned draft of the project-specific Codex skill.
- [tasks/todo.md](tasks/todo.md): active and historical task checklist.
- [tasks/lessons.md](tasks/lessons.md): corrections and lessons that should guide future work.

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
under `docs/models/` whenever adding a new adapter.

Multilingual test-audio planning lives in
`docs/test_audio_matrix.md`.

## Python SDK

External callers can use `SemanticASR` directly instead of invoking the CLI.
The loaded models are reused by the `SemanticASR` instance across single-file
calls:

```python
from semantic_asr import SemanticASR

asr = SemanticASR.from_config("configs/zh_cn.json")
output = asr.transcribe(
    wav_path="data/test/short.wav",
    uttid="short",
    outdir="output/sdk/zh_cn",
    formats=("json", "srt", "csv", "textgrid"),
)

print(output["result"]["text"])
print(output["outputs"])
```

Batch processing is also exposed through the SDK. It uses the same worker and
GPU assignment behavior as `semantic_asr/run_batch.py`:

```python
results = asr.transcribe_batch(
    wav_scp="data/test/wav.scp",
    outdir="output/sdk/batch",
    num_workers=1,
    devices="4,5,6,7",
)
```

Model support queries are available from the package root:

```python
from semantic_asr import list_models, list_model_languages

print(list_models("yue_hk"))
print(list_model_languages("qwen3_asr_1_7b", role="asr"))
```

## Unified CLI

After installing the package in editable mode, the unified command is
available as `semantic-asr`:

```bash
pip install -e .
```

Single file:

```bash
semantic-asr transcribe \
  --config configs/zh_cn.json \
  --wav-path data/test/short.wav \
  --uttid short \
  --outdir output/cli/zh_cn \
  --formats json,srt,csv,textgrid \
  --devices 4
```

Batch:

```bash
semantic-asr batch \
  --config configs/hakka.json \
  --wav-scp data/test/wav.scp \
  --outdir output/cli/hakka \
  --num-workers 1 \
  --devices 4,5,6,7
```

`--num-workers` is exposed by the unified CLI and is passed through to the
existing batch runner. It means pipeline worker processes per visible GPU and
defaults to `1`; model throughput should normally come from VAD-segment batch
decode inside the ASR adapter. `--devices` temporarily sets
`CUDA_VISIBLE_DEVICES` before model loading or worker assignment. Without
installation, the same CLI can be called as `python -m semantic_asr.cli ...`.

Model queries:

```bash
semantic-asr models yue_hk --role punc
semantic-asr model-languages qwen3_asr_1_7b --role asr
```

## HTTP Service

For ordinary remote users who cannot access this machine directly, run the
asynchronous HTTP service. The API process receives uploads and records jobs;
separate worker processes claim queued jobs and run ASR on assigned GPUs.

```bash
export SEMANTIC_ASR_API_KEYS='user-token:user,admin-token:admin:admin'
export SEMANTIC_ASR_ALLOWED_CONFIGS=zh_cn,ar_sa,hakka

semantic-asr-server
```

Start workers on the GPU machine:

```bash
semantic-asr-worker --device 4
semantic-asr-worker --device 5
semantic-asr-worker --device 6
semantic-asr-worker --device 7
```

Submit a remote job:

```bash
curl -X POST http://server:10086/v1/jobs \
  -H "Authorization: Bearer user-token" \
  -F "audio=@demo.wav" \
  -F "config=zh_cn" \
  -F "formats=json,srt,csv,textgrid"
```

See `semantic_asr_service/README.md` for status and download endpoints.

## Run A Config

```bash
python semantic_asr/run_pipeline.py \
  --config configs/zh_cn.json \
  --wav_path data/test/short.wav \
  --uttid short \
  --outdir output/experiments/zh_cn
```

The runner writes:

- `<uttid>.json`
- `result.jsonl`
- `asr_csv/<uttid>.csv`
- `asr_srt/<uttid>.srt`
- `asr_tg/<uttid>.TextGrid`
- `resolved_config.json`

## Run A Batch

`wav.scp` may contain one wav path per line. Output names use each audio
basename.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python semantic_asr/run_batch.py \
  --config configs/hakka.json \
  --wav_scp data/test/wav.scp \
  --outdir output/hakka \
  --num_workers 1
```

Workers are assigned round-robin across visible GPUs. `--num_workers` means
workers per visible GPU and defaults to `1`; increase it only when a single
model instance does not saturate the GPU. If
`CUDA_VISIBLE_DEVICES` is unset, the batch runner assumes eight device slots
`0..7`.

## Config Profiles

Profiles are named by language or project scenario. The current checked-in
profiles are:

- `configs/zh_cn.json`
- `configs/ar_sa.json`
- `configs/de_de.json`
- `configs/en_us.json`
- `configs/hakka.json`
- `configs/hi_in.json`
- `configs/ja_jp.json`
- `configs/ko_kr.json`
- `configs/pt_br.json`
- `configs/ru_ru.json`
- `configs/th_th.json`
- `configs/vi_vn.json`

Use `semantic_asr/run_pipeline.py` for single files and
`semantic_asr/run_batch.py` for `wav.scp` batches. Old combination-specific
runner scripts have been removed.

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
internally. See `docs/language_configuration.md`.

## Development Contract

Every new model adapter should include:

- installation/download/environment notes under `docs/models/`;
- a standalone example or test that verifies the model output shape;
- registry/config coverage if the model is used in a pipeline profile;
- batching notes when the model supports batch inference.

Before marking work complete, run:

```bash
conda run -n fireredasr2s python -m compileall semantic_asr
conda run -n fireredasr2s python -m unittest discover -s tests -p 'test_*.py'
```
