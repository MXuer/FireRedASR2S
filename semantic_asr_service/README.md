# Semantic ASR HTTP Service

This package exposes the local `SemanticASR` SDK as an asynchronous HTTP API.
The API process receives uploads and records jobs. Worker processes claim queued
jobs from SQLite and run ASR on assigned GPUs.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Configuration

Environment variables:

```bash
export SEMANTIC_ASR_SERVICE_DATA_DIR=service_data
export SEMANTIC_ASR_CONFIGS_DIR=configs
export SEMANTIC_ASR_ALLOWED_CONFIGS=zh_cn,ar_sa,hakka
export SEMANTIC_ASR_API_KEYS='user-token:user,admin-token:admin:admin'
export SEMANTIC_ASR_MAX_UPLOAD_MB=2048
export SEMANTIC_ASR_MAX_AUDIO_SECONDS=0
export SEMANTIC_ASR_SERVICE_PORT=10086
export SEMANTIC_ASR_TRANSLATION_BASE_URL=http://127.0.0.1:10087
export SEMANTIC_ASR_TRANSLATION_MODEL=hunyuan-mt
export SEMANTIC_ASR_TRANSLATION_TARGETS=zh_cn,en_us
export SEMANTIC_ASR_AUTO_TRANSLATE_TARGETS=zh_cn
export SEMANTIC_ASR_TRANSLATION_BATCH_SIZE=64
export SEMANTIC_ASR_TRANSLATION_MAX_CONCURRENCY=24
export SEMANTIC_ASR_TRANSLATION_TIMEOUT_S=300
export SEMANTIC_ASR_TRANSLATION_REQUEST_MODE=concurrent_single
export SEMANTIC_ASR_DEMO_USER_HEADER=1
export SEMANTIC_ASR_DEMO_WORKER_DEVICES=6,7
export SEMANTIC_ASR_DEMO_WORKERS_PER_DEVICE=2
```

`SEMANTIC_ASR_ALLOWED_CONFIGS` uses config profile names. A request with
`config=zh_cn` resolves to `configs/zh_cn.json`. Users cannot upload arbitrary
pipeline configs through the service.

## Start The API

```bash
semantic-asr-server
```

Equivalent module form:

```bash
python -m semantic_asr_service.app
```

The API listens on `0.0.0.0:10086` by default. Override it with
`SEMANTIC_ASR_SERVICE_PORT`.

## Start Workers

Start one or more workers on the GPU machine:

```bash
semantic-asr-worker --device 6
semantic-asr-worker --device 7
```

Each worker sets `CUDA_VISIBLE_DEVICES` before loading models. The demo startup
script defaults to GPUs `6,7` with two ASR workers per GPU:

```bash
SEMANTIC_ASR_DEMO_WORKER_DEVICES=6,7 SEMANTIC_ASR_DEMO_WORKERS_PER_DEVICE=2 \
  scripts/start_demo_service.sh
```

Keep Hunyuan-MT translation on GPU `5` so slow translation does not compete
with ASR workers.

## Submit A Job

```bash
curl -X POST http://server:10086/v1/jobs \
  -H "Authorization: Bearer user-token" \
  -F "audio=@demo.wav" \
  -F "config=zh_cn" \
  -F "formats=json,srt,csv,textgrid"
```

Response:

```json
{
  "job_id": "abc123",
  "status": "queued"
}
```

## Query And Download

```bash
curl -H "Authorization: Bearer user-token" \
  http://server:10086/v1/jobs/abc123

curl -H "Authorization: Bearer user-token" \
  -O http://server:10086/v1/jobs/abc123/artifacts/srt

curl -H "Authorization: Bearer user-token" \
  -O http://server:10086/v1/jobs/abc123/result
```

Available endpoints:

- `POST /v1/jobs`
- `GET /v1/jobs`
- `GET /v1/me`
- `GET /v1/jobs/{job_id}`
- `GET /v1/jobs/{job_id}/result`
- `GET /v1/jobs/{job_id}/audio`
- `GET /v1/jobs/{job_id}/artifacts/{json|srt|csv|textgrid}`
- `POST /v1/jobs/{job_id}/translations/stream`
- `GET /v1/configs`
- `GET /v1/models?language=zh_cn`
- `GET /demo`
- `GET /health`

## Web Demo

The same API service also serves a browser demo at:

```text
http://server:10086/demo
```

For local demo use, start the API and workers together:

```bash
scripts/start_demo_service.sh
```

The script starts one API process and `SEMANTIC_ASR_DEMO_WORKER_COUNT`
workers on `SEMANTIC_ASR_DEMO_WORKER_DEVICE`. Defaults are two workers on GPU
7. It also checks whether the configured translation service is reachable.

The page lets a user enter an API token, choose an allowed language/profile,
upload one or more local audio files, submit jobs, poll status and download
JSON/SRT/CSV/TextGrid artifacts. It calls the existing `/v1/jobs` endpoints;
there is no separate backend.

The demo persists the entered user name, API token and selected profile in the
browser. In demo mode, the API token is the access gate and the `User Name`
field is the per-PM job namespace sent as `X-Semantic-ASR-User`. Changing the
name really switches the job list for normal tokens. Admin tokens ignore that
header and can see all jobs. `GET /v1/jobs` is paginated so the review area does
not get pushed far below a long table. Uploads include a best-effort local-path
hint (`webkitRelativePath` when available, otherwise the file name) so the job
table can show a recognizable source name instead of only a job id. Browsers do
not expose real absolute local paths and cannot reopen arbitrary local files by
path after refresh. During the current page session the demo uses the browser
`File` object first; after refresh it fetches the saved uploaded audio from
`GET /v1/jobs/{job_id}/audio`.

After a job succeeds, click `View` to review the result in the browser. The
demo decodes the selected or saved uploaded audio file, draws a waveform,
overlays sentence cut intervals from the JSON result and lets the user click a
segment to seek and play the corresponding audio. Segment-click playback stops
at that segment's end instead of continuing into the next segment.

For long audio review, use the waveform zoom slider or mouse wheel over the
waveform to zoom in and out. Drag the waveform horizontally to pan through the
timeline. The sentence/time-text list is shown below the waveform so the audio
area stays wide.

When translation is configured, workers can translate completed ASR sentence
text with Hunyuan-MT and store it as a sidecar JSON under the job output
directory. The default demo startup translates to `zh_cn` automatically through
`SEMANTIC_ASR_AUTO_TRANSLATE_TARGETS=zh_cn`. Translation does not change
sentence timestamps or waveform intervals.

The Review panel defaults to bilingual display and tries to load the cached
`zh_cn` translation when a completed job is opened. It no longer exposes manual
Text/Target/Translate controls. `SEMANTIC_ASR_TRANSLATION_BATCH_SIZE` controls
how many sentences are processed per window, while
`SEMANTIC_ASR_TRANSLATION_MAX_CONCURRENCY` controls how many requests may hit
the translation service at the same time. `SEMANTIC_ASR_TRANSLATION_BASE_URL`
also accepts comma-separated service replicas such as
`http://127.0.0.1:10087,http://127.0.0.1:10088,http://127.0.0.1:10089`; requests
are round-robin distributed across them. The default client concurrency is `24`
for the smaller `HY-MT1.5-1.8B-FP8` model, with a `300` second per-request
timeout for long or busy batches. If latency gets worse or GPU memory becomes
tight, lower the concurrency with the environment variable. `json_batch`
remains available for programmatic translation requests, but the demo now
relies on the precomputed cache.

`systemd` and `supervisor` are process managers. They are useful when this
service should survive SSH logout, machine reboot, or crashes. The startup
script is simpler and better for experiments; a production deployment should
wrap the same API/worker commands in `systemd` units or a `supervisord`
program group.

Example HY-MT1.5-1.8B-FP8 OpenAI-compatible service:

```bash
export MODEL_PATH=/path/to/HY-MT1.5-1.8B-FP8
CUDA_VISIBLE_DEVICES=5 python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 10087 \
  --trust-remote-code \
  --model "${MODEL_PATH}" \
  --served-model-name hunyuan-mt \
  --dtype bfloat16
```

On this machine, vLLM is not installed in the ASR environment. A minimal
transformers-based OpenAI-compatible wrapper is available instead. It defaults
to `tencent/HY-MT1.5-1.8B-FP8`, GPU `5`, and server-side
`HUNYUAN_MT_MAX_CONCURRENT=24`. Set `HUNYUAN_MT_PORTS=10087,10088,10089` to run
multiple model replicas on the same GPU:

```bash
scripts/start_hunyuan_mt_service.sh
```

For the current long-running demo deployment on this machine, run three
HY-MT1.5 replicas on GPU `5` as a detached background process:

```bash
setsid -f bash -c 'cd /data/duhu/FireRedASR2S && HUNYUAN_MT_PORTS=10087,10088,10089 HUNYUAN_MT_DEVICE=5 HUNYUAN_MT_MAX_CONCURRENT=8 scripts/start_hunyuan_mt_service.sh > service_data/hunyuan_mt15_multi.log 2>&1'
```

The wrapper prepares a local runtime copy of the FP8 config because the
Hunyuan-MT model card requires renaming `ignored_layers` to `ignore` when using
the FP8 model with transformers/compressed-tensors. It keeps large safetensors
files as symlinks to the Hugging Face cache.

Backfill cached translations for historical jobs that already succeeded before
auto-translation was enabled or before the translation model was ready:

```bash
export SEMANTIC_ASR_SERVICE_DATA_DIR=service_data/demo
export SEMANTIC_ASR_TRANSLATION_BASE_URL=http://127.0.0.1:10087,http://127.0.0.1:10088,http://127.0.0.1:10089
export SEMANTIC_ASR_TRANSLATION_TARGETS=zh_cn,en_us
export SEMANTIC_ASR_TRANSLATION_REQUEST_MODE=concurrent_single
export SEMANTIC_ASR_TRANSLATION_BATCH_SIZE=64
export SEMANTIC_ASR_TRANSLATION_MAX_CONCURRENCY=24
export SEMANTIC_ASR_TRANSLATION_TIMEOUT_S=300

python -m semantic_asr_service.backfill_translations --target zh_cn --dry-run
python -m semantic_asr_service.backfill_translations --target zh_cn
```

The backfill command only translates jobs whose DB status is `succeeded`, whose
main result JSON exists, and whose target translation cache is missing. It does
not rerun ASR or overwrite an existing translation cache.

For the current demo stack, prefer the wrapper script instead of starting API
and workers by hand. It starts the API on port `10086`, ASR workers on GPUs
`6,7`, and points translation requests at the separate Hunyuan-MT service:

```bash
scripts/start_demo_service.sh
```

Detached background demo startup with the three translation replicas:

```bash
setsid -f bash -c 'cd /data/duhu/FireRedASR2S && SEMANTIC_ASR_TRANSLATION_BASE_URL=http://127.0.0.1:10087,http://127.0.0.1:10088,http://127.0.0.1:10089 SEMANTIC_ASR_TRANSLATION_BATCH_SIZE=64 SEMANTIC_ASR_TRANSLATION_MAX_CONCURRENCY=24 SEMANTIC_ASR_TRANSLATION_TIMEOUT_S=300 scripts/start_demo_service.sh > service_data/demo_service_10086.log 2>&1'
```
