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
export SEMANTIC_ASR_TRANSLATION_BATCH_SIZE=32
export SEMANTIC_ASR_TRANSLATION_REQUEST_MODE=concurrent_single
export SEMANTIC_ASR_DEMO_USER_HEADER=1
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
semantic-asr-worker --device 4
semantic-asr-worker --device 5
semantic-asr-worker --device 6
semantic-asr-worker --device 7
```

Each worker sets `CUDA_VISIBLE_DEVICES` before loading models. For the first
production pass, use one worker per GPU, then increase concurrency only after
checking model memory usage.

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

When translation is configured, the Review panel can translate sentence text
with Hunyuan-MT and switch display between original, translated and bilingual
text. Translation is stored as a sidecar JSON under the job output directory and
does not change sentence timestamps or waveform intervals.

The demo calls `POST /v1/jobs/{job_id}/translations/stream` and updates the
segment list as each translated sentence arrives. The default startup script
uses `SEMANTIC_ASR_TRANSLATION_REQUEST_MODE=concurrent_single`, which sends one
sentence per request while dispatching up to
`SEMANTIC_ASR_TRANSLATION_BATCH_SIZE` requests concurrently. `json_batch` is
still available for a single structured JSON request containing multiple
sentences, but it cannot stream sentence-by-sentence UI updates.

`systemd` and `supervisor` are process managers. They are useful when this
service should survive SSH logout, machine reboot, or crashes. The startup
script is simpler and better for experiments; a production deployment should
wrap the same API/worker commands in `systemd` units or a `supervisord`
program group.

Example Hunyuan-MT-7B-fp8 OpenAI-compatible service:

```bash
export MODEL_PATH=/path/to/Hunyuan-MT-7B-fp8
CUDA_VISIBLE_DEVICES=6 python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 10087 \
  --trust-remote-code \
  --model "${MODEL_PATH}" \
  --served-model-name hunyuan-mt \
  --dtype bfloat16
```

On this machine, vLLM is not installed in the ASR environment. A minimal
transformers-based OpenAI-compatible wrapper is available instead:

```bash
export MODEL_PATH=/home/duhu/.cache/huggingface/hub/models--tencent--Hunyuan-MT-7B-fp8/snapshots/81e5a3f7199524570ba75e61360e990ba88665e4
CUDA_VISIBLE_DEVICES=6 conda run -n llm python -m semantic_asr_service.hunyuan_mt_server \
  --model-path "${MODEL_PATH}" \
  --runtime-model-path service_data/hunyuan_mt_fp8_patched \
  --served-model-name hunyuan-mt \
  --host 127.0.0.1 \
  --port 10087
```

The wrapper prepares a local runtime copy of the FP8 config because the
Hunyuan-MT model card requires renaming `ignored_layers` to `ignore` when using
the FP8 model with transformers/compressed-tensors. It keeps large safetensors
files as symlinks to the Hugging Face cache.

For a demo backed by GPU 7 with two workers:

```bash
export SEMANTIC_ASR_SERVICE_DATA_DIR=service_data/demo
export SEMANTIC_ASR_ALLOWED_CONFIGS=zh_cn,en_us,vi_vn,ar_sa,de_de
export SEMANTIC_ASR_API_KEYS='demo-token:demo,admin-token:admin:admin'
export SEMANTIC_ASR_SERVICE_PORT=10086

semantic-asr-server
```

In two additional shells:

```bash
semantic-asr-worker --device 7
semantic-asr-worker --device 7
```
