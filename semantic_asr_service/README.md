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
- `GET /v1/jobs/{job_id}`
- `GET /v1/jobs/{job_id}/result`
- `GET /v1/jobs/{job_id}/artifacts/{json|srt|csv|textgrid}`
- `GET /v1/models?language=zh_cn`
- `GET /health`
