# Lazy-Loaded Model Sidecar Plan

## Status

Design note only. This is not implemented yet.

## Problem

The main `fireredasr2s` environment can no longer safely host every model:

- some ASR models need newer `torch` or `transformers`;
- ONNX models may need a different CUDA/cuDNN stack;
- keeping every model resident wastes GPU memory;
- changing the main environment to satisfy one model can break stable models.

The main service should stay stable. New or dependency-heavy models should run
outside it.

## Recommended Shape

Keep the API, WebUI, database, job queue and core pipeline in the main process.
Run dependency-heavy models through small local sidecar services.

```text
Web/API/job queue
  -> main semantic_asr pipeline
      -> local sidecar call when a stage needs an isolated model
          -> lazy-load model
          -> run batch
          -> keep warm briefly
          -> unload after idle TTL
```

## Minimal Lifecycle Policy

Start with the simplest policy that protects GPU memory:

```json
{
  "idle_ttl_s": 900,
  "max_loaded_models_per_gpu": 1,
  "load_on_first_request": true,
  "unload_when_idle": true
}
```

Rules:

- A sidecar process is lightweight and can stay running.
- The model inside the sidecar is loaded only after the first request.
- If the requested model is already loaded, reuse it.
- If a different model is loaded and idle, unload it before loading the new one.
- If the loaded model is busy, queue the request instead of loading a second
  copy on the same GPU.
- After `idle_ttl_s`, release model objects and clear GPU cache.

PyTorch unload path:

```python
del model
del processor
gc.collect()
torch.cuda.empty_cache()
```

## Sidecar Boundary

The main pipeline should not import sidecar model packages. It should call a
small HTTP API with normalized inputs and outputs.

ASR sidecar request:

```json
{
  "language": "vi_vn",
  "segments": [
    {"id": "0", "wav_path": "/tmp/job/0.wav", "start_ms": 0, "end_ms": 30000}
  ],
  "batch_size": 16
}
```

ASR sidecar response:

```json
{
  "segments": [
    {
      "id": "0",
      "text": "...",
      "tokens": [
        {"text": "...", "start_ms": 120, "end_ms": 260}
      ]
    }
  ]
}
```

If a model has no native timestamps, `tokens` can be omitted and the main
pipeline keeps using the configured forced aligner.

## Deployment Pattern

One sidecar environment per dependency family:

```text
env-main: API, WebUI, core pipeline, stable adapters
env-asr-indic: IndicConformer / ONNXRuntime GPU
env-asr-omni: omniASR and Python-version-specific models
env-asr-cohere: Cohere Transcribe and its Transformers stack
env-translation: Hunyuan-MT or other translation services
```

Example launch:

```bash
conda activate env-asr-indic
CUDA_VISIBLE_DEVICES=4 python -m sidecars.model_server --port 11001
```

The main config references the endpoint:

```json
{
  "asr": {
    "name": "http_asr",
    "params": {
      "base_url": "http://127.0.0.1:11001",
      "batch_size": 16
    }
  }
}
```

## What Stays In The Main Environment

Keep these in the main environment unless they become a proven problem:

- config parsing and language/model registry;
- VAD orchestration and segment preparation;
- sentence-boundary fusion;
- output writers;
- WebUI/API/job database;
- stable existing adapters that already work in `fireredasr2s`.

## What Moves To Sidecars

Move a model to a sidecar when any of these are true:

- it needs a conflicting `torch`, `transformers`, `onnxruntime`, `protobuf` or
  Python version;
- it is large and low-frequency;
- it has model-specific server/runtime code;
- it would force risky upgrades in the main environment.

## First Implementation Step Later

Do not build a full scheduler first.

The first useful step is one generic adapter:

```text
semantic_asr/adapters/http_asr.py
```

and one tiny model server that supports:

```text
GET /health
POST /v1/asr
POST /v1/unload
```

Everything else can wait until there is a real model that needs it.

