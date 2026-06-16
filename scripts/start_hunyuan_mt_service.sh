#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH="${HUNYUAN_MT_MODEL_PATH:-/home/duhu/.cache/huggingface/hub/models--tencent--Hunyuan-MT-7B-fp8/snapshots/81e5a3f7199524570ba75e61360e990ba88665e4}"
RUNTIME_MODEL_PATH="${HUNYUAN_MT_RUNTIME_MODEL_PATH:-service_data/hunyuan_mt_fp8_patched}"
SERVED_MODEL_NAME="${HUNYUAN_MT_SERVED_MODEL_NAME:-hunyuan-mt}"
HOST="${HUNYUAN_MT_HOST:-127.0.0.1}"
PORT="${HUNYUAN_MT_PORT:-10087}"
DEVICE="${HUNYUAN_MT_DEVICE:-5}"
MAX_CONCURRENT="${HUNYUAN_MT_MAX_CONCURRENT:-1}"
CONDA_ENV="${HUNYUAN_MT_CONDA_ENV:-llm}"

echo "[hunyuan-mt] root=${ROOT_DIR}"
echo "[hunyuan-mt] model=${MODEL_PATH}"
echo "[hunyuan-mt] api=http://${HOST}:${PORT}"
echo "[hunyuan-mt] device=${DEVICE} max_concurrent=${MAX_CONCURRENT}"

CUDA_VISIBLE_DEVICES="${DEVICE}" conda run -n "${CONDA_ENV}" python -u -m semantic_asr_service.hunyuan_mt_server \
  --model-path "${MODEL_PATH}" \
  --runtime-model-path "${RUNTIME_MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-concurrent "${MAX_CONCURRENT}"
