#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_ID="${HUNYUAN_MT_MODEL_ID:-Tencent-Hunyuan/HY-MT1.5-1.8B-FP8}"
MODEL_PATH="${HUNYUAN_MT_MODEL_PATH:-}"
RUNTIME_MODEL_PATH="${HUNYUAN_MT_RUNTIME_MODEL_PATH:-service_data/hy_mt15_18b_fp8_patched}"
SERVED_MODEL_NAME="${HUNYUAN_MT_SERVED_MODEL_NAME:-hunyuan-mt}"
HOST="${HUNYUAN_MT_HOST:-127.0.0.1}"
PORT="${HUNYUAN_MT_PORT:-10087}"
DEVICE="${HUNYUAN_MT_DEVICE:-5}"
MAX_CONCURRENT="${HUNYUAN_MT_MAX_CONCURRENT:-4}"
CONDA_ENV="${HUNYUAN_MT_CONDA_ENV:-llm}"

if [[ -z "${MODEL_PATH}" ]]; then
  CACHE_ROOT="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
  CACHE_MODEL_DIR="${CACHE_ROOT}/models--${MODEL_ID//\//--}"
  MODEL_PATH="$(find "${CACHE_MODEL_DIR}/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
fi

if [[ -z "${MODEL_PATH}" || ! -d "${MODEL_PATH}" ]]; then
  echo "[hunyuan-mt] model snapshot not found for ${MODEL_ID}" >&2
  echo "[hunyuan-mt] set HUNYUAN_MT_MODEL_PATH=/path/to/model_snapshot after download completes" >&2
  exit 1
fi

echo "[hunyuan-mt] root=${ROOT_DIR}"
echo "[hunyuan-mt] model_id=${MODEL_ID}"
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
