#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_ID="${HUNYUAN_MT_MODEL_ID:-tencent/HY-MT1.5-1.8B-FP8}"
MODEL_PATH="${HUNYUAN_MT_MODEL_PATH:-}"
RUNTIME_MODEL_PATH="${HUNYUAN_MT_RUNTIME_MODEL_PATH:-service_data/hy_mt15_18b_fp8_patched}"
SERVED_MODEL_NAME="${HUNYUAN_MT_SERVED_MODEL_NAME:-hunyuan-mt}"
HOST="${HUNYUAN_MT_HOST:-127.0.0.1}"
PORTS="${HUNYUAN_MT_PORTS:-${HUNYUAN_MT_PORT:-10087}}"
DEVICE="${HUNYUAN_MT_DEVICE:-5}"
MAX_CONCURRENT="${HUNYUAN_MT_MAX_CONCURRENT:-24}"
CONDA_ENV="${HUNYUAN_MT_CONDA_ENV:-llm}"

if [[ -z "${MODEL_PATH}" ]]; then
  CACHE_ROOT="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
  CACHE_MODEL_DIR="${CACHE_ROOT}/models--${MODEL_ID//\//--}"
  MODEL_PATH="$(find "${CACHE_MODEL_DIR}/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "${MODEL_PATH}" ]]; then
    LOWER_MODEL_ID="${MODEL_ID,,}"
    CACHE_MODEL_DIR="${CACHE_ROOT}/models--${LOWER_MODEL_ID//\//--}"
    MODEL_PATH="$(find "${CACHE_MODEL_DIR}/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
  fi
fi

if [[ -z "${MODEL_PATH}" || ! -d "${MODEL_PATH}" ]]; then
  echo "[hunyuan-mt] model snapshot not found for ${MODEL_ID}" >&2
  echo "[hunyuan-mt] set HUNYUAN_MT_MODEL_PATH=/path/to/model_snapshot after download completes" >&2
  exit 1
fi

echo "[hunyuan-mt] root=${ROOT_DIR}"
echo "[hunyuan-mt] model_id=${MODEL_ID}"
echo "[hunyuan-mt] model=${MODEL_PATH}"
echo "[hunyuan-mt] ports=${PORTS}"
echo "[hunyuan-mt] device=${DEVICE} max_concurrent=${MAX_CONCURRENT}"

IFS=',' read -ra PORT_LIST <<< "${PORTS}"
PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT INT TERM

for port in "${PORT_LIST[@]}"; do
  port="${port//[[:space:]]/}"
  if [[ -z "${port}" ]]; then
    continue
  fi
  instance_runtime_path="${RUNTIME_MODEL_PATH}_${port}"
  echo "[hunyuan-mt] starting api=http://${HOST}:${port} runtime=${instance_runtime_path}"
  CUDA_VISIBLE_DEVICES="${DEVICE}" conda run --no-capture-output -n "${CONDA_ENV}" python -u -m semantic_asr_service.hunyuan_mt_server \
    --model-path "${MODEL_PATH}" \
    --runtime-model-path "${instance_runtime_path}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${port}" \
    --max-concurrent "${MAX_CONCURRENT}" &
  PIDS+=("$!")
done

wait
