#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_DIR="${SEMANTIC_ASR_SERVICE_DATA_DIR:-service_data/demo}"
API_KEYS="${SEMANTIC_ASR_API_KEYS:-dev-token:dev,admin-token:admin:admin}"
HOST="${SEMANTIC_ASR_SERVICE_HOST:-0.0.0.0}"
PORT="${SEMANTIC_ASR_SERVICE_PORT:-10086}"
CONFIGS_DIR="${SEMANTIC_ASR_CONFIGS_DIR:-configs}"
ALLOWED_CONFIGS="${SEMANTIC_ASR_ALLOWED_CONFIGS:-zh_cn,en_us,vi_vn,ar_sa,de_de,ko_kr,ja_jp,pt_br,ru_ru,th_th,hi_in,hakka}"
TRANSLATION_BASE_URL="${SEMANTIC_ASR_TRANSLATION_BASE_URL:-http://127.0.0.1:10087}"
TRANSLATION_MODEL="${SEMANTIC_ASR_TRANSLATION_MODEL:-hunyuan-mt}"
TRANSLATION_TARGETS="${SEMANTIC_ASR_TRANSLATION_TARGETS:-zh_cn,en_us}"
AUTO_TRANSLATE_TARGETS="${SEMANTIC_ASR_AUTO_TRANSLATE_TARGETS:-zh_cn}"
TRANSLATION_BATCH_SIZE="${SEMANTIC_ASR_TRANSLATION_BATCH_SIZE:-8}"
TRANSLATION_MAX_CONCURRENCY="${SEMANTIC_ASR_TRANSLATION_MAX_CONCURRENCY:-1}"
TRANSLATION_REQUEST_MODE="${SEMANTIC_ASR_TRANSLATION_REQUEST_MODE:-concurrent_single}"
DEMO_USER_HEADER="${SEMANTIC_ASR_DEMO_USER_HEADER:-1}"
WORKER_DEVICES="${SEMANTIC_ASR_DEMO_WORKER_DEVICES:-6,7}"
WORKERS_PER_DEVICE="${SEMANTIC_ASR_DEMO_WORKERS_PER_DEVICE:-4}"
CONDA_ENV="${SEMANTIC_ASR_CONDA_ENV:-fireredasr2s}"

PIDS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT INT TERM

echo "[demo] root=${ROOT_DIR}"
echo "[demo] data_dir=${DATA_DIR}"
echo "[demo] api=http://${HOST}:${PORT}/demo"
echo "[demo] worker_devices=${WORKER_DEVICES} workers_per_device=${WORKERS_PER_DEVICE}"

if command -v curl >/dev/null 2>&1; then
  if curl -fsS --max-time 2 "${TRANSLATION_BASE_URL}/health" >/dev/null 2>&1; then
    echo "[demo] translation service reachable at ${TRANSLATION_BASE_URL}"
  else
    echo "[demo] translation service is not reachable at ${TRANSLATION_BASE_URL}; ASR still works, translation will return 503"
  fi
fi

export SEMANTIC_ASR_SERVICE_DATA_DIR="${DATA_DIR}"
export SEMANTIC_ASR_API_KEYS="${API_KEYS}"
export SEMANTIC_ASR_SERVICE_HOST="${HOST}"
export SEMANTIC_ASR_SERVICE_PORT="${PORT}"
export SEMANTIC_ASR_CONFIGS_DIR="${CONFIGS_DIR}"
export SEMANTIC_ASR_ALLOWED_CONFIGS="${ALLOWED_CONFIGS}"
export SEMANTIC_ASR_TRANSLATION_BASE_URL="${TRANSLATION_BASE_URL}"
export SEMANTIC_ASR_TRANSLATION_MODEL="${TRANSLATION_MODEL}"
export SEMANTIC_ASR_TRANSLATION_TARGETS="${TRANSLATION_TARGETS}"
export SEMANTIC_ASR_AUTO_TRANSLATE_TARGETS="${AUTO_TRANSLATE_TARGETS}"
export SEMANTIC_ASR_TRANSLATION_BATCH_SIZE="${TRANSLATION_BATCH_SIZE}"
export SEMANTIC_ASR_TRANSLATION_MAX_CONCURRENCY="${TRANSLATION_MAX_CONCURRENCY}"
export SEMANTIC_ASR_TRANSLATION_REQUEST_MODE="${TRANSLATION_REQUEST_MODE}"
export SEMANTIC_ASR_DEMO_USER_HEADER="${DEMO_USER_HEADER}"

conda run -n "${CONDA_ENV}" python -u -m semantic_asr_service.app &
PIDS+=("$!")
echo "[demo] started api pid=${PIDS[-1]}"

IFS=',' read -ra DEVICE_LIST <<< "${WORKER_DEVICES}"
for device in "${DEVICE_LIST[@]}"; do
  device="${device//[[:space:]]/}"
  if [[ -z "${device}" ]]; then
    continue
  fi
  for index in $(seq 1 "${WORKERS_PER_DEVICE}"); do
    CUDA_VISIBLE_DEVICES="${device}" conda run -n "${CONDA_ENV}" python -u -m semantic_asr_service.worker --device "${device}" &
    PIDS+=("$!")
    echo "[demo] started worker device=${device} index=${index}/${WORKERS_PER_DEVICE} pid=${PIDS[-1]}"
  done
done

echo "[demo] ready. Press Ctrl-C to stop API and workers."
wait
