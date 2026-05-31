#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <tgvf_checkpoint.pt> [all|readout,query,distribution]" >&2
  exit 2
fi

CHECKPOINT="$1"
TASKS="${2:-${TASKS:-all}}"

if [[ "${TASKS}" == "all" ]]; then
  TASKS="readout,query,distribution"
fi

EVAL_JSONL="${EVAL_JSONL:-data/tgvf_v3_teacher_50k/val_2k.jsonl}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Thinking}"
PROCESSOR_ID="${PROCESSOR_ID:-}"
VARIANT="${VARIANT:-tgvf_v2_bidirectional}"
DEVICE="${DEVICE:-cuda:0}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
NUM_FVT="${NUM_FVT:-none}"
CAPTURE_LAYER="${CAPTURE_LAYER:--1}"
MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION:-512}"
FVT_POSITION_MODE="${FVT_POSITION_MODE:-native_source_grid}"
SEED="${SEED:-20260525}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
CHECKPOINT_STEM="${CHECKPOINT_NAME%.pt}"
OUT_ROOT="${OUT_ROOT:-eval_outputs/tgvf_v3_eval_${CHECKPOINT_STEM}_${RUN_ID}}"

READOUT_MAX_SAMPLES="${READOUT_MAX_SAMPLES:-2000}"
DISTRIBUTION_MAX_SAMPLES="${DISTRIBUTION_MAX_SAMPLES:-2000}"
QUERY_MAX_GROUPS="${QUERY_MAX_GROUPS:-420}"
QUERY_REQUIRE_GROUPS="${QUERY_REQUIRE_GROUPS:-0}"
QUERY_MIN_TARGETS_PER_IMAGE="${QUERY_MIN_TARGETS_PER_IMAGE:-3}"
PROGRESS="${PROGRESS:-1}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"
SHARD_KEY="${SHARD_KEY:-image}"
USE_FVT_CACHE="${USE_FVT_CACHE:-1}"
FVT_CACHE_DIR="${FVT_CACHE_DIR:-${OUT_ROOT}/cache/fvt_cache}"
WANDB_LOG_EVAL="${WANDB_LOG_EVAL:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-tgvf-v3}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TAGS="${WANDB_TAGS:-eval,tgvf-v3,qwen3}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_JSONL}" ]]; then
  echo "Missing eval JSONL: ${EVAL_JSONL}" >&2
  exit 1
fi
if [[ "${EVAL_WORKERS}" -lt 1 ]]; then
  echo "EVAL_WORKERS must be >= 1" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/logs"

common_args=(
  --model-id "${MODEL_ID}"
  --tgvf-checkpoint "${CHECKPOINT}"
  --variant "${VARIANT}"
  --eval-jsonl "${EVAL_JSONL}"
  --device "${DEVICE}"
  --device-map "${DEVICE_MAP}"
  --dtype "${DTYPE}"
  --attn-implementation "${ATTN_IMPL}"
  --num-foveated-tokens "${NUM_FVT}"
  --capture-layer "${CAPTURE_LAYER}"
  --max-image-resolution "${MAX_IMAGE_RESOLUTION}"
  --fvt-position-mode "${FVT_POSITION_MODE}"
  --seed "${SEED}"
)

if [[ -n "${PROCESSOR_ID}" ]]; then
  common_args+=(--processor-id "${PROCESSOR_ID}")
fi

if [[ "${USE_FVT_CACHE}" == "1" ]]; then
  common_args+=(--use-fvt-cache --fvt-cache-dir "${FVT_CACHE_DIR}")
fi

if [[ "${PROGRESS}" == "0" ]]; then
  common_args+=(--no-progress)
fi

cat > "${OUT_ROOT}/run_config.txt" <<EOF
checkpoint=${CHECKPOINT}
tasks=${TASKS}
eval_jsonl=${EVAL_JSONL}
model_id=${MODEL_ID}
processor_id=${PROCESSOR_ID}
variant=${VARIANT}
device=${DEVICE}
device_map=${DEVICE_MAP}
dtype=${DTYPE}
attn_impl=${ATTN_IMPL}
num_fvt=${NUM_FVT}
capture_layer=${CAPTURE_LAYER}
max_image_resolution=${MAX_IMAGE_RESOLUTION}
fvt_position_mode=${FVT_POSITION_MODE}
seed=${SEED}
readout_max_samples=${READOUT_MAX_SAMPLES}
distribution_max_samples=${DISTRIBUTION_MAX_SAMPLES}
query_max_groups=${QUERY_MAX_GROUPS}
query_require_groups=${QUERY_REQUIRE_GROUPS}
query_min_targets_per_image=${QUERY_MIN_TARGETS_PER_IMAGE}
progress=${PROGRESS}
eval_workers=${EVAL_WORKERS}
shard_key=${SHARD_KEY}
use_fvt_cache=${USE_FVT_CACHE}
fvt_cache_dir=${FVT_CACHE_DIR}
wandb_log_eval=${WANDB_LOG_EVAL}
wandb_project=${WANDB_PROJECT}
wandb_run_name=${WANDB_RUN_NAME}
wandb_group=${WANDB_GROUP}
wandb_tags=${WANDB_TAGS}
EOF

has_task() {
  local needle="$1"
  [[ ",${TASKS}," == *",${needle},"* ]]
}

run_one() {
  local name="$1"
  shift
  local log_file="${OUT_ROOT}/logs/${name}.log"
  echo "[v3 eval] ${name} -> ${log_file}"
  "$@" 2>&1 | tee "${log_file}"
}

run_sharded() {
  local name="$1"
  local module="$2"
  local out_subdir="$3"
  local max_samples="$4"
  shift 4
  local extra=("$@")

  if [[ "${EVAL_WORKERS}" == "1" ]]; then
    run_one "${name}" \
      python -m "${module}" \
        "${common_args[@]}" \
        --output-dir "${OUT_ROOT}/${out_subdir}" \
        --max-samples "${max_samples}" \
        "${extra[@]}" \
        --overwrite-output-dir
    return
  fi

  local progress_dir="${OUT_ROOT}/progress/${name}"
  mkdir -p "${progress_dir}"
  local pids=()
  local progress_files=()
  for shard in $(seq 0 $((EVAL_WORKERS - 1))); do
    local shard_name="${name}_shard_${shard}_of_${EVAL_WORKERS}"
    local shard_out="${OUT_ROOT}/${out_subdir}/shard_${shard}_of_${EVAL_WORKERS}"
    local log_file="${OUT_ROOT}/logs/${shard_name}.log"
    local progress_file="${progress_dir}/shard_${shard}.json"
    progress_files+=("${progress_file}")
    rm -f "${progress_file}"
    echo "[v3 eval] ${shard_name} -> ${log_file}"
    python -m "${module}" \
      "${common_args[@]}" \
      --output-dir "${shard_out}" \
      --max-samples "${max_samples}" \
      --num-shards "${EVAL_WORKERS}" \
      --shard-index "${shard}" \
      --shard-key "${SHARD_KEY}" \
      --no-progress \
      --progress-file "${progress_file}" \
      "${extra[@]}" \
      --overwrite-output-dir \
      > "${log_file}" 2>&1 &
    pids+=("$!")
  done

  local monitor_pid=""
  if [[ "${PROGRESS}" != "0" ]]; then
    python -m eval.progress_monitor \
      --label "v3_${name}" \
      --progress-files "${progress_files[@]}" \
      --poll-interval 1.0 &
    monitor_pid="$!"
  fi

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi

  if [[ "${status}" != "0" ]]; then
    echo "[fail] ${name} one or more shards failed. See ${OUT_ROOT}/logs/" >&2
    exit "${status}"
  fi

  local merge_task="${name}"
  if [[ "${name}" == "distribution" ]]; then
    merge_task="distribution"
  fi
  echo "[merge] ${name} -> ${OUT_ROOT}/${out_subdir}"
  local merge_args=(
    --task "${merge_task}"
    --input-root "${OUT_ROOT}/${out_subdir}"
    --output-dir "${OUT_ROOT}/${out_subdir}"
  )
  if [[ "${name}" == "query" ]]; then
    merge_args+=(--max-groups "${QUERY_MAX_GROUPS}" --require-groups "${QUERY_REQUIRE_GROUPS}")
  fi
  python -m eval.merge_results "${merge_args[@]}" \
    > "${OUT_ROOT}/logs/${name}_merge.log" 2>&1
}

if has_task readout; then
  run_sharded readout eval.eval_v3_readout readout "${READOUT_MAX_SAMPLES}"
fi

if has_task query; then
  query_extra=(
    --min-targets-per-image "${QUERY_MIN_TARGETS_PER_IMAGE}"
    --save-score-matrices
  )
  if [[ "${EVAL_WORKERS}" == "1" ]]; then
    query_extra+=(--max-groups "${QUERY_MAX_GROUPS}" --require-groups "${QUERY_REQUIRE_GROUPS}")
  else
    query_extra+=(--require-groups 0)
  fi
  run_sharded query eval.eval_v3_query_sensitivity query_sensitivity "${READOUT_MAX_SAMPLES}" "${query_extra[@]}"
fi

if has_task distribution; then
  run_sharded distribution eval.eval_v3_fvt_distribution fvt_distribution "${DISTRIBUTION_MAX_SAMPLES}"
fi

if [[ "${WANDB_LOG_EVAL}" == "1" ]]; then
  wandb_args=(--eval-root "${OUT_ROOT}")
  if [[ -n "${WANDB_PROJECT}" ]]; then wandb_args+=(--project "${WANDB_PROJECT}"); fi
  if [[ -n "${WANDB_ENTITY}" ]]; then wandb_args+=(--entity "${WANDB_ENTITY}"); fi
  if [[ -n "${WANDB_MODE}" ]]; then wandb_args+=(--mode "${WANDB_MODE}"); fi
  if [[ -n "${WANDB_RUN_NAME}" ]]; then wandb_args+=(--run-name "${WANDB_RUN_NAME}"); fi
  if [[ -n "${WANDB_GROUP}" ]]; then wandb_args+=(--group "${WANDB_GROUP}"); fi
  if [[ -n "${WANDB_TAGS}" ]]; then wandb_args+=(--tags "${WANDB_TAGS}"); fi
  python -m eval.upload_eval_wandb "${wandb_args[@]}" \
    > "${OUT_ROOT}/logs/wandb_upload.log" 2>&1
fi

echo "[v3 eval complete] ${OUT_ROOT}"
