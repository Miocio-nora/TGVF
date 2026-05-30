#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <tgvf_checkpoint.pt> [all|readout,query,distribution,end2end]" >&2
  exit 2
fi

CHECKPOINT="$1"
TASKS="${2:-${TASKS:-all}}"

if [[ "${TASKS}" == "all" ]]; then
  TASKS="readout,query,distribution,end2end"
fi

EVAL_JSONL="${EVAL_JSONL:-data/tgvf_teacher/generated/runs/teacher_run_val_000001/final/tgvf_teacher_items.accepted_balanced_2k_val.jsonl}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2-VL-2B-Instruct}"
PROCESSOR_PATH="${PROCESSOR_PATH:-}"
VARIANT="${VARIANT:-foveal_cross_merger}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bf16}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
NUM_FVT="${NUM_FVT:-16}"
CAPTURE_LAYER="${CAPTURE_LAYER:--1}"
SEED="${SEED:-20260525}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
CHECKPOINT_STEM="${CHECKPOINT_NAME%.pt}"
OUT_ROOT="${OUT_ROOT:-eval_outputs/tgvf_eval_${CHECKPOINT_STEM}_${RUN_ID}}"

# Default validation scale.
READOUT_MAX_SAMPLES="${READOUT_MAX_SAMPLES:-2000}"
DISTRIBUTION_MAX_SAMPLES="${DISTRIBUTION_MAX_SAMPLES:-2000}"
QUERY_MAX_GROUPS="${QUERY_MAX_GROUPS:-420}"
QUERY_REQUIRE_GROUPS="${QUERY_REQUIRE_GROUPS:-420}"
QUERY_MIN_TARGETS_PER_IMAGE="${QUERY_MIN_TARGETS_PER_IMAGE:-3}"
END2END_MAX_SAMPLES="${END2END_MAX_SAMPLES:-300}"
END2END_MODE="${END2END_MODE:-forced}"
END2END_ANSWER_MAX_NEW_TOKENS="${END2END_ANSWER_MAX_NEW_TOKENS:-64}"
END2END_CAPTURE_MAX_NEW_TOKENS="${END2END_CAPTURE_MAX_NEW_TOKENS:-128}"
INCLUDE_DIRECT_QWEN="${INCLUDE_DIRECT_QWEN:-1}"
PROGRESS="${PROGRESS:-1}"

# Current eval path is sample-wise; parallel shards are the practical speedup.
# B200 memory is large enough for several Qwen2-VL-2B processes on one device.
EVAL_WORKERS="${EVAL_WORKERS:-10}"
SHARD_KEY="${SHARD_KEY:-image}"
STRICT_QUERY_REQUIRE="${STRICT_QUERY_REQUIRE:-0}"

# Cache is on by default because readout/query/distribution reuse the same D tensors.
USE_FVT_CACHE="${USE_FVT_CACHE:-1}"
FVT_CACHE_DIR="${FVT_CACHE_DIR:-${OUT_ROOT}/cache/fvt_cache}"
WANDB_LOG_EVAL="${WANDB_LOG_EVAL:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TAGS="${WANDB_TAGS:-eval,tgvf}"

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
  --model-path "${MODEL_PATH}"
  --tgvf-checkpoint "${CHECKPOINT}"
  --variant "${VARIANT}"
  --eval-jsonl "${EVAL_JSONL}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --attn-implementation "${ATTN_IMPL}"
  --num-foveated-tokens "${NUM_FVT}"
  --capture-layer "${CAPTURE_LAYER}"
  --seed "${SEED}"
)

if [[ -n "${PROCESSOR_PATH}" ]]; then
  common_args+=(--processor-path "${PROCESSOR_PATH}")
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
model_path=${MODEL_PATH}
processor_path=${PROCESSOR_PATH}
variant=${VARIANT}
device=${DEVICE}
dtype=${DTYPE}
attn_impl=${ATTN_IMPL}
num_fvt=${NUM_FVT}
capture_layer=${CAPTURE_LAYER}
seed=${SEED}
readout_max_samples=${READOUT_MAX_SAMPLES}
distribution_max_samples=${DISTRIBUTION_MAX_SAMPLES}
query_max_groups=${QUERY_MAX_GROUPS}
query_require_groups=${QUERY_REQUIRE_GROUPS}
query_min_targets_per_image=${QUERY_MIN_TARGETS_PER_IMAGE}
end2end_max_samples=${END2END_MAX_SAMPLES}
end2end_mode=${END2END_MODE}
include_direct_qwen=${INCLUDE_DIRECT_QWEN}
progress=${PROGRESS}
eval_workers=${EVAL_WORKERS}
shard_key=${SHARD_KEY}
strict_query_require=${STRICT_QUERY_REQUIRE}
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

ceil_div() {
  local numerator="$1"
  local denominator="$2"
  echo $(( (numerator + denominator - 1) / denominator ))
}

run_one() {
  local name="$1"
  shift
  local log_file="${OUT_ROOT}/logs/${name}.log"
  echo "[eval] ${name} -> ${log_file}"
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
    echo "[eval] ${shard_name} -> ${log_file}"
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
      --label "${name}" \
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
    python -m eval.progress_monitor \
      --label "${name}" \
      --progress-files "${progress_files[@]}" \
      --poll-interval 0.1 &
    local final_monitor_pid="$!"
    sleep 0.2
    kill "${final_monitor_pid}" 2>/dev/null || true
    wait "${final_monitor_pid}" 2>/dev/null || true
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
  run_sharded readout eval.eval_readout readout "${READOUT_MAX_SAMPLES}"
fi

if has_task query; then
  query_extra=(
    --min-targets-per-image "${QUERY_MIN_TARGETS_PER_IMAGE}"
    --save-score-matrices
  )
  if [[ "${EVAL_WORKERS}" == "1" ]]; then
    query_extra+=(--max-groups "${QUERY_MAX_GROUPS}" --require-groups "${QUERY_REQUIRE_GROUPS}")
  else
    # Do not divide QUERY_MAX_GROUPS per shard. Image-hash shards are uneven,
    # so a local cap of ceil(total/workers) can under-evaluate globally.
    query_extra+=(--require-groups 0)
  fi
  run_sharded query eval.eval_query_sensitivity query_sensitivity "${READOUT_MAX_SAMPLES}" "${query_extra[@]}"
  if [[ "${QUERY_REQUIRE_GROUPS}" -gt 0 ]]; then
    python - "${OUT_ROOT}/query_sensitivity/query_sensitivity_report.json" "${QUERY_REQUIRE_GROUPS}" <<'PY_QUERY_REQUIRE'
import json
import sys
from pathlib import Path
report_path = Path(sys.argv[1])
required = int(sys.argv[2])
report = json.loads(report_path.read_text())
evaluated = int(report.get("num_groups_evaluated", 0))
if evaluated < required:
    raise SystemExit(f"query groups evaluated {evaluated} < required {required}")
PY_QUERY_REQUIRE
  fi
fi

if has_task distribution; then
  run_sharded distribution eval.eval_fvt_distribution fvt_distribution "${DISTRIBUTION_MAX_SAMPLES}" --save-histograms
fi

if has_task end2end; then
  end2end_extra=(
    --mode "${END2END_MODE}"
    --answer-max-new-tokens "${END2END_ANSWER_MAX_NEW_TOKENS}"
    --capture-max-new-tokens "${END2END_CAPTURE_MAX_NEW_TOKENS}"
  )
  if [[ "${INCLUDE_DIRECT_QWEN}" == "0" ]]; then
    end2end_extra+=(--skip-direct-qwen)
  fi
  run_sharded end2end eval.eval_tgvf_end2end "end2end_${END2END_MODE}" "${END2END_MAX_SAMPLES}" "${end2end_extra[@]}"
fi

if [[ "${WANDB_LOG_EVAL}" == "1" && -n "${WANDB_PROJECT}" ]]; then
  upload_args=(
    --eval-root "${OUT_ROOT}"
    --project "${WANDB_PROJECT}"
    --run-name "${WANDB_RUN_NAME:-$(basename "${OUT_ROOT}")}"
    --group "${WANDB_GROUP}"
    --tags "${WANDB_TAGS}"
  )
  if [[ -n "${WANDB_ENTITY}" ]]; then
    upload_args+=(--entity "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_MODE}" ]]; then
    upload_args+=(--mode "${WANDB_MODE}")
  fi
  python -m eval.upload_eval_wandb "${upload_args[@]}"
fi

echo "Eval output root: ${OUT_ROOT}"
