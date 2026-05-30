#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
if [[ "${MODE}" != "full" && "${MODE}" != "smoke" ]]; then
  echo "Usage: $0 [full|smoke]" >&2
  exit 2
fi

TRAIN_FILE="${TRAIN_FILE:-data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001/final/tgvf_teacher_items.accepted.jsonl}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2-VL-2B-Instruct}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-outputs/tgvf_fvt/20k_v1_m128_ablation_${RUN_ID}}"
WANDB_PROJECT="${WANDB_PROJECT:-tgvf}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-}"
WANDB_GROUP="${WANDB_GROUP:-tgvf_20k_v1_m128_ablation_${RUN_ID}}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
MANIFOLD_WEIGHT="${MANIFOLD_WEIGHT:-0.01}"
SAME_IMAGE_NEGATIVE_WEIGHT="${SAME_IMAGE_NEGATIVE_WEIGHT:-1.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
SEED="${SEED:-20260525}"
PROGRESS="${PROGRESS:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-}"
EVAL_TASKS="${EVAL_TASKS:-all}"
EVAL_JSONL="${EVAL_JSONL:-data/tgvf_teacher/generated/runs/teacher_run_val_visual_cue_v1_000001/final/tgvf_teacher_items.accepted.jsonl}"
EVAL_WORKERS="${EVAL_WORKERS:-10}"
READOUT_MAX_SAMPLES="${READOUT_MAX_SAMPLES:-2007}"
DISTRIBUTION_MAX_SAMPLES="${DISTRIBUTION_MAX_SAMPLES:-2007}"
QUERY_MAX_GROUPS="${QUERY_MAX_GROUPS:-419}"
QUERY_REQUIRE_GROUPS="${QUERY_REQUIRE_GROUPS:-419}"
READOUT_DROPOUT="${READOUT_DROPOUT:-0.0}"

if [[ "${MODE}" == "smoke" ]]; then
  EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-0}"
  WANDB_MODE="${WANDB_MODE:-disabled}"
  LR_SCHEDULER="${LR_SCHEDULER:-none}"
  ABLATION_MAX_STEPS="${ABLATION_MAX_STEPS:-2}"
  SAVE_EVERY="${SAVE_EVERY:-2}"
  LOG_EVERY="${LOG_EVERY:-1}"
  ABLATION_BATCH_SIZE="${ABLATION_BATCH_SIZE:-2}"
  ABLATION_GRAD_ACCUM="${ABLATION_GRAD_ACCUM:-1}"
else
  EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-1}"
  WANDB_MODE="${WANDB_MODE:-online}"
  LR_SCHEDULER="${LR_SCHEDULER:-warmup_cosine}"
  ABLATION_BATCH_SIZE="${ABLATION_BATCH_SIZE:-6}"
  if [[ -z "${ABLATION_MAX_STEPS:-}" ]]; then
    ABLATION_MAX_STEPS=$(python - "${TRAIN_FILE}" "${ABLATION_BATCH_SIZE}" <<'PY_STEPS'
import json
import sys
from collections import defaultdict
path = sys.argv[1]
batch_size = int(sys.argv[2])
by_image = defaultdict(int)
with open(path) as handle:
    for line in handle:
        if not line.strip():
            continue
        record = json.loads(line)
        key = record.get("image_id") or record.get("stable_image_uid") or record.get("image")
        by_image[key] += 1
steps = 0
for count in by_image.values():
    if count < 2:
        continue
    full, remainder = divmod(count, batch_size)
    steps += full
    if remainder >= 2:
        steps += 1
print(steps)
PY_STEPS
)
  fi
  SAVE_EVERY="${SAVE_EVERY:-5000}"
  LOG_EVERY="${LOG_EVERY:-25}"
  ABLATION_GRAD_ACCUM="${ABLATION_GRAD_ACCUM:-1}"
fi

mkdir -p "${RUN_ROOT}/logs"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing train file: ${TRAIN_FILE}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_JSONL}" ]]; then
  echo "Missing eval JSONL: ${EVAL_JSONL}" >&2
  exit 1
fi

pids=()
names=()
logs=()
maxes=()
states=()
monitor_pid=""

cleanup() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
  fi
  if [[ ${#pids[@]} -gt 0 ]]; then
    echo "Stopping launched TGVF jobs..." >&2
    kill "${pids[@]}" 2>/dev/null || true
    wait 2>/dev/null || true
  fi
}
trap cleanup INT TERM

latest_logged_step() {
  local log_file="$1"
  if [[ ! -f "${log_file}" ]]; then
    echo 0
    return
  fi
  awk -F': ' '/"step":/ {gsub(/,/, "", $2); step=$2} END {print step + 0}' "${log_file}"
}

write_state() {
  local state_file="$1"
  local state="$2"
  printf '%s\n' "${state}" > "${state_file}"
}

read_state() {
  local state_file="$1"
  if [[ -f "${state_file}" ]]; then
    head -n 1 "${state_file}"
  else
    echo pending
  fi
}

progress_bar() {
  local step="$1"
  local max_steps="$2"
  local width=18
  local filled=0
  if [[ "${max_steps}" -gt 0 ]]; then
    filled=$(( step * width / max_steps ))
  fi
  if [[ "${filled}" -gt "${width}" ]]; then
    filled="${width}"
  fi
  local empty=$(( width - filled ))
  printf '%*s' "${filled}" '' | tr ' ' '#'
  printf '%*s' "${empty}" '' | tr ' ' '-'
}

monitor_progress() {
  if [[ "${PROGRESS}" == "0" ]]; then
    return
  fi
  while true; do
    local running=0
    local line="[progress]"
    for idx in "${!pids[@]}"; do
      local pid="${pids[$idx]}"
      local name="${names[$idx]}"
      local log="${logs[$idx]}"
      local max_steps="${maxes[$idx]}"
      local state_file="${states[$idx]}"
      local step
      step=$(latest_logged_step "${log}")
      if kill -0 "${pid}" 2>/dev/null; then
        running=1
      fi
      local phase
      phase=$(read_state "${state_file}")
      local pct="0.0"
      if [[ "${max_steps}" -gt 0 ]]; then
        pct=$(awk -v s="${step}" -v m="${max_steps}" 'BEGIN { v=100*s/m; if (v > 100) v=100; printf "%.1f", v }')
      fi
      line+=" ${name} [$(progress_bar "${step}" "${max_steps}")] ${step}/${max_steps} ${pct}% ${phase} |"
    done
    printf '\r%s' "${line% |}"
    if [[ "${running}" -eq 0 ]]; then
      printf '\n'
      break
    fi
    sleep "${PROGRESS_INTERVAL}"
  done
}

latest_checkpoint() {
  local out_dir="$1"
  python - "${out_dir}" <<'PY_CKPT'
import re
import sys
from pathlib import Path
out_dir = Path(sys.argv[1])
checkpoints = list(out_dir.glob("checkpoint_step_*.pt"))
if not checkpoints:
    sys.exit(1)
def step(path: Path) -> int:
    match = re.search(r"checkpoint_step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1
print(max(checkpoints, key=step))
PY_CKPT
}

run_eval_for_job() {
  local name="$1"
  local out_dir="$2"
  local gpu="$3"
  local variant="$4"
  local num_fvt="$5"
  local checkpoint
  if ! checkpoint=$(latest_checkpoint "${out_dir}"); then
    echo "[eval-skip] ${name}: no checkpoint found in ${out_dir}" >&2
    return 1
  fi
  local checkpoint_stem
  checkpoint_stem="$(basename "${checkpoint}" .pt)"
  local eval_root="${out_dir}/eval_${checkpoint_stem}"
  echo "[eval] ${name}: ${checkpoint} -> ${eval_root}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  OUT_ROOT="${eval_root}" \
  EVAL_JSONL="${EVAL_JSONL}" \
  MODEL_PATH="${MODEL_NAME_OR_PATH}" \
  VARIANT="${variant}" \
  DEVICE="cuda:0" \
  DTYPE="${TORCH_DTYPE}" \
  ATTN_IMPL="${ATTN_IMPL}" \
  NUM_FVT="${num_fvt}" \
  SEED="${SEED}" \
  EVAL_WORKERS="${EVAL_WORKERS}" \
  READOUT_MAX_SAMPLES="${READOUT_MAX_SAMPLES}" \
  DISTRIBUTION_MAX_SAMPLES="${DISTRIBUTION_MAX_SAMPLES}" \
  QUERY_MAX_GROUPS="${QUERY_MAX_GROUPS}" \
  QUERY_REQUIRE_GROUPS="${QUERY_REQUIRE_GROUPS}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_ENTITY="${WANDB_ENTITY}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_GROUP="${WANDB_GROUP}" \
  WANDB_RUN_NAME="${WANDB_GROUP}_${name}_eval" \
  WANDB_TAGS="eval,20k,visual_cue_v1,m128_ablation,${name},${variant}" \
  eval/run_tgvf_eval_suite.sh "${checkpoint}" "${EVAL_TASKS}"
}

wandb_args() {
  local run_name="$1"
  local tags="$2"
  local args=(--wandb-mode "${WANDB_MODE}" --wandb-run-name "${run_name}" --wandb-group "${WANDB_GROUP}" --wandb-tags "${tags}")
  if [[ -n "${WANDB_PROJECT}" ]]; then
    args+=(--wandb-project "${WANDB_PROJECT}")
  fi
  if [[ -n "${WANDB_ENTITY}" ]]; then
    args+=(--wandb-entity "${WANDB_ENTITY}")
  fi
  printf '%q ' "${args[@]}"
}

launch_experiment() {
  local gpu="$1"
  local exp_name="$2"
  local variant="$3"
  local num_fvt="$4"
  local same_image_mode="$5"
  local tags="$6"
  local out_dir="${RUN_ROOT}/${exp_name}"
  local log_file="${RUN_ROOT}/logs/${exp_name}.log"
  local eval_log_file="${RUN_ROOT}/logs/${exp_name}_eval.log"
  local state_file="${RUN_ROOT}/logs/${exp_name}.state"
  mkdir -p "${out_dir}"

  local wb
  wb=$(wandb_args "${WANDB_GROUP}_${exp_name}" "${tags}")

  local cmd="python scripts/train_tgvf_fvt.py \
    --train-file '${TRAIN_FILE}' \
    --output-dir '${out_dir}' \
    --model-name-or-path '${MODEL_NAME_OR_PATH}' \
    --variant '${variant}' \
    --num-foveated-tokens '${num_fvt}' \
    --device cuda:0 \
    --torch-dtype '${TORCH_DTYPE}' \
    --attn-implementation '${ATTN_IMPL}' \
    --batch-size '${ABLATION_BATCH_SIZE}' \
    --gradient-accumulation-steps '${ABLATION_GRAD_ACCUM}' \
    --learning-rate '${LEARNING_RATE}' \
    --lr-scheduler '${LR_SCHEDULER}' \
    --warmup-ratio '${WARMUP_RATIO}' \
    --warmup-steps '${WARMUP_STEPS}' \
    --min-lr-ratio '${MIN_LR_RATIO}' \
    --max-grad-norm '${MAX_GRAD_NORM}' \
    --max-steps '${ABLATION_MAX_STEPS}' \
    --save-every '${SAVE_EVERY}' \
    --log-every '${LOG_EVERY}' \
    --loss-gen 1.0 \
    --loss-visual-token-manifold '${MANIFOLD_WEIGHT}' \
    --loss-same-image-negative '${SAME_IMAGE_NEGATIVE_WEIGHT}' \
    --same-image-negative-mode '${same_image_mode}' \
    --readout-prompt-target-dropout '${READOUT_DROPOUT}' \
    --loss-contrastive-alignment 0.0 \
    --group-batches-by-image \
    --seed '${SEED}' \
    ${wb}"

  printf '%s\n' "CUDA_VISIBLE_DEVICES=${gpu} ${cmd}" > "${out_dir}/command.sh"
  echo "[launch] GPU ${gpu}: ${exp_name} -> ${log_file}"
  (
    write_state "${state_file}" training
    set +e
    bash -lc "CUDA_VISIBLE_DEVICES=${gpu} ${cmd}" > "${log_file}" 2>&1
    code=$?
    set -e
    if [[ "${code}" != "0" ]]; then
      write_state "${state_file}" failed
      exit "${code}"
    fi
    if [[ "${EVAL_AFTER_TRAIN}" == "1" ]]; then
      write_state "${state_file}" eval
      echo "[eval-launch] GPU ${gpu}: ${exp_name} -> ${eval_log_file}"
      set +e
      run_eval_for_job "${exp_name}" "${out_dir}" "${gpu}" "${variant}" "${num_fvt}" > "${eval_log_file}" 2>&1
      code=$?
      set -e
      if [[ "${code}" != "0" ]]; then
        write_state "${state_file}" failed
        exit "${code}"
      fi
    fi
    write_state "${state_file}" done
  ) &
  pids+=("$!")
  names+=("${exp_name}")
  logs+=("${log_file}")
  maxes+=("${ABLATION_MAX_STEPS}")
  states+=("${state_file}")
}

cat > "${RUN_ROOT}/run_config.txt" <<EOF
mode=${MODE}
run_id=${RUN_ID}
run_root=${RUN_ROOT}
train_file=${TRAIN_FILE}
model=${MODEL_NAME_OR_PATH}
wandb_project=${WANDB_PROJECT}
wandb_group=${WANDB_GROUP}
learning_rate=${LEARNING_RATE}
lr_scheduler=${LR_SCHEDULER}
warmup_ratio=${WARMUP_RATIO}
warmup_steps=${WARMUP_STEPS}
min_lr_ratio=${MIN_LR_RATIO}
ablation_max_steps=${ABLATION_MAX_STEPS}
ablation_batch_size=${ABLATION_BATCH_SIZE}
ablation_grad_accum=${ABLATION_GRAD_ACCUM}
losses_all=L_gen + L_visual_token_manifold + L_same_image_negative
readout_dropout=${READOUT_DROPOUT}
save_every=${SAVE_EVERY}
log_every=${LOG_EVERY}
progress=${PROGRESS}
progress_interval=${PROGRESS_INTERVAL}
eval_after_train=${EVAL_AFTER_TRAIN}
eval_tasks=${EVAL_TASKS}
eval_jsonl=${EVAL_JSONL}
eval_workers=${EVAL_WORKERS}
readout_max_samples=${READOUT_MAX_SAMPLES}
distribution_max_samples=${DISTRIBUTION_MAX_SAMPLES}
query_max_groups=${QUERY_MAX_GROUPS}
query_require_groups=${QUERY_REQUIRE_GROUPS}
EOF

launch_experiment 0 token_direct_m128 token_direct 128 cyclic_margin "train,20k,visual_cue_v1,m128_ablation,token_direct,m128,cyclic_margin"
launch_experiment 1 token_direct_num_output_none token_direct none cyclic_margin "train,20k,visual_cue_v1,m128_ablation,token_direct,num_output_none,cyclic_margin"
launch_experiment 2 foveal_cross_merger_matrixce_m128 foveal_cross_merger 128 matrix_ce "train,20k,visual_cue_v1,m128_ablation,foveal_cross_merger,m128,matrix_ce"
launch_experiment 3 pooled_m128 pooled 128 cyclic_margin "train,20k,visual_cue_v1,m128_ablation,pooled,m128,cyclic_margin"

if [[ "${PROGRESS}" != "0" ]]; then
  monitor_progress &
  monitor_pid="$!"
fi

status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  name="${names[$idx]}"
  log="${logs[$idx]}"
  state_file="${states[$idx]}"
  if wait "${pid}"; then
    echo "[done] ${name}"
  else
    code=$?
    phase=$(read_state "${state_file}")
    echo "[fail] ${name} exited with ${code} during ${phase}; train log: ${log}; eval log: ${RUN_ROOT}/logs/${name}_eval.log" >&2
    status=1
  fi
done

if [[ -n "${monitor_pid}" ]]; then
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  printf '\n'
fi

trap - INT TERM

echo "Run root: ${RUN_ROOT}"
exit "${status}"
