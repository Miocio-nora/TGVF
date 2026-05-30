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
RUN_ROOT="${RUN_ROOT:-outputs/tgvf_fvt/20k_v2_bidirectional_cyclic_negative_streaming_${RUN_ID}}"
GPU="${GPU:-3}"
WANDB_PROJECT="${WANDB_PROJECT:-tgvf}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-}"
WANDB_GROUP="${WANDB_GROUP:-tgvf_20k_v2_bidirectional_cyclic_negative_streaming_${RUN_ID}}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
NUM_FVT="${NUM_FVT:-none}"
MANIFOLD_WEIGHT="${MANIFOLD_WEIGHT:-0.01}"
SAME_IMAGE_NEGATIVE_WEIGHT="${SAME_IMAGE_NEGATIVE_WEIGHT:-1.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
SEED="${SEED:-20260525}"
READOUT_DROPOUT="${READOUT_DROPOUT:-0.0}"
SAVE_ACTIVATIONS_ON_CPU="${SAVE_ACTIVATIONS_ON_CPU:-0}"
EMPTY_CACHE_EVERY_STEP="${EMPTY_CACHE_EVERY_STEP:-0}"
STREAMING_BACKWARD="${STREAMING_BACKWARD:-1}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-}"
EVAL_TASKS="${EVAL_TASKS:-all}"
EVAL_JSONL="${EVAL_JSONL:-data/tgvf_teacher/generated/runs/teacher_run_val_visual_cue_v1_000001/final/tgvf_teacher_items.accepted.jsonl}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
READOUT_MAX_SAMPLES="${READOUT_MAX_SAMPLES:-2007}"
DISTRIBUTION_MAX_SAMPLES="${DISTRIBUTION_MAX_SAMPLES:-2007}"
QUERY_MAX_GROUPS="${QUERY_MAX_GROUPS:-419}"
QUERY_REQUIRE_GROUPS="${QUERY_REQUIRE_GROUPS:-419}"
END2END_MAX_SAMPLES="${END2END_MAX_SAMPLES:-300}"
WANDB_LOG_EVAL="${WANDB_LOG_EVAL:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${MODE}" == "smoke" ]]; then
  EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-0}"
  WANDB_MODE="${WANDB_MODE:-disabled}"
  LR_SCHEDULER="${LR_SCHEDULER:-none}"
  MAX_STEPS="${MAX_STEPS:-2}"
  SAVE_EVERY="${SAVE_EVERY:-2}"
  LOG_EVERY="${LOG_EVERY:-1}"
  BATCH_SIZE="${BATCH_SIZE:-6}"
  GRAD_ACCUM="${GRAD_ACCUM:-1}"
else
  EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-1}"
  WANDB_MODE="${WANDB_MODE:-online}"
  LR_SCHEDULER="${LR_SCHEDULER:-warmup_cosine}"
  BATCH_SIZE="${BATCH_SIZE:-6}"
  if [[ -z "${MAX_STEPS:-}" ]]; then
    MAX_STEPS=$(python - "${TRAIN_FILE}" "${BATCH_SIZE}" <<'PY_STEPS'
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
  GRAD_ACCUM="${GRAD_ACCUM:-1}"
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing train file: ${TRAIN_FILE}" >&2
  exit 1
fi
if [[ "${EVAL_AFTER_TRAIN}" == "1" && ! -f "${EVAL_JSONL}" ]]; then
  echo "Missing eval JSONL: ${EVAL_JSONL}" >&2
  exit 1
fi

OUT_DIR="${RUN_ROOT}/tgvf_v2_bidirectional_cyclic_negative"
mkdir -p "${OUT_DIR}" "${RUN_ROOT}/logs"
LOG_FILE="${RUN_ROOT}/logs/tgvf_v2_bidirectional_cyclic_negative.log"
EVAL_LOG_FILE="${RUN_ROOT}/logs/tgvf_v2_bidirectional_cyclic_negative_eval.log"
STATE_FILE="${RUN_ROOT}/logs/tgvf_v2_bidirectional_cyclic_negative.state"
trap 'printf "%s\n" "failed" > "${STATE_FILE}"' ERR

wandb_args=(--wandb-mode "${WANDB_MODE}" --wandb-run-name "${WANDB_GROUP}_tgvf_v2_bidirectional_cyclic_negative" --wandb-group "${WANDB_GROUP}" --wandb-tags "train,20k,visual_cue_v1,tgvf_v2,tgvf_v2_bidirectional,cyclic_margin,streaming")
if [[ -n "${WANDB_PROJECT}" ]]; then
  wandb_args+=(--wandb-project "${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  wandb_args+=(--wandb-entity "${WANDB_ENTITY}")
fi

memory_args=()
if [[ "${SAVE_ACTIVATIONS_ON_CPU}" == "1" ]]; then
  memory_args+=(--save-activations-on-cpu)
fi
if [[ "${EMPTY_CACHE_EVERY_STEP}" == "1" ]]; then
  memory_args+=(--empty-cache-every-step)
fi
if [[ "${STREAMING_BACKWARD}" == "1" ]]; then
  memory_args+=(--streaming-matrix-ce-backward)
fi

train_cmd=(
  python -u scripts/train_tgvf_fvt.py
  --train-file "${TRAIN_FILE}"
  --output-dir "${OUT_DIR}"
  --model-name-or-path "${MODEL_NAME_OR_PATH}"
  --variant tgvf_v2_bidirectional
  --num-foveated-tokens "${NUM_FVT}"
  --device cuda:0
  --torch-dtype "${TORCH_DTYPE}"
  --attn-implementation "${ATTN_IMPL}"
  --batch-size "${BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --learning-rate "${LEARNING_RATE}"
  --lr-scheduler "${LR_SCHEDULER}"
  --warmup-ratio "${WARMUP_RATIO}"
  --warmup-steps "${WARMUP_STEPS}"
  --min-lr-ratio "${MIN_LR_RATIO}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --max-steps "${MAX_STEPS}"
  --save-every "${SAVE_EVERY}"
  --log-every "${LOG_EVERY}"
  --loss-gen 1.0
  --loss-visual-token-manifold "${MANIFOLD_WEIGHT}"
  --loss-same-image-negative "${SAME_IMAGE_NEGATIVE_WEIGHT}"
  --same-image-negative-mode cyclic_margin
  --readout-prompt-target-dropout "${READOUT_DROPOUT}"
  --loss-contrastive-alignment 0.0
  --group-batches-by-image
  --seed "${SEED}"
  "${memory_args[@]}"
  "${wandb_args[@]}"
)

cat > "${RUN_ROOT}/run_config.txt" <<EOF
mode=${MODE}
run_id=${RUN_ID}
run_root=${RUN_ROOT}
gpu=${GPU}
train_file=${TRAIN_FILE}
model=${MODEL_NAME_OR_PATH}
variant=tgvf_v2_bidirectional
job=tgvf_v2_bidirectional_cyclic_negative
same_image_negative_mode=cyclic_margin
streaming_backward=${STREAMING_BACKWARD}
learning_rate=${LEARNING_RATE}
lr_scheduler=${LR_SCHEDULER}
warmup_ratio=${WARMUP_RATIO}
warmup_steps=${WARMUP_STEPS}
min_lr_ratio=${MIN_LR_RATIO}
max_steps=${MAX_STEPS}
batch_size=${BATCH_SIZE}
grad_accum=${GRAD_ACCUM}
num_fvt=${NUM_FVT}
losses=L_gen + L_visual_token_manifold + L_same_image_negative_cyclic_margin
readout_dropout=${READOUT_DROPOUT}
v2_conditioning_stage=pre_qwen_visual_merger
v2_final_fvt=frozen_qwen_visual_merger
trainable_merger=0
save_every=${SAVE_EVERY}
log_every=${LOG_EVERY}
eval_after_train=${EVAL_AFTER_TRAIN}
eval_tasks=${EVAL_TASKS}
eval_jsonl=${EVAL_JSONL}
eval_workers=${EVAL_WORKERS}
readout_max_samples=${READOUT_MAX_SAMPLES}
distribution_max_samples=${DISTRIBUTION_MAX_SAMPLES}
query_max_groups=${QUERY_MAX_GROUPS}
query_require_groups=${QUERY_REQUIRE_GROUPS}
end2end_max_samples=${END2END_MAX_SAMPLES}
wandb_log_eval=${WANDB_LOG_EVAL}
wandb_project=${WANDB_PROJECT}
wandb_group=${WANDB_GROUP}
save_activations_on_cpu=${SAVE_ACTIVATIONS_ON_CPU}
empty_cache_every_step=${EMPTY_CACHE_EVERY_STEP}
pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}
EOF

printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU}" > "${OUT_DIR}/command.sh"
printf '%q ' "${train_cmd[@]}" >> "${OUT_DIR}/command.sh"
printf '\n' >> "${OUT_DIR}/command.sh"

echo "training" > "${STATE_FILE}"
echo "[launch] GPU ${GPU}: tgvf_v2_bidirectional_cyclic_negative -> ${LOG_FILE}"
CUDA_VISIBLE_DEVICES="${GPU}" "${train_cmd[@]}" > "${LOG_FILE}" 2>&1

if [[ "${EVAL_AFTER_TRAIN}" == "1" ]]; then
  checkpoint=$(python - "${OUT_DIR}" <<'PY_CKPT'
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
)
  checkpoint_stem="$(basename "${checkpoint}" .pt)"
  eval_root="${OUT_DIR}/eval_${checkpoint_stem}"
  echo "eval" > "${STATE_FILE}"
  echo "[eval] ${checkpoint} -> ${eval_root}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  OUT_ROOT="${eval_root}" \
  EVAL_JSONL="${EVAL_JSONL}" \
  MODEL_PATH="${MODEL_NAME_OR_PATH}" \
  VARIANT="tgvf_v2_bidirectional" \
  DEVICE="cuda:0" \
  DTYPE="${TORCH_DTYPE}" \
  ATTN_IMPL="${ATTN_IMPL}" \
  NUM_FVT="${NUM_FVT}" \
  SEED="${SEED}" \
  EVAL_WORKERS="${EVAL_WORKERS}" \
  READOUT_MAX_SAMPLES="${READOUT_MAX_SAMPLES}" \
  DISTRIBUTION_MAX_SAMPLES="${DISTRIBUTION_MAX_SAMPLES}" \
  QUERY_MAX_GROUPS="${QUERY_MAX_GROUPS}" \
  QUERY_REQUIRE_GROUPS="${QUERY_REQUIRE_GROUPS}" \
  END2END_MAX_SAMPLES="${END2END_MAX_SAMPLES}" \
  WANDB_LOG_EVAL="${WANDB_LOG_EVAL}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_ENTITY="${WANDB_ENTITY}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_GROUP="${WANDB_GROUP}" \
  WANDB_RUN_NAME="${WANDB_GROUP}_tgvf_v2_bidirectional_cyclic_negative_eval" \
  WANDB_TAGS="eval,20k,visual_cue_v1,tgvf_v2,pre_qwen_merger,cyclic_margin,tgvf_v2_bidirectional_cyclic_negative,tgvf_v2_bidirectional" \
  eval/run_tgvf_eval_suite.sh "${checkpoint}" "${EVAL_TASKS}" > "${EVAL_LOG_FILE}" 2>&1
fi

echo "done" > "${STATE_FILE}"
trap - ERR
echo "Run root: ${RUN_ROOT}"
