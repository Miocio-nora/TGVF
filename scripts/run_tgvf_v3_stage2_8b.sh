#!/usr/bin/env bash
set -euo pipefail

# TGVF-v3 Stage2 8B trajectory LoRA + trainable TGVF run.
#
# Smoke:
#   RUN_ID=stage2_8b_8gpu_100step_smoke NUM_GPUS=8 MAX_STEPS=100 SAVE_EVERY=100 EVAL_EVERY=100 RUN_EVAL_AFTER=0 \
#     scripts/run_tgvf_v3_stage2_8b.sh
#
# Main:
#   RUN_ID=stage2_8b_8gpu_1200step NUM_GPUS=8 scripts/run_tgvf_v3_stage2_8b.sh

RUN_ID="${RUN_ID:-v3_stage2_50k_qwen3vl_thinking_lora64_tgvftrain_plainmarkers_512}"

MODEL_ID="${MODEL_ID:-/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking}"
TRAIN_FILE="${TRAIN_FILE:-data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/final/tgvf_teacher_items.accepted.jsonl}"
VAL_FILE="${VAL_FILE:-data/tgvf_teacher/generated/runs/tgvf_v3_teacher_val_2k/final/tgvf_teacher_items.accepted.jsonl}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-outputs/tgvf_v3_stage1_8b/8b_8gpu_2000step_stage1_v3/train/checkpoint_step_2000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tgvf_v3_stage2_8b/${RUN_ID}}"

NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29576}"
TORCHRUN="${TORCHRUN:-torchrun}"
DEVICE="${DEVICE:-cuda:0}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

VARIANT="${VARIANT:-tgvf_v2_bidirectional}"
NUM_FVT="${NUM_FVT:-none}"
MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION:-512}"
FVT_POSITION_MODE="${FVT_POSITION_MODE:-native_source_grid}"
FAST_BATCHED_STAGE2="${FAST_BATCHED_STAGE2:-1}"

BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-1200}"
SAVE_EVERY="${SAVE_EVERY:-300}"
EVAL_EVERY="${EVAL_EVERY:-300}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
LOG_EVERY="${LOG_EVERY:-10}"

LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-256}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

LR_LORA="${LR_LORA:-2e-5}"
LR_TGVF="${LR_TGVF:-5e-6}"
LR_CALIBRATION="${LR_CALIBRATION:-1e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"

WANDB_PROJECT="${WANDB_PROJECT:-tgvf-v3}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-stage2-8b}"
WANDB_TAGS="${WANDB_TAGS:-stage2,v3,qwen3-8b,lora,tgvf-trainable,plain-markers}"

RUN_EVAL_AFTER="${RUN_EVAL_AFTER:-1}"
EVAL_TASKS="${EVAL_TASKS:-readout,query,distribution}"
EVAL_JSONL="${EVAL_JSONL:-${VAL_FILE}}"
EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-${OUTPUT_ROOT}/eval_stage1_regression}"
READOUT_MAX_SAMPLES="${READOUT_MAX_SAMPLES:-200}"
DISTRIBUTION_MAX_SAMPLES="${DISTRIBUTION_MAX_SAMPLES:-200}"
QUERY_MAX_GROUPS="${QUERY_MAX_GROUPS:-50}"
QUERY_REQUIRE_GROUPS="${QUERY_REQUIRE_GROUPS:-0}"
QUERY_MIN_TARGETS_PER_IMAGE="${QUERY_MIN_TARGETS_PER_IMAGE:-3}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing TRAIN_FILE: ${TRAIN_FILE}" >&2
  exit 1
fi
if [[ ! -f "${VAL_FILE}" ]]; then
  echo "Missing VAL_FILE: ${VAL_FILE}" >&2
  exit 1
fi
if [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then
  echo "Missing STAGE1_CHECKPOINT: ${STAGE1_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_ID}" && "${MODEL_ID}" == /* ]]; then
  echo "Missing local MODEL_ID directory: ${MODEL_ID}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/logs"

cat > "${OUTPUT_ROOT}/run_config.txt" <<EOF
run_id=${RUN_ID}
model_id=${MODEL_ID}
train_file=${TRAIN_FILE}
val_file=${VAL_FILE}
stage1_checkpoint=${STAGE1_CHECKPOINT}
output_root=${OUTPUT_ROOT}
num_gpus=${NUM_GPUS}
batch_size=${BATCH_SIZE}
grad_accum=${GRAD_ACCUM}
effective_global_batch_size=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))
max_steps=${MAX_STEPS}
save_every=${SAVE_EVERY}
eval_every=${EVAL_EVERY}
lora_rank=${LORA_RANK}
lora_alpha=${LORA_ALPHA}
lora_target_modules=${LORA_TARGET_MODULES}
lr_lora=${LR_LORA}
lr_tgvf=${LR_TGVF}
lr_calibration=${LR_CALIBRATION}
max_image_resolution=${MAX_IMAGE_RESOLUTION}
fvt_position_mode=${FVT_POSITION_MODE}
fast_batched_stage2=${FAST_BATCHED_STAGE2}
wandb_project=${WANDB_PROJECT}
wandb_mode=${WANDB_MODE}
wandb_group=${WANDB_GROUP}
run_eval_after=${RUN_EVAL_AFTER}
eval_tasks=${EVAL_TASKS}
eval_jsonl=${EVAL_JSONL}
eval_out_root=${EVAL_OUT_ROOT}
readout_max_samples=${READOUT_MAX_SAMPLES}
distribution_max_samples=${DISTRIBUTION_MAX_SAMPLES}
query_max_groups=${QUERY_MAX_GROUPS}
EOF

args=(
  --train-file "${TRAIN_FILE}"
  --val-file "${VAL_FILE}"
  --output-dir "${OUTPUT_ROOT}/train"
  --stage1-checkpoint "${STAGE1_CHECKPOINT}"
  --model-id "${MODEL_ID}"
  --dtype "${DTYPE}"
  --device-map "${DEVICE_MAP}"
  --device "${DEVICE}"
  --attn-implementation "${ATTN_IMPL}"
  --variant "${VARIANT}"
  --num-foveated-tokens "${NUM_FVT}"
  --max-image-resolution "${MAX_IMAGE_RESOLUTION}"
  --fvt-position-mode "${FVT_POSITION_MODE}"
  --batch-size "${BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --max-steps "${MAX_STEPS}"
  --save-every "${SAVE_EVERY}"
  --eval-every "${EVAL_EVERY}"
  --eval-max-samples "${EVAL_MAX_SAMPLES}"
  --log-every "${LOG_EVERY}"
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --lora-target-modules "${LORA_TARGET_MODULES}"
  --lr-lora "${LR_LORA}"
  --lr-tgvf "${LR_TGVF}"
  --lr-calibration "${LR_CALIBRATION}"
  --lr-scheduler "${LR_SCHEDULER}"
  --warmup-ratio "${WARMUP_RATIO}"
  --min-lr-ratio "${MIN_LR_RATIO}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-mode "${WANDB_MODE}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-name "tgvf-v3-stage2-8b-${RUN_ID}"
  --wandb-tags "${WANDB_TAGS}"
)
if [[ "${FAST_BATCHED_STAGE2}" == "1" ]]; then
  args+=(--fast-batched-stage2)
else
  args+=(--no-fast-batched-stage2)
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  args+=(--wandb-entity "${WANDB_ENTITY}")
fi

echo "[stage2] output: ${OUTPUT_ROOT}/train"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  echo "[stage2] torchrun NUM_GPUS=${NUM_GPUS} per_device_batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} effective_global=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
  /usr/bin/time -f 'WALL_SECONDS=%e' -o "${OUTPUT_ROOT}/logs/train_time.txt" \
    "${TORCHRUN}" --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT}" \
      scripts/train_tgvf_v3_stage2.py "${args[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train.log"
else
  /usr/bin/time -f 'WALL_SECONDS=%e' -o "${OUTPUT_ROOT}/logs/train_time.txt" \
    python scripts/train_tgvf_v3_stage2.py "${args[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train.log"
fi

if [[ "${RUN_EVAL_AFTER}" == "1" ]]; then
  FINAL_CHECKPOINT="${OUTPUT_ROOT}/train/checkpoint_step_${MAX_STEPS}.pt"
  if [[ ! -f "${FINAL_CHECKPOINT}" ]]; then
    echo "Missing final checkpoint for eval: ${FINAL_CHECKPOINT}" >&2
    exit 1
  fi
  echo "[stage2] running Stage1-regression eval on TGVF module: ${FINAL_CHECKPOINT}"
  echo "[stage2] note: this eval does not load the Stage2 LoRA adapter; it checks TGVF/D quality regression."
  MODEL_ID="${MODEL_ID}" \
  EVAL_JSONL="${EVAL_JSONL}" \
  OUT_ROOT="${EVAL_OUT_ROOT}" \
  TASKS="${EVAL_TASKS}" \
  VARIANT="${VARIANT}" \
  DEVICE="${DEVICE}" \
  DEVICE_MAP="${DEVICE_MAP}" \
  DTYPE="${DTYPE}" \
  ATTN_IMPL="${ATTN_IMPL}" \
  NUM_FVT="${NUM_FVT}" \
  MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION}" \
  FVT_POSITION_MODE="${FVT_POSITION_MODE}" \
  READOUT_MAX_SAMPLES="${READOUT_MAX_SAMPLES}" \
  DISTRIBUTION_MAX_SAMPLES="${DISTRIBUTION_MAX_SAMPLES}" \
  QUERY_MAX_GROUPS="${QUERY_MAX_GROUPS}" \
  QUERY_REQUIRE_GROUPS="${QUERY_REQUIRE_GROUPS}" \
  QUERY_MIN_TARGETS_PER_IMAGE="${QUERY_MIN_TARGETS_PER_IMAGE}" \
  EVAL_WORKERS="${EVAL_WORKERS}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_GROUP="${WANDB_GROUP}" \
  WANDB_TAGS="${WANDB_TAGS},stage1-regression" \
    eval/run_tgvf_v3_eval_suite.sh "${FINAL_CHECKPOINT}" "${EVAL_TASKS}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/eval_stage1_regression.log"
fi

echo "[complete] ${OUTPUT_ROOT}"
