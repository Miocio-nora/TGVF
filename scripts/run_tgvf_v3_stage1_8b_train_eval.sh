#!/usr/bin/env bash
set -euo pipefail

# TGVF-v3 Stage1 8B train + automatic evaluation.
#
# Typical usage:
#   MAX_STEPS=1000 CUDA_VISIBLE_DEVICES=0 \
#     scripts/run_tgvf_v3_stage1_8b_train_eval.sh
#
# Important overrides:
#   TRAIN_FILE=/path/to/tgvf_v3_train.jsonl
#   EVAL_JSONL=/path/to/tgvf_v3_val_2k.jsonl
#   MAX_STEPS=1000
#   EVAL_PRESET=smoke|pilot|full
#   WANDB_MODE=online|offline|disabled

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

MODEL_ID="${MODEL_ID:-/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking}"
TRAIN_FILE="${TRAIN_FILE:-data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/final/tgvf_teacher_items.accepted.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tgvf_v3_stage1_8b/${RUN_ID}}"
TRAIN_OUT="${TRAIN_OUT:-${OUTPUT_ROOT}/train}"
EVAL_OUT="${EVAL_OUT:-${OUTPUT_ROOT}/eval}"

DEVICE="${DEVICE:-cuda:0}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29573}"
TORCHRUN="${TORCHRUN:-torchrun}"

VARIANT="${VARIANT:-tgvf_v2_bidirectional}"
NUM_FVT="${NUM_FVT:-none}"
MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION:-512}"
FVT_POSITION_MODE="${FVT_POSITION_MODE:-native_source_grid}"

BATCH_SIZE="${BATCH_SIZE:-4}"
BATCH_SAMPLING="${BATCH_SAMPLING:-auto}"
DROP_INCOMPLETE_SAME_IMAGE_BATCHES="${DROP_INCOMPLETE_SAME_IMAGE_BATCHES:-1}"
CAPTURE_MODE="${CAPTURE_MODE:-teacher_forced}"
READOUT_BATCH_SIZE="${READOUT_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
CAPTURE_LAYER="${CAPTURE_LAYER:--1}"
LOSS_GEN="${LOSS_GEN:-1.0}"
LOSS_VISUAL_TOKEN_MANIFOLD="${LOSS_VISUAL_TOKEN_MANIFOLD:-0.01}"
LOSS_SAME_IMAGE_NEGATIVE="${LOSS_SAME_IMAGE_NEGATIVE:-1.0}"
SAME_IMAGE_NEGATIVE_MODE="${SAME_IMAGE_NEGATIVE_MODE:-matrix_ce}"

WANDB_PROJECT="${WANDB_PROJECT:-tgvf-v3}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-stage1-8b}"
WANDB_TAGS="${WANDB_TAGS:-stage1,v3,qwen3-8b,matrix-ce}"
WANDB_LOG_CHECKPOINTS="${WANDB_LOG_CHECKPOINTS:-0}"

EVAL_PRESET="${EVAL_PRESET:-pilot}"
EVAL_TASKS="${EVAL_TASKS:-all}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"
SHARD_KEY="${SHARD_KEY:-image}"
USE_FVT_CACHE="${USE_FVT_CACHE:-1}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing TRAIN_FILE: ${TRAIN_FILE}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_ID}" && "${MODEL_ID}" == /* ]]; then
  echo "Missing local MODEL_ID directory: ${MODEL_ID}" >&2
  exit 1
fi

EVAL_JSONL="${EVAL_JSONL:-}"
if [[ -z "${EVAL_JSONL}" ]]; then
  for candidate in \
    "data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/final/tgvf_v3_val_2k.jsonl" \
    "data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/final/tgvf_teacher_items.val_2k.jsonl" \
    "data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/final/val_2k.jsonl"; do
    if [[ -f "${candidate}" ]]; then
      EVAL_JSONL="${candidate}"
      break
    fi
  done
fi
if [[ -z "${EVAL_JSONL}" ]]; then
  EVAL_JSONL="${TRAIN_FILE}"
  echo "[warn] No v3 val JSONL found. Evaluation will use TRAIN_FILE. Set EVAL_JSONL for real validation." >&2
fi
if [[ ! -f "${EVAL_JSONL}" ]]; then
  echo "Missing EVAL_JSONL: ${EVAL_JSONL}" >&2
  exit 1
fi

case "${EVAL_PRESET}" in
  smoke)
    READOUT_MAX_SAMPLES="${READOUT_MAX_SAMPLES:-16}"
    DISTRIBUTION_MAX_SAMPLES="${DISTRIBUTION_MAX_SAMPLES:-16}"
    QUERY_MAX_GROUPS="${QUERY_MAX_GROUPS:-4}"
    QUERY_REQUIRE_GROUPS="${QUERY_REQUIRE_GROUPS:-0}"
    ;;
  pilot)
    READOUT_MAX_SAMPLES="${READOUT_MAX_SAMPLES:-200}"
    DISTRIBUTION_MAX_SAMPLES="${DISTRIBUTION_MAX_SAMPLES:-200}"
    QUERY_MAX_GROUPS="${QUERY_MAX_GROUPS:-50}"
    QUERY_REQUIRE_GROUPS="${QUERY_REQUIRE_GROUPS:-0}"
    ;;
  full)
    READOUT_MAX_SAMPLES="${READOUT_MAX_SAMPLES:-2000}"
    DISTRIBUTION_MAX_SAMPLES="${DISTRIBUTION_MAX_SAMPLES:-2000}"
    QUERY_MAX_GROUPS="${QUERY_MAX_GROUPS:-420}"
    QUERY_REQUIRE_GROUPS="${QUERY_REQUIRE_GROUPS:-0}"
    ;;
  *)
    echo "Unsupported EVAL_PRESET=${EVAL_PRESET}; use smoke, pilot, or full." >&2
    exit 2
    ;;
esac
QUERY_MIN_TARGETS_PER_IMAGE="${QUERY_MIN_TARGETS_PER_IMAGE:-3}"

mkdir -p "${OUTPUT_ROOT}/logs" "${TRAIN_OUT}" "${EVAL_OUT}"

cat > "${OUTPUT_ROOT}/run_config.txt" <<EOF
run_id=${RUN_ID}
model_id=${MODEL_ID}
train_file=${TRAIN_FILE}
eval_jsonl=${EVAL_JSONL}
output_root=${OUTPUT_ROOT}
train_out=${TRAIN_OUT}
eval_out=${EVAL_OUT}
device=${DEVICE}
device_map=${DEVICE_MAP}
num_gpus=${NUM_GPUS}
dtype=${DTYPE}
attn_impl=${ATTN_IMPL}
variant=${VARIANT}
num_fvt=${NUM_FVT}
max_image_resolution=${MAX_IMAGE_RESOLUTION}
fvt_position_mode=${FVT_POSITION_MODE}
batch_size=${BATCH_SIZE}
batch_sampling=${BATCH_SAMPLING}
drop_incomplete_same_image_batches=${DROP_INCOMPLETE_SAME_IMAGE_BATCHES}
capture_mode=${CAPTURE_MODE}
readout_batch_size=${READOUT_BATCH_SIZE}
num_workers=${NUM_WORKERS}
global_batch_size=$((BATCH_SIZE * NUM_GPUS))
max_steps=${MAX_STEPS}
save_every=${SAVE_EVERY}
log_every=${LOG_EVERY}
learning_rate=${LEARNING_RATE}
lr_scheduler=${LR_SCHEDULER}
warmup_steps=${WARMUP_STEPS}
min_lr_ratio=${MIN_LR_RATIO}
loss_gen=${LOSS_GEN}
loss_visual_token_manifold=${LOSS_VISUAL_TOKEN_MANIFOLD}
loss_same_image_negative=${LOSS_SAME_IMAGE_NEGATIVE}
same_image_negative_mode=${SAME_IMAGE_NEGATIVE_MODE}
eval_preset=${EVAL_PRESET}
eval_tasks=${EVAL_TASKS}
readout_max_samples=${READOUT_MAX_SAMPLES}
distribution_max_samples=${DISTRIBUTION_MAX_SAMPLES}
query_max_groups=${QUERY_MAX_GROUPS}
query_require_groups=${QUERY_REQUIRE_GROUPS}
query_min_targets_per_image=${QUERY_MIN_TARGETS_PER_IMAGE}
eval_workers=${EVAL_WORKERS}
use_fvt_cache=${USE_FVT_CACHE}
wandb_project=${WANDB_PROJECT}
wandb_mode=${WANDB_MODE}
wandb_group=${WANDB_GROUP}
wandb_tags=${WANDB_TAGS}
EOF

train_args=(
  --train-file "${TRAIN_FILE}"
  --output-dir "${TRAIN_OUT}"
  --model-id "${MODEL_ID}"
  --dtype "${DTYPE}"
  --device-map "${DEVICE_MAP}"
  --device "${DEVICE}"
  --attn-implementation "${ATTN_IMPL}"
  --variant "${VARIANT}"
  --num-foveated-tokens "${NUM_FVT}"
  --batch-size "${BATCH_SIZE}"
  --batch-sampling "${BATCH_SAMPLING}"
  --capture-mode "${CAPTURE_MODE}"
  --readout-batch-size "${READOUT_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --max-steps "${MAX_STEPS}"
  --save-every "${SAVE_EVERY}"
  --log-every "${LOG_EVERY}"
  --learning-rate "${LEARNING_RATE}"
  --lr-scheduler "${LR_SCHEDULER}"
  --warmup-steps "${WARMUP_STEPS}"
  --min-lr-ratio "${MIN_LR_RATIO}"
  --capture-layer "${CAPTURE_LAYER}"
  --max-image-resolution "${MAX_IMAGE_RESOLUTION}"
  --fvt-position-mode "${FVT_POSITION_MODE}"
  --loss-gen "${LOSS_GEN}"
  --loss-visual-token-manifold "${LOSS_VISUAL_TOKEN_MANIFOLD}"
  --loss-same-image-negative "${LOSS_SAME_IMAGE_NEGATIVE}"
  --same-image-negative-mode "${SAME_IMAGE_NEGATIVE_MODE}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-mode "${WANDB_MODE}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-name "tgvf-v3-stage1-8b-${RUN_ID}"
  --wandb-tags "${WANDB_TAGS}"
)
if [[ "${DROP_INCOMPLETE_SAME_IMAGE_BATCHES}" == "1" ]]; then
  train_args+=(--drop-incomplete-same-image-batches)
else
  train_args+=(--no-drop-incomplete-same-image-batches)
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  train_args+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ "${WANDB_LOG_CHECKPOINTS}" == "1" ]]; then
  train_args+=(--wandb-log-checkpoints)
fi

echo "[train] output: ${TRAIN_OUT}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  echo "[train] torchrun NUM_GPUS=${NUM_GPUS} local_batch=${BATCH_SIZE} global_batch=$((BATCH_SIZE * NUM_GPUS))"
  /usr/bin/time -f 'WALL_SECONDS=%e' -o "${OUTPUT_ROOT}/logs/train_time.txt" \
    "${TORCHRUN}" --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT}" \
      scripts/train_tgvf_v3_stage1.py "${train_args[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train.log"
else
  /usr/bin/time -f 'WALL_SECONDS=%e' -o "${OUTPUT_ROOT}/logs/train_time.txt" \
    python scripts/train_tgvf_v3_stage1.py "${train_args[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train.log"
fi

CHECKPOINT="${TRAIN_OUT}/checkpoint_step_${MAX_STEPS}.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Expected checkpoint missing: ${CHECKPOINT}" >&2
  exit 1
fi

echo "[eval] checkpoint: ${CHECKPOINT}"
EVAL_JSONL="${EVAL_JSONL}" \
MODEL_ID="${MODEL_ID}" \
DEVICE="${DEVICE}" \
DEVICE_MAP="${DEVICE_MAP}" \
DTYPE="${DTYPE}" \
ATTN_IMPL="${ATTN_IMPL}" \
VARIANT="${VARIANT}" \
NUM_FVT="${NUM_FVT}" \
MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION}" \
FVT_POSITION_MODE="${FVT_POSITION_MODE}" \
OUT_ROOT="${EVAL_OUT}" \
READOUT_MAX_SAMPLES="${READOUT_MAX_SAMPLES}" \
DISTRIBUTION_MAX_SAMPLES="${DISTRIBUTION_MAX_SAMPLES}" \
QUERY_MAX_GROUPS="${QUERY_MAX_GROUPS}" \
QUERY_REQUIRE_GROUPS="${QUERY_REQUIRE_GROUPS}" \
QUERY_MIN_TARGETS_PER_IMAGE="${QUERY_MIN_TARGETS_PER_IMAGE}" \
EVAL_WORKERS="${EVAL_WORKERS}" \
SHARD_KEY="${SHARD_KEY}" \
USE_FVT_CACHE="${USE_FVT_CACHE}" \
WANDB_LOG_EVAL=1 \
WANDB_PROJECT="${WANDB_PROJECT}" \
WANDB_ENTITY="${WANDB_ENTITY}" \
WANDB_MODE="${WANDB_MODE}" \
WANDB_RUN_NAME="tgvf-v3-eval-8b-${RUN_ID}" \
WANDB_GROUP="${WANDB_GROUP}" \
WANDB_TAGS="eval,tgvf-v3,qwen3-8b,${EVAL_PRESET}" \
  ./eval/run_tgvf_v3_eval_suite.sh "${CHECKPOINT}" "${EVAL_TASKS}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/eval.log"

echo "[complete] ${OUTPUT_ROOT}"
