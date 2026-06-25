#!/usr/bin/env bash
set -euo pipefail

# Protocol C Stage1 training for encoder-side TGVF reencode:
#   tgvf_encoder_bidir_8_16_24 + optional DeepStack-compatible outputs.
# Default mirrors the clean Protocol C Stage1 setup unless overridden.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

RUN_ID="${RUN_ID:-protocol_c_stage1_encoder_reencode_deepstack_$(date +%Y%m%d_%H%M%S)}"
MODEL_ID="${MODEL_ID:-/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking}"
TRAIN_FILE="${TRAIN_FILE:-data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/final/tgvf_teacher_items.accepted.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-data/tgvf_teacher/generated/runs/tgvf_v3_teacher_val_2k/final/tgvf_teacher_items.accepted.jsonl}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tgvf_v3_protocol_c_stage1_8b/${RUN_ID}}"
TRAIN_OUT="${TRAIN_OUT:-${OUTPUT_ROOT}/train}"
EVAL_OUT="${EVAL_OUT:-${OUTPUT_ROOT}/eval}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"

NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29579}"
TORCHRUN="${TORCHRUN:-torchrun}"
DEVICE="${DEVICE:-cuda:0}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-10}"
READOUT_BATCH_SIZE="${READOUT_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"

VARIANT="${VARIANT:-tgvf_encoder_bidir_8_16_24}"
NUM_FVT="${NUM_FVT:-none}"
MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION:-512}"
FVT_POSITION_MODE="${FVT_POSITION_MODE:-native_source_grid}"
BATCH_SAMPLING="${BATCH_SAMPLING:-same_image}"
CAPTURE_MODE="${CAPTURE_MODE:-teacher_forced}"
ENCODER_ADAPTER_LAYERS="${ENCODER_ADAPTER_LAYERS:-8,16,24}"
ENCODER_ADAPTER_LAYER_INDEX_BASE="${ENCODER_ADAPTER_LAYER_INDEX_BASE:-0}"
ENCODER_ADAPTER_GATE_INIT="${ENCODER_ADAPTER_GATE_INIT:-0.0}"
ENCODER_ADAPTER_SHARE_WEIGHTS="${ENCODER_ADAPTER_SHARE_WEIGHTS:-0}"
ENCODER_REENCODE_DEEPSTACK_COMPATIBLE="${ENCODER_REENCODE_DEEPSTACK_COMPATIBLE:-1}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
LOSS_GEN="${LOSS_GEN:-1.0}"
LOSS_VISUAL_TOKEN_MANIFOLD="${LOSS_VISUAL_TOKEN_MANIFOLD:-0.1}"
LOSS_SAME_IMAGE_NEGATIVE="${LOSS_SAME_IMAGE_NEGATIVE:-1.0}"
SAME_IMAGE_NEGATIVE_MODE="${SAME_IMAGE_NEGATIVE_MODE:-matrix_ce}"

WANDB_PROJECT="${WANDB_PROJECT:-tgvf-v3}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-protocol_c_stage1_encoder_reencode}"
WANDB_TAGS="${WANDB_TAGS:-protocol_c,stage1,encoder_reencode,deepstack,qwen3-8b,matrix-ce}"
WANDB_LOG_CHECKPOINTS="${WANDB_LOG_CHECKPOINTS:-0}"

RUN_EVAL="${RUN_EVAL:-1}"
EVAL_TASKS="${EVAL_TASKS:-all}"
READOUT_MAX_SAMPLES="${READOUT_MAX_SAMPLES:-200}"
DISTRIBUTION_MAX_SAMPLES="${DISTRIBUTION_MAX_SAMPLES:-200}"
QUERY_MAX_GROUPS="${QUERY_MAX_GROUPS:-50}"
QUERY_REQUIRE_GROUPS="${QUERY_REQUIRE_GROUPS:-0}"
QUERY_MIN_TARGETS_PER_IMAGE="${QUERY_MIN_TARGETS_PER_IMAGE:-3}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"
USE_FVT_CACHE="${USE_FVT_CACHE:-1}"

mkdir -p "${TRAIN_OUT}" "${EVAL_OUT}" "${LOG_DIR}"

cat > "${OUTPUT_ROOT}/run_config.txt" <<EOF
run_id=${RUN_ID}
model_id=${MODEL_ID}
train_file=${TRAIN_FILE}
eval_jsonl=${EVAL_JSONL}
output_root=${OUTPUT_ROOT}
train_out=${TRAIN_OUT}
eval_out=${EVAL_OUT}
tgvf_protocol=protocol_c_thinking_special
variant=${VARIANT}
encoder_adapter_layers=${ENCODER_ADAPTER_LAYERS}
encoder_adapter_layer_index_base=${ENCODER_ADAPTER_LAYER_INDEX_BASE}
encoder_adapter_gate_init=${ENCODER_ADAPTER_GATE_INIT}
encoder_adapter_share_weights=${ENCODER_ADAPTER_SHARE_WEIGHTS}
encoder_reencode_deepstack_compatible=${ENCODER_REENCODE_DEEPSTACK_COMPATIBLE}
num_gpus=${NUM_GPUS}
batch_size=${BATCH_SIZE}
grad_accum=${GRAD_ACCUM}
global_batch_size=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))
max_steps=${MAX_STEPS}
save_every=${SAVE_EVERY}
learning_rate=${LEARNING_RATE}
lr_scheduler=${LR_SCHEDULER}
warmup_steps=${WARMUP_STEPS}
min_lr_ratio=${MIN_LR_RATIO}
batch_sampling=${BATCH_SAMPLING}
same_image_negative_mode=${SAME_IMAGE_NEGATIVE_MODE}
resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}
EOF

echo "[Protocol C Encoder Stage1] output: ${OUTPUT_ROOT}"
echo "[Protocol C Encoder Stage1] global_batch_size: $((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))"
echo "[Protocol C Encoder Stage1] deepstack compatible: ${ENCODER_REENCODE_DEEPSTACK_COMPATIBLE}"

train_args=(
  --train-file "${TRAIN_FILE}"
  --output-dir "${TRAIN_OUT}"
  --model-id "${MODEL_ID}"
  --dtype "${DTYPE}"
  --device-map "${DEVICE_MAP}"
  --device "${DEVICE}"
  --attn-implementation "${ATTN_IMPL}"
  --tgvf-protocol protocol_c_thinking_special
  --variant "${VARIANT}"
  --num-foveated-tokens "${NUM_FVT}"
  --encoder-adapter-layers "${ENCODER_ADAPTER_LAYERS}"
  --encoder-adapter-layer-index-base "${ENCODER_ADAPTER_LAYER_INDEX_BASE}"
  --encoder-adapter-gate-init "${ENCODER_ADAPTER_GATE_INIT}"
  --batch-size "${BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
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
  --max-image-resolution "${MAX_IMAGE_RESOLUTION}"
  --fvt-position-mode "${FVT_POSITION_MODE}"
  --loss-gen "${LOSS_GEN}"
  --loss-visual-token-manifold "${LOSS_VISUAL_TOKEN_MANIFOLD}"
  --loss-same-image-negative "${LOSS_SAME_IMAGE_NEGATIVE}"
  --same-image-negative-mode "${SAME_IMAGE_NEGATIVE_MODE}"
  --drop-incomplete-same-image-batches
  --wandb-project "${WANDB_PROJECT}"
  --wandb-mode "${WANDB_MODE}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-name "tgvf-v3-stage1-c-encoder-reencode-${RUN_ID}"
  --wandb-tags "${WANDB_TAGS}"
)
if [[ "${ENCODER_ADAPTER_SHARE_WEIGHTS}" == "1" ]]; then
  train_args+=(--encoder-adapter-share-weights)
else
  train_args+=(--no-encoder-adapter-share-weights)
fi
if [[ "${ENCODER_REENCODE_DEEPSTACK_COMPATIBLE}" == "1" ]]; then
  train_args+=(--encoder-reencode-deepstack-compatible)
else
  train_args+=(--no-encoder-reencode-deepstack-compatible)
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  train_args+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ "${WANDB_LOG_CHECKPOINTS}" == "1" ]]; then
  train_args+=(--wandb-log-checkpoints)
fi

/usr/bin/time -f 'WALL_SECONDS=%e' -o "${LOG_DIR}/train_time.txt" \
  "${TORCHRUN}" --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT}" \
    scripts/train_tgvf_v3_stage1.py "${train_args[@]}" \
  2>&1 | tee "${LOG_DIR}/train.log"

CHECKPOINT="${TRAIN_OUT}/checkpoint_step_${MAX_STEPS}.pt"
PROCESSOR_ID="${TRAIN_OUT}/processor_step_${MAX_STEPS}"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Expected checkpoint missing: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "${PROCESSOR_ID}" ]]; then
  echo "Expected processor missing: ${PROCESSOR_ID}" >&2
  exit 1
fi

python - <<PY
import torch
ckpt = torch.load('${CHECKPOINT}', map_location='cpu')
rows = ckpt.get('protocol_c_token_rows')
assert rows is not None, 'missing protocol_c_token_rows'
assert 'input_embeddings' in rows, 'missing input_embeddings rows'
assert 'output_embeddings' in rows, 'missing output_embeddings rows'
print('[token-row-check]', rows['token_ids'], tuple(rows['input_embeddings'].shape), tuple(rows['output_embeddings'].shape))
print('[tgvf-config]', ckpt.get('config', {}).get('tgvf'))
PY

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "[Protocol C Encoder Stage1 Eval] checkpoint: ${CHECKPOINT}"
  EVAL_JSONL="${EVAL_JSONL}" \
  MODEL_ID="${MODEL_ID}" \
  PROCESSOR_ID="${PROCESSOR_ID}" \
  TGVF_PROTOCOL=protocol_c_thinking_special \
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
  USE_FVT_CACHE="${USE_FVT_CACHE}" \
  WANDB_LOG_EVAL=1 \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_RUN_NAME="tgvf-v3-eval-stage1-c-encoder-reencode-${RUN_ID}" \
  WANDB_GROUP="${WANDB_GROUP}" \
  WANDB_TAGS="eval,protocol_c,stage1,encoder_reencode" \
    ./eval/run_tgvf_v3_eval_suite.sh "${CHECKPOINT}" "${EVAL_TASKS}" \
    2>&1 | tee "${LOG_DIR}/eval.log"
fi

echo "[Protocol C Encoder Stage1] complete: ${OUTPUT_ROOT}"
