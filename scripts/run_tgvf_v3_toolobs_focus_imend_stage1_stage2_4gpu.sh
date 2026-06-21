#!/usr/bin/env bash
set -euo pipefail

# Full aligned Protocol C tool-observation + focus-action <|im_end|> refresh:
#   1. Stage1 with teacher-forced focus action ending in <|im_end|>
#   2. Stage2 from that Stage1 checkpoint
#   3. Stage2 protocol eval
# Defaults use GPUs 4-7 to avoid the interactive chats on 1/2.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MODEL_ID="${MODEL_ID:-/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking}"

STAGE1_RUN_ID="${STAGE1_RUN_ID:-protocol_c_toolobs_stage1_v4data_clean_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_${STAMP}}"
STAGE1_ROOT="${STAGE1_ROOT:-outputs/tgvf_v3_protocol_c_stage1_8b/${STAGE1_RUN_ID}}"
STAGE1_CKPT="${STAGE1_CKPT:-${STAGE1_ROOT}/train/checkpoint_step_2000.pt}"
STAGE1_PROCESSOR="${STAGE1_PROCESSOR:-${STAGE1_ROOT}/train/processor_step_2000}"

STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_clean_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_${STAMP}}"
STAGE2_OUT="${STAGE2_OUT:-outputs/tgvf_v3_protocol_c/${STAGE2_RUN_NAME}}"

STAGE1_TRAIN_FILE="${STAGE1_TRAIN_FILE:-data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl}"
STAGE1_EVAL_JSONL="${STAGE1_EVAL_JSONL:-data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl}"
STAGE2_TRAIN_FILE="${STAGE2_TRAIN_FILE:-data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl}"
STAGE2_VAL_FILE="${STAGE2_VAL_FILE:-data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl}"

MASTER_PORT_STAGE1="${MASTER_PORT_STAGE1:-29631}"
MASTER_PORT_STAGE2="${MASTER_PORT_STAGE2:-29632}"

WANDB_PROJECT="${WANDB_PROJECT:-tgvf-v3}"
WANDB_MODE="${WANDB_MODE:-online}"

RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_PROTOCOL_EVAL="${RUN_PROTOCOL_EVAL:-1}"

mkdir -p "${STAGE1_ROOT}" "${STAGE2_OUT}"
cat > "${STAGE2_OUT}/chain_config.txt" <<CFG
stamp=${STAMP}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
model_id=${MODEL_ID}
stage1_run_id=${STAGE1_RUN_ID}
stage1_root=${STAGE1_ROOT}
stage1_ckpt=${STAGE1_CKPT}
stage1_processor=${STAGE1_PROCESSOR}
stage2_run_name=${STAGE2_RUN_NAME}
stage2_out=${STAGE2_OUT}
stage1_train_file=${STAGE1_TRAIN_FILE}
stage1_eval_jsonl=${STAGE1_EVAL_JSONL}
stage2_train_file=${STAGE2_TRAIN_FILE}
stage2_val_file=${STAGE2_VAL_FILE}
protocol=protocol_c_tool_observation
focus_action_im_end=true
CFG

if [[ "${RUN_STAGE1}" == "1" ]]; then
  echo "[chain] Stage1 focus-imend start: ${STAGE1_ROOT}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  RUN_ID="${STAGE1_RUN_ID}" \
  MODEL_ID="${MODEL_ID}" \
  TRAIN_FILE="${STAGE1_TRAIN_FILE}" \
  EVAL_JSONL="${STAGE1_EVAL_JSONL}" \
  OUTPUT_ROOT="${STAGE1_ROOT}" \
  NUM_GPUS=4 \
  MASTER_PORT="${MASTER_PORT_STAGE1}" \
  TGVF_PROTOCOL=protocol_c_tool_observation \
  FOCUS_ACTION_IM_END=1 \
  BATCH_SIZE=4 \
  GRAD_ACCUM=2 \
  MAX_STEPS=2000 \
  SAVE_EVERY=500 \
  MAX_IMAGE_RESOLUTION=512 \
  LEARNING_RATE=1e-4 \
  LR_SCHEDULER=cosine \
  WARMUP_STEPS=100 \
  MIN_LR_RATIO=0.1 \
  LOSS_SAME_IMAGE_NEGATIVE=1.0 \
  SAME_IMAGE_NEGATIVE_MODE=matrix_ce \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_GROUP=protocol_c_toolobs_focus_imend_stage1 \
  WANDB_TAGS=protocol_c_toolobs,stage1,focus_imend,v4data_clean,token_rows,matrix-ce,gbs32 \
  RUN_EVAL=1 \
  bash scripts/run_tgvf_v3_protocol_c_stage1_4gpu_train_eval.sh
fi

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "Missing Stage1 checkpoint: ${STAGE1_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${STAGE1_PROCESSOR}" ]]; then
  echo "Missing Stage1 processor: ${STAGE1_PROCESSOR}" >&2
  exit 1
fi

if [[ "${RUN_STAGE2}" == "1" ]]; then
  echo "[chain] Stage2 focus-imend from Stage1 start: ${STAGE2_OUT}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  MODEL_ID="${MODEL_ID}" \
  STAGE1_CKPT="${STAGE1_CKPT}" \
  PROCESSOR_ID="${STAGE1_PROCESSOR}" \
  TRAIN_FILE="${STAGE2_TRAIN_FILE}" \
  VAL_FILE="${STAGE2_VAL_FILE}" \
  RUN_NAME="${STAGE2_RUN_NAME}" \
  OUT_DIR="${STAGE2_OUT}" \
  NPROC_PER_NODE=4 \
  TGVF_PROTOCOL=protocol_c_tool_observation \
  TARGET_FOCUS_RATIO=0.8 \
  BATCH_SIZE=16 \
  GRAD_ACCUM=2 \
  MAX_STEPS=1200 \
  SAVE_EVERY=300 \
  EVAL_EVERY=300 \
  TRAIN_EVAL_MAX_SAMPLES=128 \
  MAX_IMAGE_RESOLUTION=512 \
  LR_LORA=2e-5 \
  LR_TGVF=5e-6 \
  LR_CALIBRATION=1e-5 \
  WARMUP_STEPS=100 \
  MIN_LR_RATIO=0.1 \
  LOSS_VALUE_SPAN=1.0 \
  MASTER_PORT="${MASTER_PORT_STAGE2}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_GROUP=protocol_c_toolobs_focus_imend_stage2 \
  WANDB_TAGS=protocol_c_toolobs,stage2,focus_imend,v4data_clean,open_answer,focus80,value1,stage1_focus_imend \
  WANDB_RUN_NAME="${STAGE2_RUN_NAME}" \
  RUN_PROTOCOL_EVAL="${RUN_PROTOCOL_EVAL}" \
  PROTOCOL_EVAL_MAX_FOCUS=128 \
  PROTOCOL_EVAL_MAX_NO_FOCUS=128 \
  bash scripts/run_tgvf_v3_protocol_c_stage2_from_stage1c_4gpu.sh
fi

echo "[chain] complete"
echo "[chain] Stage1: ${STAGE1_CKPT}"
echo "[chain] Stage2: ${STAGE2_OUT}/checkpoint_step_1200.pt"
