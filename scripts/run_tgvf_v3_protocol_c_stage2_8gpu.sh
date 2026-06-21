#!/usr/bin/env bash
set -euo pipefail

# TGVF-v3.1 Protocol C Stage2 8GPU training + protocol eval.
#
# Defaults match the current Stage2 setup, except training sampling is changed to
# focus:no-focus ~= 0.8:0.2 through --target-focus-ratio 0.8.
#
# Override any variable from the shell, for example:
#   WANDB_MODE=offline MAX_STEPS=100 RUN_NAME=protocol_c_smoke bash scripts/run_tgvf_v3_protocol_c_stage2_8gpu.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TGVF_QWEN3_CAPTURE_GENERATE="${TGVF_QWEN3_CAPTURE_GENERATE:-1}"

MODEL_ID="${MODEL_ID:-/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking}"
STAGE1_CKPT="${STAGE1_CKPT:-outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617/train/checkpoint_step_2000.pt}"
PROCESSOR_ID="${PROCESSOR_ID:-outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617/train/processor_step_2000}"
TRAIN_FILE="${TRAIN_FILE:-data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl}"
VAL_FILE="${VAL_FILE:-data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl}"

RUN_NAME="${RUN_NAME:-protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_8gpu_bs16_focus80_from_clean_imend_stage1_1200step}"
OUT_DIR="${OUT_DIR:-outputs/tgvf_v3_protocol_c/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-1200}"
SAVE_EVERY="${SAVE_EVERY:-300}"
EVAL_EVERY="${EVAL_EVERY:-300}"
TRAIN_EVAL_MAX_SAMPLES="${TRAIN_EVAL_MAX_SAMPLES:-128}"
LOG_EVERY="${LOG_EVERY:-10}"

TARGET_FOCUS_RATIO="${TARGET_FOCUS_RATIO:-0.8}"
MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION:-512}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
TGVF_PROTOCOL="${TGVF_PROTOCOL:-protocol_c_tool_observation}"

LR_LORA="${LR_LORA:-2e-5}"
LR_TGVF="${LR_TGVF:-5e-6}"
LR_CALIBRATION="${LR_CALIBRATION:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-256}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

WANDB_PROJECT="${WANDB_PROJECT:-tgvf-v3}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-protocol_c_toolobs_focus_imend_stage2}"
WANDB_TAGS="${WANDB_TAGS:-protocol_c_toolobs,stage2,8gpu,focus80,focus_imend,v4data_clean,open_answer,stage1_focus_imend}"

RUN_PROTOCOL_EVAL="${RUN_PROTOCOL_EVAL:-1}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_DEVICE_MAP="${EVAL_DEVICE_MAP:-cuda:0}"
PROTOCOL_EVAL_MAX_FOCUS="${PROTOCOL_EVAL_MAX_FOCUS:-128}"
PROTOCOL_EVAL_MAX_NO_FOCUS="${PROTOCOL_EVAL_MAX_NO_FOCUS:-128}"
PROTOCOL_EVAL_MAX_ACTION_TOKENS="${PROTOCOL_EVAL_MAX_ACTION_TOKENS:-128}"
PROTOCOL_EVAL_MAX_ANSWER_TOKENS="${PROTOCOL_EVAL_MAX_ANSWER_TOKENS:-128}"
PROTOCOL_EVAL_BLOCKS="${PROTOCOL_EVAL_BLOCKS:-no_focus_direct,force_focus_targets,teacher_forced_post_tgvf,force_end2end,free_router_end2end}"
PROTOCOL_EVAL_D_CONDITIONS="${PROTOCOL_EVAL_D_CONDITIONS:-correct_D,no_D,random_D,wrong_same_image_D,wrong_diff_image_D}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

echo "[Protocol C Stage2] output: ${OUT_DIR}"
echo "[Protocol C Stage2] model: ${MODEL_ID}"
echo "[Protocol C Stage2] stage1: ${STAGE1_CKPT}"
echo "[Protocol C Stage2] processor: ${PROCESSOR_ID}"
echo "[Protocol C Stage2] train: ${TRAIN_FILE}"
echo "[Protocol C Stage2] val: ${VAL_FILE}"
echo "[Protocol C Stage2] tgvf_protocol: ${TGVF_PROTOCOL}"
echo "[Protocol C Stage2] target_focus_ratio: ${TARGET_FOCUS_RATIO}"
echo "[Protocol C Stage2] effective_global_batch_size: $((NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUM))"

set -o pipefail
/usr/bin/time -v torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" scripts/train_tgvf_v3_stage2.py \
  --train-file "${TRAIN_FILE}" \
  --val-file "${VAL_FILE}" \
  --output-dir "${OUT_DIR}" \
  --stage1-checkpoint "${STAGE1_CKPT}" \
  --model-id "${MODEL_ID}" \
  --processor-id "${PROCESSOR_ID}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --tgvf-protocol "${TGVF_PROTOCOL}" \
  --target-focus-ratio "${TARGET_FOCUS_RATIO}" \
  --batch-size "${BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  --max-steps "${MAX_STEPS}" \
  --save-every "${SAVE_EVERY}" \
  --eval-every "${EVAL_EVERY}" \
  --eval-max-samples "${TRAIN_EVAL_MAX_SAMPLES}" \
  --max-image-resolution "${MAX_IMAGE_RESOLUTION}" \
  --fast-batched-stage2 \
  --lr-lora "${LR_LORA}" \
  --lr-tgvf "${LR_TGVF}" \
  --lr-calibration "${LR_CALIBRATION}" \
  --lr-scheduler cosine \
  --warmup-steps "${WARMUP_STEPS}" \
  --min-lr-ratio "${MIN_LR_RATIO}" \
  --weight-decay 0.01 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --mask-original-image-after-tgvf \
  --fvt-position-mode native_source_grid \
  --num-workers 0 \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-mode "${WANDB_MODE}" \
  --wandb-tags "${WANDB_TAGS}" \
  --log-every "${LOG_EVERY}" \
  2>&1 | tee "${LOG_DIR}/train.log"

FINAL_CKPT="${OUT_DIR}/checkpoint_step_${MAX_STEPS}.pt"
if [[ ! -f "${FINAL_CKPT}" ]]; then
  echo "Expected checkpoint not found: ${FINAL_CKPT}" >&2
  exit 1
fi

if [[ "${RUN_PROTOCOL_EVAL}" == "1" ]]; then
  EVAL_OUT_DIR="${OUT_DIR}/protocol_eval_step_${MAX_STEPS}"
  mkdir -p "${EVAL_OUT_DIR}"
  echo "[Protocol C Eval] checkpoint: ${FINAL_CKPT}"
  echo "[Protocol C Eval] output: ${EVAL_OUT_DIR}"
  /usr/bin/time -v python eval/eval_v3_stage2_protocol.py \
    --stage2-checkpoint "${FINAL_CKPT}" \
    --eval-jsonl "${VAL_FILE}" \
    --output-dir "${EVAL_OUT_DIR}" \
    --model-id "${MODEL_ID}" \
    --processor-id "${PROCESSOR_ID}" \
    --device "${EVAL_DEVICE}" \
    --device-map "${EVAL_DEVICE_MAP}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --tgvf-protocol "${TGVF_PROTOCOL}" \
    --blocks "${PROTOCOL_EVAL_BLOCKS}" \
    --d-conditions "${PROTOCOL_EVAL_D_CONDITIONS}" \
    --max-focus "${PROTOCOL_EVAL_MAX_FOCUS}" \
    --max-no-focus "${PROTOCOL_EVAL_MAX_NO_FOCUS}" \
    --max-action-tokens "${PROTOCOL_EVAL_MAX_ACTION_TOKENS}" \
    --max-answer-tokens "${PROTOCOL_EVAL_MAX_ANSWER_TOKENS}" \
    2>&1 | tee "${LOG_DIR}/protocol_eval_step_${MAX_STEPS}.log"
fi

echo "[Protocol C Stage2] done: ${OUT_DIR}"
