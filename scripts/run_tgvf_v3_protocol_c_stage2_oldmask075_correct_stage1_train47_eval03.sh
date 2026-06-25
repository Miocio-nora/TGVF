#!/usr/bin/env bash
set -euo pipefail

cd /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/src:$(pwd)/third_party/VLMEvalKit"
export TGVF_TOOLOBS_ACTION_STOP=im_end
export TGVF_BLOCK_FOCUS_IN_CONTINUATION="${TGVF_BLOCK_FOCUS_IN_CONTINUATION:-1}"
export TGVF_QWEN3_CAPTURE_GENERATE="${TGVF_QWEN3_CAPTURE_GENERATE:-1}"
export TGVF_STOP_REPETITIVE_CONTINUATION="${TGVF_STOP_REPETITIVE_CONTINUATION:-1}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MODEL_ID="${MODEL_ID:-/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"
BENCH_GPUS="${BENCH_GPUS:-${EVAL_GPUS}}"

STAGE1_RUN_ID="${STAGE1_RUN_ID:-protocol_c_toolobs_stage1_v4data_clean_imend_bidirectional_2gpu_bs4_accum4_gbs32_2000step_20260619_014148}"
STAGE1_ROOT="${STAGE1_ROOT:-outputs/tgvf_v3_protocol_c_stage1_8b/${STAGE1_RUN_ID}}"
STAGE1_CKPT="${STAGE1_CKPT:-${STAGE1_ROOT}/train/checkpoint_step_2000.pt}"
STAGE1_PROCESSOR="${STAGE1_PROCESSOR:-${STAGE1_ROOT}/train/processor_step_2000}"

STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_oldmaskprob075_throughanswer_from_20260619_stage1_train47_eval03_4gpu_bs16_accum2_focus80_value1_1200step_${STAMP}}"
STAGE2_OUT="${STAGE2_OUT:-outputs/tgvf_v3_protocol_c/${STAGE2_RUN_NAME}}"
STAGE2_CKPT="${STAGE2_CKPT:-${STAGE2_OUT}/checkpoint_step_1200.pt}"
STAGE2_PROCESSOR="${STAGE2_PROCESSOR:-${STAGE2_OUT}/processor_step_1200}"

TRAIN_FILE="${TRAIN_FILE:-data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl}"
VAL_FILE="${VAL_FILE:-data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl}"
BENCH_RUN_ROOT="${BENCH_RUN_ROOT:-outputs/full_benchmarks_oldmaskprob075_throughanswer_correct_stage1_4gpu_${STAMP}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MAX_STEPS="${MAX_STEPS:-1200}"
MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION:-512}"
MASK_ORIGINAL_IMAGE_AFTER_TGVF_PROB="${MASK_ORIGINAL_IMAGE_AFTER_TGVF_PROB:-0.75}"
MASK_ORIGINAL_IMAGE_AFTER_TGVF_SCOPE="${MASK_ORIGINAL_IMAGE_AFTER_TGVF_SCOPE:-through_answer}"
RUN_PROTOCOL_EVAL="${RUN_PROTOCOL_EVAL:-1}"
RUN_FULL_BENCHMARKS="${RUN_FULL_BENCHMARKS:-1}"

if [[ "${MASK_ORIGINAL_IMAGE_AFTER_TGVF_SCOPE}" != "through_answer" ]]; then
  echo "This wrapper is intended for old attention mask behavior: through_answer" >&2
  exit 1
fi

mkdir -p "${STAGE2_OUT}/logs" "${BENCH_RUN_ROOT}/logs"
cat > "${STAGE2_OUT}/run_wrapper_config.txt" <<CFG
stamp=${STAMP}
train_gpus=${TRAIN_GPUS}
eval_gpus=${EVAL_GPUS}
bench_gpus=${BENCH_GPUS}
model_id=${MODEL_ID}
stage1_run_id=${STAGE1_RUN_ID}
stage1_ckpt=${STAGE1_CKPT}
stage1_processor=${STAGE1_PROCESSOR}
stage2_run_name=${STAGE2_RUN_NAME}
stage2_out=${STAGE2_OUT}
stage2_ckpt=${STAGE2_CKPT}
stage2_processor=${STAGE2_PROCESSOR}
train_file=${TRAIN_FILE}
val_file=${VAL_FILE}
max_steps=${MAX_STEPS}
max_image_resolution=${MAX_IMAGE_RESOLUTION}
mask_original_image_after_tgvf=1
mask_original_image_after_tgvf_prob=${MASK_ORIGINAL_IMAGE_AFTER_TGVF_PROB}
mask_original_image_after_tgvf_scope=${MASK_ORIGINAL_IMAGE_AFTER_TGVF_SCOPE}
run_protocol_eval=${RUN_PROTOCOL_EVAL}
run_full_benchmarks=${RUN_FULL_BENCHMARKS}
bench_run_root=${BENCH_RUN_ROOT}
CFG

echo "[oldmask075-correct-stage1] Stage2 output: ${STAGE2_OUT}"
echo "[oldmask075-correct-stage1] Stage1 checkpoint: ${STAGE1_CKPT}"
echo "[oldmask075-correct-stage1] train_gpus: ${TRAIN_GPUS}"
echo "[oldmask075-correct-stage1] eval_gpus: ${EVAL_GPUS}"
echo "[oldmask075-correct-stage1] mask scope: ${MASK_ORIGINAL_IMAGE_AFTER_TGVF_SCOPE}"
echo "[oldmask075-correct-stage1] mask prob: ${MASK_ORIGINAL_IMAGE_AFTER_TGVF_PROB}"

CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MODEL_ID="${MODEL_ID}" \
STAGE1_CKPT="${STAGE1_CKPT}" \
PROCESSOR_ID="${STAGE1_PROCESSOR}" \
TRAIN_FILE="${TRAIN_FILE}" \
VAL_FILE="${VAL_FILE}" \
RUN_NAME="${STAGE2_RUN_NAME}" \
OUT_DIR="${STAGE2_OUT}" \
MAX_STEPS="${MAX_STEPS}" \
MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION}" \
MASK_ORIGINAL_IMAGE_AFTER_TGVF=1 \
MASK_ORIGINAL_IMAGE_AFTER_TGVF_PROB="${MASK_ORIGINAL_IMAGE_AFTER_TGVF_PROB}" \
MASK_ORIGINAL_IMAGE_AFTER_TGVF_SCOPE="${MASK_ORIGINAL_IMAGE_AFTER_TGVF_SCOPE}" \
RUN_PROTOCOL_EVAL=0 \
WANDB_GROUP="${WANDB_GROUP:-protocol_c_toolobs_oldmask075_correct_stage1_stage2}" \
WANDB_TAGS="${WANDB_TAGS:-protocol_c_toolobs,stage2,4gpu,focus80,v4data_clean,open_answer,20260619_stage1,old_attention_mask,maskprob075}" \
bash scripts/run_tgvf_v3_protocol_c_stage2_from_stage1c_4gpu.sh \
  2>&1 | tee "${STAGE2_OUT}/logs/stage2_wrapper.log"

if [[ ! -f "${STAGE2_CKPT}" ]]; then
  echo "Missing Stage2 checkpoint: ${STAGE2_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${STAGE2_PROCESSOR}" ]]; then
  echo "Missing Stage2 processor: ${STAGE2_PROCESSOR}" >&2
  exit 1
fi

if [[ "${RUN_PROTOCOL_EVAL}" == "1" ]]; then
  EVAL_OUT_DIR="${STAGE2_OUT}/protocol_eval_step_${MAX_STEPS}"
  mkdir -p "${EVAL_OUT_DIR}"
  echo "[oldmask075-correct-stage1] Protocol eval output: ${EVAL_OUT_DIR}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" \
  /usr/bin/time -v python eval/eval_v3_stage2_protocol.py \
    --stage2-checkpoint "${STAGE2_CKPT}" \
    --eval-jsonl "${VAL_FILE}" \
    --output-dir "${EVAL_OUT_DIR}" \
    --model-id "${MODEL_ID}" \
    --processor-id "${STAGE2_PROCESSOR}" \
    --device cuda:0 \
    --device-map cuda:0 \
    --attn-implementation sdpa \
    --tgvf-protocol protocol_c_tool_observation \
    --blocks no_focus_direct,force_focus_targets,teacher_forced_post_tgvf,force_end2end,free_router_end2end \
    --d-conditions correct_D,no_D,random_D,wrong_same_image_D,wrong_diff_image_D \
    --max-focus 128 \
    --max-no-focus 128 \
    --max-action-tokens 128 \
    --max-answer-tokens 128 \
    2>&1 | tee "${STAGE2_OUT}/logs/protocol_eval_step_${MAX_STEPS}.log"
fi

if [[ "${RUN_FULL_BENCHMARKS}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" \
  BENCH_GPUS="${BENCH_GPUS}" \
  NUM_GPUS="${NUM_GPUS}" \
  NUM_SHARDS="${NUM_SHARDS}" \
  MODEL_ID="${MODEL_ID}" \
  STAMP="${STAMP}" \
  RUN_CHAIN=0 \
  RUN_VSTAR=1 \
  RUN_EXTERNAL_BENCHMARKS=1 \
  RUN_FREE=1 \
  RUN_SOFTFORCE=1 \
  PROTOCOL_TOKEN_ROW_MODE=row_only \
  STAGE1_RUN_ID="${STAGE1_RUN_ID}" \
  STAGE1_CKPT="${STAGE1_CKPT}" \
  STAGE1_PROCESSOR="${STAGE1_PROCESSOR}" \
  STAGE2_RUN_NAME="${STAGE2_RUN_NAME}" \
  STAGE2_OUT="${STAGE2_OUT}" \
  STAGE2_CKPT="${STAGE2_CKPT}" \
  STAGE2_PROCESSOR="${STAGE2_PROCESSOR}" \
  BENCH_RUN_ROOT="${BENCH_RUN_ROOT}" \
  MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION}" \
  bash scripts/run_tgvf_v3_fullmask_stage1_stage2_fullbench_2gpu.sh \
    2>&1 | tee "${BENCH_RUN_ROOT}/logs/fullbench_wrapper.log"
fi

echo "[oldmask075-correct-stage1] Stage2: ${STAGE2_CKPT}"
echo "[oldmask075-correct-stage1] Protocol eval: ${STAGE2_OUT}/protocol_eval_step_${MAX_STEPS}"
echo "[oldmask075-correct-stage1] Full benchmarks: ${BENCH_RUN_ROOT}"
