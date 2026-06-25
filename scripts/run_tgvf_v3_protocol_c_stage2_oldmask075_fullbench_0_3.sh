#!/usr/bin/env bash
set -euo pipefail

cd /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export BENCH_GPUS="${BENCH_GPUS:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/src:$(pwd)/third_party/VLMEvalKit"
export TGVF_TOOLOBS_ACTION_STOP=im_end
export TGVF_BLOCK_FOCUS_IN_CONTINUATION="${TGVF_BLOCK_FOCUS_IN_CONTINUATION:-1}"
export TGVF_QWEN3_CAPTURE_GENERATE="${TGVF_QWEN3_CAPTURE_GENERATE:-1}"
export TGVF_STOP_REPETITIVE_CONTINUATION="${TGVF_STOP_REPETITIVE_CONTINUATION:-1}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MODEL_ID="${MODEL_ID:-/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking}"

STAGE1_RUN_ID="${STAGE1_RUN_ID:-protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617}"
STAGE1_ROOT="${STAGE1_ROOT:-outputs/tgvf_v3_protocol_c_stage1_8b/${STAGE1_RUN_ID}}"
STAGE1_CKPT="${STAGE1_CKPT:-${STAGE1_ROOT}/train/checkpoint_step_2000.pt}"
STAGE1_PROCESSOR="${STAGE1_PROCESSOR:-${STAGE1_ROOT}/train/processor_step_2000}"

STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_oldmaskprob075_throughanswer_from_clean_rowonly_stage1_4gpu_bs16_accum2_focus80_value1_1200step_${STAMP}}"
STAGE2_OUT="${STAGE2_OUT:-outputs/tgvf_v3_protocol_c/${STAGE2_RUN_NAME}}"
STAGE2_CKPT="${STAGE2_CKPT:-${STAGE2_OUT}/checkpoint_step_1200.pt}"
STAGE2_PROCESSOR="${STAGE2_PROCESSOR:-${STAGE2_OUT}/processor_step_1200}"

BENCH_RUN_ROOT="${BENCH_RUN_ROOT:-outputs/full_benchmarks_oldmaskprob075_throughanswer_4gpu_${STAMP}}"
RUN_FULL_BENCHMARKS="${RUN_FULL_BENCHMARKS:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MAX_STEPS="${MAX_STEPS:-1200}"
MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION:-512}"
MASK_ORIGINAL_IMAGE_AFTER_TGVF_PROB="${MASK_ORIGINAL_IMAGE_AFTER_TGVF_PROB:-0.75}"
MASK_ORIGINAL_IMAGE_AFTER_TGVF_SCOPE="${MASK_ORIGINAL_IMAGE_AFTER_TGVF_SCOPE:-through_answer}"

if [[ "${MASK_ORIGINAL_IMAGE_AFTER_TGVF_SCOPE}" != "through_answer" ]]; then
  echo "This wrapper is intended for old attention mask behavior: through_answer" >&2
  exit 1
fi

mkdir -p "${STAGE2_OUT}/logs" "${BENCH_RUN_ROOT}/logs"
cat > "${STAGE2_OUT}/run_wrapper_config.txt" <<CFG
stamp=${STAMP}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
bench_gpus=${BENCH_GPUS}
model_id=${MODEL_ID}
stage1_run_id=${STAGE1_RUN_ID}
stage1_ckpt=${STAGE1_CKPT}
stage1_processor=${STAGE1_PROCESSOR}
stage2_run_name=${STAGE2_RUN_NAME}
stage2_out=${STAGE2_OUT}
stage2_ckpt=${STAGE2_CKPT}
stage2_processor=${STAGE2_PROCESSOR}
max_steps=${MAX_STEPS}
max_image_resolution=${MAX_IMAGE_RESOLUTION}
mask_original_image_after_tgvf=1
mask_original_image_after_tgvf_prob=${MASK_ORIGINAL_IMAGE_AFTER_TGVF_PROB}
mask_original_image_after_tgvf_scope=${MASK_ORIGINAL_IMAGE_AFTER_TGVF_SCOPE}
run_protocol_eval=1
run_full_benchmarks=${RUN_FULL_BENCHMARKS}
bench_run_root=${BENCH_RUN_ROOT}
CFG

echo "[oldmask075] Stage2 output: ${STAGE2_OUT}"
echo "[oldmask075] Stage1 checkpoint: ${STAGE1_CKPT}"
echo "[oldmask075] mask scope: ${MASK_ORIGINAL_IMAGE_AFTER_TGVF_SCOPE}"
echo "[oldmask075] mask prob: ${MASK_ORIGINAL_IMAGE_AFTER_TGVF_PROB}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MODEL_ID="${MODEL_ID}" \
STAGE1_CKPT="${STAGE1_CKPT}" \
PROCESSOR_ID="${STAGE1_PROCESSOR}" \
RUN_NAME="${STAGE2_RUN_NAME}" \
OUT_DIR="${STAGE2_OUT}" \
MAX_STEPS="${MAX_STEPS}" \
MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION}" \
MASK_ORIGINAL_IMAGE_AFTER_TGVF=1 \
MASK_ORIGINAL_IMAGE_AFTER_TGVF_PROB="${MASK_ORIGINAL_IMAGE_AFTER_TGVF_PROB}" \
MASK_ORIGINAL_IMAGE_AFTER_TGVF_SCOPE="${MASK_ORIGINAL_IMAGE_AFTER_TGVF_SCOPE}" \
RUN_PROTOCOL_EVAL=1 \
WANDB_GROUP="${WANDB_GROUP:-protocol_c_toolobs_oldmask075_stage2}" \
WANDB_TAGS="${WANDB_TAGS:-protocol_c_toolobs,stage2,4gpu,focus80,v4data_clean,open_answer,row_only_stage1,old_attention_mask,maskprob075}" \
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

if [[ "${RUN_FULL_BENCHMARKS}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
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

echo "[oldmask075] Stage2: ${STAGE2_CKPT}"
echo "[oldmask075] Protocol eval: ${STAGE2_OUT}/protocol_eval_step_${MAX_STEPS}"
echo "[oldmask075] Full benchmarks: ${BENCH_RUN_ROOT}"
