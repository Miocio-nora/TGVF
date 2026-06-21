#!/usr/bin/env bash
set -euo pipefail

cd /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/src"
export TGVF_TOOLOBS_ACTION_STOP=im_end
export TGVF_BLOCK_FOCUS_IN_CONTINUATION="${TGVF_BLOCK_FOCUS_IN_CONTINUATION:-1}"
export TGVF_QWEN3_CAPTURE_GENERATE="${TGVF_QWEN3_CAPTURE_GENERATE:-1}"

CKPT="outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/checkpoint_step_1200.pt"
PROC="outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/processor_step_1200"
ROOT="${ROOT:-outputs/other_benchmarks_512_clean_rowonly_gpu0_3_imend_20260617}"
SHARDS="${SHARDS:-4}"
mkdir -p "$ROOT"

run_benchmark() {
  local bench="$1"
  local tier="$2"
  local out="$ROOT/$bench"
  mkdir -p "$out"
  echo "=== START clean $bench tier=$tier $(date -Is) ==="
  for shard in $(seq 0 $((SHARDS - 1))); do
    (
      export CUDA_VISIBLE_DEVICES="$shard"
      python eval/eval_v3_mmmu_force.py \
        --benchmark "$bench" \
        --tier "$tier" \
        --output-dir "$out/shard_$shard" \
        --eval-mode free \
        --d-conditions correct_D \
        --no-include-stage2-direct \
        --post-tgvf-continuation natural_continue \
        --stage2-checkpoint "$CKPT" \
        --processor-id "$PROC" \
        --tgvf-protocol protocol_c_tool_observation \
        --max-image-resolution 512 \
        --max-action-tokens 128 \
        --max-answer-tokens 512 \
        --device cuda:0 \
        --device-map cuda:0 \
        --attn-implementation sdpa \
        --num-shards "$SHARDS" \
        --shard-index "$shard" \
        --log-every 20
    ) > "$out/shard_$shard.log" 2>&1 &
  done
  wait
  echo "=== DONE clean $bench $(date -Is) ==="
}

run_benchmark blink medium
run_benchmark mmmu_pro medium
run_benchmark mathvista full
