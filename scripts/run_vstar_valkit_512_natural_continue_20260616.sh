#!/usr/bin/env bash
set -euo pipefail

cd /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/src:$(pwd)/third_party/VLMEvalKit"

# Strict im_end-trained tool-observation path:
# action generation should emit <|im_end|>, and the continuation should also stop on <|im_end|>.
export TGVF_TOOLOBS_ACTION_STOP=im_end
export TGVF_BLOCK_FOCUS_IN_CONTINUATION="${TGVF_BLOCK_FOCUS_IN_CONTINUATION:-1}"
export TGVF_QWEN3_CAPTURE_GENERATE="${TGVF_QWEN3_CAPTURE_GENERATE:-1}"
export MASTER_PORT="${MASTER_PORT:-29616}"

RUN_ROOT="${RUN_ROOT:-outputs/vlmevalkit/vstar_512_natural_continue_20260616}"
TRACE_ROOT="${TRACE_ROOT:-$RUN_ROOT/tgvf_traces}"
mkdir -p "$RUN_ROOT" "$TRACE_ROOT"
export TGVF_VLMEVAL_TRACE_DIR="$TRACE_ROOT"

torchrun --nproc-per-node=4 third_party/VLMEvalKit/run.py \
  --data VStarBench \
  --model \
    Qwen3-VL-8B-Thinking-HF-Local-512 \
    TGVF-Qwen3VL-8B-ToolObs-FocusImEnd-FromFocusImEndStage1-Free-512 \
  --work-dir "$RUN_ROOT" \
  --mode all \
  --judge exact_matching
