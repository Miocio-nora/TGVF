#!/usr/bin/env bash
set -euo pipefail

cd /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/src:$(pwd)/third_party/VLMEvalKit"
export TGVF_TOOLOBS_ACTION_STOP="${TGVF_TOOLOBS_ACTION_STOP:-im_end}"
export TGVF_QWEN3_CAPTURE_GENERATE="${TGVF_QWEN3_CAPTURE_GENERATE:-1}"
export MASTER_PORT="${MASTER_PORT:-29614}"

RUN_ROOT="${RUN_ROOT:-outputs/vlmevalkit/vstar_max16k_exact_mcq_latest_and_qwen_20260614}"
mkdir -p "$RUN_ROOT"

torchrun --nproc-per-node=4 third_party/VLMEvalKit/run.py \
  --data VStarBench \
  --model \
    Qwen3-VL-8B-Thinking-HF-Local-Max16KPixels \
    TGVF-Qwen3VL-8B-ToolObs-FocusImEnd-FromFocusImEndStage1-Free-Max16KPixels \
  --work-dir "$RUN_ROOT" \
  --mode all \
  --judge exact_matching
