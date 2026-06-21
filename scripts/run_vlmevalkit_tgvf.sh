#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/third_party/VLMEvalKit:$ROOT/src:$ROOT:${PYTHONPATH:-}"
export TGVF_QWEN3_CAPTURE_GENERATE="${TGVF_QWEN3_CAPTURE_GENERATE:-1}"

DATASET="${DATASET:-VStarBench}"
MODEL="${MODEL:-TGVF-Qwen3VL-8B-ToolObs-Stage2-Force-512}"
WORK_DIR="${WORK_DIR:-outputs/vlmevalkit}"
MODE="${MODE:-all}"

python third_party/VLMEvalKit/run.py \
  --data "$DATASET" \
  --model "$MODEL" \
  --work-dir "$WORK_DIR" \
  --mode "$MODE" \
  "$@"
