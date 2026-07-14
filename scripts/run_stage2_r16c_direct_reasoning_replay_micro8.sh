#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm

export STAMP=20260714_232021
export RUN_VARIANT=stage2_r16c_direct_reasoning_replay_micro8
export RUN_ROOT="$ROOT/outputs/clean_ablation/stage2_r16c_direct_reasoning_replay_micro8_4gpu_${STAMP}"
export MICRO_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=4
export SMOKE_MASTER_PORT=29753
export TRAIN_MASTER_PORT=29754

exec "$ROOT/scripts/run_stage2_r16c_direct_reasoning_replay.sh" "${1:-all}"
