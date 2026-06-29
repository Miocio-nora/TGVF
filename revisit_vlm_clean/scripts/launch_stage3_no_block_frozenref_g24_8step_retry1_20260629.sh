#!/usr/bin/env bash
set -euo pipefail

ROOT="/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm"
OUT="${ROOT}/outputs/stage3_grpo/all5_hint_no_block_frozenref_4gpu_g24_res768_8step_retry1_20260629"
PLAN="${OUT}/step_000001/stage3_grpo_training_plan.json"
LAUNCH="${OUT}/launch_tmux.sh"

cd "${ROOT}"

if [[ ! -f "${PLAN}" ]]; then
  echo "missing prepared Stage3 plan: ${PLAN}" >&2
  exit 1
fi

if [[ ! -x "${LAUNCH}" ]]; then
  echo "missing executable launch script: ${LAUNCH}" >&2
  exit 1
fi

exec "${LAUNCH}"
