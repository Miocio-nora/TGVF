#!/usr/bin/env bash
set -euo pipefail

cd /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm

exec outputs/stage3_grpo/all5_hint_no_block_frozenref_rewardgate_softprompt8_4gpu_g20_res512_1step_worstmem_20260630_024144/launch_tmux.sh
