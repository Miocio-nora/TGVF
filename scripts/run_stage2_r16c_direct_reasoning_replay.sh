#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
STAMP=20260714_225235
RUN_ROOT="$ROOT/outputs/clean_ablation/stage2_r16c_direct_reasoning_replay_4gpu_${STAMP}"
SMOKE_DIR="$RUN_ROOT/smoke/stage2_micro4"
TRAIN_DIR="$RUN_ROOT/main/stage2_micro4"

DATA_ROOT="$ROOT/data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer_original_reasoning_v1"
TRAIN_FILE="$DATA_ROOT/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl"
VAL_FILE="$DATA_ROOT/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl"
STAGE1="$ROOT/outputs/clean_training/qwen3_stage1_ddeepstack_norm01_wandb_4gpu_20260702_180906/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt"
MODEL=/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking

export PYTHONPATH="$ROOT/revisit_vlm_clean/src:$ROOT/src"
cd "$ROOT"

mkdir -p "$RUN_ROOT/logs" "$SMOKE_DIR" "$TRAIN_DIR"

write_plan() {
  local output_dir=$1
  local run_id=$2
  local max_steps=$3
  local save_every=$4
  local eval_every=$5
  local wandb_mode=$6

  python -m revisit_vlm_clean.cli.train_stage2 \
    --run-id "$run_id" \
    --train-file "$TRAIN_FILE" \
    --val-file "$VAL_FILE" \
    --stage1-checkpoint "$STAGE1" \
    --output-dir "$output_dir" \
    --model-id "$MODEL" \
    --processor-id "$MODEL" \
    --protocol protocol_c_tool_observation \
    --global-batch 128 \
    --world-size 4 \
    --micro-batch-size 4 \
    --gradient-accumulation-steps 8 \
    --max-steps "$max_steps" \
    --save-every "$save_every" \
    --eval-every "$eval_every" \
    --seed 20260525 \
    --max-image-resolution 512 \
    --max-seq-len 2048 \
    --dtype bfloat16 \
    --attn-implementation sdpa \
    --variant tgvf_v2_bidirectional \
    --use-stage1-tgvf-config \
    --fast-batched-stage2 \
    --fvt-position-mode native_source_grid \
    --target-focus-ratio 0.8 \
    --mask-original-image-after-tgvf \
    --mask-original-image-after-tgvf-prob 0.75 \
    --mask-original-image-after-tgvf-scope through_answer \
    --deepstack-enabled \
    --deepstack-original-image-scope through_answer \
    --d-deepstack-enabled \
    --lora-rank 16 \
    --lora-alpha 64 \
    --lora-dropout 0.05 \
    --lora-bias none \
    --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --protocol-token-training-mode row_only \
    --lr-lora 2e-5 \
    --lr-tgvf 5e-6 \
    --lr-calibration 1e-5 \
    --lr-scheduler cosine \
    --warmup-ratio 0.03 \
    --warmup-steps 100 \
    --min-lr-ratio 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --weight-decay 0.01 \
    --max-grad-norm 1.0 \
    --loss-evidence-state 0.2 \
    --loss-focus-target 1.5 \
    --loss-evidence 1.0 \
    --loss-value-span 1.0 \
    --loss-answer 1.0 \
    --loss-no-focus-evidence-state 0.2 \
    --loss-no-focus-answer 1.0 \
    --loss-visual-token-manifold 0.0 \
    --no-matrix-ce-preservation \
    --wandb-project tgvf-clean-qwen3-deepstack \
    --wandb-mode "$wandb_mode" \
    --write-plan
}

prepare() {
  write_plan \
    "$SMOKE_DIR" \
    "stage2_r16c_direct_reasoning_replay_smoke_4gpu_${STAMP}" \
    1 1 1 disabled
  python -m revisit_vlm_clean.training.stage2_executor \
    --plan "$SMOKE_DIR/training_plan.json" \
    --prepare-execution

  write_plan \
    "$TRAIN_DIR" \
    "stage2_r16c_direct_reasoning_replay_main_4gpu_${STAMP}" \
    1200 300 300 online
  python -m revisit_vlm_clean.training.stage2_executor \
    --plan "$TRAIN_DIR/training_plan.json" \
    --prepare-execution
}

smoke() {
  local execution_dir="$SMOKE_DIR/clean_training_execution"
  local checkpoint="$execution_dir/checkpoint_step_1.pt"
  local progress="$execution_dir/training_progress.jsonl"
  if [[ -f "$checkpoint" ]]; then
    echo "[$(date -Is)] smoke already complete: $checkpoint"
    return
  fi
  echo "[$(date -Is)] Stage2 R16-C direct-reasoning replay smoke start"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  MASTER_PORT=29751 \
  torchrun --nproc-per-node 4 \
    -m revisit_vlm_clean.training.stage2_executor \
    --plan "$SMOKE_DIR/training_plan.json" \
    --launch-training \
    2>&1 | tee "$RUN_ROOT/logs/smoke.log"

  test -f "$checkpoint"
  jq -e 'select(.event == "finish" and .optimizer_steps_completed == 1)' \
    "$progress" >/dev/null
  jq -e '
    select(.event == "optimizer_step") |
    .debug_summary.focus_count > 0 and
    .debug_summary.no_focus_count > 0 and
    .debug_summary.no_focus_loss_token_weight > 0 and
    .debug_summary.deepstack_training_enabled == true and
    .debug_summary.matrix_ce_enabled == false and
    .loss_no_focus != null
  ' "$progress" >/dev/null
  echo "[$(date -Is)] Stage2 R16-C direct-reasoning replay smoke passed"
}

train() {
  local execution_dir="$TRAIN_DIR/clean_training_execution"
  local checkpoint="$execution_dir/checkpoint_step_1200.pt"
  if [[ -f "$checkpoint" ]]; then
    echo "[$(date -Is)] training already complete: $checkpoint"
    return
  fi
  echo "[$(date -Is)] Stage2 R16-C direct-reasoning replay training start"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  MASTER_PORT=29752 \
  torchrun --nproc-per-node 4 \
    -m revisit_vlm_clean.training.stage2_executor \
    --plan "$TRAIN_DIR/training_plan.json" \
    --launch-training \
    2>&1 | tee "$RUN_ROOT/logs/train.log"
  test -f "$checkpoint"
  echo "[$(date -Is)] Stage2 R16-C direct-reasoning replay training done"
}

case "${1:-all}" in
  prepare)
    prepare
    ;;
  smoke)
    smoke
    ;;
  train)
    train
    ;;
  all)
    prepare
    smoke
    train
    ;;
  *)
    echo "usage: $0 [prepare|smoke|train|all]" >&2
    exit 2
    ;;
esac
