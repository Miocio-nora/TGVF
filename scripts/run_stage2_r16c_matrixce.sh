#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
STAMP=20260714_122949
RUN_ROOT="$ROOT/outputs/clean_ablation/stage2_r16c_matrixce_4gpu_${STAMP}"
SMOKE_DIR="$RUN_ROOT/smoke/stage2_micro4"
TRAIN_DIR="$RUN_ROOT/main/stage2_micro4"
DIAG_DIR="$TRAIN_DIR/internal_diagnostics_step1200"
BENCH_ROOT="$ROOT/outputs/clean_benchmarks/qwen3_stage2_r16c_matrixce_coredev2511_${STAMP}"

TRAIN_FILE="$ROOT/data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl"
VAL_FILE="$ROOT/data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl"
INTERNAL_FILE="$ROOT/data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl"
STAGE1="$ROOT/outputs/clean_training/qwen3_stage1_ddeepstack_norm01_wandb_4gpu_20260702_180906/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt"
MODEL=/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking
MANIFEST="$ROOT/revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json"
MANIFEST_HASH=a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579
BENCHMARK_DATA=/home/dredvpn009/Flash_Storage/datasets/benchmarks

export PYTHONPATH="$ROOT/revisit_vlm_clean/src:$ROOT/src"
cd "$ROOT"

mkdir -p "$RUN_ROOT/logs" "$SMOKE_DIR" "$TRAIN_DIR" "$DIAG_DIR/logs" "$BENCH_ROOT"

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
    --matrix-ce-preservation \
    --loss-same-image-matrix-ce 1.0 \
    --matrix-ce-group-size 4 \
    --matrix-ce-readout-batch-size 4 \
    --wandb-project tgvf-clean-qwen3-deepstack \
    --wandb-mode "$wandb_mode" \
    --write-plan
}

prepare() {
  write_plan \
    "$SMOKE_DIR" \
    "stage2_r16c_matrixce_smoke_4gpu_${STAMP}" \
    1 1 1 disabled
  python -m revisit_vlm_clean.training.stage2_executor \
    --plan "$SMOKE_DIR/training_plan.json" \
    --prepare-execution

  write_plan \
    "$TRAIN_DIR" \
    "stage2_r16c_matrixce_main_4gpu_${STAMP}" \
    1200 300 300 online
  python -m revisit_vlm_clean.training.stage2_executor \
    --plan "$TRAIN_DIR/training_plan.json" \
    --prepare-execution
}

smoke() {
  local checkpoint="$SMOKE_DIR/clean_training_execution/checkpoint_step_1.pt"
  if [[ -f "$checkpoint" ]]; then
    echo "[$(date -Is)] smoke already complete: $checkpoint"
    return
  fi
  echo "[$(date -Is)] Stage2 R16-C + Matrix-CE smoke start"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  MASTER_PORT=29743 \
  torchrun --nproc-per-node 4 \
    -m revisit_vlm_clean.training.stage2_executor \
    --plan "$SMOKE_DIR/training_plan.json" \
    --launch-training \
    2>&1 | tee "$RUN_ROOT/logs/smoke.log"
  test -f "$checkpoint"
  jq -e 'select(.event == "finish" and .optimizer_steps_completed == 1)' \
    "$SMOKE_DIR/clean_training_execution/training_progress.jsonl" >/dev/null
  echo "[$(date -Is)] Stage2 R16-C + Matrix-CE smoke passed"
}

train() {
  local checkpoint="$TRAIN_DIR/clean_training_execution/checkpoint_step_1200.pt"
  if [[ -f "$checkpoint" ]]; then
    echo "[$(date -Is)] training already complete: $checkpoint"
    return
  fi
  echo "[$(date -Is)] Stage2 R16-C + Matrix-CE training start"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  MASTER_PORT=29744 \
  torchrun --nproc-per-node 4 \
    -m revisit_vlm_clean.training.stage2_executor \
    --plan "$TRAIN_DIR/training_plan.json" \
    --launch-training \
    2>&1 | tee "$RUN_ROOT/logs/train.log"
  test -f "$checkpoint"
  echo "[$(date -Is)] Stage2 R16-C + Matrix-CE training done"
}

diagnostics() {
  local checkpoint="$TRAIN_DIR/clean_training_execution/checkpoint_step_1200.pt"
  local summary="$DIAG_DIR/query_sensitivity/query_sensitivity_report.json"
  if [[ -f "$summary" ]]; then
    echo "[$(date -Is)] internal diagnostics already complete: $summary"
    return
  fi
  echo "[$(date -Is)] internal diagnostics start"
  CUDA_VISIBLE_DEVICES=0 \
  python -m revisit_vlm_clean.cli.stage_diagnostics \
    --run-id "stage2_r16c_matrixce_step1200_internal_${STAMP}" \
    --stage stage2 \
    --checkpoint "$checkpoint" \
    --eval-jsonl "$INTERNAL_FILE" \
    --output-dir "$DIAG_DIR" \
    --model-id "$MODEL" \
    --processor-id "$MODEL" \
    --protocol protocol_c_tool_observation \
    --variant tgvf_v2_bidirectional \
    --num-foveated-tokens none \
    --encoder-adapter-type bidirectional \
    --max-image-resolution 512 \
    --fvt-position-mode native_source_grid \
    --dtype bfloat16 \
    --attn-implementation sdpa \
    --device cuda:0 \
    --device-map cuda:0 \
    --tasks readout,query,distribution \
    --readout-max-samples 200 \
    --distribution-max-samples 200 \
    --query-max-groups 50 \
    --query-require-groups 0 \
    --query-min-targets-per-image 3 \
    --eval-workers 1 \
    --stage2-load-lora \
    --seed 20260525 \
    --focus-action-im-end \
    --no-use-fvt-cache \
    --execute \
    2>&1 | tee "$RUN_ROOT/logs/diagnostics.log"
  test -f "$summary"
  echo "[$(date -Is)] internal diagnostics done"
}

benchmark_args() {
  local run_id=$1
  local mode=$2
  local output_dir=$3
  shift 3
  python -m revisit_vlm_clean.cli.dynamic_benchmark \
    --run-id "$run_id" \
    --checkpoint-path "$TRAIN_DIR/clean_training_execution/checkpoint_step_1200.pt" \
    --model-id "$MODEL" \
    --processor-id "$MODEL" \
    --mode "$mode" \
    --runner-backend tgvf_stage2_qwen3_native \
    --post-tgvf-forward-mode kv_cache \
    --subset-id core_balanced_dev_2511_seed20260625 \
    --manifest-path "$MANIFEST" \
    --manifest-hash "$MANIFEST_HASH" \
    --benchmark-root "$BENCHMARK_DATA" \
    --output-dir "$output_dir" \
    --max-image-resolution 512 \
    --max-tokens 512 \
    --scoring-backend auto \
    --dtype bfloat16 \
    --device cuda:0 \
    --device-map cuda:0 \
    --attn-implementation flash_attention_2 \
    --stage2-checkpoint "$TRAIN_DIR/clean_training_execution/checkpoint_step_1200.pt" \
    --stage2-eval-jsonl "$VAL_FILE" \
    --stage2-d-condition correct_D \
    --force-prefix-mode target_hint \
    --deepstack-enabled \
    --deepstack-original-image-scope no_block \
    --d-deepstack-enabled \
    "$@"
}

verify_benchmark_smoke() {
  local summary=$1
  jq -e '
    .n_rows == 4 and
    .n_scored == 4 and
    .answer_parse_rate == 1 and
    .append_success_rate == 1 and
    .malformed_rate == 0
  ' "$summary" >/dev/null
}

evaluate() {
  local smoke_free="$BENCH_ROOT/smoke_free"
  local smoke_soft="$BENCH_ROOT/smoke_softforce"
  local free_dir="$BENCH_ROOT/tgvf_free"
  local soft_dir="$BENCH_ROOT/tgvf_softforce"

  if [[ ! -f "$smoke_free/summary.json" ]]; then
    echo "[$(date -Is)] free benchmark smoke start"
    CUDA_VISIBLE_DEVICES=0 benchmark_args \
      "stage2_r16c_matrixce_coredev2511_smoke_free_${STAMP}" \
      tgvf_free "$smoke_free" \
      --gpus 0 --batch-size 1 --max-samples 4 --progress-every 1 \
      2>&1 | tee "$RUN_ROOT/logs/benchmark_smoke_free.log"
  fi
  verify_benchmark_smoke "$smoke_free/summary.json"

  if [[ ! -f "$smoke_soft/summary.json" ]]; then
    echo "[$(date -Is)] softforce benchmark smoke start"
    CUDA_VISIBLE_DEVICES=0 benchmark_args \
      "stage2_r16c_matrixce_coredev2511_smoke_softforce_${STAMP}" \
      tgvf_softforce "$smoke_soft" \
      --softforce-prompt-text "Use focus tool." \
      --gpus 0 --batch-size 1 --max-samples 4 --progress-every 1 \
      2>&1 | tee "$RUN_ROOT/logs/benchmark_smoke_softforce.log"
  fi
  verify_benchmark_smoke "$smoke_soft/summary.json"

  if [[ ! -f "$free_dir/summary.json" ]]; then
    echo "[$(date -Is)] complete CoreDev-2511 free start"
    CUDA_VISIBLE_DEVICES=0,1,2,3 benchmark_args \
      "stage2_r16c_matrixce_coredev2511_free_${STAMP}" \
      tgvf_free "$free_dir" \
      --gpus 0,1,2,3 --batch-size 1 --progress-every 25 \
      2>&1 | tee "$RUN_ROOT/logs/benchmark_free.log"
  fi

  if [[ ! -f "$soft_dir/summary.json" ]]; then
    echo "[$(date -Is)] complete CoreDev-2511 softforce start"
    CUDA_VISIBLE_DEVICES=0,1,2,3 benchmark_args \
      "stage2_r16c_matrixce_coredev2511_softforce_${STAMP}" \
      tgvf_softforce "$soft_dir" \
      --softforce-prompt-text "Use focus tool." \
      --gpus 0,1,2,3 --batch-size 1 --progress-every 25 \
      2>&1 | tee "$RUN_ROOT/logs/benchmark_softforce.log"
  fi
  echo "[$(date -Is)] complete R16-C + Matrix-CE workflow done"
}

usage() {
  echo "Usage: $0 {prepare|smoke|train|diagnostics|evaluate|all}"
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
  diagnostics)
    diagnostics
    ;;
  evaluate)
    evaluate
    ;;
  all)
    prepare
    smoke
    train
    diagnostics
    evaluate
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
