#!/usr/bin/env bash
set -euo pipefail

cd /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
export PYTHONPATH=revisit_vlm_clean/src:src

STAMP=20260713_111841
EXP=r16_e_rowonly_evidence02
SUITE_ID=stage2_r16e_rowonly_evidence02_${STAMP}
SUITE_ROOT=outputs/clean_ablation/$SUITE_ID
TRAIN_ROOT=outputs/clean_training/qwen3_stage2_ddeepstack_${EXP}_8gpu_${STAMP}
SMOKE_ROOT=outputs/clean_training/qwen3_stage2_ddeepstack_${EXP}_8gpu_${STAMP}_smoke
TRAIN_FILE=data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl
VAL_FILE=data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl
INTERNAL_FILE=data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl
STAGE1=outputs/clean_training/qwen3_stage1_ddeepstack_norm01_wandb_4gpu_20260702_180906/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt
MODEL=/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking
MANIFEST=revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json
MANIFEST_HASH=a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579
BENCHMARK_ROOT=/home/dredvpn009/Flash_Storage/datasets/benchmarks

mkdir -p "$SUITE_ROOT/logs"

write_plan() {
  local output_dir=$1
  local max_steps=$2
  local save_every=$3
  local eval_every=$4
  local wandb_mode=$5
  local suffix=$6

  python -m revisit_vlm_clean.cli.train_stage2 \
    --run-id "clean_qwen3_stage2_ddeepstack_${EXP}_${suffix}_${STAMP}" \
    --train-file "$TRAIN_FILE" \
    --val-file "$VAL_FILE" \
    --stage1-checkpoint "$STAGE1" \
    --output-dir "$output_dir" \
    --model-id "$MODEL" \
    --processor-id "$MODEL" \
    --protocol protocol_c_tool_observation \
    --global-batch 128 \
    --world-size 8 \
    --micro-batch-size 4 \
    --gradient-accumulation-steps 4 \
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
    --loss-evidence 0.2 \
    --loss-value-span 1.0 \
    --loss-answer 1.0 \
    --loss-no-focus-evidence-state 0.2 \
    --loss-no-focus-answer 1.0 \
    --loss-visual-token-manifold 0.0 \
    --wandb-project tgvf-clean-qwen3-deepstack \
    --wandb-mode "$wandb_mode" \
    --write-plan
}

prepare() {
  local main_dir="$TRAIN_ROOT/stage2_micro4"
  local smoke_dir="$SMOKE_ROOT/stage2_micro4"

  write_plan "$smoke_dir" 1 1 1 disabled smoke
  python -m revisit_vlm_clean.training.stage2_executor \
    --plan "$smoke_dir/training_plan.json" --prepare-execution
  write_plan "$main_dir" 1200 300 300 online main
  python -m revisit_vlm_clean.training.stage2_executor \
    --plan "$main_dir/training_plan.json" --prepare-execution
}

smoke() {
  local root="$SMOKE_ROOT/stage2_micro4"
  local checkpoint="$root/clean_training_execution/checkpoint_step_1.pt"
  if [[ -f "$checkpoint" ]]; then
    echo "[$(date -Is)] smoke skip checkpoint=$checkpoint"
    return
  fi
  echo "[$(date -Is)] smoke start"
  MASTER_PORT=29631 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --nproc-per-node 8 \
    -m revisit_vlm_clean.training.stage2_executor \
    --plan "$root/training_plan.json" --launch-training \
    > "$SUITE_ROOT/logs/${EXP}_smoke.log" 2>&1
  echo "[$(date -Is)] smoke done checkpoint=$checkpoint"
}

train() {
  local root="$TRAIN_ROOT/stage2_micro4"
  local checkpoint="$root/clean_training_execution/checkpoint_step_1200.pt"
  if [[ -f "$checkpoint" ]]; then
    echo "[$(date -Is)] train skip checkpoint=$checkpoint"
    return
  fi
  echo "[$(date -Is)] train start"
  MASTER_PORT=29632 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --nproc-per-node 8 \
    -m revisit_vlm_clean.training.stage2_executor \
    --plan "$root/training_plan.json" --launch-training \
    > "$SUITE_ROOT/logs/${EXP}_train.log" 2>&1
  echo "[$(date -Is)] train done checkpoint=$checkpoint"
}

benchmark_args() {
  local checkpoint=$1
  local mode=$2
  local output_dir=$3
  local run_id=$4
  shift 4
  python -m revisit_vlm_clean.cli.dynamic_benchmark \
    --run-id "$run_id" \
    --checkpoint-path "$checkpoint" \
    --model-id "$MODEL" \
    --processor-id "$MODEL" \
    --mode "$mode" \
    --runner-backend tgvf_stage2_qwen3_native \
    --post-tgvf-forward-mode kv_cache \
    --subset-id core_balanced_dev_2511_seed20260625 \
    --manifest-path "$MANIFEST" \
    --manifest-hash "$MANIFEST_HASH" \
    --benchmark-root "$BENCHMARK_ROOT" \
    --output-dir "$output_dir" \
    --max-image-resolution 512 \
    --max-tokens 512 \
    --scoring-backend auto \
    --dtype bfloat16 \
    --device cuda:0 \
    --device-map cuda:0 \
    --attn-implementation flash_attention_2 \
    --stage2-checkpoint "$checkpoint" \
    --stage2-eval-jsonl "$VAL_FILE" \
    --stage2-d-condition correct_D \
    --force-prefix-mode target_hint \
    --deepstack-enabled \
    --deepstack-original-image-scope no_block \
    --d-deepstack-enabled \
    "$@"
}

evaluate() {
  local checkpoint="$TRAIN_ROOT/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt"
  local internal_dir="$TRAIN_ROOT/stage2_micro4/internal_diagnostics_step1200_${STAMP}"
  local smoke_dir="outputs/clean_benchmarks/qwen3_stage2_${EXP}_coredev2511_smoke_free_${STAMP}"
  local free_dir="outputs/clean_benchmarks/qwen3_stage2_${EXP}_coredev2511_free_${STAMP}/tgvf_free"
  local soft_dir="outputs/clean_benchmarks/qwen3_stage2_${EXP}_coredev2511_softforce_${STAMP}/tgvf_softforce"
  local internal_pid=

  if [[ ! -f "$smoke_dir/summary.json" ]]; then
    echo "[$(date -Is)] eval smoke start"
    CUDA_VISIBLE_DEVICES=0 benchmark_args \
      "$checkpoint" tgvf_free "$smoke_dir" \
      "clean_qwen3_stage2_${EXP}_coredev2511_smoke_free_${STAMP}" \
      --gpus 0 --batch-size 1 --max-samples 4 --progress-every 1 \
      > "$SUITE_ROOT/logs/${EXP}_eval_smoke.log" 2>&1
    echo "[$(date -Is)] eval smoke done"
  fi

  if [[ ! -f "$internal_dir/clean_stage_diagnostic_status.json" ]]; then
    echo "[$(date -Is)] internal start"
    CUDA_VISIBLE_DEVICES=0 python -m revisit_vlm_clean.cli.stage_diagnostics \
      --run-id "clean_qwen3_stage2_${EXP}_internal_${STAMP}" \
      --stage stage2 \
      --checkpoint "$checkpoint" \
      --eval-jsonl "$INTERNAL_FILE" \
      --output-dir "$internal_dir" \
      --model-id "$MODEL" \
      --processor-id "$MODEL" \
      --protocol protocol_c_tool_observation \
      --focus-action-im-end \
      --variant tgvf_v2_bidirectional \
      --encoder-adapter-type bidirectional \
      --max-image-resolution 512 \
      --fvt-position-mode native_source_grid \
      --dtype bfloat16 \
      --attn-implementation sdpa \
      --device cuda:0 \
      --device-map cuda:0 \
      --tasks all \
      --readout-max-samples 200 \
      --distribution-max-samples 200 \
      --query-max-groups 50 \
      --query-require-groups 0 \
      --seed 20260525 \
      --no-wandb-log-eval \
      --wandb-mode disabled \
      --execute > "$SUITE_ROOT/logs/${EXP}_internal.log" 2>&1 &
    internal_pid=$!
  fi

  if [[ ! -f "$free_dir/summary.json" ]]; then
    echo "[$(date -Is)] free start"
    benchmark_args "$checkpoint" tgvf_free "$free_dir" \
      "clean_qwen3_stage2_${EXP}_coredev2511_free_${STAMP}" \
      --gpus 1,2,3,4,5,6,7 --batch-size 1 --progress-every 25 \
      > "$SUITE_ROOT/logs/${EXP}_free.log" 2>&1
    echo "[$(date -Is)] free done"
  fi

  if [[ -n "$internal_pid" ]]; then
    wait "$internal_pid"
    echo "[$(date -Is)] internal done"
  fi

  if [[ ! -f "$soft_dir/summary.json" ]]; then
    echo "[$(date -Is)] softforce start"
    benchmark_args "$checkpoint" tgvf_softforce "$soft_dir" \
      "clean_qwen3_stage2_${EXP}_coredev2511_softforce_${STAMP}" \
      --softforce-prompt-text "Use focus tool." \
      --gpus 0,1,2,3,4,5,6,7 --batch-size 1 --progress-every 25 \
      > "$SUITE_ROOT/logs/${EXP}_softforce.log" 2>&1
    echo "[$(date -Is)] softforce done"
  fi
}

case "${1:-}" in
  prepare)
    prepare
    ;;
  smoke)
    smoke
    ;;
  run)
    train
    evaluate
    ;;
  all)
    prepare
    smoke
    train
    evaluate
    ;;
  *)
    echo "usage: $0 {prepare|smoke|run|all}" >&2
    exit 2
    ;;
esac
