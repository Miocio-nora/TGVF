#!/usr/bin/env bash
set -euo pipefail

cd /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
export PYTHONPATH=revisit_vlm_clean/src:src

SUITE_ID=stage2_r16_disentanglement_20260712_121814
SUITE_ROOT=outputs/clean_ablation/$SUITE_ID
TRAIN_FILE=data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl
VAL_FILE=data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl
INTERNAL_FILE=data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl
STAGE1=outputs/clean_training/qwen3_stage1_ddeepstack_norm01_wandb_4gpu_20260702_180906/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt
MODEL=/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking
MANIFEST=revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json
MANIFEST_HASH=a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579
BENCHMARK_ROOT=/home/dredvpn009/Flash_Storage/datasets/benchmarks
EXPERIMENTS=(r16_a_rankonly r16_b_qvo r16_c_rowonly r16_d_evidence02)

declare -A TARGETS=(
  [r16_a_rankonly]=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  [r16_b_qvo]=q_proj,v_proj,o_proj
  [r16_c_rowonly]=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  [r16_d_evidence02]=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
)
declare -A TOKEN_MODES=(
  [r16_a_rankonly]=full_modules
  [r16_b_qvo]=full_modules
  [r16_c_rowonly]=row_only
  [r16_d_evidence02]=full_modules
)
declare -A EVIDENCE_WEIGHTS=(
  [r16_a_rankonly]=1.0
  [r16_b_qvo]=1.0
  [r16_c_rowonly]=1.0
  [r16_d_evidence02]=0.2
)

mkdir -p "$SUITE_ROOT/logs"

train_root() {
  printf 'outputs/clean_training/qwen3_stage2_ddeepstack_%s_8gpu_20260712_121814' "$1"
}

smoke_root() {
  printf 'outputs/clean_training/qwen3_stage2_ddeepstack_%s_8gpu_20260712_121814_smoke' "$1"
}

write_plan() {
  local exp=$1
  local output_dir=$2
  local max_steps=$3
  local save_every=$4
  local eval_every=$5
  local wandb_mode=$6
  local suffix=$7

  python -m revisit_vlm_clean.cli.train_stage2 \
    --run-id "clean_qwen3_stage2_ddeepstack_${exp}_${suffix}_20260712_121814" \
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
    --lora-target-modules "${TARGETS[$exp]}" \
    --protocol-token-training-mode "${TOKEN_MODES[$exp]}" \
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
    --loss-evidence "${EVIDENCE_WEIGHTS[$exp]}" \
    --loss-value-span 1.0 \
    --loss-answer 1.0 \
    --loss-no-focus-evidence-state 0.2 \
    --loss-no-focus-answer 1.0 \
    --loss-visual-token-manifold 0.0 \
    --wandb-project tgvf-clean-qwen3-deepstack \
    --wandb-mode "$wandb_mode" \
    --write-plan
}

prepare_exp() {
  local exp=$1
  local main_dir
  local smoke_dir
  main_dir="$(train_root "$exp")/stage2_micro4"
  smoke_dir="$(smoke_root "$exp")/stage2_micro4"

  write_plan "$exp" "$smoke_dir" 1 1 1 disabled smoke
  python -m revisit_vlm_clean.training.stage2_executor \
    --plan "$smoke_dir/training_plan.json" --prepare-execution
  write_plan "$exp" "$main_dir" 1200 300 300 online main
  python -m revisit_vlm_clean.training.stage2_executor \
    --plan "$main_dir/training_plan.json" --prepare-execution
}

smoke_exp() {
  local exp=$1
  local root
  local checkpoint
  root="$(smoke_root "$exp")/stage2_micro4"
  checkpoint="$root/clean_training_execution/checkpoint_step_1.pt"
  if [[ -f "$checkpoint" ]]; then
    echo "[$(date -Is)] smoke skip $exp checkpoint=$checkpoint"
    return
  fi
  echo "[$(date -Is)] smoke start $exp"
  MASTER_PORT=29621 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --nproc-per-node 8 \
    -m revisit_vlm_clean.training.stage2_executor \
    --plan "$root/training_plan.json" --launch-training \
    > "$SUITE_ROOT/logs/${exp}_smoke.log" 2>&1
  echo "[$(date -Is)] smoke done $exp checkpoint=$checkpoint"
}

train_exp() {
  local exp=$1
  local root
  local checkpoint
  root="$(train_root "$exp")/stage2_micro4"
  checkpoint="$root/clean_training_execution/checkpoint_step_1200.pt"
  if [[ -f "$checkpoint" ]]; then
    echo "[$(date -Is)] train skip $exp checkpoint=$checkpoint"
    return
  fi
  echo "[$(date -Is)] train start $exp"
  MASTER_PORT=29622 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --nproc-per-node 8 \
    -m revisit_vlm_clean.training.stage2_executor \
    --plan "$root/training_plan.json" --launch-training \
    > "$SUITE_ROOT/logs/${exp}_train.log" 2>&1
  echo "[$(date -Is)] train done $exp checkpoint=$checkpoint"
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

eval_exp() {
  local exp=$1
  local checkpoint
  local internal_dir
  local smoke_dir
  local free_dir
  local soft_dir
  local internal_pid
  checkpoint="$(train_root "$exp")/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt"
  internal_dir="$(train_root "$exp")/stage2_micro4/internal_diagnostics_step1200_20260712_121814"
  smoke_dir="outputs/clean_benchmarks/qwen3_stage2_${exp}_coredev2511_smoke_free_20260712_121814"
  free_dir="outputs/clean_benchmarks/qwen3_stage2_${exp}_coredev2511_free_20260712_121814/tgvf_free"
  soft_dir="outputs/clean_benchmarks/qwen3_stage2_${exp}_coredev2511_softforce_20260712_121814/tgvf_softforce"

  if [[ ! -f "$smoke_dir/summary.json" ]]; then
    echo "[$(date -Is)] eval smoke start $exp"
    CUDA_VISIBLE_DEVICES=0 benchmark_args \
      "$checkpoint" tgvf_free "$smoke_dir" \
      "clean_qwen3_stage2_${exp}_coredev2511_smoke_free_20260712_121814" \
      --gpus 0 --batch-size 1 --max-samples 4 --progress-every 1 \
      > "$SUITE_ROOT/logs/${exp}_eval_smoke.log" 2>&1
    echo "[$(date -Is)] eval smoke done $exp"
  fi

  if [[ ! -f "$internal_dir/clean_stage_diagnostic_status.json" ]]; then
    echo "[$(date -Is)] internal start $exp"
    CUDA_VISIBLE_DEVICES=0 python -m revisit_vlm_clean.cli.stage_diagnostics \
      --run-id "clean_qwen3_stage2_${exp}_internal_20260712_121814" \
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
      --execute > "$SUITE_ROOT/logs/${exp}_internal.log" 2>&1 &
    internal_pid=$!
  else
    internal_pid=
    echo "[$(date -Is)] internal skip $exp"
  fi

  if [[ ! -f "$free_dir/summary.json" ]]; then
    echo "[$(date -Is)] free start $exp"
    benchmark_args "$checkpoint" tgvf_free "$free_dir" \
      "clean_qwen3_stage2_${exp}_coredev2511_free_20260712_121814" \
      --gpus 1,2,3,4,5,6,7 --batch-size 1 --progress-every 25 \
      > "$SUITE_ROOT/logs/${exp}_free.log" 2>&1
    echo "[$(date -Is)] free done $exp"
  fi

  if [[ -n "$internal_pid" ]]; then
    wait "$internal_pid"
    echo "[$(date -Is)] internal done $exp"
  fi

  if [[ ! -f "$soft_dir/summary.json" ]]; then
    echo "[$(date -Is)] softforce start $exp"
    benchmark_args "$checkpoint" tgvf_softforce "$soft_dir" \
      "clean_qwen3_stage2_${exp}_coredev2511_softforce_20260712_121814" \
      --softforce-prompt-text "Use focus tool." \
      --gpus 0,1,2,3,4,5,6,7 --batch-size 1 --progress-every 25 \
      > "$SUITE_ROOT/logs/${exp}_softforce.log" 2>&1
    echo "[$(date -Is)] softforce done $exp"
  fi
}

prepare_all() {
  local exp
  for exp in "${EXPERIMENTS[@]}"; do
    echo "[$(date -Is)] prepare start $exp"
    prepare_exp "$exp"
    echo "[$(date -Is)] prepare done $exp"
  done
}

smoke_all() {
  local exp
  for exp in "${EXPERIMENTS[@]}"; do
    smoke_exp "$exp"
  done
}

run_all() {
  local exp
  for exp in "${EXPERIMENTS[@]}"; do
    train_exp "$exp"
    eval_exp "$exp"
  done
}

case "${1:-}" in
  prepare)
    prepare_all
    ;;
  smoke)
    smoke_all
    ;;
  run)
    run_all
    ;;
  all)
    prepare_all
    smoke_all
    run_all
    ;;
  *)
    echo "usage: $0 {prepare|smoke|run|all}" >&2
    exit 2
    ;;
esac
