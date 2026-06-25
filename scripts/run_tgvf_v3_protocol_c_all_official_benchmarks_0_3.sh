#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src:$(pwd)"
export TGVF_QWEN3_CAPTURE_GENERATE="${TGVF_QWEN3_CAPTURE_GENERATE:-1}"

CKPT="${CKPT:-outputs/tgvf_v3_protocol_c/tgvf_v3_protocol_c_stage2_8b_4gpu_bs16_accum2_focus80_stage1c_tokenrows_1200step_rerun_after_reboot/checkpoint_step_1200.pt}"
MODEL_ID="${MODEL_ID:-/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking}"
PROCESSOR_ID="${PROCESSOR_ID:-}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/nvmesv/dredvpn009/datasets/benchmarks}"
RUN_ROOT="${RUN_ROOT:-outputs/tgvf_v3_protocol_c/tgvf_v3_protocol_c_stage2_8b_4gpu_bs16_accum2_focus80_stage1c_tokenrows_1200step_rerun_after_reboot/official_benchmarks_all_except_ovo_20260604}"
BENCHMARKS=(vstar_bench hr_bench_4k ocrbench_v2 blink mmmu_pro mathvista mathverse)
GPUS=(0 1 2 3)
NUM_SHARDS=4
SCORING_BACKEND="${SCORING_BACKEND:-auto}"
OFFICIAL_LLM_MODE="${OFFICIAL_LLM_MODE:-disabled}"
ATTN="${ATTN:-sdpa}"
MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION:-512}"
ORIGINAL_MAX_TOKENS="${ORIGINAL_MAX_TOKENS:-512}"
ACTION_MAX_TOKENS="${ACTION_MAX_TOKENS:-128}"
ANSWER_MAX_TOKENS="${ANSWER_MAX_TOKENS:-256}"
POST_TGVF_CONTINUATION="${POST_TGVF_CONTINUATION:-evidence_then_answer}"
POST_TGVF_FORWARD_MODE="${POST_TGVF_FORWARD_MODE:-no_kv_full_sequence}"
TGVF_PROTOCOL="${TGVF_PROTOCOL:-protocol_c_thinking_special}"

mkdir -p "$RUN_ROOT/logs"

run_shards() {
  local bench="$1"
  local mode="$2"
  local out="$RUN_ROOT/$bench/$mode"
  mkdir -p "$out"
  echo "[$(date -Is)] start $bench $mode post_tgvf_forward_mode=$POST_TGVF_FORWARD_MODE" | tee -a "$RUN_ROOT/logs/run.log"
  for shard in 0 1 2 3; do
    local gpu="${GPUS[$shard]}"
    local shard_out="$out/shard_$shard"
    mkdir -p "$shard_out"
    if [[ "$mode" == "original" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" python eval/eval_qwen3_mmmu_base_direct.py \
        --benchmark "$bench" \
        --benchmark-root "$BENCHMARK_ROOT" \
        --scoring-backend "$SCORING_BACKEND" \
        --official-llm-mode "$OFFICIAL_LLM_MODE" \
        --output-dir "$shard_out" \
        --tier full \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$shard" \
        --model-id "$MODEL_ID" \
        --device cuda:0 \
        --device-map cuda:0 \
        --attn-implementation "$ATTN" \
        --max-image-resolution "$MAX_IMAGE_RESOLUTION" \
        --max-answer-tokens "$ORIGINAL_MAX_TOKENS" \
        --enable-thinking \
        --log-every 25 \
        > "$shard_out/run.log" 2>&1 &
    else
      local eval_mode="$mode"
      local processor_args=()
      if [[ -n "$PROCESSOR_ID" ]]; then
        processor_args=(--processor-id "$PROCESSOR_ID")
      fi
      CUDA_VISIBLE_DEVICES="$gpu" python eval/eval_v3_mmmu_force.py \
        --benchmark "$bench" \
        --benchmark-root "$BENCHMARK_ROOT" \
        --scoring-backend "$SCORING_BACKEND" \
        --official-llm-mode "$OFFICIAL_LLM_MODE" \
        --output-dir "$shard_out" \
        --tier full \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$shard" \
        --stage2-checkpoint "$CKPT" \
        --model-id "$MODEL_ID" \
        "${processor_args[@]}" \
        --device cuda:0 \
        --device-map cuda:0 \
        --attn-implementation "$ATTN" \
        --tgvf-protocol "$TGVF_PROTOCOL" \
        --max-image-resolution "$MAX_IMAGE_RESOLUTION" \
        --max-action-tokens "$ACTION_MAX_TOKENS" \
        --max-answer-tokens "$ANSWER_MAX_TOKENS" \
        --d-conditions correct_D \
        --eval-mode "$eval_mode" \
        --no-include-stage2-direct \
        --post-tgvf-continuation "$POST_TGVF_CONTINUATION" \
        --post-tgvf-forward-mode "$POST_TGVF_FORWARD_MODE" \
        --log-every 25 \
        > "$shard_out/run.log" 2>&1 &
    fi
  done
  wait
  if [[ "$mode" == "original" ]]; then
    python scripts/merge_tgvf_v3_external_benchmark_shards.py \
      --mode-dir "$out" \
      --row-file benchmark_base_direct_rows.jsonl \
      --summary-name merged_summary.json \
      > "$out/merge.log" 2>&1
  else
    python scripts/merge_tgvf_v3_external_benchmark_shards.py \
      --mode-dir "$out" \
      --row-file benchmark_force_rows.jsonl \
      --summary-name merged_summary.json \
      > "$out/merge.log" 2>&1
  fi
  echo "[$(date -Is)] done $bench $mode" | tee -a "$RUN_ROOT/logs/run.log"
}

for bench in "${BENCHMARKS[@]}"; do
  run_shards "$bench" original
  run_shards "$bench" force
  run_shards "$bench" free
done

python - <<PY
import json
from pathlib import Path
root = Path("${RUN_ROOT}")
rows = []
for bench_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "logs"):
    for mode in ("original", "force", "free"):
        summary_path = bench_dir / mode / "merged_summary.json"
        if not summary_path.exists():
            rows.append({"benchmark": bench_dir.name, "mode": mode, "status": "missing"})
            continue
        summary = json.loads(summary_path.read_text())
        if mode == "original":
            method = summary.get("by_method", {}).get("base_direct_qwen3", {})
        elif mode == "force":
            method = summary.get("by_method", {}).get("force_correct_D", {})
        else:
            method = summary.get("free_policy_correct_D") or {}
        rows.append({
            "benchmark": bench_dir.name,
            "mode": mode,
            "n": method.get("n"),
            "accuracy": method.get("accuracy"),
            "answer_parse_rate": method.get("answer_parse_rate"),
            "trigger_rate": method.get("trigger_rate"),
            "focus_valid_rate": method.get("focus_valid_rate"),
        })
(root / "all_benchmarks_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(rows, indent=2, ensure_ascii=False))
PY
