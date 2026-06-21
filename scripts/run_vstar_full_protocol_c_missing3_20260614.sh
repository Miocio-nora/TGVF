#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src:$(pwd)"
export TGVF_QWEN3_CAPTURE_GENERATE="${TGVF_QWEN3_CAPTURE_GENERATE:-1}"

MODEL_ID="${MODEL_ID:-/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/nvmesv/dredvpn009/datasets/benchmarks}"
RUN_ROOT="${RUN_ROOT:-outputs/vstar_full_protocol_c_missing3_20260614_answer_only}"
GPUS_CSV="${GPUS_CSV:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
NUM_SHARDS="${NUM_SHARDS:-4}"
SCORING_BACKEND="${SCORING_BACKEND:-auto}"
OFFICIAL_LLM_MODE="${OFFICIAL_LLM_MODE:-disabled}"
ATTN="${ATTN:-sdpa}"
MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION:-512}"
ACTION_MAX_TOKENS="${ACTION_MAX_TOKENS:-128}"
ANSWER_MAX_TOKENS="${ANSWER_MAX_TOKENS:-256}"
POST_TGVF_CONTINUATION="${POST_TGVF_CONTINUATION:-answer_only}"
TIER="${TIER:-full}"
LIMIT="${LIMIT:-}"
LOG_EVERY="${LOG_EVERY:-25}"

if [[ "${#GPUS[@]}" -ne "$NUM_SHARDS" ]]; then
  echo "GPUS_CSV must contain NUM_SHARDS entries; got GPUS_CSV=$GPUS_CSV NUM_SHARDS=$NUM_SHARDS" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs"

run_variant() {
  local name="$1"
  local ckpt="$2"
  local processor="$3"
  local protocol="$4"
  local toolobs_stop="$5"
  local out="$RUN_ROOT/$name"
  mkdir -p "$out"
  echo "[$(date -Is)] start variant=$name protocol=$protocol stop=$toolobs_stop tier=$TIER limit=${LIMIT:-none} continuation=$POST_TGVF_CONTINUATION" | tee -a "$RUN_ROOT/logs/run.log"
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPUS[$shard]}"
    local shard_out="$out/shard_$shard"
    mkdir -p "$shard_out"
    (
      export TGVF_TOOLOBS_ACTION_STOP="$toolobs_stop"
      cmd=(
        python eval/eval_v3_mmmu_force.py
        --benchmark vstar_bench
        --benchmark-root "$BENCHMARK_ROOT"
        --scoring-backend "$SCORING_BACKEND"
        --official-llm-mode "$OFFICIAL_LLM_MODE"
        --output-dir "$shard_out"
        --tier "$TIER"
        --num-shards "$NUM_SHARDS"
        --shard-index "$shard"
        --stage2-checkpoint "$ckpt"
        --model-id "$MODEL_ID"
        --processor-id "$processor"
        --device cuda:0
        --device-map cuda:0
        --attn-implementation "$ATTN"
        --tgvf-protocol "$protocol"
        --max-image-resolution "$MAX_IMAGE_RESOLUTION"
        --max-action-tokens "$ACTION_MAX_TOKENS"
        --max-answer-tokens "$ANSWER_MAX_TOKENS"
        --d-conditions correct_D
        --eval-mode both
        --post-tgvf-continuation "$POST_TGVF_CONTINUATION"
        --log-every "$LOG_EVERY"
      )
      if [[ -n "$LIMIT" ]]; then
        cmd+=(--limit "$LIMIT")
      fi
      CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$shard_out/run.log" 2>&1
    ) &
  done
  wait
  python scripts/merge_tgvf_v3_external_benchmark_shards.py \
    --mode-dir "$out" \
    --row-file benchmark_force_rows.jsonl \
    --summary-name merged_summary.json \
    > "$out/merge.log" 2>&1
  echo "[$(date -Is)] done variant=$name" | tee -a "$RUN_ROOT/logs/run.log"
}

run_variant \
  new_c_v4data_from_v4stage1 \
  outputs/tgvf_v3_protocol_c/protocol_c_stage2_v4data_from_v4stage1_bidirectional_4gpu_bs16_accum2_focus80_1200step/checkpoint_step_1200.pt \
  outputs/tgvf_v3_protocol_c/protocol_c_stage2_v4data_from_v4stage1_bidirectional_4gpu_bs16_accum2_focus80_1200step/processor_step_1200 \
  protocol_c_thinking_special \
  focus_end

run_variant \
  focus_imend_from_toolobs_stage1 \
  outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_multifocus_focus_imend_from_toolobs_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step/checkpoint_step_1200.pt \
  outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_multifocus_focus_imend_from_toolobs_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step/processor_step_1200 \
  protocol_c_tool_observation \
  im_end

run_variant \
  focus_imend_from_focus_imend_stage1 \
  outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_multifocus_focus_imend_from_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260611_021244/checkpoint_step_1200.pt \
  outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_multifocus_focus_imend_from_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260611_021244/processor_step_1200 \
  protocol_c_tool_observation \
  im_end

python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/vstar_full_protocol_c_missing3_20260614_answer_only")
rows = []
for variant_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "logs"):
    summary_path = variant_dir / "merged_summary.json"
    if not summary_path.exists():
        rows.append({"variant": variant_dir.name, "status": "missing"})
        continue
    summary = json.loads(summary_path.read_text())
    by = summary.get("by_method", {})
    free_policy = summary.get("free_policy_correct_D") or {}
    rows.append({
        "variant": variant_dir.name,
        "n_total_rows": summary.get("n_total_rows"),
        "direct_n": (by.get("direct_qwen3") or {}).get("n"),
        "direct_acc": (by.get("direct_qwen3") or {}).get("accuracy"),
        "force_n": (by.get("force_correct_D") or {}).get("n"),
        "force_acc": (by.get("force_correct_D") or {}).get("accuracy"),
        "force_parse": (by.get("force_correct_D") or {}).get("answer_parse_rate"),
        "force_focus_valid": (by.get("force_correct_D") or {}).get("focus_valid_rate"),
        "force_append": (by.get("force_correct_D") or {}).get("append_success_rate"),
        "free_policy_n": free_policy.get("n"),
        "free_policy_acc": free_policy.get("accuracy"),
        "free_trigger": free_policy.get("trigger_rate"),
        "free_correct_n": (by.get("free_correct_D") or {}).get("n"),
        "free_correct_acc": (by.get("free_correct_D") or {}).get("accuracy"),
        "free_direct_or_miss_n": (by.get("free_direct_or_miss") or {}).get("n"),
        "free_direct_or_miss_acc": (by.get("free_direct_or_miss") or {}).get("accuracy"),
        "second_full_forward_used_any": summary.get("second_full_forward_used_any"),
    })
(root / "summary_table.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(rows, indent=2, ensure_ascii=False))
PY

echo "[$(date -Is)] all done" | tee -a "$RUN_ROOT/logs/run.log"
