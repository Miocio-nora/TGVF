#!/usr/bin/env bash
set -euo pipefail

cd /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/src:$(pwd)/third_party/VLMEvalKit"
export TGVF_TOOLOBS_ACTION_STOP=im_end
export TGVF_BLOCK_FOCUS_IN_CONTINUATION="${TGVF_BLOCK_FOCUS_IN_CONTINUATION:-1}"
export TGVF_QWEN3_CAPTURE_GENERATE="${TGVF_QWEN3_CAPTURE_GENERATE:-1}"
export TGVF_STOP_REPETITIVE_CONTINUATION="${TGVF_STOP_REPETITIVE_CONTINUATION:-1}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-outputs/no_kv_benchmarks/qwen3_other_benchmarks_19_vs_075_nokv_${STAMP}}"
MODEL_ID="${MODEL_ID:-/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/nvmesv/dredvpn009/datasets/benchmarks}"
BENCH_GPUS="${BENCH_GPUS:-4,5,6,7}"
NUM_SHARDS="${NUM_SHARDS:-4}"
BENCHMARKS="${BENCHMARKS:-vstar_bench hr_bench_4k ocrbench_v2}"

SCORING_BACKEND="${SCORING_BACKEND:-auto}"
OFFICIAL_LLM_MODE="${OFFICIAL_LLM_MODE:-disabled}"
ATTN="${ATTN:-sdpa}"
MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION:-512}"
ACTION_MAX_TOKENS="${ACTION_MAX_TOKENS:-128}"
ANSWER_MAX_TOKENS="${ANSWER_MAX_TOKENS:-512}"
POST_TGVF_CONTINUATION="${POST_TGVF_CONTINUATION:-natural_continue}"
POST_TGVF_FORWARD_MODE="${POST_TGVF_FORWARD_MODE:-no_kv_full_sequence}"
TGVF_PROTOCOL="${TGVF_PROTOCOL:-protocol_c_tool_observation}"
SOFTFORCE_SUFFIX="${SOFTFORCE_SUFFIX:-use focus tool}"

CKPT_19="${CKPT_19:-outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt}"
PROCESSOR_19="${PROCESSOR_19:-outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200}"
CKPT_075="${CKPT_075:-outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_oldmaskprob075_throughanswer_from_20260619_stage1_train47_eval03_4gpu_bs16_accum2_focus80_value1_1200step_20260623_005426/checkpoint_step_1200.pt}"
PROCESSOR_075="${PROCESSOR_075:-outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_oldmaskprob075_throughanswer_from_20260619_stage1_train47_eval03_4gpu_bs16_accum2_focus80_value1_1200step_20260623_005426/processor_step_1200}"

IFS=',' read -r -a GPUS <<< "${BENCH_GPUS}"
mkdir -p "${RUN_ROOT}/logs"

cat > "${RUN_ROOT}/run_config.txt" <<CFG
stamp=${STAMP}
run_root=${RUN_ROOT}
bench_gpus=${BENCH_GPUS}
num_shards=${NUM_SHARDS}
model_id=${MODEL_ID}
benchmark_root=${BENCHMARK_ROOT}
benchmarks=${BENCHMARKS}
ckpt_19=${CKPT_19}
processor_19=${PROCESSOR_19}
ckpt_075=${CKPT_075}
processor_075=${PROCESSOR_075}
max_image_resolution=${MAX_IMAGE_RESOLUTION}
max_action_tokens=${ACTION_MAX_TOKENS}
max_answer_tokens=${ANSWER_MAX_TOKENS}
post_tgvf_continuation=${POST_TGVF_CONTINUATION}
post_tgvf_forward_mode=${POST_TGVF_FORWARD_MODE}
tgvf_protocol=${TGVF_PROTOCOL}
softforce_suffix=${SOFTFORCE_SUFFIX}
scoring_backend=${SCORING_BACKEND}
official_llm_mode=${OFFICIAL_LLM_MODE}
CFG

run_one() {
  local ckpt_label="$1"
  local ckpt="$2"
  local processor="$3"
  local bench="$4"
  local mode="$5"
  local suffix="$6"
  local out="${RUN_ROOT}/${ckpt_label}/${bench}/${mode}"
  mkdir -p "${out}"
  echo "[$(date -Is)] start ${ckpt_label} ${bench} ${mode} forward=${POST_TGVF_FORWARD_MODE}" | tee -a "${RUN_ROOT}/logs/run.log"
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPUS[$shard]}"
    local shard_out="${out}/shard_${shard}"
    mkdir -p "${shard_out}"
    CUDA_VISIBLE_DEVICES="${gpu}" python eval/eval_v3_mmmu_force.py \
      --benchmark "${bench}" \
      --benchmark-root "${BENCHMARK_ROOT}" \
      --scoring-backend "${SCORING_BACKEND}" \
      --official-llm-mode "${OFFICIAL_LLM_MODE}" \
      --output-dir "${shard_out}" \
      --tier full \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard}" \
      --stage2-checkpoint "${ckpt}" \
      --model-id "${MODEL_ID}" \
      --processor-id "${processor}" \
      --device cuda:0 \
      --device-map cuda:0 \
      --attn-implementation "${ATTN}" \
      --tgvf-protocol "${TGVF_PROTOCOL}" \
      --max-image-resolution "${MAX_IMAGE_RESOLUTION}" \
      --max-action-tokens "${ACTION_MAX_TOKENS}" \
      --max-answer-tokens "${ANSWER_MAX_TOKENS}" \
      --d-conditions correct_D \
      --eval-mode free \
      --no-include-stage2-direct \
      --post-tgvf-continuation "${POST_TGVF_CONTINUATION}" \
      --post-tgvf-forward-mode "${POST_TGVF_FORWARD_MODE}" \
      --question-suffix "${suffix}" \
      --log-every 25 \
      > "${shard_out}/run.log" 2>&1 &
  done
  wait
  python scripts/merge_tgvf_v3_external_benchmark_shards.py \
    --mode-dir "${out}" \
    --row-file benchmark_force_rows.jsonl \
    --summary-name merged_summary.json \
    > "${out}/merge.log" 2>&1
  echo "[$(date -Is)] done ${ckpt_label} ${bench} ${mode}" | tee -a "${RUN_ROOT}/logs/run.log"
}

for bench in ${BENCHMARKS}; do
  run_one "ckpt19" "${CKPT_19}" "${PROCESSOR_19}" "${bench}" "free" ""
  run_one "ckpt19" "${CKPT_19}" "${PROCESSOR_19}" "${bench}" "free_softforce_focus" "${SOFTFORCE_SUFFIX}"
  run_one "ckpt075" "${CKPT_075}" "${PROCESSOR_075}" "${bench}" "free" ""
  run_one "ckpt075" "${CKPT_075}" "${PROCESSOR_075}" "${bench}" "free_softforce_focus" "${SOFTFORCE_SUFFIX}"
done

python - <<PY
import json
from pathlib import Path

root = Path("${RUN_ROOT}")
rows = []
for ckpt_label in ("ckpt19", "ckpt075"):
    for bench_dir in sorted((root / ckpt_label).iterdir() if (root / ckpt_label).exists() else []):
        if not bench_dir.is_dir():
            continue
        for mode in ("free", "free_softforce_focus"):
            summary_path = bench_dir / mode / "merged_summary.json"
            if not summary_path.exists():
                rows.append({"ckpt": ckpt_label, "benchmark": bench_dir.name, "mode": mode, "status": "missing"})
                continue
            summary = json.loads(summary_path.read_text())
            method = summary.get("free_policy_correct_D") or {}
            rows.append({
                "ckpt": ckpt_label,
                "benchmark": bench_dir.name,
                "mode": mode,
                "n": method.get("n"),
                "accuracy": method.get("accuracy"),
                "answer_parse_rate": method.get("answer_parse_rate"),
                "trigger_rate": method.get("trigger_rate"),
                "post_tgvf_forward_mode": "${POST_TGVF_FORWARD_MODE}",
            })
(root / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(rows, indent=2, ensure_ascii=False))
PY
