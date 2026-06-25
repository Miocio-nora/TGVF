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
MODEL_ID="${MODEL_ID:-/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/nvmesv/dredvpn009/datasets/benchmarks}"
SOURCE_BENCHMARK_ROOT="${SOURCE_BENCHMARK_ROOT:-${BENCHMARK_ROOT}}"
STAGE2_CKPT="${STAGE2_CKPT:-outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_oldmaskprob075_throughanswer_from_20260619_stage1_train47_eval03_4gpu_bs16_accum2_focus80_value1_1200step_20260623_005426/checkpoint_step_1200.pt}"
STAGE2_PROCESSOR="${STAGE2_PROCESSOR:-outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_oldmaskprob075_throughanswer_from_20260619_stage1_train47_eval03_4gpu_bs16_accum2_focus80_value1_1200step_20260623_005426/processor_step_1200}"
RUN_ROOT="${RUN_ROOT:-outputs/no_kv_benchmarks/qwen3_blink120_oldmask075_correct_stage1_nokv_${STAMP}}"
BENCH_GPUS="${BENCH_GPUS:-4,5,6,7}"
NUM_SHARDS="${NUM_SHARDS:-4}"
POPULATION_ID="${POPULATION_ID:-blink_counting_val_120}"

SCORING_BACKEND="${SCORING_BACKEND:-auto}"
OFFICIAL_LLM_MODE="${OFFICIAL_LLM_MODE:-disabled}"
ATTN="${ATTN:-sdpa}"
MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION:-512}"
ACTION_MAX_TOKENS="${ACTION_MAX_TOKENS:-128}"
ANSWER_MAX_TOKENS="${ANSWER_MAX_TOKENS:-512}"
POST_TGVF_CONTINUATION="${POST_TGVF_CONTINUATION:-natural_continue}"
POST_TGVF_FORWARD_MODE="${POST_TGVF_FORWARD_MODE:-no_kv_full_sequence}"
TGVF_PROTOCOL="${TGVF_PROTOCOL:-protocol_c_tool_observation}"
QUESTION_SUFFIX_SOFTFORCE="${QUESTION_SUFFIX_SOFTFORCE:-use focus tool}"

IFS=',' read -r -a GPUS <<< "${BENCH_GPUS}"
mkdir -p "${RUN_ROOT}/logs"
COUNTING_ROOT="${RUN_ROOT}/benchmark_root_counting_only"
mkdir -p "${COUNTING_ROOT}/blink/snapshot"
ln -sfn "${SOURCE_BENCHMARK_ROOT}/blink/snapshot/Counting" "${COUNTING_ROOT}/blink/snapshot/Counting"
ln -sfn "${SOURCE_BENCHMARK_ROOT}/blink/official_code" "${COUNTING_ROOT}/blink/official_code"
BENCHMARK_ROOT="${COUNTING_ROOT}"

cat > "${RUN_ROOT}/run_config.txt" <<CFG
stamp=${STAMP}
bench_gpus=${BENCH_GPUS}
num_shards=${NUM_SHARDS}
model_id=${MODEL_ID}
source_benchmark_root=${SOURCE_BENCHMARK_ROOT}
benchmark_root=${BENCHMARK_ROOT}
stage2_ckpt=${STAGE2_CKPT}
stage2_processor=${STAGE2_PROCESSOR}
benchmark=blink
population_id=${POPULATION_ID}
tier=full
limit=None
blink_subset=Counting
max_image_resolution=${MAX_IMAGE_RESOLUTION}
max_action_tokens=${ACTION_MAX_TOKENS}
max_answer_tokens=${ANSWER_MAX_TOKENS}
post_tgvf_continuation=${POST_TGVF_CONTINUATION}
post_tgvf_forward_mode=${POST_TGVF_FORWARD_MODE}
tgvf_protocol=${TGVF_PROTOCOL}
question_suffix_softforce=${QUESTION_SUFFIX_SOFTFORCE}
CFG

run_mode() {
  local mode="$1"
  local suffix="$2"
  local out="${RUN_ROOT}/blink/${mode}"
  mkdir -p "${out}"
  echo "[$(date -Is)] start blink ${mode} population=${POPULATION_ID} limit=None forward=${POST_TGVF_FORWARD_MODE}" | tee -a "${RUN_ROOT}/logs/run.log"
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPUS[$shard]}"
    local shard_out="${out}/shard_${shard}"
    mkdir -p "${shard_out}"
    CUDA_VISIBLE_DEVICES="${gpu}" python eval/eval_v3_mmmu_force.py \
      --benchmark blink \
      --benchmark-root "${BENCHMARK_ROOT}" \
      --scoring-backend "${SCORING_BACKEND}" \
      --official-llm-mode "${OFFICIAL_LLM_MODE}" \
      --output-dir "${shard_out}" \
      --tier full \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard}" \
      --stage2-checkpoint "${STAGE2_CKPT}" \
      --model-id "${MODEL_ID}" \
      --processor-id "${STAGE2_PROCESSOR}" \
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
      --log-every 10 \
      > "${shard_out}/run.log" 2>&1 &
  done
  wait
  python scripts/merge_tgvf_v3_external_benchmark_shards.py \
    --mode-dir "${out}" \
    --row-file benchmark_force_rows.jsonl \
    --summary-name merged_summary.json \
    > "${out}/merge.log" 2>&1
  echo "[$(date -Is)] done blink ${mode}" | tee -a "${RUN_ROOT}/logs/run.log"
}

run_mode free ""
run_mode free_softforce_focus "${QUESTION_SUFFIX_SOFTFORCE}"

python - <<PY
import json
from pathlib import Path
root = Path("${RUN_ROOT}")
rows = []
for mode in ("free", "free_softforce_focus"):
    summary = json.loads((root / "blink" / mode / "merged_summary.json").read_text())
    method = summary.get("free_policy_correct_D") or {}
    rows.append({
        "benchmark": "blink",
        "mode": mode,
        "n": method.get("n"),
        "accuracy": method.get("accuracy"),
        "answer_parse_rate": method.get("answer_parse_rate"),
        "trigger_rate": method.get("trigger_rate"),
        "post_tgvf_forward_mode": "${POST_TGVF_FORWARD_MODE}",
    })
(root / f"summary_blink120_${POST_TGVF_FORWARD_MODE}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(rows, indent=2, ensure_ascii=False))
PY
