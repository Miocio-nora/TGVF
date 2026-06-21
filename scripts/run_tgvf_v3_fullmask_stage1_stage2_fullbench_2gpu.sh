#!/usr/bin/env bash
set -euo pipefail

cd /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/src:$(pwd)/third_party/VLMEvalKit"
export TGVF_TOOLOBS_ACTION_STOP=im_end
export TGVF_BLOCK_FOCUS_IN_CONTINUATION="${TGVF_BLOCK_FOCUS_IN_CONTINUATION:-1}"
export TGVF_QWEN3_CAPTURE_GENERATE="${TGVF_QWEN3_CAPTURE_GENERATE:-1}"
export TGVF_STOP_REPETITIVE_CONTINUATION="${TGVF_STOP_REPETITIVE_CONTINUATION:-1}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MODEL_ID="${MODEL_ID:-/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/nvmesv/dredvpn009/datasets/benchmarks}"
NUM_GPUS="${NUM_GPUS:-2}"
PROTOCOL_TOKEN_ROW_MODE="${PROTOCOL_TOKEN_ROW_MODE:-full_mask}"

STAGE1_RUN_ID="${STAGE1_RUN_ID:-protocol_c_toolobs_stage1_v4data_clean_imend_fullmask_bidirectional_2gpu_bs4_accum4_gbs32_2000step_${STAMP}}"
STAGE1_ROOT="${STAGE1_ROOT:-outputs/tgvf_v3_protocol_c_stage1_8b/${STAGE1_RUN_ID}}"
STAGE1_CKPT="${STAGE1_CKPT:-${STAGE1_ROOT}/train/checkpoint_step_2000.pt}"
STAGE1_PROCESSOR="${STAGE1_PROCESSOR:-${STAGE1_ROOT}/train/processor_step_2000}"

STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_fullmask_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_${STAMP}}"
STAGE2_OUT="${STAGE2_OUT:-outputs/tgvf_v3_protocol_c/${STAGE2_RUN_NAME}}"
STAGE2_CKPT="${STAGE2_CKPT:-${STAGE2_OUT}/checkpoint_step_1200.pt}"
STAGE2_PROCESSOR="${STAGE2_PROCESSOR:-${STAGE2_OUT}/processor_step_1200}"

BENCH_RUN_ROOT="${BENCH_RUN_ROOT:-outputs/full_benchmarks_fullmask_2gpu_${STAMP}}"
RUN_CHAIN="${RUN_CHAIN:-1}"
RUN_VSTAR="${RUN_VSTAR:-1}"
RUN_EXTERNAL_BENCHMARKS="${RUN_EXTERNAL_BENCHMARKS:-1}"
RUN_FREE="${RUN_FREE:-1}"
RUN_SOFTFORCE="${RUN_SOFTFORCE:-1}"

SCORING_BACKEND="${SCORING_BACKEND:-auto}"
OFFICIAL_LLM_MODE="${OFFICIAL_LLM_MODE:-disabled}"
ATTN="${ATTN:-sdpa}"
MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION:-512}"
ACTION_MAX_TOKENS="${ACTION_MAX_TOKENS:-128}"
ANSWER_MAX_TOKENS="${ANSWER_MAX_TOKENS:-512}"
POST_TGVF_CONTINUATION="${POST_TGVF_CONTINUATION:-natural_continue}"
TGVF_PROTOCOL="${TGVF_PROTOCOL:-protocol_c_tool_observation}"
QUESTION_SUFFIX_SOFTFORCE="${QUESTION_SUFFIX_SOFTFORCE:-use focus tool}"
BENCHMARKS="${BENCHMARKS:-hr_bench_4k ocrbench_v2 blink}"

IFS=',' read -r -a GPUS <<< "${BENCH_GPUS:-${CUDA_VISIBLE_DEVICES}}"
NUM_SHARDS="${NUM_SHARDS:-${#GPUS[@]}}"
if (( NUM_SHARDS < 1 )); then
  echo "NUM_SHARDS must be >= 1" >&2
  exit 1
fi
if (( NUM_SHARDS > ${#GPUS[@]} )); then
  echo "NUM_SHARDS=${NUM_SHARDS} exceeds GPU count ${#GPUS[@]} from ${BENCH_GPUS:-${CUDA_VISIBLE_DEVICES}}" >&2
  exit 1
fi

mkdir -p "${BENCH_RUN_ROOT}/logs"
cat > "${BENCH_RUN_ROOT}/run_config.txt" <<CFG
stamp=${STAMP}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
bench_gpus=${BENCH_GPUS:-${CUDA_VISIBLE_DEVICES}}
num_gpus=${NUM_GPUS}
num_shards=${NUM_SHARDS}
model_id=${MODEL_ID}
benchmark_root=${BENCHMARK_ROOT}
protocol_token_row_mode=${PROTOCOL_TOKEN_ROW_MODE}
stage1_run_id=${STAGE1_RUN_ID}
stage1_ckpt=${STAGE1_CKPT}
stage1_processor=${STAGE1_PROCESSOR}
stage2_run_name=${STAGE2_RUN_NAME}
stage2_ckpt=${STAGE2_CKPT}
stage2_processor=${STAGE2_PROCESSOR}
benchmarks=${BENCHMARKS}
run_vstar=${RUN_VSTAR}
run_external_benchmarks=${RUN_EXTERNAL_BENCHMARKS}
run_free=${RUN_FREE}
run_softforce=${RUN_SOFTFORCE}
max_image_resolution=${MAX_IMAGE_RESOLUTION}
max_action_tokens=${ACTION_MAX_TOKENS}
max_answer_tokens=${ANSWER_MAX_TOKENS}
post_tgvf_continuation=${POST_TGVF_CONTINUATION}
question_suffix_softforce=${QUESTION_SUFFIX_SOFTFORCE}
CFG

log() {
  echo "[$(date -Is)] $*" | tee -a "${BENCH_RUN_ROOT}/logs/run.log"
}

if [[ "${RUN_CHAIN}" == "1" ]]; then
  log "chain start stage1+stage2 row_mode=${PROTOCOL_TOKEN_ROW_MODE}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NUM_GPUS="${NUM_GPUS}" \
  MODEL_ID="${MODEL_ID}" \
  STAMP="${STAMP}" \
  STAGE1_RUN_ID="${STAGE1_RUN_ID}" \
  STAGE1_ROOT="${STAGE1_ROOT}" \
  STAGE1_CKPT="${STAGE1_CKPT}" \
  STAGE1_PROCESSOR="${STAGE1_PROCESSOR}" \
  STAGE2_RUN_NAME="${STAGE2_RUN_NAME}" \
  STAGE2_OUT="${STAGE2_OUT}" \
  PROTOCOL_TOKEN_ROW_MODE="${PROTOCOL_TOKEN_ROW_MODE}" \
  RUN_VSTAR=0 \
  bash scripts/run_tgvf_v3_toolobs_open_answer_stage1_stage2_eval_2gpu.sh \
    2>&1 | tee -a "${BENCH_RUN_ROOT}/logs/chain.log"
  log "chain done"
fi

if [[ ! -f "${STAGE2_CKPT}" ]]; then
  echo "Missing Stage2 checkpoint: ${STAGE2_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${STAGE2_PROCESSOR}" ]]; then
  echo "Missing Stage2 processor: ${STAGE2_PROCESSOR}" >&2
  exit 1
fi

run_vstar_mode() {
  local mode="$1"
  local suffix="$2"
  local out="${BENCH_RUN_ROOT}/vstar_bench/${mode}"
  local trace="${out}/tgvf_traces"
  mkdir -p "${out}" "${trace}"
  log "start vstar_bench ${mode}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  TGVF_STAGE2_CHECKPOINT="${STAGE2_CKPT}" \
  TGVF_PROCESSOR_ID="${STAGE2_PROCESSOR}" \
  TGVF_MAX_IMAGE_RESOLUTION="${MAX_IMAGE_RESOLUTION}" \
  TGVF_MAX_ACTION_TOKENS="${ACTION_MAX_TOKENS}" \
  TGVF_MAX_ANSWER_TOKENS="${ANSWER_MAX_TOKENS}" \
  TGVF_POST_TGVF_CONTINUATION="${POST_TGVF_CONTINUATION}" \
  TGVF_VLMEVAL_TRACE_DIR="${trace}" \
  TGVF_PROMPT_SUFFIX="${suffix}" \
  MASTER_PORT="${MASTER_PORT_VSTAR:-29763}" \
  torchrun --nproc-per-node="${NUM_GPUS}" third_party/VLMEvalKit/run.py \
    --data VStarBench \
    --model TGVF-Qwen3VL-8B-ToolObs-Env-Free-512 \
    --work-dir "${out}" \
    --mode all \
    --judge exact_matching \
    > "${out}/run.log" 2>&1
  log "done vstar_bench ${mode}"
}

run_external_mode() {
  local bench="$1"
  local mode="$2"
  local suffix="$3"
  local out="${BENCH_RUN_ROOT}/${bench}/${mode}"
  mkdir -p "${out}"
  log "start ${bench} ${mode}"
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPUS[$shard]}"
    local shard_out="${out}/shard_${shard}"
    mkdir -p "${shard_out}"
    (
      CUDA_VISIBLE_DEVICES="${gpu}" python eval/eval_v3_mmmu_force.py \
        --benchmark "${bench}" \
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
        --question-suffix "${suffix}" \
        --log-every 25
    ) > "${shard_out}/run.log" 2>&1 &
  done
  wait
  python scripts/merge_tgvf_v3_external_benchmark_shards.py \
    --mode-dir "${out}" \
    --row-file benchmark_force_rows.jsonl \
    --summary-name merged_summary.json \
    > "${out}/merge.log" 2>&1
  log "done ${bench} ${mode}"
}

if [[ "${RUN_VSTAR}" == "1" ]]; then
  if [[ "${RUN_FREE}" == "1" ]]; then
    run_vstar_mode free ""
  fi
  if [[ "${RUN_SOFTFORCE}" == "1" ]]; then
    run_vstar_mode free_softforce_focus "${QUESTION_SUFFIX_SOFTFORCE}"
  fi
fi

if [[ "${RUN_EXTERNAL_BENCHMARKS}" == "1" ]]; then
  for bench in ${BENCHMARKS}; do
    if [[ "${RUN_FREE}" == "1" ]]; then
      run_external_mode "${bench}" free ""
    fi
    if [[ "${RUN_SOFTFORCE}" == "1" ]]; then
      run_external_mode "${bench}" free_softforce_focus "${QUESTION_SUFFIX_SOFTFORCE}"
    fi
  done
fi

python - <<PY
import json
from pathlib import Path

root = Path("${BENCH_RUN_ROOT}")
rows = []

for bench in ["hr_bench_4k", "ocrbench_v2", "blink"]:
    for mode in ["free", "free_softforce_focus"]:
        path = root / bench / mode / "merged_summary.json"
        if not path.exists():
            rows.append({"benchmark": bench, "mode": mode, "status": "missing"})
            continue
        summary = json.loads(path.read_text())
        method = summary.get("free_policy_correct_D") or {}
        rows.append({
            "benchmark": bench,
            "mode": mode,
            "status": "done",
            "n": method.get("n"),
            "accuracy": method.get("accuracy"),
            "answer_parse_rate": method.get("answer_parse_rate"),
            "trigger_rate": method.get("trigger_rate"),
        })

for mode in ["free", "free_softforce_focus"]:
    status_files = sorted((root / "vstar_bench" / mode).glob("**/status.json"))
    status_files = [p for p in status_files if p.parent.name.startswith("T") or p.name == "status.json"]
    if not status_files:
        rows.append({"benchmark": "vstar_bench", "mode": mode, "status": "missing"})
        continue
    data = json.loads(status_files[-1].read_text())
    bench = data.get("datasets", {}).get("VStarBench", {})
    metrics = bench.get("metrics", {})
    rows.append({
        "benchmark": "vstar_bench",
        "mode": mode,
        "status": bench.get("status"),
        "accuracy": metrics.get("split=none|Overall"),
        "direct_attributes": metrics.get("split=none|direct_attributes"),
        "relative_position": metrics.get("split=none|relative_position"),
        "world_size": data.get("world_size"),
        "status_file": str(status_files[-1]),
    })

(root / "summary_free_softforce.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\\n")
print(json.dumps(rows, indent=2, ensure_ascii=False))
PY

log "complete"
echo "Stage1: ${STAGE1_CKPT}"
echo "Stage2: ${STAGE2_CKPT}"
echo "Benchmarks: ${BENCH_RUN_ROOT}"
