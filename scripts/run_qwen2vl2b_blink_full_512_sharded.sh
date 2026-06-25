#!/usr/bin/env bash
set -uo pipefail

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-qwen2vl2b_blink_full_512_${RUN_STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_outputs/${RUN_NAME}}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/nvmesv/dredvpn009/datasets/benchmarks}"
TOOLS_ROOT="${TOOLS_ROOT:-${BENCHMARK_ROOT}/_tools}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2-VL-2B-Instruct}"
GPUS_CSV="${GPUS:-4,5,6,7}"

IFS=',' read -r -a GPUS_ARR <<< "${GPUS_CSV}"
NUM_SHARDS="${#GPUS_ARR[@]}"
if [[ "${NUM_SHARDS}" -lt 1 ]]; then
  echo "No GPUs provided via GPUS=${GPUS_CSV}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/merged"
{
  echo "run_name=${RUN_NAME}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "model_path=${MODEL_PATH}"
  echo "benchmark_root=${BENCHMARK_ROOT}"
  echo "gpus=${GPUS_CSV}"
  echo "num_shards=${NUM_SHARDS}"
  echo "max_image_resolution=512"
} | tee "${OUTPUT_ROOT}/logs/launch.txt"

pids=()
status=0
for shard_index in "${!GPUS_ARR[@]}"; do
  gpu="${GPUS_ARR[${shard_index}]}"
  log_path="${OUTPUT_ROOT}/logs/shard_${shard_index}.log"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONPATH="src:."
    python -m tgvf_eval.run \
      --benchmark blink \
      --tier full \
      --method direct_qwen \
      --model-path "${MODEL_PATH}" \
      --dtype bf16 \
      --device cuda:0 \
      --answer-max-new-tokens 128 \
      --image-budget mid \
      --max-image-resolution 512 \
      --benchmark-root "${BENCHMARK_ROOT}" \
      --tools-root "${TOOLS_ROOT}" \
      --scoring-backend project \
      --official-llm-mode disabled \
      --output-root "${OUTPUT_ROOT}" \
      --run-id "shard_${shard_index}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard_index}" \
      --no-resume \
      --no-progress
  ) > "${log_path}" 2>&1 &
  pids+=("$!")
  echo "started shard_${shard_index} on gpu ${gpu}, pid ${pids[-1]}, log ${log_path}" | tee -a "${OUTPUT_ROOT}/logs/launch.txt"
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

export OUTPUT_ROOT RUN_NAME
python - <<'PY'
import json
import os
from collections import defaultdict
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
rows = []
for path in sorted(root.glob("runs/shard_*/predictions/blink__direct_qwen.jsonl")):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                row.setdefault("shard_run", path.parts[-3])
                rows.append(row)

merged_dir = root / "merged"
merged_dir.mkdir(parents=True, exist_ok=True)
prediction_path = merged_dir / "blink__direct_qwen.jsonl"
with prediction_path.open("w") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def mean(values):
    return None if not values else sum(values) / len(values)

scored = [float(row["score"]) for row in rows if row.get("score") is not None]
by_subtask = defaultdict(list)
for row in rows:
    score = row.get("score")
    if score is None:
        continue
    subtask = (row.get("metadata") or {}).get("sub_task") or "unknown"
    by_subtask[str(subtask)].append(float(score))

summary = {
    "run_name": os.environ["RUN_NAME"],
    "num_rows": len(rows),
    "num_scored": len(scored),
    "accuracy": mean(scored),
    "num_errors": sum(1 for row in rows if row.get("error")),
    "by_subtask": {
        key: {
            "num_scored": len(values),
            "accuracy": mean(values),
        }
        for key, values in sorted(by_subtask.items())
    },
}
(merged_dir / "blink__direct_qwen.merged_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

exit "${status}"
