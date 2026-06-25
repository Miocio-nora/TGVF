#!/usr/bin/env bash
set -uo pipefail

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-vpt_qwen2vl2b_clip_blink_full_512_${RUN_STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_outputs/${RUN_NAME}}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/nvmesv/dredvpn009/datasets/benchmarks}"
TOOLS_ROOT="${TOOLS_ROOT:-${BENCHMARK_ROOT}/_tools}"
MODEL_ID="${MODEL_ID:-rp-yu/Qwen2-VL-2b-VPT-CLIP}"
BASE_MODEL_ID="${BASE_MODEL_ID:-Qwen/Qwen2-VL-2B-Instruct}"
GPUS_CSV="${GPUS:-4,5,6,7}"
METHOD="${METHOD:-vpt_clip_direct}"
VPT_TRIGGER_PROMPT="${VPT_TRIGGER_PROMPT:-none}"
VPT_FORCE_ACTION="${VPT_FORCE_ACTION:-none}"

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
  echo "model_id=${MODEL_ID}"
  echo "base_model_id=${BASE_MODEL_ID}"
  echo "benchmark_root=${BENCHMARK_ROOT}"
  echo "gpus=${GPUS_CSV}"
  echo "num_shards=${NUM_SHARDS}"
  echo "max_image_resolution=512"
  echo "method=${METHOD}"
  echo "vpt_trigger_prompt=${VPT_TRIGGER_PROMPT}"
  echo "vpt_force_action=${VPT_FORCE_ACTION}"
} | tee "${OUTPUT_ROOT}/logs/launch.txt"

pids=()
status=0
for shard_index in "${!GPUS_ARR[@]}"; do
  gpu="${GPUS_ARR[${shard_index}]}"
  log_path="${OUTPUT_ROOT}/logs/shard_${shard_index}.log"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONPATH="src:."
    python scripts/eval_vpt_qwen2_blink_full_512.py \
      --model-id "${MODEL_ID}" \
      --base-model-id "${BASE_MODEL_ID}" \
      --benchmark-root "${BENCHMARK_ROOT}" \
      --tools-root "${TOOLS_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --run-id "shard_${shard_index}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard_index}" \
      --device cuda:0 \
      --dtype bf16 \
      --max-image-resolution 512 \
      --max-new-tokens 128 \
      --method "${METHOD}" \
      --vpt-second-round auto \
      --vpt-trigger-prompt "${VPT_TRIGGER_PROMPT}" \
      --vpt-force-action "${VPT_FORCE_ACTION}" \
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

export OUTPUT_ROOT RUN_NAME METHOD
python - <<'PY'
import json
import os
from collections import defaultdict
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
method = os.environ["METHOD"]
rows = []
for path in sorted(root.glob(f"runs/shard_*/predictions/blink__{method}.jsonl")):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                row.setdefault("shard_run", path.parts[-3])
                rows.append(row)

merged_dir = root / "merged"
merged_dir.mkdir(parents=True, exist_ok=True)
prediction_path = merged_dir / f"blink__{method}.jsonl"
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
    "method": method,
    "num_rows": len(rows),
    "num_scored": len(scored),
    "accuracy": mean(scored),
    "num_errors": sum(1 for row in rows if row.get("error")),
    "trigger_rate": mean([1.0 if row.get("triggered") else 0.0 for row in rows]),
    "second_round_rate": mean([1.0 if row.get("second_full_forward_used") else 0.0 for row in rows]),
    "by_subtask": {
        key: {
            "num_scored": len(values),
            "accuracy": mean(values),
        }
        for key, values in sorted(by_subtask.items())
    },
}
(merged_dir / f"blink__{method}.merged_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

exit "${status}"
