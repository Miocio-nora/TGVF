# Stage3 RL-GRPO

Stage3 optimizes the clean TGVF tool policy after the SFT cold start. The
training sample unit is one QA prompt, loaded from the Stage3 RL pool such as:

```text
revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl
```

The RL data does not need teacher CoT. `reference_target` / `target_spec` is
only the clean target for forced ON-clean probes. Free rollouts may choose no
tool, one tool call, or multiple tool calls up to `max_tool_calls`.

## What Is Implemented

The clean Stage3 framework currently provides:

- Stage3 GRPO config, sample, rollout, reward, probe, and judge-cache schemas.
- A balanced prompt sampler using tool-need buckets.
- A rollout engine interface with a deterministic `fake` backend for smoke.
- A `native_single_focus` backend for real Stage2/TGVF rollout and GRPO
  training updates.
- Forced OFF / forced ON-clean probe cache generation.
- Offline local Qwen-VL judge runner for FocusEvidence and GroundedReasoning
  cache construction.
- A `native_single_focus` rollout wrapper that reuses clean-native Stage2
  checkpoint loading, focus capture, TGVF D construction, and post-tool
  continuation for rollout/probe dumps.
- Sampled native decode plumbing for Stage3 rollouts (`temperature` / `top_p`)
  with generated-token logprobs captured from focus and continuation spans.
- Five-part reward plumbing:
  - answer correctness;
  - tool decision;
  - focus evidence judge score;
  - grounded reasoning judge score;
  - protocol gate penalty.
- Cache-first judge wrapper that writes pending judge requests instead of
  blocking training.
- GRPO group advantage and clipped policy-loss math.
- A configurable multi-step GRPO training loop controlled by `max_steps`,
  `save_steps`, `per_device_prompt_batch_size`, and
  `gradient_accumulation_steps`.
- Fake and native GRPO update paths that write rollout debug rows, reward
  breakdowns, per-step metrics, launch summaries, and checkpoints.
- A native update path that replays policy/reference logprobs for sampled free
  rollouts, performs optimizer steps, and writes Stage2-shaped native
  checkpoints.

The fake rollout backend is only for verifying the plumbing. `native_single_focus`
can run real sampled Stage2/TGVF trajectories and record generation-time
logprobs. Its launch path now replays policy/ref logprobs with gradients for the
sampled trajectories, runs GRPO updates for the configured number of steps, and
saves checkpoints on the configured schedule. The current implementation uses
step-local rollout accumulation before each optimizer update; it does not yet
implement memory-saving micro-batch backward accumulation.
The judge runner is real and can use local Qwen3/Qwen2.5-VL style models through
Hugging Face `transformers`.

## Attention Backend Default

Prefer `--attn-implementation flash_attention_2` whenever the path supports it.
The current clean Stage3 native path has passed a hard-focus one-step smoke with
`flash_attention_2`, including `forced_on_clean`, focus append, TGVF D
construction, post-D continuation, replay, optimizer update, and checkpoint
save. Direct offline judge runs can also request FlashAttention-2 through the
same flag.

Keep `sdpa` for Stage1/Stage2-style paths that rely on custom weak-strict
additive 4D masks until those masks have a Flash-compatible implementation or a
separate compatibility proof. The Stage3 stepwise wrapper also needs a
`--judge-attn-implementation` pass-through before its automatic judge shards can
be controlled from the stepwise command.

## Plan

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.train_stage3_grpo \
  --write-plan \
  --run-id stage3_grpo_smoke \
  --rl-data-path revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl \
  --policy-checkpoint /path/to/stage2/checkpoint \
  --output-dir revisit_vlm_clean/outputs/stage3_grpo/stage3_grpo_smoke \
  --runtime-backend fake \
  --group-size 4 \
  --max-tool-calls 1
```

This writes:

- `stage3_grpo_training_plan.json`
- `stage3_grpo_training_plan.txt`
- `stage3_grpo_dataset_identity.json`

Add W&B logging at plan time when launching real runs:

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.train_stage3_grpo \
  --write-plan \
  --run-id stage3_grpo_native_run \
  --rl-data-path revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl \
  --policy-checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt \
  --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking \
  --processor-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking \
  --attn-implementation flash_attention_2 \
  --output-dir outputs/stage3_grpo/stage3_grpo_native_run \
  --runtime-backend native_single_focus \
  --wandb-project tgvf-stage3-grpo \
  --wandb-mode online \
  --wandb-tags stage3,grpo,native
```

W&B receives the Stage3 config, rollout/reward/probe/judge/train parameters,
RL dataset identity, Stage2 checkpoint identity, git identity, scalar
train/reward metrics, and an output artifact containing plan, preflight, launch
result, train metrics, rollout debug, and reward breakdown. Checkpoint upload is
intentionally separate because native checkpoints are large; enable it only when
needed:

```bash
--wandb-log-checkpoint-artifact
```

## Preflight

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.training.stage3_grpo_executor \
  --plan revisit_vlm_clean/outputs/stage3_grpo/stage3_grpo_smoke/stage3_grpo_training_plan.json \
  --preflight-only
```

Preflight validates the plan schema, RL data identity, checkpoint identity,
output path, rollout backend, reward config, probe config, and judge mode.

## Probe Cache

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.training.stage3_grpo_executor \
  --plan revisit_vlm_clean/outputs/stage3_grpo/stage3_grpo_smoke/stage3_grpo_training_plan.json \
  --precompute-probes \
  --limit-prompts 100
```

Forced probes estimate:

```text
Delta_tool = mean_correct(ON_clean) - mean_correct(OFF)
```

The cache is used by `ToolDecision` reward. If no probe row exists, the current
default falls back to the teacher/data-generation tool hint with reduced weight.

## Offline Local Judge

Training should consume judge caches, not block on a 32B/70B judge model inside
the optimizer loop. First produce `judge_pending.jsonl` with rollout/reward
smoke or training dumps.

Prepare and preflight local judge model directories first:

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.stage3_grpo_prepare_judge_models \
  --output-dir outputs/stage3_grpo/judge_models_qwen3_plan \
  --model-root /nvmesv/dredvpn009/models/hf \
  --target-set stage3_qwen3_recommended \
  --preflight-only
```

The recommended Qwen3 set is:

```text
qwen3_vl_32b_thinking
qwen3_vl_235b_a22b_thinking_fp8
```

The second model is the large Qwen3-VL judge option. There is also a fallback
target set for 70B-class non-Qwen3-VL judges:

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.stage3_grpo_prepare_judge_models \
  --output-dir outputs/stage3_grpo/judge_models_72b_fallback_plan \
  --model-root /nvmesv/dredvpn009/models/hf \
  --target-set stage3_72b_fallback \
  --preflight-only
```

To actually download, make the large side effect explicit:

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.stage3_grpo_prepare_judge_models \
  --output-dir outputs/stage3_grpo/judge_models_qwen3_download \
  --model-root /nvmesv/dredvpn009/models/hf \
  --target-set stage3_qwen3_recommended \
  --execute-download \
  --allow-large-download
```

Then score pending rows offline:

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.stage3_grpo_judge \
  --pending-path revisit_vlm_clean/outputs/stage3_grpo/stage3_grpo_smoke/judge_pending.jsonl \
  --output-dir revisit_vlm_clean/outputs/stage3_grpo/stage3_grpo_smoke/offline_judge \
  --backend local_qwen_vl \
  --model-preset qwen3_vl_32b_thinking \
  --model-root /nvmesv/dredvpn009/models/hf \
  --require-local-model \
  --preflight-only
```

The default local model root is:

```text
/nvmesv/dredvpn009/models/hf
```

For a 32B Qwen3-VL judge, the local directory should normally be:

```text
/nvmesv/dredvpn009/models/hf/Qwen3-VL-32B-Thinking
```

After preflight passes, run the same command with `--execute`:

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.stage3_grpo_judge \
  --pending-path revisit_vlm_clean/outputs/stage3_grpo/stage3_grpo_smoke/judge_pending.jsonl \
  --output-dir revisit_vlm_clean/outputs/stage3_grpo/stage3_grpo_smoke/offline_judge \
  --backend local_qwen_vl \
  --model-preset qwen3_vl_32b_thinking \
  --model-root /nvmesv/dredvpn009/models/hf \
  --require-local-model \
  --device-map auto \
  --dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --max-new-tokens 512 \
  --execute
```

Qwen3-VL Thinking models may emit a short reasoning preamble before the final
JSON object. Keep `--max-new-tokens` high enough for the JSON tail; the local
32B smoke used `512`.

For a 70B-class visual judge, use an actual local VLM checkpoint path. Qwen2.5-VL
72B is supported as a preset; if a Qwen3 70B-class VLM checkpoint is added
locally, pass its directory explicitly with `--model-id`:

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.stage3_grpo_judge \
  --pending-path .../judge_pending.jsonl \
  --output-dir .../offline_judge_72b \
  --backend local_qwen_vl \
  --model-preset qwen25_vl_72b_instruct \
  --model-root /nvmesv/dredvpn009/models/hf \
  --require-local-model \
  --device-map auto \
  --dtype bfloat16 \
  --execute
```

Convenience presets also exist:

```text
qwen3_vl_2b_thinking
qwen3_vl_4b_thinking
qwen3_vl_8b_thinking
qwen3_vl_32b_thinking
qwen3_vl_32b_thinking_fp8
qwen3_vl_30b_a3b_thinking
qwen3_vl_235b_a22b_thinking
qwen3_vl_235b_a22b_thinking_fp8
qwen25_vl_72b_instruct
qvq_72b_preview
```

Use `--model-id` for the actual local checkpoint path when available; it
overrides presets. By default, `--require-local-model` prevents accidental
remote loading. If a model is missing, preflight reports candidate local paths
and a `huggingface-cli download ... --local-dir ...` command. The runner writes:

- `focus_judge_cache.jsonl`
- `grounding_judge_cache.jsonl`
- `judge_raw_outputs.jsonl`
- `judge_preflight_report.json`
- `judge_run_summary.json`

Then point the training plan to these caches:

```bash
--judge-model /path/to/qwen-vl-judge \
--focus-cache-path .../offline_judge/focus_judge_cache.jsonl \
--grounding-cache-path .../offline_judge/grounding_judge_cache.jsonl
```

## Rollout Smoke

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.training.stage3_grpo_executor \
  --plan revisit_vlm_clean/outputs/stage3_grpo/stage3_grpo_smoke/stage3_grpo_training_plan.json \
  --rollout-only
```

This writes rollout debug rows and reward breakdowns without optimizer updates.

## GRPO Training Launch

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.training.stage3_grpo_executor \
  --plan revisit_vlm_clean/outputs/stage3_grpo/stage3_grpo_smoke/stage3_grpo_training_plan.json \
  --launch-training
```

With `runtime_backend=fake`, this runs the configured multi-step GRPO loop
without loading the 8B model. With `runtime_backend=native_single_focus`, it
loads the Stage2 checkpoint, runs real sampled rollouts, replays
policy/reference logprobs for the sampled tokens, performs GRPO optimizer
updates, and writes native checkpoints.

The launch writes:

- `rollout_debug.jsonl`
- `reward_breakdown.jsonl`
- `judge_pending.jsonl`
- `train_metrics.json`
- `train_metrics.jsonl`
- `checkpoint_step_*.pt`
- `LATEST_CHECKPOINT.txt`
- `stage3_grpo_launch_result.json`

## Reward Notes

`R_total` is:

```text
w_answer * R_answer
+ w_tool   * R_tool_decision
+ w_focus  * R_focus_evidence
+ w_ground * R_grounded_reasoning
+ R_protocol_gate
```

Answer reward is deterministic where possible. Focus and grounding rewards are
cache-backed judge scores. In cache-only mode, missing judge rows are logged to
`judge_pending.jsonl` and receive the configured cache-miss reward.

## Non-Goals In This Implementation

This code does not yet implement distributed GRPO, periodic benchmark eval, or
advanced replay optimizations. Online multimodal judging is intentionally not in
the train loop; the implemented path is offline local-judge cache construction.
