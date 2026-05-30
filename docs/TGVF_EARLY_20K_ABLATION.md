# TGVF Early 20k Ablation

This document records the early 20k TGVF FVT ablation task. Keep ablation-specific run plans and results here instead of mixing them into the general training architecture document.

## Purpose

This is an initial training-path ablation, not the final model report.

The goal is to compare:

- the three implemented TGVF FVT module variants;
- whether enabling `L_same_image_negative` changes early convergence behavior;
- the final preferred baseline, `foveal_cross_merger` with the conservative two-loss objective.

All runs use the same balanced 20k teacher-guide dataset.

## Dataset

Training file:

```text
data/tgvf_teacher/generated/runs/teacher_run_000001/final/tgvf_teacher_items.accepted_balanced_20k.jsonl
```

Balanced source mix:

```text
Visual Genome       40%  8000 samples
TextVQA + TextOCR   30%  6000 samples
DocVQA              20%  4000 samples
ChartQA             10%  2000 samples
```

This balanced file should be preferred over the raw accepted file for the 20k ablation, because the raw generation reached 20k early and can overrepresent whichever sources produced more accepted samples per image.

## Variants

The training entrypoint supports:

```text
--variant token_direct
--variant pooled
--variant foveal_cross_merger
```

### token_direct

Uses `TokenFovealCrossAttention`.

Each target hidden-state token acts directly as a query into Qwen2-VL pre-merge visual tokens. This keeps token-level target information and is mainly an ablation for whether direct token queries are enough.

### pooled

Uses `PooledFovealCrossAttention`.

The target hidden states are pooled into a global foveation query, which conditions a fixed number of FVT output slots. This is simpler and has fixed append length.

### foveal_cross_merger

Uses `FovealCrossMerger`.

This is the preferred structural design. It pools the target query, creates `M x R` sub-slots, cross-attends those sub-slots to pre-merge visual tokens, then uses a Qwen-style merger MLP to produce `M` Foveated Visual Tokens.

Default:

```text
M = 16
```

## Loss Settings

Implemented losses:

```text
L_gen
L_visual_token_manifold
L_same_image_negative
L_contrastive_alignment
```

This ablation runs two loss configurations.

### Three-Loss Ablation

Used for all three variants:

```text
L_total =
  L_gen
+ 0.01  * L_visual_token_manifold
+ 1.0   * L_same_image_negative
```

Flags:

```text
--loss-gen 1.0
--loss-visual-token-manifold 0.01
--readout-prompt-target-dropout 0.3
--loss-same-image-negative 1.0
--same-image-negative-mode cyclic_margin
--loss-contrastive-alignment 0.0
--group-batches-by-image
```

`--group-batches-by-image` is required for this ablation because `L_same_image_negative` needs at least two teacher items from the same image in a batch. Without grouped batches, the loss often skips. The historical runs used `--same-image-negative-mode cyclic_margin`; new runs can use `matrix_ce` to compare each `desc_i` against all same-image `D_j` values in the batch.

### Final Two-Loss Baseline

Used only for the preferred variant:

```text
variant = foveal_cross_merger
```

Objective:

```text
L_total =
  L_gen
+ 0.01  * L_visual_token_manifold
```

Flags:

```text
--loss-gen 1.0
--loss-visual-token-manifold 0.01
--readout-prompt-target-dropout 0.3
--loss-same-image-negative 0.0
--loss-contrastive-alignment 0.0
```

This is the conservative default v0 training objective.

## Four-GPU Layout

The one-click script launches one independent job per B200 GPU:

```text
GPU 0: token_direct_three_loss
GPU 1: pooled_three_loss
GPU 2: foveal_cross_merger_three_loss
GPU 3: foveal_cross_merger_two_loss_final
```

This is not DDP. It is four independent single-GPU experiments running concurrently.

Script:

```text
scripts/run_tgvf_20k_4gpu_ablation.sh
```

## Smoke Run

Smoke mode verifies that all four jobs can load data, construct TGVF inputs, compute losses, backprop into the TGVF module, log, and checkpoint.

Command:

```bash
scripts/run_tgvf_20k_4gpu_ablation.sh smoke
```

Smoke defaults:

```text
WANDB_MODE=disabled
LR_SCHEDULER=none
ABLATION_MAX_STEPS=2
FINAL_MAX_STEPS=2
SAVE_EVERY=2
LOG_EVERY=1
ABLATION_BATCH_SIZE=2
ABLATION_GRAD_ACCUM=1
FINAL_BATCH_SIZE=1
FINAL_GRAD_ACCUM=1
```

## Full 20k Run

Command:

```bash
scripts/run_tgvf_20k_4gpu_ablation.sh full
```

Full-mode defaults:

```text
WANDB_MODE=online
LEARNING_RATE=1e-4
LR_SCHEDULER=warmup_cosine
WARMUP_RATIO=0.03
WARMUP_STEPS=0
MIN_LR_RATIO=0.1
ATTN_IMPL=flash_attention_2
TORCH_DTYPE=bfloat16
NUM_FVT=16
MAX_GRAD_NORM=1.0
SEED=20260525
SAVE_EVERY=5000
LOG_EVERY=25
PROGRESS=1
PROGRESS_INTERVAL=30
EVAL_AFTER_TRAIN=0
EVAL_WORKERS=10
QUERY_MAX_GROUPS=420
QUERY_REQUIRE_GROUPS=420
```

Three-loss ablation defaults:

```text
ABLATION_BATCH_SIZE=6
ABLATION_GRAD_ACCUM=1
ABLATION_MAX_STEPS=auto
```

`ABLATION_MAX_STEPS=auto` computes the number of grouped same-image batches from the training JSONL. With the current balanced 20k file and batch size 6, this is about one grouped epoch over same-image eligible samples. Full mode saves checkpoints every 5000 steps by default to avoid producing many `.pt` files; override with `SAVE_EVERY=...` if needed. The shell prints a compact multi-job progress bar by default; set `PROGRESS=0` to disable it.

Final two-loss baseline defaults:

```text
FINAL_BATCH_SIZE=1
FINAL_GRAD_ACCUM=8
FINAL_MAX_STEPS=20000
```

Optional post-training eval:

```bash
EVAL_AFTER_TRAIN=1 scripts/run_tgvf_20k_4gpu_ablation.sh full
```

When enabled, each successfully completed module is evaluated from its latest `checkpoint_step_*.pt`. The eval output is written inside that module directory, for example:

```text
foveal_cross_merger_three_loss/eval_checkpoint_step_5176/
```

The eval suite runs the four tasks `readout,query,distribution,end2end`, uses the same visible training GPU for that module (`DEVICE=cuda:0` under the corresponding `CUDA_VISIBLE_DEVICES`), defaults to `EVAL_WORKERS=10`, and defaults to `QUERY_MAX_GROUPS=420` plus `QUERY_REQUIRE_GROUPS=420`. If W&B is enabled for the training run, merged eval metrics and the eval output directory are uploaded to W&B as a separate eval run in the same group.

## Learning Rate Schedule

Smoke mode uses constant LR:

```text
LR_SCHEDULER=none
```

Full mode uses warmup plus cosine decay:

```text
LR_SCHEDULER=warmup_cosine
LEARNING_RATE=1e-4
WARMUP_RATIO=0.03
MIN_LR_RATIO=0.1
```

The final LR is:

```text
1e-4 * 0.1 = 1e-5
```

Scheduler steps happen only after `optimizer.step()`, not after every micro-batch.

`--max-steps` is interpreted as micro-batch / dataloader steps. Therefore:

```text
estimated_optimizer_steps = ceil(max_steps / gradient_accumulation_steps)
```

## Output Layout

Default run root:

```text
outputs/tgvf_fvt/20k_4gpu_ablation_${RUN_ID}
```

Each experiment gets its own subdirectory:

```text
token_direct_three_loss/
pooled_three_loss/
foveal_cross_merger_three_loss/
foveal_cross_merger_two_loss_final/
```

Each subdirectory contains:

```text
command.sh
checkpoint_step_*.pt
```

Logs:

```text
outputs/tgvf_fvt/20k_4gpu_ablation_${RUN_ID}/logs/*.log
```

Run configuration:

```text
outputs/tgvf_fvt/20k_4gpu_ablation_${RUN_ID}/run_config.txt
```

## W&B

Full mode logs online by default:

```text
WANDB_MODE=online
WANDB_PROJECT=tgvf
WANDB_GROUP=tgvf_20k_4gpu_ablation_${RUN_ID}
```

Smoke mode disables W&B by default:

```text
WANDB_MODE=disabled
```

Useful logged fields include:

```text
train/loss_total
train/loss_gen
train/loss_visual_token_manifold
train/loss_same_image_negative
train/loss_contrastive_alignment
train/learning_rate
train/base_learning_rate
train/optimizer_step
train/batch_step
train/grad_norm
```

## One-Command Examples

Smoke:

```bash
RUN_ID=smoke_check scripts/run_tgvf_20k_4gpu_ablation.sh smoke
```

Full run:

```bash
RUN_ID=full_20k_ablation scripts/run_tgvf_20k_4gpu_ablation.sh full
```

Full run with W&B disabled:

```bash
RUN_ID=full_20k_ablation_no_wandb \
WANDB_MODE=disabled \
scripts/run_tgvf_20k_4gpu_ablation.sh full
```

Override max steps for a quick full-mode scheduler check:

```bash
RUN_ID=full_mode_short_check \
WANDB_MODE=disabled \
ABLATION_MAX_STEPS=10 \
FINAL_MAX_STEPS=10 \
scripts/run_tgvf_20k_4gpu_ablation.sh full
```

## Current Interpretation

This ablation is a formal 20k training-plan run when executed in `full` mode on:

```text
tgvf_teacher_items.accepted_balanced_20k.jsonl
```

But it is still an early task ablation, not the final TGVF training recipe. Its purpose is to decide which variant/loss combination is worth promoting into the next training stage.


## Foveal Ablation v2

`run_tgvf_20k_foveal_ablation_v2.sh` runs four single-GPU experiments concurrently:

```text
GPU 0: foveal_cross_merger
GPU 1: target_slot_foveal_cross_merger
GPU 2: target_slot_foveal_cross_merger_dropout
GPU 3: target_slot_foveal_cross_merger_matrixce
```

All four use:

```text
L_gen + L_visual_token_manifold + L_same_image_negative
--group-batches-by-image
```

The per-run differences are:

```text
foveal_cross_merger:
  variant=foveal_cross_merger
  readout_prompt_target_dropout=BASE_READOUT_DROPOUT, default 0.0
  same_image_negative_mode=cyclic_margin

target_slot_foveal_cross_merger:
  variant=target_slot_foveal_cross_merger
  readout_prompt_target_dropout=BASE_READOUT_DROPOUT, default 0.0
  same_image_negative_mode=cyclic_margin

target_slot_foveal_cross_merger_dropout:
  variant=target_slot_foveal_cross_merger
  readout_prompt_target_dropout=DROPOUT_READOUT_DROPOUT, default 0.3
  same_image_negative_mode=cyclic_margin

target_slot_foveal_cross_merger_matrixce:
  variant=target_slot_foveal_cross_merger
  readout_prompt_target_dropout=MATRIXCE_READOUT_DROPOUT, default 0.0
  same_image_negative_mode=matrix_ce
```

Full mode runs post-training eval by default:

```bash
scripts/run_tgvf_20k_foveal_ablation_v2.sh full
```

Smoke mode does not run eval by default:

```bash
scripts/run_tgvf_20k_foveal_ablation_v2.sh smoke
```

Eval outputs are written inside each model directory, for example:

```text
outputs/tgvf_fvt/20k_foveal_ablation_v2_${RUN_ID}/target_slot_foveal_cross_merger_matrixce/eval_checkpoint_step_*/
```

The eval defaults match the main suite:

```text
EVAL_WORKERS=10
QUERY_MAX_GROUPS=420
QUERY_REQUIRE_GROUPS=420
```

Each GPU worker now owns its whole lifecycle: train, then immediately eval that model on the same visible GPU. Workers do not wait for the other GPUs before starting eval. Top-level progress reports the worker phase (`training`, `eval`, `done`, or `failed`) next to the training step count, and eval stdout/stderr is saved to `logs/<experiment>_eval.log`.

Set `EVAL_AFTER_TRAIN=0` to skip automatic eval, or set `EVAL_TASKS=readout,query` to run a subset.

## Token Direct Ablation v2

`run_tgvf_20k_token_direct_ablation_v2.sh` mirrors the foveal v2 runner but keeps `variant=token_direct` for all four jobs:

```text
GPU 0: token_direct
GPU 1: token_direct_dropout
GPU 2: token_direct_matrixce
GPU 3: token_direct_dropout_matrixce
```

The per-run differences are:

```text
token_direct:
  readout_prompt_target_dropout=BASE_READOUT_DROPOUT, default 0.0
  same_image_negative_mode=cyclic_margin

token_direct_dropout:
  readout_prompt_target_dropout=DROPOUT_READOUT_DROPOUT, default 0.3
  same_image_negative_mode=cyclic_margin

token_direct_matrixce:
  readout_prompt_target_dropout=MATRIXCE_READOUT_DROPOUT, default 0.0
  same_image_negative_mode=matrix_ce

token_direct_dropout_matrixce:
  readout_prompt_target_dropout=DROPOUT_READOUT_DROPOUT, default 0.3
  same_image_negative_mode=matrix_ce
```

Full mode runs post-training eval by default:

```bash
scripts/run_tgvf_20k_token_direct_ablation_v2.sh full
```

Smoke mode does not run eval by default:

```bash
scripts/run_tgvf_20k_token_direct_ablation_v2.sh smoke
```

Eval defaults and worker lifecycle are the same as the foveal v2 runner: each GPU evaluates its own completed model immediately, with eval logs under `logs/<experiment>_eval.log`.

```text
EVAL_WORKERS=10
QUERY_MAX_GROUPS=420
QUERY_REQUIRE_GROUPS=420
```

## Visual Cue v1 Matrix Ablation

`run_tgvf_20k_v1_matrix_ablation.sh` uses the unbalanced visual-cue v1 teacher data directly:

```text
TRAIN_FILE=data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001/final/tgvf_teacher_items.accepted.jsonl
EVAL_JSONL=data/tgvf_teacher/generated/runs/teacher_run_val_visual_cue_v1_000001/final/tgvf_teacher_items.accepted.jsonl
```

It launches four single-GPU jobs:

```text
GPU 0: token_direct
GPU 1: token_direct_matrixce
GPU 2: target_slot_foveal_cross_merger_matrixce
GPU 3: foveal_cross_merger_matrixce
```

The `token_direct` baseline uses `same_image_negative_mode=cyclic_margin`; the other three use `same_image_negative_mode=matrix_ce`. Full mode runs post-training eval by default, with each GPU evaluating its own completed model immediately.

Command:

```bash
scripts/run_tgvf_20k_v1_matrix_ablation.sh full
```

The v1 validation file has 419 image groups with at least 3 targets when all 2007 accepted validation samples are used, so this runner defaults to:

```text
READOUT_MAX_SAMPLES=2007
DISTRIBUTION_MAX_SAMPLES=2007
QUERY_MAX_GROUPS=419
QUERY_REQUIRE_GROUPS=419
```

## Visual Cue v1 M128 Ablation

`run_tgvf_20k_v1_m128_ablation.sh` uses the visual-cue v1 train/eval files and compares four single-GPU jobs:

```text
GPU 0: token_direct_m128
GPU 1: token_direct_num_output_none
GPU 2: foveal_cross_merger_matrixce_m128
GPU 3: pooled_m128
```

The per-run settings are:

```text
token_direct_m128:
  variant=token_direct
  num_foveated_tokens=128
  same_image_negative_mode=cyclic_margin

token_direct_num_output_none:
  variant=token_direct
  num_foveated_tokens=none
  same_image_negative_mode=cyclic_margin

foveal_cross_merger_matrixce_m128:
  variant=foveal_cross_merger
  num_foveated_tokens=128
  same_image_negative_mode=matrix_ce

pooled_m128:
  variant=pooled
  num_foveated_tokens=128
  same_image_negative_mode=cyclic_margin
```

Command:

```bash
scripts/run_tgvf_20k_v1_m128_ablation.sh full
```

`num_foveated_tokens=none` is only valid for `token_direct`; it leaves `TokenFovealCrossAttention` in per-target-token output mode instead of adaptively pooling to fixed `M`.
