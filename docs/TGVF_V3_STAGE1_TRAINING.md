# TGVF-v3 Stage1 Training Record

This document records the current TGVF-v3 Stage1 training setup.

Stage1 here means projector / foveal-module pretraining for the Qwen3-VL-Thinking based TGVF-v3 path. It does not include Stage2 LoRA trajectory training, dataset generation, benchmark full runs, crop baselines, or multi-foveation.

## Current purpose

TGVF-v3 Stage1 trains the TGVF visual module so that:

```text
image + question + <EVIDENCE_STATE> + <FOCUS>target</FOCUS>
    -> contextual target hidden states H_q
    -> Qwen3 visual features V_pre / V_merge
    -> TGVF visual tokens D
    -> frozen Qwen3 reads D as target-specific evidence
```

The supervised readout target is the v3 evidence segment:

```text
<TGVF>[D visual embeddings]</TGVF>
<EVIDENCE>evidence_description</EVIDENCE>
```

`<ANSWER>` is not required for the current Stage1 loss.

## Main files

```text
scripts/train_tgvf_v3_stage1.py
scripts/run_tgvf_v3_stage1_8b_train_eval.sh
src/revisit_vlm/tgvf_v3_stage1.py
src/revisit_vlm/qwen3_vl_tgvf.py
```

The default 8B train + eval launcher is:

```bash
scripts/run_tgvf_v3_stage1_8b_train_eval.sh
```

## Current default training recipe

The 8B script is configured for v3 Stage1 with:

```text
MODEL_ID=/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking
VARIANT=tgvf_v2_bidirectional
NUM_FVT=none
MAX_IMAGE_RESOLUTION=512
FVT_POSITION_MODE=native_source_grid

BATCH_SIZE=4
NUM_GPUS=1 by default, set NUM_GPUS=8 for the full run
global batch on 8 GPUs = 32
BATCH_SAMPLING=auto
same-image matrix CE enabled by default

LEARNING_RATE=1e-4
LR_SCHEDULER=cosine
WARMUP_STEPS=100
MIN_LR_RATIO=0.1
MAX_STEPS=2000
SAVE_EVERY=500

CAPTURE_MODE=teacher_forced
READOUT_BATCH_SIZE=4
WANDB_MODE=online
WANDB_PROJECT=tgvf-v3
```

For the v2 bidirectional variant, `NUM_FVT=none` means the module emits a dynamic number of D tokens matching the source visual token count after Qwen's native merge/grid behavior. In recent 512px examples this was typically around 228-252 D tokens, but it varies by image grid.

## Full 8B 2000-step command

Use this for the current official v3 Stage1 8B run:

```bash
RUN_ID=8b_8gpu_2000step_stage1_v3 \
NUM_GPUS=8 \
WANDB_GROUP=stage1-8b-2000 \
EVAL_PRESET=pilot \
EVAL_TASKS=all \
scripts/run_tgvf_v3_stage1_8b_train_eval.sh
```

If the validation file is not in one of the default locations, pass it explicitly:

```bash
EVAL_JSONL=/path/to/tgvf_v3_val_2k.jsonl
```

Default validation probe paths are:

```text
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/final/tgvf_v3_val_2k.jsonl
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/final/tgvf_teacher_items.val_2k.jsonl
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/final/val_2k.jsonl
```

If no validation file is found, the script warns and falls back to the train JSONL for evaluation. For formal reporting, use a real image-disjoint v3 val split.

## Dataset

Current training file:

```text
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/final/tgvf_teacher_items.accepted.jsonl
```

Known stats from the current 50k accepted dataset:

```text
total accepted rows: 50,022
focus rows used by Stage1: 35,542
direct/control rows: 14,480
focus image groups: 9,186
groups with >=3 focus targets: 8,867
groups with >=4 focus targets: 6,398
```

Stage1 loads only:

```text
need_focus = true
trajectory_type = single_focus
evidence_state = need_local_visual_evidence
```

Direct/no-focus rows are reserved for Stage2.

## Trainable and frozen modules

Frozen:

```text
Qwen3-VL vision encoder
Qwen3-VL LLM
Qwen3 native visual path
Qwen3 merger path
```

Trainable:

```text
TGVF module / projector / foveal refiner
associated TGVF calibration layers if present
```

Recent debug logs confirm:

```text
qwen_frozen = true
finite_rate = 1.0
```

## H_q capture mode

Current Stage1 default:

```text
CAPTURE_MODE=teacher_forced
```

This is Stage1-only. Since the teacher target is known during supervised training, Stage1 extracts contextual target hidden states with a teacher-forced focus trajectory instead of a slow token-by-token decode loop.

The forced prefix is:

```text
<EVIDENCE_STATE>need_local_visual_evidence</EVIDENCE_STATE>
<FOCUS> target </FOCUS>
```

Only the target span inside `<FOCUS>...</FOCUS>` is used for `H_q`. Marker tokens are excluded.

Deployment/inference still uses the single-pass visible focus capture path; the teacher-forced shortcut is not the runtime TGVF-v3 inference path.

## Vision features

For each focus sample, Stage1 obtains:

```text
H_q: target hidden states from frozen Qwen3 text/vision context
V_pre: Qwen3 visual feature tap
V_merge: Qwen3 merged visual features
D: TGVF output visual tokens
```

Recent 8B logs at 512px showed examples like:

```text
target_hidden_shape: [8-13, 4096]
pre_merge_visual_shape: [912-1008, 1152]
merged_visual_shape: [912-1008, 1152]
foveated_visual_tokens_shape: [228-252, 4096]
```

## TGVF append and position mode

Current append/readout mode:

```text
readout_append_mode = qwen3_visual_special_tokens_embedding_replace
```

D is inserted into a bracketed TGVF visual block using Qwen visual special-token style wrapping.

Current position mode:

```text
FVT_POSITION_MODE=native_source_grid
position_ids_source=qwen3_native_source_grid_full_trajectory
```

This is the v3 default. The older name `inherit_source_visual_positions` remains available but is not the default.

## Weak-strict mask status

The weak-strict original-image-key mask is enabled by default:

```text
--mask-original-image-after-tgvf default = true
attention_mask_mode = weak_strict_original_image_keys_4d
```

The intended behavior is:

```text
Before TGVF:
  question / evidence_state / focus may attend to original image tokens.

From TGVF onward:
  TGVF / evidence / answer positions cannot attend to original image token keys.

Non-image context remains visible under normal causal rules.
```

Recent 8B run logs confirmed the mask was active:

```json
{
  "attention_mask_mode": "weak_strict_original_image_keys_4d",
  "image_keys_blocked_for_tgvf_evidence_answer": true,
  "pre_tgvf_queries_keep_original_image_keys": true
}
```

`mask_span_summary` also confirmed that post-TGVF query positions were blocked from original image keys while pre-TGVF queries kept access.

## Losses

Current default loss stack:

```text
loss_gen = 1.0
loss_visual_token_manifold = 0.01
loss_same_image_negative = 1.0
same_image_negative_mode = matrix_ce
```

The main supervised loss is evidence generation/readout loss over `<EVIDENCE>`.

Same-image matrix CE is enabled by default when:

```text
BATCH_SIZE > 1
loss_same_image_negative > 0
same_image_negative_mode = matrix_ce
```

With local `BATCH_SIZE=4`, each local rank samples same-image groups of 4 focus targets when possible.

Matrix CE scores:

```text
score[i, j] = -NLL(evidence_i | target_i, D_j)
```

The diagonal should be correct.

## Throughput fixes applied for v3 Stage1

The initial online Stage1 path was too slow because it performed many small serial forwards:

```text
per-sample token decode for H_q
per-sample vision feature extraction
serial KxK readout scoring for matrix CE
large JSON norm logging
```

The current v3 Stage1 path includes these speed fixes:

```text
teacher-forced Stage1 H_q extraction
per-batch same-image vision feature cache
chunked batched readout forward for evidence scoring
chunked batched matrix CE readout
compact norm diagnostics logging
DataLoader options exposed, although defaults are still conservative
```

This changed 8B 8GPU training from an unusable roughly 195s/step path to a practical single-digit seconds-per-step range on the 100-step run.

## Smoke and speed results

### 8B single GPU bs4 matrix CE smoke

Command used local group4 smoke data:

```text
/tmp/tgvf_v3_stage1_group4.jsonl
```

Result after speed fixes:

```text
success
loss_total = 3.796875
loss_gen = 2.390625
loss_same_image_negative = 1.40625
peak_memory_gb = 93.43
qwen_frozen = true
finite_rate = 1.0
mask active = true
```

### 8B 8GPU 10-step speedcheck

Output:

```text
/tmp/tgvf_v3_stage1_speedcheck_8b_8gpu_10
```

Result:

```text
wall time = 153.24s
global batch = 32
peak_memory_gb around 94.23 on rank0
```

This includes torchrun startup, 8 rank model loading, and the first-step overhead.

### 8B 8GPU 100-step speedfix run

Output:

```text
outputs/tgvf_v3_stage1_8b/8b_8gpu_100step_speedfix_20260531_181242
```

W&B:

```text
project: https://wandb.ai/mio_nora/tgvf-v3
run: https://wandb.ai/mio_nora/tgvf-v3/runs/6ilgxjyq
```

Checkpoints:

```text
checkpoint_step_50.pt
checkpoint_step_100.pt
```

Wall time:

```text
839.17s for 100 steps
```

Key step 100 metrics:

```text
loss_total = 2.625
loss_gen = 1.3046875
loss_same_image_negative = 1.3203125
grad_norm = 4.0625
learning_rate = 1e-05
peak_memory_gb = 97.59
finite_rate = 1.0
qwen_frozen = true
mask active = true
```

The 100-step smoke used `MAX_STEPS=100` and `WARMUP_STEPS=100`, so the LR schedule in that run is not representative of the 2000-step schedule. It was used for speed, OOM, checkpoint, W&B, and mask verification.

## Current 2000-step time estimate

Using the 8B 8GPU 100-step wall time:

```text
839.17s / 100 steps = 8.39s/step
2000 steps ~= 16,783s ~= 4.66 hours
```

Practical estimate:

```text
training only: about 4.7-5.0 hours
training + pilot eval: add eval time
```

This is much faster than the old pre-speedfix online path.

## W&B logging

Training logs to W&B by default in the 8B script:

```text
WANDB_PROJECT=tgvf-v3
WANDB_MODE=online
WANDB_GROUP configurable
```

Metrics include:

```text
loss_total
loss_gen
loss_same_image_negative
loss_visual_token_manifold
grad_norm
learning_rate
peak_memory_gb
finite_rate
source_visual_token_count
answer_token_count
qwen_frozen
mask flags
shape stats
attention diagnostics
norm diagnostics summary
```

Checkpoint artifact upload is available with:

```bash
WANDB_LOG_CHECKPOINTS=1
```

It is off by default to avoid large artifact uploads unless requested.

## Evaluation after training

The 8B script automatically runs:

```bash
./eval/run_tgvf_v3_eval_suite.sh "${CHECKPOINT}" "${EVAL_TASKS}"
```

Current supported v3 eval tasks include:

```text
readout
query
distribution
```

Default for full launcher:

```text
EVAL_PRESET=pilot
EVAL_TASKS=all
```

Use a real v3 validation JSONL for formal evaluation.

## Known caveats

1. `CAPTURE_MODE=teacher_forced` is a Stage1 training optimization, not the deployment inference capture mode.
2. The current pipeline still performs online frozen Qwen3 feature extraction. A future offline H_q/V_pre/V_merge cache could further reduce cost.
3. DDP utilization still has some sawtooth behavior because image/token lengths vary by rank, but it is now practical.
4. `visual_token_manifold_active` is currently false when D dim is 4096 and V_merge dim is 1152. The manifold loss safely contributes zero in that mismatch case.
5. The 100-step speed run used a short schedule and should not be interpreted as convergence evidence.
6. Formal reports should use an image-disjoint v3 validation split, not train-file fallback.

## Current acceptance status

The current v3 Stage1 path satisfies:

```text
8B 8GPU run starts and completes 100 steps
checkpoints save correctly
W&B online logging works
Qwen3 backbone is frozen
TGVF module receives gradients
same-image matrix CE is active
weak-strict mask is active
native_source_grid position mode is active
finite losses observed
peak memory fits on B200 with local bs4
2000-step training script is prepared
```
