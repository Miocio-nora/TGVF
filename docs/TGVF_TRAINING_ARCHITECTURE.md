# TGVF FVT Training Architecture

This document summarizes the current TGVF FVT training path, the model framework, and the three trainable module variants.

## Current Status

The current training code supports:

- single-card training
- 4-GPU DDP training with `torchrun`
- W&B logging from rank 0 only
- frozen Qwen2-VL
- training only the new TGVF/FVT module
- balanced 20k teacher-guide JSONL input
- optional post-training eval suite upload to W&B from the 20k ablation shell

Current validated run:

```text
train file:
  data/tgvf_teacher/generated/runs/teacher_run_000001/final/tgvf_teacher_items.accepted_balanced_20k.jsonl

variant:
  foveal_cross_merger

DDP GPUs:
  CUDA_VISIBLE_DEVICES=0,1,2,3

validated progress:
  reached step 550 after the tokenizer-boundary capture fix

rank-0 checkpoint:
  outputs/tgvf_fvt/teacher_run_000001_foveal_cross_merger_ddp0123_runcheck2/checkpoint_step_500.pt

checkpoint contains:
  tgvf_module, config, optimizer, global_step

Qwen weights saved:
  no
```

A bug was found during the first long 20k DDP run: Qwen tokenization can merge the last target character and the opening `<` of `<|/foveate|>` into one token, for example `'.<'`. The capture parser now keeps the token-id fast path and adds a decoded-text fallback that maps character spans back to generated-token hidden states. Regression coverage is in `tests/test_tgvf_capture.py`.

## Training Goal

The v0 objective trains the newly added Foveated Visual Token module so that it can turn:

```text
target hidden states H_q
pre-merge Qwen2-VL visual tokens V_pre
```

into:

```text
Foveated Visual Tokens D
```

The frozen Qwen LLM should be able to read the bracketed FVTs in a fresh readout context and generate the teacher evidence description.

This stage does not train Qwen2-VL itself.

## Frozen and Trainable Parts

Frozen:

- Qwen2-VL visual encoder
- Qwen2-VL original visual merger
- Qwen2-VL language model
- tokenizer and processor

Trainable:

- selected TGVF FVT module
- its cross-attention projections
- its slots or merger MLP, depending on variant

No LoRA, Qwen finetuning, visual encoder finetuning, OCR, crop pipeline, or cache surgery is used in this stage.

## Data Input

Training consumes teacher-guide JSONL records with at least:

```text
image
question
target
evidence_description
```

Current balanced 20k file:

```text
data/tgvf_teacher/generated/runs/teacher_run_000001/final/tgvf_teacher_items.accepted_balanced_20k.jsonl
```

The balanced source mix is:

```text
Visual Genome      8000
TextVQA + TextOCR  6000
DocVQA             4000
ChartQA            2000
```

The `target` should be a neutral local visual pointer and should not leak the answer. The `evidence_description` contains the visible detail that the FVTs are trained to support.

## Training Flow

For each sample, training performs three conceptual branches.

### 1. Query Construction Branch

Use frozen Qwen2-VL with the original image and question.

Force the assistant to generate:

```text
<|foveate|>{target}<|/foveate|>
```

During the same single-pass decoding path, collect:

```text
H_q:     target token hidden states, [T, d_lm]
V_pre:   pre-merge Qwen visual tokens, [N, d_v]
V_merge: original merged Qwen visual tokens, [N_merge, d_lm]
```

These tensors are detached before entering the trainable TGVF module. This branch is not where the generation loss is computed.

### 2. Foveal Module Branch

Run the selected TGVF module:

```text
D = foveal_module(
  target_hidden_states=H_q,
  pre_merge_visual_tokens=V_pre,
)
```

Expected output:

```text
D: [M, d_lm]
```

In the current 2B setup:

```text
d_lm = 1536
d_v  = 1280
M    = 16 by default
```

### 3. Fresh Readout Branch

Start a fresh text context. Do not reuse the original image/question KV cache.

Build a prompt that tells frozen Qwen it is reading foveated visual evidence. By default training uses target prompt dropout: 70% of readout examples include the target, and 30% omit it. The dropout is controlled by `--readout-prompt-target-dropout` and defaults to `0.3`.

Target-conditioned prompt:

```text
System: You are reading foveated visual evidence...
User:
Target: {target}
Describe the visible evidence for this target.

<|vision_start|>[D_1] ... [D_M]<|vision_end|>

Assistant:
{evidence_description}
```

Generic prompt after target dropout:

```text
System: You are reading foveated visual evidence...
User:
Describe the visible evidence in the foveated visual tokens.

<|vision_start|>[D_1] ... [D_M]<|vision_end|>

Assistant:
{evidence_description}
```

Normal text tokens use Qwen text embeddings. The image-placeholder positions inside the bracketed visual span are replaced by `D`.

Compute cross-entropy only on `evidence_description` tokens. Prompt tokens, target tokens, bracket tokens, and FVT placeholder tokens are masked with `IGNORE_INDEX`.

Qwen parameters stay frozen, but this branch is not under `torch.no_grad()`, because gradients must flow through frozen Qwen computations back into the input embeddings `D`, then into the TGVF module.

## Losses

Implemented losses:

```text
L_gen
L_visual_token_manifold
L_same_image_negative
L_contrastive_alignment
```

Default v0 objective:

```text
L_total = L_gen + 0.1 * L_visual_token_manifold
```

Default enabled:

```text
--loss-gen 1.0
--loss-visual-token-manifold 0.01
--readout-prompt-target-dropout 0.3
```

Default disabled:

```text
--loss-same-image-negative 0.0
--same-image-negative-mode cyclic_margin
--same-image-negative-margin 1.0
--loss-contrastive-alignment 0.0
```

### L_gen

Fresh-readout cross-entropy over only `evidence_description` tokens.

Purpose: make the frozen Qwen LLM able to read useful evidence from the injected FVTs. The default readout prompt includes `Target: ...` for 70% of training examples and uses a generic D-only prompt for 30%, reducing reliance on target text plus a generic `D`.

### L_visual_token_manifold

Matches simple statistics of FVTs to original Qwen merged visual tokens:

```text
MSE(mean(D), mean(V_merge)) + MSE(std(D), std(V_merge))
```

Purpose: keep generated FVTs near Qwen's visual-token manifold.

### L_same_image_negative

Disabled by default. It is intended for query sensitivity when a batch contains multiple teacher items from the same image. Use `--group-batches-by-image` so same-image negatives are present reliably.

Two modes are supported:

```text
--same-image-negative-mode cyclic_margin
--same-image-negative-mode matrix_ce
```

`cyclic_margin` is the original objective. For each same-image group it pairs each item with the next item in the group and applies:

```text
max(0, margin - logP(desc_i | target_i, D_i) + logP(desc_i | target_i, D_j))
```

`matrix_ce` builds a full same-image score matrix for each group:

```text
score[i, j] = logP(desc_i | target_i, D_j)
L_query = CrossEntropy(score[i, :], label=i)
```

This makes every description prefer its own `D_i` over all other same-image `D_j` values in the batch. It is stronger than the cyclic margin mode but costs `O(n^2)` readout forwards per same-image group, with diagonal scores reused from `L_gen`.

### L_contrastive_alignment

Disabled by default.

Future batch contrastive loss between pooled FVT representation and evidence-description text representation.

## Model Variants

The `--variant` flag supports:

```text
token_direct
pooled
foveal_cross_merger
```

### token_direct

Module:

```text
TokenFovealCrossAttention
```

Core idea:

Use all target token hidden states directly as queries.

Shape flow:

```text
H_q:   [T, d_lm]
V_pre: [N, d_v]
Q = projection(H_q)
K,V = projection(V_pre)
cross-attention output -> [T, d_lm]
optional fixed-length output -> [M, d_lm]
```

Role:

This is the most token-preserving ablation. It keeps target-token granularity and asks each target token to retrieve relevant visual evidence.

Pros:

- preserves token-level query information
- useful for ablation against pooled variants
- small checkpoint compared with the merger variant

Cons:

- variable target length can make behavior less stable
- less structurally similar to Qwen's visual merger

### pooled

Module:

```text
PooledFovealCrossAttention
```

Core idea:

Mean-pool target hidden states into one global foveation query, then condition a fixed number of output slots.

Shape flow:

```text
H_q:      [T, d_lm]
q_global: [d_lm]
slots:    [M, attn_dim]
V_pre:    [N, d_v]
cross-attention output -> [M, d_lm]
```

Role:

This is a stable fixed-length FVT baseline.

Pros:

- fixed number of FVTs
- simpler than the cross-merger
- easier to append and batch

Cons:

- target-token details are compressed into one pooled query
- does not imitate Qwen's patch-merger structure

### foveal_cross_merger

Module:

```text
FovealCrossMerger
```

Core idea:

Use a pooled global target query to condition `M x R` sub-slots. Each sub-slot cross-attends to pre-merge visual tokens. Then each group of `R` sub-slots is flattened and passed through a Qwen-style merger MLP to produce one FVT.

Definitions:

```text
M = number of final FVTs, default 16
R = spatial_merge_size ** 2, usually 4 for Qwen2-VL
```

Shape flow:

```text
H_q:         [T, d_lm]
q_global:    [d_lm]
sub_slots:   [M, R, d_v]
flatten:     [M * R, d_v]
V_pre:       [N, d_v]
cross-attn:  [M * R, d_v]
group:       [M, R, d_v]
flatten:     [M, R * d_v]
merger MLP:  [M, d_lm]
```

Role:

This is the preferred main design for v0.

Pros:

- uses dense pre-merge visual features
- structurally closer to Qwen2-VL's patch-merger idea
- produces fixed-length FVTs
- strongest candidate for later training extensions

Cons:

- largest checkpoint and most parameters among the three variants
- more compute and memory than the other variants

## Early 20k Ablation

The initial four-job 20k ablation is documented separately:

```text
docs/TGVF_EARLY_20K_ABLATION.md
```

Keep future ablation-specific run plans, commands, defaults, and results in that document or in another category-specific document, not in this general architecture summary.

## LR Scheduler

The training script supports two scheduler modes:

```text
--lr-scheduler none
--lr-scheduler warmup_cosine
```

Smoke/debug runs use constant LR by default:

```text
lr_scheduler = none
learning_rate = 1e-4
```

Full 20k runs launched through `scripts/run_tgvf_20k_4gpu_ablation.sh full` use warmup + cosine decay by default:

```text
lr_scheduler = warmup_cosine
learning_rate = 1e-4
warmup_ratio = 0.03
warmup_steps = 0
min_lr_ratio = 0.1
min_learning_rate = 1e-5
```

`--max-steps` in `scripts/train_tgvf_fvt.py` means micro-batch / dataloader steps. Scheduler length is therefore computed as:

```text
estimated_optimizer_steps = ceil(max_steps / gradient_accumulation_steps)
```

Scheduler stepping happens only after `optimizer.step()`, not on every micro-batch. With gradient accumulation, this means:

```text
backward every micro-batch
optimizer.step every gradient_accumulation_steps micro-batches
scheduler.step once after optimizer.step
```

Example for final two-loss full run:

```text
max_steps = 20000
gradient_accumulation_steps = 8
estimated_optimizer_steps = 2500
warmup_ratio = 0.03
warmup_steps = 75
LR warms up to 1e-4, then cosine decays to 1e-5
```

Logs and W&B include:

```text
learning_rate
base_learning_rate
optimizer_step
batch_step
lr_scheduler
warmup_steps
estimated_optimizer_steps
min_learning_rate
max_steps_semantics
```

Checkpoints include scheduler state when a scheduler is active, plus `optimizer_step`. The default training script save interval is 1000 steps, while the 20k ablation shell sets `SAVE_EVERY=5000` in full mode to keep checkpoint counts small. The ablation shell can also run the eval suite after training with `EVAL_AFTER_TRAIN=1`; eval outputs are placed inside each module directory and uploaded to W&B when W&B is enabled. Resume is available through:

```bash
python scripts/train_tgvf_fvt.py   ...   --resume-from-checkpoint path/to/checkpoint_step_N.pt
```

## Single-Card Training Command

One pass over the balanced 20k file on one B200:

```bash
python scripts/train_tgvf_fvt.py \
  --train-file data/tgvf_teacher/generated/runs/teacher_run_000001/final/tgvf_teacher_items.accepted_balanced_20k.jsonl \
  --output-dir outputs/tgvf_fvt/teacher_run_000001_foveal_cross_merger \
  --model-name-or-path Qwen/Qwen2-VL-2B-Instruct \
  --variant foveal_cross_merger \
  --num-foveated-tokens 16 \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --max-steps 20000 \
  --save-every 5000 \
  --log-every 50 \
  --progress
```

## 4-GPU DDP Training Command

One pass over the balanced 20k file on GPUs 0,1,2,3:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/train_tgvf_fvt.py \
  --train-file data/tgvf_teacher/generated/runs/teacher_run_000001/final/tgvf_teacher_items.accepted_balanced_20k.jsonl \
  --output-dir outputs/tgvf_fvt/teacher_run_000001_foveal_cross_merger_ddp0123 \
  --model-name-or-path Qwen/Qwen2-VL-2B-Instruct \
  --variant foveal_cross_merger \
  --num-foveated-tokens 16 \
  --device auto \
  --torch-dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --batch-size 1 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-4 \
  --max-steps 5000 \
  --save-every 5000 \
  --log-every 25 \
  --progress \
  --wandb-project tgvf \
  --wandb-run-name fvt_train_teacher_run_000001_ddp0123 \
  --wandb-tags train,fvt,foveal_cross_merger,balanced_20k,ddp0123
```

DDP semantics:

```text
--max-steps is per rank
4 ranks * 5000 steps * batch 1 = about 20,000 samples total
--gradient-accumulation-steps 2 gives effective global batch 8
```

If `--gradient-accumulation-steps 8` is used under 4-GPU DDP, the effective global batch becomes 32.

## W&B Behavior

W&B is optional and enabled only when `--wandb-project` is provided.

In DDP mode:

- rank 0 initializes W&B
- rank 0 logs metrics
- rank 0 saves checkpoints
- non-zero ranks do not initialize W&B
- scalar losses and grad norm are averaged across ranks before rank-0 logging

The smoke commands often use `--wandb-mode disabled` only to avoid creating short debug runs.

## Convergence Validation Plan

The immediate next goal is not final quality, but checking whether the training objective is numerically meaningful.

Recommended validation sequence:

1. Run `foveal_cross_merger` on 4 GPUs for 500 to 1000 steps.
2. Watch W&B curves:
   - `train/loss_gen`
   - `train/loss_total`
   - `train/loss_visual_token_manifold`
   - `train/grad_norm`
3. Confirm `loss_gen` trends downward over a moving window, not just individual noisy samples.
4. Confirm `grad_norm` is finite and not collapsing to NaN/Inf.
5. Save checkpoints at least every 500 steps.
6. Use `debug_tgvf_readout.py` or an equivalent readout debug utility on a fixed small sample set before and after training.

Important interpretation notes:

- Per-step loss is noisy because image resolution, target type, and evidence length vary widely.
- Odd steps can show `grad_norm = 0` when gradient accumulation is active; only optimizer-step intervals report clipped grad norm.
- `L_visual_token_manifold` is weighted by 0.1 by default for current Stage1 runs. Earlier 0.01 runs left this term at only about 1-2% of total loss and did not prevent D scale drift.
- The current v0 objective tests whether FVTs are readable by frozen Qwen in a fresh context. It does not yet prove final TGVF inference accuracy.

## Same-Image Negative Loss Smoke Experiments

Two small overfit experiments were run to verify that `L_same_image_negative` is active and optimizable.

### Experiment A: L_gen + L_same_image_negative

Fixture:

```text
outputs/tgvf_fvt/same_image_negative_overfit_vg2385028.jsonl
```

This fixture contains 6 teacher items from one image:

```text
visual_genome:2385028
/home/dredvpn009/Flash_Storage/datasets/visual_genome/VG_100K_2/2385028.jpg
```

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_tgvf_fvt.py \
  --train-file outputs/tgvf_fvt/same_image_negative_overfit_vg2385028.jsonl \
  --output-dir outputs/tgvf_fvt/same_image_negative_overfit_fcm_vg2385028 \
  --model-name-or-path Qwen/Qwen2-VL-2B-Instruct \
  --variant foveal_cross_merger \
  --num-foveated-tokens 8 \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --batch-size 6 \
  --gradient-accumulation-steps 1 \
  --learning-rate 3e-4 \
  --max-steps 40 \
  --save-every 40 \
  --log-every 5 \
  --loss-gen 1.0 \
  --loss-visual-token-manifold 0.01 \
  --loss-same-image-negative 1.0 \
  --loss-contrastive-alignment 0.0 \
  --wandb-mode disabled
```

Observed values:

```text
step 1:
  loss_total: 3.7467
  loss_gen: 2.8744
  loss_same_image_negative: 0.8692

step 10:
  loss_total: 2.9491
  loss_gen: 2.0762
  loss_same_image_negative: 0.8495

step 15:
  loss_total: 2.0587
  loss_gen: 1.6896
  loss_same_image_negative: 0.3365

step 30:
  loss_total: 0.7124
  loss_gen: 0.5861
  loss_same_image_negative: 0.0783

step 40:
  loss_total: 0.8377
  loss_gen: 0.1350
  loss_same_image_negative: 0.6550
```

Interpretation:

`L_gen` clearly overfits the tiny same-image set. `L_same_image_negative` is active and can drop sharply, but it is noisy in this setup because the whole 6-item group is shuffled each epoch and the cyclic wrong-D pairing changes.

Checkpoint:

```text
outputs/tgvf_fvt/same_image_negative_overfit_fcm_vg2385028/checkpoint_step_40.pt
```

### Experiment B: L_same_image_negative as the Main Loss

Fixture:

```text
outputs/tgvf_fvt/same_image_negative_overfit_vg2385028_3items.jsonl
```

This fixture contains the first 3 items from the same image. `L_gen` was computed for logging but disabled in the total loss.

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_tgvf_fvt.py \
  --train-file outputs/tgvf_fvt/same_image_negative_overfit_vg2385028_3items.jsonl \
  --output-dir outputs/tgvf_fvt/same_image_negative_only_overfit_fcm_vg2385028_3items \
  --model-name-or-path Qwen/Qwen2-VL-2B-Instruct \
  --variant foveal_cross_merger \
  --num-foveated-tokens 4 \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --batch-size 3 \
  --gradient-accumulation-steps 1 \
  --learning-rate 3e-4 \
  --max-steps 50 \
  --save-every 50 \
  --log-every 5 \
  --loss-gen 0.0 \
  --loss-visual-token-manifold 0.01 \
  --loss-same-image-negative 1.0 \
  --loss-contrastive-alignment 0.0 \
  --wandb-mode disabled
```

Observed values:

```text
step 1:
  loss_total: 1.1279
  loss_same_image_negative: 1.1248

step 5:
  loss_total: 0.6544
  loss_same_image_negative: 0.6489

step 10:
  loss_total: 0.0158
  loss_same_image_negative: 0.0000

step 20:
  loss_total: 0.0351
  loss_same_image_negative: 0.0000

step 50:
  loss_total: 0.0118
  loss_same_image_negative: 0.0000
```

Interpretation:

This verifies that the same-image negative margin objective can converge on a controlled tiny same-image batch. The remaining total loss after step 10 is almost entirely the weighted visual-token manifold term.

Checkpoint:

```text
outputs/tgvf_fvt/same_image_negative_only_overfit_fcm_vg2385028_3items/checkpoint_step_50.pt
```

Both checkpoints contain only:

```text
tgvf_module
config
optimizer
global_step
```

No Qwen2-VL weights are saved.

## Inference Relation

Training and inference intentionally differ.

Training uses a fresh readout branch to force evidence to be recoverable from `D`.

Inference uses:

```text
single-pass foveation capture
-> generate FVTs from H_q and V_pre
-> append bracketed FVTs to the preserved generation state
-> continue answer generation
```

Inference must not rerun the full prompt plus foveation span, must not modify old KV cache entries, and must not rerun the visual tower for appended FVTs.
