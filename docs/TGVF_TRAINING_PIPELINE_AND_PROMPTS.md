# TGVF Training Pipeline And Prompts

This document records the current TGVF/FVT training path and every prompt or
instruction that participates in training.

Code entry points:

- `scripts/train_tgvf_fvt.py`
- `src/revisit_vlm/tgvf_training.py`
- `src/revisit_vlm/tgvf_capture.py`
- `src/revisit_vlm/tgvf_foveal.py`

Current v2 experiment launcher:

- `scripts/run_tgvf_20k_v2_matrixce.sh`
- `scripts/run_tgvf_20k_v2_bidirectional_cyclic_negative.sh`

## What Is Trained

Qwen2-VL is frozen. The training script freezes:

- Qwen2-VL visual encoder
- Qwen2-VL original visual merger
- Qwen2-VL language model
- tokenizer and processor

Only the selected TGVF/FVT module is trainable.

Training consumes teacher-guide JSONL records with at least:

```text
image
question
target
evidence_description
```

The current v2 matrix-CE script defaults to:

```text
train file:
  data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001/final/tgvf_teacher_items.accepted.jsonl

model:
  Qwen/Qwen2-VL-2B-Instruct

variants:
  tgvf_v2_vpt_gating
  tgvf_v2_cross_attention
  tgvf_v2_bidirectional
  tgvf_v2_bidirectional cyclic-negative control

num_foveated_tokens:
  none

batch_size:
  6

gradient_accumulation_steps:
  1

losses:
  matrix-CE jobs:
    L_gen = 1.0
    L_visual_token_manifold = 0.1
    L_same_image_negative = 1.0, matrix_ce
  bidirectional cyclic-negative control:
    L_gen = 1.0
    L_visual_token_manifold = 0.1
    L_same_image_negative = 1.0, cyclic_margin
  L_contrastive_alignment = 0.0

readout_prompt_target_dropout:
  0.0

memory mode:
  matrix-CE jobs:
    streaming_matrix_ce_backward = 1
  bidirectional cyclic-negative control:
    streaming_matrix_ce_backward = 0
  empty_cache_every_step = 0
```

The matrix-CE jobs use streaming backward because full `B x B` readout graphs
with real image grids can exceed 178GB. The extra bidirectional cyclic-negative
job on GPU 3 does not use streaming. `empty_cache_every_step` remains an
emergency OOM switch and is not enabled by default.

The single-job bidirectional cyclic-negative launcher defaults to `GPU=3` and
`STREAMING_BACKWARD=1`, so it uses streaming backward for `cyclic_margin` and
then runs the same automatic evaluation suite in `full` mode.

Automatic eval in `full` mode:

```text
EVAL_AFTER_TRAIN=1
EVAL_TASKS=all
EVAL_JSONL=data/tgvf_teacher/generated/runs/teacher_run_val_visual_cue_v1_000001/final/tgvf_teacher_items.accepted.jsonl
EVAL_WORKERS=4
READOUT_MAX_SAMPLES=2007
DISTRIBUTION_MAX_SAMPLES=2007
QUERY_MAX_GROUPS=419
QUERY_REQUIRE_GROUPS=419
END2END_MAX_SAMPLES=300
```

The `smoke` mode keeps `EVAL_AFTER_TRAIN=0` by default. Set
`EVAL_AFTER_TRAIN=1 EVAL_TASKS=readout READOUT_MAX_SAMPLES=1 EVAL_WORKERS=1` for
a small automatic-eval wiring check.

The base Python training script still has a default
`--readout-prompt-target-dropout 0.3`, but the current 20k v1/v2 shell scripts set
the matrix-CE runs to `0.0` unless overridden.

## High-Level Pipeline

For each micro-batch, `training_step` does:

```text
teacher JSONL samples
-> frozen Qwen capture pass with original image + question
-> force <|foveate|>{target}<|/foveate|>
-> capture target token hidden states H_q
-> capture Qwen pre-merge visual tokens V_pre
-> capture original merged visual tokens V_merge
-> v2: V_pre' = TGVF_conditioner(H_q, V_pre)
-> v2: D = frozen Qwen visual.merger(V_pre')
-> fresh readout context containing D as pseudo-image tokens
-> frozen Qwen predicts evidence_description
-> losses backprop only through D into the TGVF module
```

There is no Qwen finetuning in this stage.

## Stage 1: Capture Prompt

Training calls `collect_training_features`, which builds:

```python
forced_text = f"<|foveate|>{sample.target}<|/foveate|>"
```

Then `ForcedFoveationWrapper` forces Qwen's generated tokens to exactly this
sequence. Qwen does not freely choose the target during training.

The model still receives the original image and a text instruction through
`build_tgvf_prompt(question)`. There is no system message. The message shape is:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": build_tgvf_prompt(question)},
        ],
    }
]
```

Then Qwen's processor applies:

```python
processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

The exact capture instruction text is:

```text
{question}

If fine-grained visual evidence is needed, output exactly one foveation request:
<|foveate|>visual target description<|/foveate|>
After <|/foveate|>, stop.
Do not answer the question yet.
Do not emit intent, mode, scope, JSON, tool metadata, explanations, or natural-language commentary.
Only emit the foveation request.
```

Because the logits are forced, the generated assistant text becomes:

```text
<|foveate|>{sample.target}<|/foveate|>
```

Training captures hidden states only for the target span inside the markers:

```text
H_q = hidden states of {sample.target}
```

It also captures:

```text
V_pre   = visual tokens immediately before Qwen PatchMerger
V_merge = original Qwen merged visual tokens
grid    = original image_grid_thw
```

## Stage 2: TGVF Module

The trainable module receives:

```text
target_hidden_states = H_q
pre_merge_visual_tokens = V_pre
```

and emits:

```text
v1-style variants: D = foveated_visual_tokens
v2 variants: V_pre' = conditioned_pre_merge_visual_tokens
```

Current v2 variants condition visual tokens before Qwen's original visual
merger. They do not train a new merger:

- `tgvf_v2_vpt_gating`: mean-pools target tokens into one condition vector, applies
  a dot-product residual to each pre-merge visual token.
- `tgvf_v2_cross_attention`: each visual token attends to the target-token
  sequence, then applies a gated residual delta before merge.
- `tgvf_v2_bidirectional`: target attends to vision, vision attends back to the
  enriched target tokens, then applies a gated residual delta before merge.

After the v2 conditioner, the common training/eval/inference path calls the
frozen Qwen2-VL visual merger:

```text
D = frozen_qwen.visual.merger(V_pre')
```

The frozen merger is not saved in the TGVF checkpoint and is not passed to the
optimizer. Gradients still flow through its frozen computation into the
conditioner.

For v2, `--num-foveated-tokens none` means the output token count is not a fixed
M. It is derived from the original pre-merge visual grid after Qwen's own
spatial merge, so it matches the original merged image-token count.

Older v1-style variants are still available:

- `token_direct`
- `pooled`
- `foveal_cross_merger`
- `target_slot_foveal_cross_merger`

Those fixed-M variants generally use a fake near-square pseudo-image grid during
fresh readout. `token_direct` can be dynamic when `--num-foveated-tokens none`,
but it still does not preserve the original image grid unless it is a v2 output.

## Stage 3: Fresh Readout Prompt

The readout branch is a fresh context. It does not reuse the original image,
question, or generation cache.

It manually builds Qwen chat tokens instead of calling `apply_chat_template`.
There is no system message. The FVT span is inserted as Qwen-native visual
placeholder tokens:

```text
<|vision_start|><|image_pad|> ... <|image_pad|><|vision_end|>
```

The embeddings for the `<|image_pad|>` positions are replaced with `D`.

### Target Present

When the target is not dropped, the exact readout prefix is:

```text
<|im_start|>user
<|vision_start|><|image_pad|> ... <|image_pad|><|vision_end|>
The visual tokens above are focused evidence for the target:
{target}

Describe only what is visible in this focused evidence.
<|im_end|>
<|im_start|>assistant
```

Then training appends the supervised answer:

```text
{evidence_description}
```

Cross-entropy labels are active only on `{evidence_description}`. The prompt,
target text, visual brackets, and image-placeholder token ids are masked with
`IGNORE_INDEX`.

### Target Dropped

If `readout_prompt_target_dropout` samples a dropout event, the target is omitted
and the exact readout prefix is:

```text
<|im_start|>user
<|vision_start|><|image_pad|> ... <|image_pad|><|vision_end|>
The visual tokens above are focused evidence.

Describe only what is visible in this focused evidence.
<|im_end|>
<|im_start|>assistant
```

Then training still supervises:

```text
{evidence_description}
```

In the current v2 matrix-CE launcher, target dropout defaults to `0.0`, so this
target-dropped path is normally disabled unless the environment overrides
`READOUT_DROPOUT`.

## Real Image Ids And Positions

The readout path uses Qwen-native image placeholder ids and image modality ids.

For all FVT readout:

```text
token ids:
  <|vision_start|>, repeated <|image_pad|>, <|vision_end|>

mm_token_type_ids:
  text tokens = 0
  FVT image_pad span = 1
```

For v2 outputs, training requires the original source grid:

```text
image_grid_thw = feature.image_grid_thw
position_ids_source = source_image_grid_mrope
fake_image_grid_thw = None
```

This is intentional. v2 should preserve the original image-token count and use
the original image grid for mRoPE positions.

If a v2 sample has no `image_grid_thw`, training raises:

```text
TGVF v2 requires source image_grid_thw for image-position readout
```

For non-v2 outputs, the code falls back to:

```text
position_ids_source = manual_fake_grid_mrope
fake_image_grid_thw = near_square_fake_grid
```

That fallback is mainly for old fixed-M experiments.

## Losses

Total loss is:

```text
L_total =
  loss_gen_weight * L_gen
  + loss_visual_token_manifold_weight * L_visual_token_manifold
  + loss_same_image_negative_weight * L_same_image_negative
  + loss_contrastive_alignment_weight * L_contrastive_alignment
```

### L_gen

`L_gen` is the frozen-Qwen language-model cross entropy on the readout answer:

```text
input:
  fresh readout prompt + D pseudo-image tokens

label:
  evidence_description only
```

This is the main training signal. Gradients flow through frozen Qwen operations
back to the input embeddings at the FVT span, then into the TGVF module.

### L_visual_token_manifold

This compares simple statistics of D with original Qwen merged visual tokens:

```text
MSE(mean(D), mean(V_merge)) + MSE(std(D), std(V_merge))
```

It is a weak regularizer to keep FVT embeddings on a Qwen-like visual-token
manifold.

For current Qwen3/Qwen2 tool-observation Stage1 runs, the default weight is
`0.1`. Earlier `0.01` runs left this term at only about 1-2% of total loss and
did not prevent D scale drift.

### L_same_image_negative

This requires batches grouped by image via `--group-batches-by-image`.

For `matrix_ce`, each same-image group builds a score matrix:

```text
row i = target/evidence pair i
col j = D from sample j
score[i, j] = log likelihood of evidence_i when reading D_j
```

The diagonal is the positive pair. Cross entropy over each row trains the model
to make the correct D for each target/evidence pair more likely than wrong D
from the same image.

The current v2 script uses:

```text
--same-image-negative-mode matrix_ce
--loss-same-image-negative 1.0
--streaming-matrix-ce-backward
```

The streaming mode computes score gradients without retaining all readout graphs
at once.

### L_contrastive_alignment

This is implemented but disabled in current scripts.

When enabled, Qwen encodes each `evidence_description` as text, mean-pools D, and
applies an InfoNCE-style cross entropy between pooled D and evidence text
embeddings.

No prompt or instruction is used for this loss; the evidence descriptions are
tokenized directly as text.

## What Training Does Not Use

Training does not use the benchmark inference answer-turn instructions from
`build_fvt_answer_instruction`.

It does not train on:

```text
Use this evidence to answer the original multiple-choice question.
Output exactly one option letter.
```

Those are inference/evaluation instructions, not TGVF module training prompts.

Training also does not append FVTs to the original question cache. That happens
in inference. Training readout is a fresh standalone evidence-description task.
