# TGVF / Target-Guided Visual Foveation

This repository implements **TGVF-v3**, the Qwen3-VL-Thinking based version of
Target-Guided Visual Foveation.

TGVF-v3 is an active-perception VLM pipeline. Instead of treating foveation as a
crop/zoom operation, the model emits an explicit local visual target during
reasoning, uses the hidden states of that target as a query, generates
target-conditioned visual evidence tokens, appends them as new visual evidence,
and then answers.

Core flow:

```text
image + question
-> <EVIDENCE_STATE>
-> <FOCUS>local visual target</FOCUS>
-> target hidden states H_q
-> H_q + Qwen3 visual features V_pre
-> TGVF/FVT visual tokens D
-> <TGVF>[visual embeddings]</TGVF>
-> <EVIDENCE>...</EVIDENCE>
-> <ANSWER>...</ANSWER>
```

TGVF-v3 is additive to the older v2 paths. Old Qwen2/v2 code should remain
available; v3 behavior is selected through scripts/configs.

## Protocol

TGVF-v3 uses compact visible action traces, not long supervised chain-of-thought:

```text
<EVIDENCE_STATE>need_local_visual_evidence</EVIDENCE_STATE>
<FOCUS>the date-like small text below the barcode</FOCUS>
<TGVF>
[foveated visual embeddings inserted by the runtime]
</TGVF>
<EVIDENCE>The small text below the barcode reads EXP 08/2026.</EVIDENCE>
<ANSWER>EXP 08/2026</ANSWER>
```

Direct/no-focus samples use:

```text
<EVIDENCE_STATE>sufficient_visual_evidence</EVIDENCE_STATE>
<ANSWER>dog</ANSWER>
```

Important conventions:

- Markers are currently plain text strings, not tokenizer special tokens.
- The tokenizer is not resized for Stage2.
- Only text inside `<FOCUS>...</FOCUS>` is used as the foveation query.
- `<FOCUS>` and `</FOCUS>` marker tokens are excluded from `H_q`.
- `<TGVF>` brackets visual embeddings; it is not a textual description.
- TGVF tokens are appended as new evidence and do not mutate original image tokens.

## Current v3 status

Implemented pieces:

- v3 teacher dataset construction path with schema validation, filtering,
  flattening, reporting, and resume/ledger support.
- Qwen3-VL-Thinking deployment smoke path.
- Qwen3 focus-action parsing and target hidden-state capture utilities.
- Qwen3 visual feature tap and TGVF append smoke path.
- Stage1 v3 projector/foveal-module training with weak-strict image-key mask.
- Stage2 v3 trajectory training with LoRA + trainable TGVF module.
- Fast batched Stage2 path inspired by VPT-style batched readout training.
- W&B logging for training loss, loss breakdown, mask rates, focus/no-focus
  counts, dataset stats, LR, grad norm, and memory.

Current 50k Stage2 split:

```text
train records: 50,022
focus samples: 35,542  (71.05%)
no-focus samples: 14,480  (28.95%)
```

Validation split:

```text
val records: 2,023
focus samples: 1,382
no-focus samples: 641
```

Default local model used in current runs:

```text
/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking
```

Stage1 checkpoint used for Stage2:

```text
outputs/tgvf_v3_stage1_8b/8b_8gpu_2000step_stage1_v3/train/checkpoint_step_2000.pt
```

## Install

Install the package in editable mode:

```bash
pip install -e ".[dev]"
```

Install a PyTorch build matching the CUDA environment before installing this
package. For Qwen3-VL training, use the local environment that already has
Transformers, PEFT, W&B, and the required CUDA attention backend installed.

## Important paths

Core v3 code:

```text
src/revisit_vlm/tgvf_v3_teacher.py
src/revisit_vlm/tgvf_v3_stage1.py
src/revisit_vlm/tgvf_v3_stage2.py
src/revisit_vlm/tgvf_v3_stage2_fast.py
src/revisit_vlm/qwen3_vl_tgvf.py
```

Stage2 entrypoints:

```text
scripts/train_tgvf_v3_stage2.py
scripts/run_tgvf_v3_stage2_8b.sh
```

Documentation:

```text
docs/TGVF_V3_STAGE1_TRAINING.md
docs/TGVF_V3_STAGE2_TRAINING.md
```

Current data:

```text
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/final/tgvf_teacher_items.accepted.jsonl
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_val_2k/final/tgvf_teacher_items.accepted.jsonl
```

## Stage2 8B full training

Recommended current full run:

```bash
RUN_ID=stage2_8b_fast_8gpu_1200step_full \
NUM_GPUS=8 \
BATCH_SIZE=16 \
GRAD_ACCUM=1 \
MAX_STEPS=1200 \
SAVE_EVERY=300 \
EVAL_EVERY=300 \
EVAL_MAX_SAMPLES=128 \
LOG_EVERY=10 \
RUN_EVAL_AFTER=1 \
WANDB_MODE=online \
scripts/run_tgvf_v3_stage2_8b.sh
```

Background run:

```bash
mkdir -p logs

nohup env \
RUN_ID=stage2_8b_fast_8gpu_1200step_full \
NUM_GPUS=8 \
BATCH_SIZE=16 \
GRAD_ACCUM=1 \
MAX_STEPS=1200 \
SAVE_EVERY=300 \
EVAL_EVERY=300 \
EVAL_MAX_SAMPLES=128 \
LOG_EVERY=10 \
RUN_EVAL_AFTER=1 \
WANDB_MODE=online \
scripts/run_tgvf_v3_stage2_8b.sh \
> logs/stage2_8b_fast_8gpu_1200step_full.log 2>&1 &
```

Current Stage2 defaults:

```text
model: Qwen3-VL-8B-Thinking
per-device batch: 16
num GPUs: 8
effective global batch: 128
max steps: 1200
save every: 300
eval every: 300
fast batched Stage2: enabled
LoRA: enabled
TGVF module: trainable
Qwen base / vision encoder: frozen
matrix CE: disabled
same-image negative: disabled
special tokens: not added
tokenizer resize: false
markers: plain text
mask: weak-strict original image-key block for focus samples after TGVF
```

Recent 100-step smoke result:

```text
8GPU x batch16
100 steps completed
wall time: 1771.91s
final loss_total: 0.8832
peak memory: about 26-30GB per rank
focus mask active rate: 1.0
no-focus mask active rate: 0.0
no OOM
```

## Stage1 training

Stage1 trains the TGVF projector/foveal module while freezing Qwen3-VL. It uses
focus samples only and supervises readout through the v3 trajectory-style
`<FOCUS> -> <TGVF> -> <EVIDENCE>` format.

Stage1 policy:

```text
trainable: TGVF projector / foveal module / calibration layers
frozen: Qwen3 vision encoder, Qwen3 LLM, Qwen visual merger/path
default mask: block original image visual-token keys for TGVF/EVIDENCE/ANSWER
same-image negative / matrix CE: enabled when configured for Stage1
max image resolution: 512 in current 8B runs
```

See:

```text
docs/TGVF_V3_STAGE1_TRAINING.md
```

## Dataset construction

The v3 teacher dataset uses image-level teacher outputs with both focus and
direct/no-focus items, then flattens them into one JSONL row per accepted item.

Focus rows contain:

```text
image
image_id
source_dataset
source_profile
question
need_focus = true
evidence_state = need_local_visual_evidence
trajectory_type = single_focus
target
target_style
target_cues
evidence_description
short_answer / answer
value_span_text
evidence_type
locality
answer_type
visual_difficulty
visibility
target_leakage_risk
evidence_specificity
confidence
```

No-focus rows contain:

```text
need_focus = false
evidence_state = sufficient_visual_evidence
trajectory_type = direct_answer
target = ""
target_style = none
target_cues = []
answer
```

Filtering rules reject generic or leaking targets such as:

```text
the image
the scene
the object
something
the answer
targets containing the answer value
medium/high leakage rows
low-confidence rows
sensitive personal information
```

Valid visual-cue targets are allowed when they are local and locatable:

```text
the small green object near the left edge
the dark rectangular patch near the bottom
the date-like text cluster below the barcode
```

## W&B logging

Stage2 logs training metrics to W&B when `WANDB_MODE=online`.

Current logged fields include:

```text
train/loss_total
train/loss_focus
train/loss_no_focus
train/loss_visual_token_manifold
train/grad_norm
train/peak_memory_gb
train/focus_count
train/no_focus_count
train/focus_ratio
train/no_focus_ratio
train/focus_loss_token_weight
train/no_focus_loss_token_weight
train/focus_sample_mask_active_rate
train/no_focus_mask_active_rate
train/value_span_match_rate
train/effective_global_batch_size
train/world_size
train/lr_group_0
train/lr_group_1
train/special_tokens_added
train/tokenizer_resized
train/markers_are_plain_text
train/matrix_ce_enabled
train/same_image_negative_enabled
train_dataset/*
val_dataset/*
```

## Design constraints

Keep these constraints intact when changing v3:

- Do not make crop/zoom the core method.
- Do not use a fixed generic target.
- Do not train long visible chain-of-thought as the main format.
- Do not add protocol markers as tokenizer special tokens unless explicitly
  planned and migrated.
- Do not use matrix CE or same-image O(K^2) loss in Stage2 by default.
- Do not mutate old cached image tokens; append TGVF tokens as new evidence.
- Preserve old v2 paths and make v3 behavior config-driven.

## Legacy Qwen2 scaffold

This repository started as an editable Qwen2-VL scaffold. The older LoRA/full
fine-tuning entrypoints and Qwen2 data utilities are retained for compatibility,
but the current research path is TGVF-v3 with Qwen3-VL-Thinking.
