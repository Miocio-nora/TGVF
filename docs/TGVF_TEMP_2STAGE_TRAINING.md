# TGVF Temporary 2-Stage Training Notes

This note records the current mainline 2-stage run that produced:

```text
outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_multifocus_focus_imend_from_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260611_021244/checkpoint_step_1200.pt
```

The run uses the pre-merger adaptor path: `tgvf_v2_bidirectional`. The TGVF module conditions Qwen3 pre-merger visual tokens with the focus-target hidden states, then the frozen Qwen visual merger converts the conditioned pre-merger tokens into LLM-side visual tokens.

Source-of-truth configs:

```text
outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260611_021244/train/config.json
outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_multifocus_focus_imend_from_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260611_021244/config.json
```

## Shared Setup

Base model:

```text
/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking
```

Protocol:

```text
protocol_c_tool_observation
```

Protocol tokens:

```text
<|focus_start|> 151669
<|focus_end|>   151670
<|tgvf_start|>  151671
<|tgvf_end|>    151672
```

Runtime shape:

```text
assistant:
<think>...</think>
<|focus_start|>visual descriptor<|focus_end|><|im_end|>

tool:
<|tgvf_start|>
[D visual embeddings]
<|tgvf_end|><|im_end|>

assistant:
<think>focused evidence / reasoning</think>
answer
```

Image processing:

```text
max_image_resolution: 512
max_pixels: 262144
fvt_position_mode: native_source_grid
attention implementation: sdpa
dtype: bfloat16
```

TGVF module:

```text
variant: tgvf_v2_bidirectional
d_lm: 4096
d_v: 1152
spatial_merge_size: 2
num_foveated_tokens: dynamic, equal to source merged visual token count
```

The module runs bidirectional target/vision attention on `pre_merge_visual_tokens`:

```text
H_q: focus descriptor hidden states, shape [T, 4096]
V_pre: Qwen3 pre-merger visual tokens, shape [N_pre, 1152]
D_pre = V_pre + gated_delta(H_q, V_pre)
D = frozen_qwen_visual_merger(D_pre), shape [N_merge, 4096]
```

## Data

The teacher source is `tgvf_v4_teacher_50k`, converted into Protocol-C-compatible Stage1/Stage2 JSONL.

Stage1 focus data:

```text
train: data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl
eval:  data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl
train rows: 40021
eval rows: 867
```

Stage2 trajectory data:

```text
train: data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl
val:   data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl
train total: 46919
train focus: 39678
train no_focus/direct: 7241
train single_focus: 39335
train multi_focus: 343
val total: 1002
val focus: 857
val no_focus/direct: 145
val single_focus: 847
val multi_focus: 10
```

## Stage1

Checkpoint:

```text
outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260611_021244/train/checkpoint_step_2000.pt
```

Processor output:

```text
outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260611_021244/train/processor_step_2000
```

Training command path:

```text
scripts/run_tgvf_v3_toolobs_focus_imend_stage1_stage2_4gpu.sh
  -> scripts/run_tgvf_v3_protocol_c_stage1_4gpu_train_eval.sh
  -> scripts/train_tgvf_v3_stage1.py
```

Main Stage1 parameters:

```text
NUM_GPUS: 4
BATCH_SIZE: 4
GRAD_ACCUM: 2
global_batch_size: 32
MAX_STEPS: 2000
SAVE_EVERY: 500
LEARNING_RATE: 1e-4
LR_SCHEDULER: cosine
WARMUP_STEPS: 100
MIN_LR_RATIO: 0.1
FOCUS_ACTION_IM_END: 1
capture_mode: teacher_forced
batch_sampling: same_image
drop_incomplete_same_image_batches: true
```

Trainable parameters:

```text
TGVF module
protocol token rows in input embeddings and lm_head
```

Stage1 flow:

1. Build the Qwen3 image/question prompt.
2. Teacher-force the focus action with the target descriptor.
3. Capture the target hidden states `H_q` for the descriptor text.
4. Capture Qwen3 visual features before and after the visual merger.
5. Run `tgvf_v2_bidirectional(H_q, V_pre)` to produce `D_pre`.
6. Run the frozen Qwen visual merger to produce `D`.
7. Append the tool-observation TGVF span and train the readout text against `evidence_description`.

Stage1 loss:

```text
loss_total =
  1.0  * loss_gen
+ 1.0  * loss_same_image_negative
+ 0.01 * loss_visual_token_manifold
```

`loss_gen` is token-level LM cross entropy on the focused evidence readout after the inserted TGVF visual embeddings.

`loss_same_image_negative` is same-image matrix CE. Within a same-image batch group, each question should assign higher likelihood to its own `D` than to other `D` tensors from the same image.

`loss_visual_token_manifold` matches the mean and standard deviation of generated `D` against frozen Qwen merged visual tokens.

Stage1 mask:

```text
attention_mask_mode: weak_strict_original_image_keys_4d
mask_original_image_after_tgvf: true
```

The mask is causal plus key blocking. Queries before the inserted TGVF span still attend to the original image keys. Queries from the post-TGVF readout region cannot attend to the original image keys, so the readout has to use the inserted `D` evidence.

## Stage2

Final checkpoint:

```text
outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_multifocus_focus_imend_from_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260611_021244/checkpoint_step_1200.pt
```

Training command path:

```text
scripts/run_tgvf_v3_toolobs_focus_imend_stage1_stage2_4gpu.sh
  -> scripts/run_tgvf_v3_protocol_c_stage2_from_stage1c_4gpu.sh
  -> scripts/train_tgvf_v3_stage2.py
  -> src/revisit_vlm/tgvf_v3_stage2_fast.py
```

Main Stage2 parameters:

```text
NPROC_PER_NODE: 4
BATCH_SIZE: 16
GRAD_ACCUM: 2
effective_global_batch_size: 128
MAX_STEPS: 1200
SAVE_EVERY: 300
EVAL_EVERY: 300
TRAIN_EVAL_MAX_SAMPLES: 128
TARGET_FOCUS_RATIO: 0.8
fast_batched_stage2: true
gradient_checkpointing: true
max_seq_len: 2048
```

Optimizer and LR:

```text
optimizer: AdamW
betas: [0.9, 0.95]
eps: 1e-8
weight_decay: 0.01
lr_lora: 2e-5
lr_tgvf: 5e-6
lr_scheduler: cosine
warmup_steps: 100
min_lr_ratio: 0.1
```

LoRA:

```text
rank: 64
alpha: 256
dropout: 0.05
target_modules: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
modules_to_save: embed_tokens,lm_head
```

Trainable parameters:

```text
LoRA parameters
TGVF module
embed_tokens and lm_head as modules_to_save
```

Stage2 flow for focus samples:

1. Build the original image/question prompt.
2. Teacher-force the focus action:
   `think + <|focus_start|>target<|focus_end|><|im_end|>`.
3. Capture `H_q` from the target descriptor tokens.
4. Run the TGVF module on `H_q` and Qwen3 pre-merger visual features.
5. Finalize with frozen Qwen visual merger.
6. Insert TGVF visual embeddings in the tool-observation span.
7. Train the assistant continuation after the tool output.

Stage2 flow for no-focus samples:

1. Build the original image/question prompt.
2. Train the direct assistant continuation without a TGVF tool span.

Stage2 loss weights:

```text
focus evidence_state:      0.2
focus target descriptor:   1.5
focused evidence readout:  1.0
value span inside evidence: 1.0
answer:                    1.0
no_focus evidence_state:   0.2
no_focus answer:           1.0
```

The fast Stage2 path computes weighted LM loss over labeled continuation tokens. Focus and no-focus losses are combined by token-weighted averaging:

```text
loss_total =
  (loss_focus * focus_loss_token_weight + loss_no_focus * no_focus_loss_token_weight)
  / (focus_loss_token_weight + no_focus_loss_token_weight)
```

Stage2 label/mask behavior:

```text
Original prompt tokens: IGNORE_INDEX
TGVF placeholder/tool visual tokens: IGNORE_INDEX
Focus action tokens: labeled with weighted CE
Post-tool evidence/readout tokens: labeled with weighted CE
Answer tokens: labeled with weighted CE
No-focus output tokens: labeled with weighted CE
```

Stage2 attention mask:

```text
focus samples: weak_strict_original_image_keys_4d
no-focus samples: standard native attention mask
```

For focus samples, the mask blocks original image token keys starting after the focus action, i.e. after the model has emitted the focus descriptor and before it consumes the TGVF tool observation/readout. This makes the post-focus continuation depend on the inserted TGVF visual embeddings rather than directly rereading the original image tokens.

For multi-focus samples, the same rule starts after the first focus action, so later tool observations and readouts cannot use the original image keys directly.

## Recorded Run State

Stage2 final W&B/train log summary at step 1200:

```text
train/loss_total: 0.71973
train/loss_focus: 0.73203
train/loss_no_focus: 0.4395
train/focus_ratio: 0.8
train/focus_sample_mask_active_rate: 1.0
train/no_focus_mask_active_rate: 0.0
train/value_span_match_rate: 0.65
val/loss: 0.78427
val/focus_ratio: 0.86719
val/focus_sample_mask_active_rate: 0.86719
val/no_focus_mask_active_rate: 0.0
val/value_span_match_rate: 0.54688
```

The Stage2 run completed successfully:

```text
elapsed wall time: 8:12:38
exit status: 0
```
