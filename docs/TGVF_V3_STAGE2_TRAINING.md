# TGVF-v3 Stage2 Training Notes

This document records the current TGVF-v3 Stage2 trajectory training implementation and the initial smoke results.

## Scope

Stage2 trains the v3 trajectory/action/readout behavior on the existing 50k teacher dataset.

Implemented scope:

- Qwen3-VL-Thinking backbone with base weights frozen.
- Plain-text v3 markers, with no tokenizer resize and no new special tokens.
- LLM LoRA trainable.
- Stage1 TGVF module loaded and kept trainable.
- Focus and no-focus samples loaded from the same v3 50k JSONL.
- Teacher-forced focus span capture for focus samples.
- TGVF visual embeddings inserted inside bracketed `<TGVF>` visual blocks.
- Weak-strict original-image-key mask for focus samples after TGVF.
- No original-image-key mask for no-focus/direct samples.
- Weighted trajectory CE losses.
- No matrix CE, no same-image O(K^2), no contrastive loss by default.
- Optional post-training Stage1-regression eval on the trained TGVF module.

Not implemented in this pass:

- Full Stage2 generation/protocol validation suite that loads the LoRA adapter.
- Full benchmark evaluation.
- Multi-foveation.
- New tokenizer special tokens.

## Dataset

Train split:

- Path: `data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/final/tgvf_teacher_items.accepted.jsonl`
- Total records: 50,022
- Focus samples: 35,542, 71.05%
- No-focus/direct samples: 14,480, 28.95%

Val split:

- Path: `data/tgvf_teacher/generated/runs/tgvf_v3_teacher_val_2k/final/tgvf_teacher_items.accepted.jsonl`
- Total records: 2,023
- Focus samples: 1,382, 68.31%
- No-focus/direct samples: 641, 31.69%

## Stage1 checkpoint

Default Stage2 initialization uses the completed 8B Stage1 checkpoint:

`outputs/tgvf_v3_stage1_8b/8b_8gpu_2000step_stage1_v3/train/checkpoint_step_2000.pt`

## Code paths

- Dataset / step logic: `src/revisit_vlm/tgvf_v3_stage2.py`
- Training entrypoint: `scripts/train_tgvf_v3_stage2.py`
- 8B run script: `scripts/run_tgvf_v3_stage2_8b.sh`

## Default training configuration

Default script settings:

- Model: `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`
- Max image resolution: 512
- Variant: `tgvf_v2_bidirectional`
- FVT position mode: `native_source_grid`
- LoRA rank: 64
- LoRA alpha: 256
- LoRA dropout: 0.05
- LoRA targets: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- LR LoRA: 2e-5
- LR TGVF: 5e-6
- LR calibration: 1e-5
- Scheduler: cosine
- Warmup ratio: 0.03
- Min LR ratio: 0.1
- Max steps: 1200
- Save every: 300
- Eval every: 300
- Batch per device: 16
- Gradient accumulation: 1

For 8 GPUs, effective global batch size is 128. This keeps the intended global batch while avoiding the older bs=1/grad_accum=16 micro-step overhead.

## Loss configuration

Focus samples:

- evidence_state: 0.2
- focus_target: 1.5
- evidence: 1.0
- value_span: 1.0
- answer: 1.0

No-focus samples:

- evidence_state: 0.2
- answer: 1.0

Disabled by default:

- matrix CE
- same-image negative O(K^2)
- contrastive alignment
- visual token manifold regularization

## Mask policy

Focus samples use `weak_strict_original_image_keys_4d`:

- question / choices / evidence_state / focus target can attend original image tokens.
- TGVF / evidence / answer query positions cannot attend original image-token keys.
- non-image text context remains visible under causal masking.

No-focus samples use standard causal masking and retain access to original image tokens.



## Fast batched Stage2 path

The original Stage2 implementation was correct but too slow because it processed every sample independently inside the batch. The fast path follows the reference VPT project structure more closely: the expensive Qwen forwards are batched.

Implemented fast path:

- `src/revisit_vlm/tgvf_v3_stage2_fast.py`
- enabled by default through `--fast-batched-stage2`
- disabled with `--no-fast-batched-stage2` or `FAST_BATCHED_STAGE2=0`

Fast path dataflow:

```text
base image/question inputs
-> batched Qwen3 visual feature extraction for V_pre
-> batched first Qwen3 forward for focus samples
-> extract hidden states inside plain-text <FOCUS>...</FOCUS> spans
-> per-sample TGVF module creates D
-> batched focus final forward with D scattered into <TGVF> visual placeholders
-> batched no-focus final forward through native Qwen3 image path
-> weighted trajectory CE
```

Important details:

- No tokenizer special tokens are added.
- Markers remain plain text.
- Focus marker tokens are excluded from H_q.
- Focus final forward uses weak-strict original-image-key mask.
- No-focus final forward uses native Qwen3 image inputs and no weak-strict mask.
- Original image embeddings for focus final forward are produced by running frozen Qwen visual merger on source V_pre, rather than using raw `<|image_pad|>` token embeddings.
- Stage2 training uses two Qwen passes for focus samples. This is allowed for training; inference remains single-pass/KV-resume.

## Fast path smoke results

Single-GPU bs4 fast smoke:

- Result: passed
- Focus count: 4
- No-focus count: 0
- Peak memory: 20.36 GB
- Focus mask active rate: 1.0
- FVT shapes observed: `[160, 4096]`, `[252, 4096]`

Single-GPU bs16 mixed fast smoke:

- Result: passed
- Focus count: 12
- No-focus count: 4
- Peak memory: 27.56 GB
- Focus mask active rate: 1.0
- No-focus mask active rate: 0.0
- Value span match rate: 0.9375

8GPU bs16 5-step speed smoke:

- Path: `outputs/tgvf_v3_stage2_8b/stage2_8b_fast_8gpu_5step_speed`
- Result: passed
- Wall time: 137.03 seconds, including model load and final checkpoint save
- World size: 8
- Per-device batch: 16
- Effective global batch: 128
- Peak memory range observed in logs: 26.48-29.03 GB
- Focus mask active rate: 1.0
- No-focus mask active rate: 0.0
- Matrix CE: disabled
- Same-image negative: disabled

This is materially better than the previous sample-wise path, where the 8GPU bs16 run completed step 1 but remained too slow to wait for step 10. The fast path is now the default for Stage2.

## Batch-size probe on B200

Hardware check: 8x NVIDIA B200, about 183GB memory per GPU.

Single-GPU 1-step probes with default rank64/all-linear LoRA showed:

| per-device batch | grad accum | effective batch, 1 GPU | peak memory | result |
| --- | --- | ---: | ---: | --- |
| 1 | 1 | 1 | 19.67 GB | passed |
| 2 | 1 | 2 | 19.71 GB | passed |
| 4 | 1 | 4 | 19.68 GB | passed |
| 8 | 1 | 8 | 20.32 GB | passed |
| 16 | 1 | 16 | 21.88 GB | passed |

Conclusion: peak memory is low because the current Stage2 step processes samples sequentially inside the batch instead of doing a fully vectorized batched multimodal forward. Increasing batch size mainly reduces micro-step/optimizer/all-reduce overhead and preserves the desired effective global batch. It does not fill B200 memory.

Default was changed from bs=1/grad_accum=16 to bs=16/grad_accum=1. On 8 GPUs this keeps global batch 128. Filling memory further would require a separate vectorized batching refactor or a deliberate increase in image resolution/sequence length, which would change the training setting.

## Smoke results

Static checks:

- `python -m py_compile src/revisit_vlm/tgvf_v3_stage2.py scripts/train_tgvf_v3_stage2.py`: passed
- `bash -n scripts/run_tgvf_v3_stage2_8b.sh`: passed

No-focus 1-step smoke, low-rank LoRA:

- Loss: 2.6275
- Grad norm: 1.6765
- Peak memory: 16.72 GB
- Focus count: 0
- No-focus count: 1
- No-focus mask active rate: 0.0
- Special tokens added: false
- Tokenizer resized: false
- Matrix CE enabled: false

Focus-only 1-step smoke, low-rank LoRA:

- Loss: 2.5320
- Grad norm: 10.6442
- Peak memory: 17.07 GB
- Focus count: 1
- Focus mask active rate: 1.0
- Mask mode: `weak_strict_original_image_keys_4d`
- Masked image key count: 228
- FVT shape: `[228, 4096]`
- Target hidden shape: `[8, 4096]`
- Value span matched: true
- Special tokens added: false
- Tokenizer resized: false
- Matrix CE enabled: false

Default script 1-step smoke, rank64/all-linear LoRA:

- Command: `RUN_ID=stage2_8b_default_1step_smoke NUM_GPUS=1 MAX_STEPS=1 GRAD_ACCUM=1 SAVE_EVERY=1 EVAL_EVERY=0 RUN_EVAL_AFTER=0 WANDB_MODE=disabled scripts/run_tgvf_v3_stage2_8b.sh`
- Loss: 1.7075
- Grad norm: 20.2981
- Peak memory: 19.67 GB
- Focus count: 1
- Focus mask active rate: 1.0
- Mask mode: `weak_strict_original_image_keys_4d`
- Masked image key count: 252
- FVT shape: `[252, 4096]`
- Target hidden shape: `[8, 4096]`
- Value span matched: true
- Special tokens added: false
- Tokenizer resized: false
- Matrix CE enabled: false

## Run commands

100-step smoke on 8 GPUs:

```bash
RUN_ID=stage2_8b_8gpu_100step_smoke \
NUM_GPUS=8 \
MAX_STEPS=100 \
SAVE_EVERY=100 \
EVAL_EVERY=100 \
RUN_EVAL_AFTER=0 \
scripts/run_tgvf_v3_stage2_8b.sh
```

Main 1200-step run on 8 GPUs:

```bash
RUN_ID=stage2_8b_8gpu_1200step \
NUM_GPUS=8 \
scripts/run_tgvf_v3_stage2_8b.sh
```

Disable post-training Stage1-regression eval:

```bash
RUN_EVAL_AFTER=0 scripts/run_tgvf_v3_stage2_8b.sh
```

## Evaluation note

`scripts/run_tgvf_v3_stage2_8b.sh` can run the existing v3 readout/query/distribution suite after training. This loads the `tgvf_module` from the Stage2 checkpoint and checks whether Stage1-style D/readout/query quality regressed.

This post-training eval currently does not load the Stage2 LoRA adapter, so it should not be interpreted as full Stage2 protocol-generation validation.
