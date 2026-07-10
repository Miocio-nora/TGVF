# Stage2 Narrow-LoRA Ablation

## Status

- Experiment status: running.
- Mainline status: unchanged.
- The authoritative golden recipe remains the 2026-07-03 Stage2 D-DeepStack
  run until this ablation finishes and is evaluated.
- This experiment does not change Stage1 or any default Stage2 value.

## Question

Can a narrower Stage2 language adaptation surface preserve more of the
original Qwen3-VL reasoning distribution while retaining tool triggering and
TGVF D readout capability?

## Baseline

- Golden Stage1 checkpoint:
  `outputs/clean_training/qwen3_stage1_ddeepstack_norm01_wandb_4gpu_20260702_180906/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt`
- Golden Stage2 plan:
  `outputs/clean_training/qwen3_stage2_ddeepstack_norm01_from_stage1_ddeepstack_8gpu_20260703_005210/stage2_micro4/training_plan.json`
- Golden Stage2 plan sha256:
  `527d5f51c8c2b47b4533d2a03110adfdc31dff1a6f2c7979324b7df860c591f9`

## Experimental Diff

| Variable | Golden | Narrow-LoRA ablation |
|---|---:|---:|
| LoRA rank | 64 | 16 |
| LoRA alpha | 256 | 64 |
| LoRA scaling | 4 | 4 |
| LoRA targets | q/k/v/o + gate/up/down | q/v/o |
| Protocol token training | full embed/lm-head modules | protocol rows only |
| Evidence CE weight | 1.0 | 0.2 |

All other data, model, batch, optimizer, schedule, mask, DeepStack, D
DeepStack, TGVF, and CE settings match the golden Stage2 plan.

## Isolation

- The row-only path is opt-in through
  `--protocol-token-training-mode row_only`.
- The default remains `full_modules` for backward compatibility and mainline
  reproducibility.
- The ablation code commit is
  `7485a45ec35f6f11aee55e95bf05f5f3dee9a1c8`.

## Verification

- Related tests: 98 passed.
- One-step 8-GPU smoke completed with checkpoint save and validation.
- Smoke loss: 4.6484375.
- Smoke checkpoint reloaded through the clean Stage2 diagnostic path and
  completed a readout execution.
- LLM trainable optimizer parameters:
  - Golden: 1,417,093,120.
  - Narrow-LoRA: 12,419,072.
- TGVF trainable parameters remain 72,055,808.
- Both input embedding and LM-head protocol rows are trainable; full vocabulary
  matrices are frozen.

## Formal Run

- Plan:
  `outputs/clean_training/qwen3_stage2_ddeepstack_narrow_lora_r16_qvo_rowonly_ev02_from_golden_stage1_8gpu_20260710_160934/stage2_micro4/training_plan.json`
- Plan sha256:
  `fd265202eaaac6546503321d01c0fe78543e2bc1eb85cf7c29fa73af41d08af3`
- GPUs: 0-7.
- Batch: `8 * micro4 * accumulation4 = 128`.
- Steps: 1200.
- W&B run: `f1s4su0z`.
- Output:
  `outputs/clean_training/qwen3_stage2_ddeepstack_narrow_lora_r16_qvo_rowonly_ev02_from_golden_stage1_8gpu_20260710_160934/stage2_micro4`

## Decision Rule

This experiment can motivate a mainline change only after Stage2 internal
diagnostics and the standard free/softforce benchmark evaluation show that it
preserves or improves tool behavior while recovering original-model reasoning
performance. Until then it remains an ablation result.
