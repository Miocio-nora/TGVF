# Stage2 Narrow-LoRA Ablation

## Status

- Experiment status: completed and evaluated.
- Mainline status: unchanged.
- The authoritative golden recipe remains the 2026-07-03 Stage2 D-DeepStack
  run.
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
- Final checkpoint:
  `outputs/clean_training/qwen3_stage2_ddeepstack_narrow_lora_r16_qvo_rowonly_ev02_from_golden_stage1_8gpu_20260710_160934/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`
- Final checkpoint sha256:
  `943094a006209907962a56da3c158ab084fe6dddb6cffc075ab09024de901d1c`
- Completed `1200` optimizer steps in `8514.64` seconds.

## Internal Results

The final checkpoint was evaluated on the same 200-row/46-group diagnostic
identity as the golden D-DeepStack Stage2 checkpoint.

| Metric | Golden | Narrow-LoRA | Delta |
|---|---:|---:|---:|
| Correct-D NLL | 1.6373 | 1.5840 | -0.0534 |
| Correct-D beats wrong-same | 40.50% | 51.50% | +11.00 pp |
| Query Top-1 | 23.50% | 28.00% | +4.50 pp |
| Query Top-2 | 49.00% | 53.50% | +4.50 pp |
| Query MRR | 0.5004 | 0.5396 | +0.0392 |
| D/V-merge norm ratio | 2.1198 | 2.1659 | +0.0461 |

All FVT values were finite and no collapse warning was raised. The narrower
adaptation surface improved internal D sensitivity.

## CoreDev-2511 Results

Both modes use the exact complete 2511-row manifest and the golden evaluation
identity: native Stage2 backend, FlashAttention-2, DeepStack `no_block`, D
DeepStack enabled, post-D KV continuation, resolution 512, and unified
`max_tokens=512`.

| Mode | Golden acc | Narrow acc | Delta | Golden trigger | Narrow trigger |
|---|---:|---:|---:|---:|---:|
| Free | 37.04% | 36.94% | -0.11 pp | 29.79% | 20.91% |
| Softforce | 37.92% | 35.02% | -2.90 pp | 40.98% | 20.19% |

The summary scorer excluded two free and three softforce empty-generation
failures. Counting these rows as wrong gives strict accuracies of 36.91% and
34.98%, respectively. There were no D append failures.

### Per-Benchmark Accuracy

| Benchmark | Golden free | Narrow free | Delta | Golden soft | Narrow soft | Delta |
|---|---:|---:|---:|---:|---:|---:|
| VStar | 52.36% | 56.54% | +4.19 pp | 50.79% | 52.36% | +1.57 pp |
| HR-Bench 4K | 52.50% | 56.00% | +3.50 pp | 59.50% | 52.00% | -7.50 pp |
| BLINK | 56.90% | 55.71% | -1.19 pp | 55.48% | 54.52% | -0.95 pp |
| OCRBench v2 | 25.02% | 25.46% | +0.43 pp | 24.54% | 24.55% | +0.00 pp |
| MMMU-Pro | 35.67% | 32.11% | -3.56 pp | 38.00% | 29.77% | -8.23 pp |
| MathVista | 49.67% | 48.16% | -1.51 pp | 49.00% | 46.15% | -2.85 pp |
| MathVerse | 16.00% | 16.00% | +0.00 pp | 19.00% | 14.23% | -4.77 pp |

### Output Tokens

| Mode | Mean | Direct mean | Triggered mean | p50 | p90 | p95 | Hit 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Golden free | 72.77 | 72.15 | 74.24 | 42 | 141 | 250 | 57 (2.27%) |
| Narrow free | 75.56 | 79.32 | 61.33 | 47 | 132 | 241 | 76 (3.03%) |
| Golden softforce | 69.56 | 75.06 | 61.63 | 40 | 131 | 233 | 46 (1.83%) |
| Narrow softforce | 74.55 | 77.32 | 63.56 | 47 | 129 | 234 | 73 (2.91%) |

| Benchmark | Narrow free mean | Narrow softforce mean |
|---|---:|---:|
| VStar | 27.86 | 27.91 |
| HR-Bench 4K | 33.44 | 30.82 |
| BLINK | 42.46 | 43.49 |
| OCRBench v2 | 127.87 | 127.92 |
| MMMU-Pro | 73.78 | 70.73 |
| MathVista | 60.78 | 59.51 |
| MathVerse | 85.59 | 83.20 |

Mean output length increased, but the increase is concentrated in direct-path
and OCR long-tail outputs. Triggered free outputs became shorter than golden,
and max-token hits increased. The mathematical benchmark accuracy therefore
does not support interpreting the higher total mean as recovered reasoning.

## Conclusion

- Internal D/readout sensitivity improved.
- Complete CoreDev-2511 free accuracy remained effectively flat.
- Softforce accuracy fell by 2.90 points, and softforce failed to increase the
  aggregate trigger rate over free.
- MMMU-Pro, MathVista, and MathVerse did not recover original reasoning
  behavior despite the higher average output length.
- This combined ablation changes LoRA capacity/coverage, protocol-token row
  training, and evidence weight together. It does not identify which one is
  responsible for the trigger regression.
- The golden D-DeepStack Stage2 recipe remains authoritative.

## Decision Rule

The promotion condition was not met. Although internal diagnostics improved,
external softforce/tool behavior and reasoning-heavy benchmark accuracy
regressed. No mainline defaults are changed by this result.
