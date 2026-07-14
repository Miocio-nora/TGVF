# Stage2 Matrix-CE Preservation

Status: implemented and evaluated as an opt-in experiment. Matrix-CE preserves
internal target-specific D discrimination, but both completed 1200-step runs
regress external CoreDev-2511 accuracy. The authoritative Golden Stage2 default
remains unchanged.

## Objective

Stage1 learns to distinguish target-conditioned D tokens among questions that
share an image. Existing Stage2 training applies weighted trajectory CE but no
same-image discrimination loss. Internal evaluation showed that Stage2 LoRA can
destroy the target-specific D readout even when the underlying Stage1 TGVF
representation remains intact.

This path preserves the Stage1 discrimination objective during Stage2 without
changing the ordinary Stage2 sample stream or weighted CE.

## Loss Contract

For a same-image group of K single-focus rows, row `i` keeps its own question,
focus action, and post-D readout target. Candidate column `j` supplies D from
row `j`:

```text
score[i, j] = log P(post-D readout_i | prompt_i, action_i, D_j)
L_matrix_ce = CE(score, diagonal_target)
L_total = L_weighted_stage2_ce + lambda_matrix_ce * L_matrix_ce
```

The score span starts immediately after D and stops before the final answer.
It includes the post-D evidence/reasoning readout and its protocol boundaries,
but excludes the answer. When D-DeepStack is enabled, the candidate's D merger
tokens and D-DeepStack branch features are swapped together.

## Sampling Contract

- The ordinary Stage2 cursor is unchanged.
- An auxiliary deterministic cursor selects only `single_focus` rows.
- Every Matrix-CE group contains distinct rows from one image.
- The image is encoded once per group; cached V_pre, V_merge, and original
  DeepStack features are reused across its K questions.
- Original-image access is always blocked over the Matrix-CE readout span
  (`probability=1.0`) so the objective cannot bypass D. This is independent of
  the ordinary Stage2 CE mask probability.
- Image groups are rank-owned by `sha1(image_key) % world_size`.
- Group order and member order are shuffled from the recorded seed.
- Incomplete group remainders are dropped for that epoch; shuffling rotates
  which remainder is omitted.
- The default experimental group size is `K=4`, matching Golden Stage1.

## Configuration

The feature is default-off. The clean Stage2 CLI exposes:

```text
--matrix-ce-preservation
--loss-same-image-matrix-ce 1.0
--matrix-ce-group-size 4
--matrix-ce-readout-batch-size 4
```

The launch plan records the effective loss weight, group/readout sizes, sample
scope, score span, candidate swap contract, and deterministic sampler identity.
Runtime logs include Matrix-CE loss, Top-1, positive and negative score means,
positive-negative margin, and hashed auxiliary sample traces.

The executable training path uses
`sequential_weighted_ce_then_matrix_ce`: it backpropagates the ordinary
weighted Stage2 CE before constructing the auxiliary Matrix-CE graph. The two
gradient contributions are accumulated into the same optimizer update, so the
objective is unchanged while the two large graphs do not coexist in memory.
The auxiliary cursor contributes one independently sampled same-image group
per optimizer step with the full configured Matrix-CE weight. This is an
unbiased, higher-variance estimate of averaging one group per accumulation
micro-step and is the recorded low-cost pilot mode.

## Verification

CPU regression suite:

```text
74 passed
```

Coverage includes default-off plan behavior, opt-in plan serialization,
same-image single-focus filtering, candidate D and D-DeepStack swapping, clean
executor plumbing, and existing Stage1/Stage2/DeepStack regressions.

Golden-identity one-step smoke:

```text
output: outputs/clean_training/
  smoke_stage2_matrix_ce_preservation_golden_gpu0_20260713_v3
elapsed: 42.12 s
weighted focus CE: 3.859375
same-image Matrix-CE: 6.4375
combined loss: 10.3125
Matrix Top-1: 0.25
positive score mean: -60.875
negative score mean: -61.6667
positive-negative margin: +0.7917
pre-clip grad norm: 515.54
post-clip grad norm: 0.9994
peak allocated memory: 123.45 GiB
```

The smoke completed forward, backward, optimizer step, validation, and
checkpoint publication with finite losses and gradients. Its single-group
Top-1 is not an effectiveness result.

The first 4-GPU `micro_batch=4` smoke exposed an SDPA OOM near the full
`178 GiB` device capacity because weighted CE and Matrix-CE graphs were held
simultaneously. Allocator tuning did not fix it, and FlashAttention-2 is not a
valid fallback for this path because its manual DeepStack forward triggered a
CUDA gather assertion. Sequential backward fixed the SDPA OOM. A final
`accumulation=2` gate completed with exactly one Matrix-CE group:

```text
output: outputs/clean_ablation/
  stage2_matrixce_preservation_golden_100step_4gpu_20260713_170018/
  smoke_stage2_micro4_accum2_v2
optimizer-step loss: 10.03125
mean weighted CE: 3.9375
Matrix-CE: 6.09375
peak allocated memory: 131.23 GiB
elapsed: 146.41 s
```

## Controlled Pilot

The first effectiveness run should preserve the authoritative Golden Stage2
configuration and change only Matrix-CE:

- Golden Stage1 D-DeepStack checkpoint and 50k train/test splits.
- Golden rank-64 broad LoRA and full-modules protocol-token training.
- Golden weighted span CE, target focus ratio, masks, and DeepStack scopes.
- Matrix-CE enabled with `K=4`, weight `1.0`, readout batch size `4`.
- One Matrix-CE group per optimizer step, sequential backward after weighted
  Stage2 CE.
- GPUs `0-3`; `4 * micro batch 4 * accumulation 8 = global batch 128`.
- Run `100` optimizer steps and save/evaluate at step `100` only.

Primary acceptance criteria:

1. Stage2 internal D Top-1/Top-2 and wrong-same-image accuracy improve clearly
   over Golden Stage2 and move toward Golden Stage1.
2. Stage2 weighted CE, parser health, trigger behavior, and benchmark accuracy
   do not regress materially.
3. Reasoning length and math benchmark behavior are reported separately. This
   objective targets D discrimination; it is not expected to solve reasoning
   distribution drift by itself.
4. Matrix loss scale, gradient clipping frequency, throughput, and peak memory
   remain operationally acceptable.

Only after those checks should Matrix-CE be considered for the authoritative
Stage2 configuration.

## Completed Results

All external results below use the exact CoreDev-2511 manifest (`2511` rows,
internal hash
`a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`),
resolution `512`, unified `max_tokens=512`, native Stage2, FlashAttention-2,
correct D, D-DeepStack enabled, and original-image DeepStack scope `no_block`.

Two 1200-step Matrix-CE variants were evaluated:

- `R64 + MCE`: Golden rank-64 broad LoRA and full-module protocol-token
  training plus Matrix-CE.
- `R16-C + MCE`: rank `16`, broad LoRA targets, row-only protocol-token
  training, and otherwise the same Matrix-CE and weighted CE configuration.

### Internal D Diagnostics

| Variant | Correct-D NLL | Wrong-same | Top-1 | Top-2 | MRR | D/V norm |
|---|---:|---:|---:|---:|---:|---:|
| Golden, no Stage2 MCE | 1.6373 | 40.50% | 23.50% | 49.00% | 0.5004 | 2.1198 |
| R16-C, no Stage2 MCE | 1.6097 | 34.00% | 18.50% | 41.00% | 0.4586 | 2.0905 |
| R64 + MCE | **1.5828** | 97.00% | 90.00% | 97.50% | 0.9458 | 2.1572 |
| R16-C + MCE | 1.6395 | **98.00%** | **92.00%** | **98.50%** | **0.9564** | 2.1000 |

Matrix-CE succeeds at its narrow objective: both variants retain a strong
target-specific D ranking signal after Stage2. R16-C + MCE is slightly stronger
than R64 + MCE on the ranking metrics, although not on correct-D NLL.

### CoreDev-2511 Summary

| Variant | Free acc | Soft acc | Free trigger | Soft trigger | Free parse | Soft parse | Free mean tok | Soft mean tok | Free hit-512 | Soft hit-512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Golden, no Stage2 MCE | 37.04% | **37.92%** | 29.79% | 40.98% | 99.84% | 99.48% | 72.77 | 69.56 | 57 | 46 |
| R16-C, no Stage2 MCE | **37.57%** | 37.11% | 38.87% | 50.62% | 99.76% | 99.68% | 72.24 | 66.65 | **51** | **42** |
| R64 + MCE | 34.71% | 33.43% | **47.83%** | **57.55%** | 97.69% | 98.41% | 94.38 | 73.66 | 153 | 62 |
| R16-C + MCE | 31.53% | 31.87% | 43.37% | 51.37% | 96.02% | 97.33% | 103.04 | **64.88** | 258 | 71 |

Relative to R16-C without Matrix-CE, R16-C + MCE loses `6.04/5.24` accuracy
points in free/softforce. It also underperforms R64 + MCE by `3.18/1.55`
points despite stronger internal D ranking.

### Per-Benchmark Accuracy

#### Free

| Variant | VStar | HR | BLINK | OCR | MMMU | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|
| Golden | 52.36% | 52.50% | **56.90%** | 25.02% | **35.67%** | **49.67%** | 16.00% |
| R16-C | 52.36% | **56.00%** | 55.95% | **25.91%** | 35.00% | 48.67% | **18.00%** |
| R64 + MCE | 49.74% | 55.50% | 55.71% | 24.26% | 30.33% | 45.00% | 12.00% |
| R16-C + MCE | **52.36%** | 54.50% | 52.86% | 23.14% | 27.67% | 38.33% | 4.80% |

#### Softforce

| Variant | VStar | HR | BLINK | OCR | MMMU | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|
| Golden | 50.79% | **59.50%** | 55.48% | 24.54% | **38.00%** | **49.00%** | **19.00%** |
| R16-C | **51.83%** | 59.00% | **57.38%** | **24.99%** | 29.67% | **49.00%** | 17.60% |
| R64 + MCE | **51.83%** | 54.50% | 54.05% | 24.56% | 27.00% | 42.00% | 10.00% |
| R16-C + MCE | 49.21% | 54.50% | 53.33% | 23.90% | 27.00% | 41.00% | 5.20% |

### Failure Analysis

R16-C + MCE free has `258` hit-512 rows and `100` answer-parser failures,
with zero overlap between the two sets. The long truncations are concentrated
in non-triggered reasoning: non-triggered rows average `145.28` tokens versus
`47.89` for triggered rows. The parse failures are mostly short triggered
continuations that describe a desired follow-up view but never emit a
scoreable final answer. MathVerse is the clearest reasoning regression at
`4.8/5.2%` free/softforce, down from R16-C's `18.0/17.6%`.

The result separates two requirements:

1. Matrix-CE can preserve which D belongs to which question.
2. It does not preserve the base model's reasoning distribution or guarantee
   that a valid tool continuation ends in an answer.

Therefore Matrix-CE remains useful as an auxiliary representation objective,
but it cannot be promoted alone. The next controlled direction is direct/no-
focus original-Qwen reasoning replay while keeping focus rows on the current
tool protocol.

## Result Artifacts

- R64 + MCE checkpoint:
  `outputs/clean_ablation/stage2_matrixce_preservation_golden_1200step_4gpu_20260713_200250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`
- R64 + MCE benchmark:
  `outputs/clean_benchmarks/qwen3_stage2_matrixce_preservation_coredev2511_20260714_100641`
- R16-C + MCE checkpoint:
  `outputs/clean_ablation/stage2_r16c_matrixce_4gpu_20260714_122949/main/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`
- R16-C + MCE benchmark:
  `outputs/clean_benchmarks/qwen3_stage2_r16c_matrixce_coredev2511_20260714_122949`
