# Stage2 Matrix-CE Preservation

Status: implemented as an opt-in experiment. The authoritative Golden Stage2
default remains unchanged until a controlled pilot passes internal and external
evaluation.

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
