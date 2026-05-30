# TGVF FVT Evaluation Protocol

This document records the current teacher-guide FVT evaluation suite. It is
separate from benchmark evaluation in `docs/TGVF_BENCHMARK_EVALUATION.md`.

The goal here is not to score VStar/MMMU directly. The goal is to test whether a
trained TGVF module produces focused visual tokens `D` that are useful,
target-specific, and still live on the Qwen visual-token manifold.

## Entry Point

Main script:

```text
eval/run_tgvf_eval_suite.sh <checkpoint.pt> all
```

`all` expands to:

```text
readout,query,distribution,end2end
```

The suite calls:

```text
eval/eval_readout.py
eval/eval_query_sensitivity.py
eval/eval_fvt_distribution.py
eval/eval_tgvf_end2end.py
eval/merge_results.py
eval/upload_eval_wandb.py
```

The current validation JSONL is:

```text
data/tgvf_teacher/generated/runs/teacher_run_val_visual_cue_v1_000001/final/tgvf_teacher_items.accepted.jsonl
```

## Current V2 Run

Run root:

```text
outputs/tgvf_fvt/20k_v2_matrixce_bs6_streaming_realgrid_20260529_103941
```

Checkpoints:

```text
tgvf_v2_vpt_gating_matrixce/checkpoint_step_4254.pt
tgvf_v2_cross_attention_matrixce/checkpoint_step_4254.pt
tgvf_v2_bidirectional_matrixce/checkpoint_step_4254.pt
```

Evaluation output root for each variant:

```text
<variant>/eval_checkpoint_step_4254/
```

Current evaluation settings:

```text
MODEL_PATH=Qwen/Qwen2-VL-2B-Instruct
EVAL_JSONL=data/tgvf_teacher/generated/runs/teacher_run_val_visual_cue_v1_000001/final/tgvf_teacher_items.accepted.jsonl
DTYPE=bfloat16
ATTN_IMPL=flash_attention_2
NUM_FVT=none
SEED=20260525
READOUT_MAX_SAMPLES=2007
DISTRIBUTION_MAX_SAMPLES=2007
QUERY_MAX_GROUPS=419
QUERY_REQUIRE_GROUPS=419
END2END_MAX_SAMPLES=300
EVAL_WORKERS=4
WANDB_LOG_EVAL=0
```

For v2, `NUM_FVT=none` means the final FVT count keeps the real source image
grid token count rather than compressing to a fixed `M`. The trainable module
conditions pre-merge visual tokens, then the eval path calls the frozen Qwen
visual merger to produce `D`. The readout and end-to-end paths use the source
image `image_grid_thw`, Qwen mRoPE image positions, and image modality token ids
instead of a fake fixed grid.

## Compared Conditions

The suite compares these conditions around the same target/evidence item:

```text
correct_D_plus_target
target_only_no_D
random_D_plus_target
wrong_D_same_image_plus_target
wrong_D_different_image_plus_target
```

Interpretation:

- `correct_D_plus_target`: the target text plus the FVT `D` generated from the
  matching image and matching target.
- `target_only_no_D`: target text only. This checks whether `D` adds visual
  evidence beyond text.
- `random_D_plus_target`: random visual-token-shaped `D`. This checks whether
  improvement comes from real evidence rather than just adding image-like tokens.
- `wrong_D_same_image_plus_target`: `D` from another target on the same image.
  This is the most important specificity control; it tests whether `D` is
  target-conditioned rather than merely image-conditioned.
- `wrong_D_different_image_plus_target`: `D` from a different image. This is the
  basic mismatch control.

## Readout

Readout measures forced evidence-description likelihood. For each validation
item, the evaluator asks the frozen Qwen LLM to score the teacher
`evidence_description` under the different `D` conditions.

The important metrics are:

```text
mean_delta_correct_vs_target_only
mean_delta_correct_vs_random
mean_delta_correct_vs_wrong_same
mean_delta_correct_vs_wrong_diff
pct_correct_D_beats_target_only
pct_correct_D_beats_random
pct_correct_D_beats_wrong_same
pct_correct_D_beats_wrong_diff
```

The deltas are defined as:

```text
NLL(control) - NLL(correct_D)
```

So larger positive values mean the correct `D` lowers NLL and helps the frozen
LLM read the intended visual evidence.

Primary readout focus:

- `mean_delta_correct_vs_target_only`: does `D` add information beyond target
  text?
- `mean_delta_correct_vs_wrong_same`: is `D` specific to the selected target
  within the same image?
- `pct_correct_D_beats_*`: how often the improvement is positive, not just the
  mean effect.

Output:

```text
readout/readout_eval_report.json
readout/readout_rows.jsonl
```

## Query Sensitivity

Query sensitivity is the key eval for Matrix CE and target conditioning. It
groups validation items by image, keeps images with at least three targets, and
builds a score matrix:

```text
rows = target/evidence query
cols = D generated from candidate targets on the same image
```

For a good target-conditioned module, the diagonal should be best: each query
should score lowest NLL with its own `D`, not with another target's `D`.

Important metrics:

```text
retrieval_top1
retrieval_top2
mrr
mean_diagonal_gap
median_diagonal_gap
```

Primary query focus:

- `retrieval_top1`: strict same-image target retrieval accuracy.
- `mrr`: ranking quality beyond top-1.
- `mean_diagonal_gap`: margin between the correct `D` and the best wrong same
  image `D`.

Output:

```text
query_sensitivity/query_sensitivity_report.json
query_sensitivity/query_rows.jsonl
query_sensitivity/score_matrices/
```

## FVT Distribution

Distribution eval checks whether generated `D` looks numerically compatible with
Qwen visual tokens after merge. It does not prove usefulness; it catches
manifold drift, scale mismatch, NaNs/Infs, and token collapse.

Important metrics:

```text
avg_manifold_loss
median_manifold_loss
norm_ratio_D_to_Vmerge
finite_rate
collapse_near_identical_rate
collapse_warning
```

Primary distribution focus:

- `finite_rate` should be 1.0.
- `collapse_warning` should be false.
- `norm_ratio_D_to_Vmerge` should not be extreme.
- `avg_manifold_loss` is useful mainly for comparing variants trained/evaluated
  under the same setup.

Output:

```text
fvt_distribution/fvt_distribution_report.json
fvt_distribution/fvt_distribution_rows.jsonl
```

## End-To-End Forced

End-to-end forced eval runs the teacher-guide question/answer path with a forced
target. It is diagnostic, not the main benchmark score. The frozen LLM is asked
to answer the teacher item under several conditions:

```text
direct_qwen
foveation_no_D
random_D
wrong_D
correct_TGVF_D
```

Important metrics:

```text
exact_match
substring_match
token_f1
char_f1
completion_rate
```

Highlighted comparisons:

```text
correct_D_improvement_over_no_D
correct_D_improvement_over_wrong_D
```

End-to-end is noisier than readout/query because it includes answer generation,
prompt effects, and frozen-LLM behavior. We use it as a sanity check after
readout/query/distribution have shown that `D` is useful and target-specific.

Output:

```text
end2end_forced/end2end_eval_report.json
end2end_forced/end2end_rows.jsonl
```

## What We Compare First

For the current v2 experiment, the main comparison is across the three v2
conditioning modules at the same checkpoint step and same validation JSONL:

```text
tgvf_v2_vpt_gating_matrixce
tgvf_v2_cross_attention_matrixce
tgvf_v2_bidirectional_matrixce
```

The primary decision metrics are:

```text
readout.mean_delta_correct_vs_target_only
readout.mean_delta_correct_vs_wrong_same
readout.pct_correct_D_beats_wrong_same
query.retrieval_top1
query.mrr
query.mean_diagonal_gap
distribution.finite_rate
distribution.collapse_warning
end2end_forced.correct_D_improvement_over_no_D
end2end_forced.correct_D_improvement_over_wrong_D
```

The most important evidence of success is:

```text
correct D beats target-only
correct D beats wrong same-image D
query diagonal retrieval is high
distribution has no numerical collapse
```

We should also compare against the older v1 fixed-M runs, especially the m128
ablation outputs:

```text
outputs/tgvf_fvt/20k_v1_m128_ablation_20260528_155423/*/eval_checkpoint_step_4254
```

That comparison answers a different question: whether v2 real-grid re-encoding
is better than v1 fixed-M token compression under the same teacher-guide
validation distribution.

## Practical Notes

The eval suite is sample-wise and uses shard workers for speed. v1 m128 runs
used `EVAL_WORKERS=10`. Current v2 real-grid runs are heavier because `D`
contains the real image-grid number of tokens, but the trainable module no
longer includes a learned merger. Use worker count according to memory pressure
and cache reuse.

The cache lives under:

```text
eval_checkpoint_step_4254/cache/fvt_cache/
```

Readout, query sensitivity, and distribution reuse cached `D` tensors when
possible. If the FVT implementation or real-grid behavior changes, remove the
cache or use a new output root.
