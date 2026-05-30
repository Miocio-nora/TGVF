# TGVF Evaluation v0

Evaluation scripts live in this folder and are intentionally separate from training code.

## Scripts

```text
eval/eval_readout.py
eval/eval_query_sensitivity.py
eval/eval_fvt_distribution.py
eval/eval_tgvf_end2end.py
```

## Primary Questions

1. Does correct `D` improve fresh-context readout over target-only?
2. Is correct `D` better than wrong `D` from the same image?
3. Are generated FVTs on a reasonable Qwen visual-token manifold?
4. Does end-to-end TGVF improve over no-D, random-D, and wrong-D conditions?


## Default 2k Held-Out Validation Set

Default validation file:

```text
data/tgvf_teacher/generated/runs/teacher_run_val_000001/final/tgvf_teacher_items.accepted_balanced_2k_val.jsonl
```

Default evaluation scale for the suite:

```text
readout:       2000 held-out samples
query:         500 image groups, each with >= 3 targets
distribution:  2000 held-out samples
end2end:       300 forced-target samples by default
```

Current validation file statistics:

```text
samples:       2000
images:        425
groups >= 3:   420
source mix:    visual_genome 800, textvqa+textocr 600, docvqa 400, chartqa 200
```

Therefore the strict `QUERY_REQUIRE_GROUPS=500` default will fail on this specific 2k file. To run query sensitivity on all available eligible groups from this file, use:

```bash
QUERY_REQUIRE_GROUPS=420 QUERY_MAX_GROUPS=420 \
  eval/run_tgvf_eval_suite.sh outputs/tgvf_fvt/.../checkpoint_step_N.pt query
```

To truly evaluate 500 groups with `>=3` targets each, generate a larger image-disjoint validation file.

## One-Click Eval Suite

Run all four evaluation tasks for one TGVF checkpoint:

```bash
eval/run_tgvf_eval_suite.sh outputs/tgvf_fvt/.../checkpoint_step_N.pt
```

Run only selected tasks:

```bash
eval/run_tgvf_eval_suite.sh outputs/tgvf_fvt/.../checkpoint_step_N.pt readout,query
```

Useful overrides:

```bash
RUN_ID=my_eval_001 \
DEVICE=cuda:0 \
VARIANT=foveal_cross_merger \
END2END_MAX_SAMPLES=500 \
eval/run_tgvf_eval_suite.sh outputs/tgvf_fvt/.../checkpoint_step_N.pt all
```

The suite writes outputs under:

```text
eval_outputs/tgvf_eval_<checkpoint_stem>_<RUN_ID>/
```

FVT caching is enabled by default for repeated readout/query/distribution work:

```text
USE_FVT_CACHE=1
FVT_CACHE_DIR=<OUT_ROOT>/cache/fvt_cache
```

Disable cache if you want every task to recompute capture and FVT generation:

```bash
USE_FVT_CACHE=0 eval/run_tgvf_eval_suite.sh outputs/tgvf_fvt/.../checkpoint_step_N.pt readout
```

Progress bars are enabled by default. Disable them for cleaner batch logs with:

```bash
PROGRESS=0 eval/run_tgvf_eval_suite.sh outputs/tgvf_fvt/.../checkpoint_step_N.pt all
```

Each individual eval script also supports `--no-progress`.

The eval path is currently sample-wise for Qwen capture and forced readout. `--batch-size` is reserved for future tensor batching and does not speed up the current path. The practical speedup is process-level sharding. The suite defaults to four workers on the configured device:

```text
EVAL_WORKERS=4
SHARD_KEY=image
```

When `EVAL_WORKERS>1`, worker stdout/stderr is redirected to per-shard log files and the terminal shows one aggregated progress line only. Shards write progress JSON files under `<OUT_ROOT>/progress/`, and `eval.progress_monitor` renders the combined progress.

After each sharded task succeeds, the suite automatically merges shard outputs back into the task root. For example, readout shards live under:

```text
<OUT_ROOT>/readout/shard_0_of_4/
<OUT_ROOT>/readout/shard_1_of_4/
```

Merged outputs are written to:

```text
<OUT_ROOT>/readout/readout_eval_report.json
<OUT_ROOT>/readout/readout_eval_summary.txt
<OUT_ROOT>/readout/per_sample_results.jsonl
```

The same pattern applies to query, distribution, and end-to-end eval.

This loads multiple Qwen2-VL-2B processes on the same B200. With the current memory footprint, four workers should still use far below B200 memory capacity. Reduce or increase workers with:

```bash
EVAL_WORKERS=8 eval/run_tgvf_eval_suite.sh outputs/tgvf_fvt/.../checkpoint_step_N.pt readout
```

Set `EVAL_WORKERS=1` to restore single-process behavior.

## Readout Evaluation

Teacher-forced fresh-context readout:

```bash
python -m eval.eval_readout \
  --model-path Qwen/Qwen2-VL-2B-Instruct \
  --tgvf-checkpoint outputs/tgvf_fvt/.../checkpoint_step_N.pt \
  --variant foveal_cross_merger \
  --eval-jsonl data/tgvf_teacher/generated/runs/teacher_run_val_000001/final/tgvf_teacher_items.accepted_balanced_2k_val.jsonl \
  --output-dir eval_outputs/readout_debug \
  --max-samples 2000 \
  --device cuda:0
```

Conditions:

```text
correct_D_plus_target
target_only_no_D
random_D_plus_target
wrong_D_same_image_plus_target
wrong_D_different_image_plus_target
```

## Query Sensitivity

Same-image target-specificity matrix:

```bash
python -m eval.eval_query_sensitivity \
  --model-path Qwen/Qwen2-VL-2B-Instruct \
  --tgvf-checkpoint outputs/tgvf_fvt/.../checkpoint_step_N.pt \
  --variant foveal_cross_merger \
  --eval-jsonl data/tgvf_teacher/generated/runs/teacher_run_val_000001/final/tgvf_teacher_items.accepted_balanced_2k_val.jsonl \
  --output-dir eval_outputs/query_sensitivity_debug \
  --min-targets-per-image 3 \
  --max-groups 500 \
  --require-groups 500 \
  --device cuda:0 \
  --save-score-matrices
```

The score matrix uses NLL, so lower is better.

## FVT Distribution

Visual-token manifold statistics:

```bash
python -m eval.eval_fvt_distribution \
  --model-path Qwen/Qwen2-VL-2B-Instruct \
  --tgvf-checkpoint outputs/tgvf_fvt/.../checkpoint_step_N.pt \
  --variant foveal_cross_merger \
  --eval-jsonl data/tgvf_teacher/generated/runs/teacher_run_val_000001/final/tgvf_teacher_items.accepted_balanced_2k_val.jsonl \
  --output-dir eval_outputs/fvt_distribution_debug \
  --max-samples 2000 \
  --device cuda:0 \
  --save-histograms
```

## End-to-End Evaluation

Forced-target mode isolates the trained FVT path:

```bash
python -m eval.eval_tgvf_end2end \
  --model-path Qwen/Qwen2-VL-2B-Instruct \
  --tgvf-checkpoint outputs/tgvf_fvt/.../checkpoint_step_N.pt \
  --variant foveal_cross_merger \
  --eval-jsonl data/tgvf_teacher/generated/runs/teacher_run_val_000001/final/tgvf_teacher_items.accepted_balanced_2k_val.jsonl \
  --output-dir eval_outputs/end2end_forced_debug \
  --max-samples 300 \
  --mode forced \
  --device cuda:0
```

Free-target mode tests deployed behavior:

```bash
python -m eval.eval_tgvf_end2end \
  --model-path Qwen/Qwen2-VL-2B-Instruct \
  --tgvf-checkpoint outputs/tgvf_fvt/.../checkpoint_step_N.pt \
  --variant foveal_cross_merger \
  --eval-jsonl data/tgvf_teacher/generated/runs/teacher_run_val_000001/final/tgvf_teacher_items.accepted_balanced_2k_val.jsonl \
  --output-dir eval_outputs/end2end_free_debug \
  --max-samples 300 \
  --mode free \
  --device cuda:0
```

End-to-end eval keeps the real inference constraint:

```text
single-pass capture -> FVT generation -> bracketed append -> continuation
```

It does not replay the full prompt plus foveation span.


Direct Qwen is included in end-to-end eval by default. Add `--skip-direct-qwen` when you only want foveation conditions.
