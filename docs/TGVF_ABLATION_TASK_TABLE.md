# TGVF Ablation Task Table

Date: 2026-07-05

This document is the working ablation board for the current clean TGVF line.
It records the intended task pipeline and which branches are already complete.
It is not an experiment ledger replacement; concrete run commands, hashes,
paths, and failure notes still belong in `docs/EXPERIMENT_LEDGER.md`.

## Current Golden Setting

For the next ablation wave, the current experimental golden setting is:

| Axis | Golden value |
|---|---|
| Training flow | Stage1 -> Stage1 internal eval -> Stage2 -> Stage2 internal eval -> CoreDev-2511 benchmarks |
| D DeepStack | enabled |
| Matrix CE | enabled, same-image group/readout size `4` |
| Data scale | current clean teacher 50k family |
| Stage2 recipe | current authoritative Stage2 recipe, with D DeepStack enabled |
| External eval | CoreDev-2511 complete 2,511-row manifest |
| Benchmark modes | `tgvf_free`, `tgvf_softforce` |
| Benchmark backend | clean native `tgvf_stage2_qwen3_native` |
| Benchmark runtime | FlashAttention-2, DeepStack `no_block`, post-D `kv_cache`, unified `max_tokens=512` |

Golden benchmark anchor:

| Mode | Overall Acc | Macro Avg | Trigger | Parse | Append | Ledger |
|---|---:|---:|---:|---:|---:|---|
| Golden free | 37.04% | 41.16% | 29.79% | 99.84% | 100.00% | `BENCH-20260703-084300-stage2-d-deepstack-coredev2511` |
| Golden softforce | 37.92% | 42.33% | 40.98% | 99.48% | 100.00% | `BENCH-20260703-084300-stage2-d-deepstack-coredev2511` |

Golden internal anchors:

| Stage | Metric snapshot | Ledger |
|---|---|---|
| Stage1 internal | readout NLL `1.333408`; query top1/top2/MRR `0.77 / 0.915 / 0.869`; readout correct-D beats wrong-same `0.905` | `DIAG-20260702-235026-stage1-d-deepstack-internal-diagnostics` |
| Stage2 internal | query top1/top2/MRR `0.235 / 0.490 / 0.500`; readout wrong-same `0.405`; norm ratio `2.1198` | `DIAG-20260703-083919-stage2-d-deepstack-internal-diagnostics` |

## Standard Task Pipeline

Every ablation branch should follow this pipeline unless explicitly marked as
a diagnostic-only branch:

| Step | Required output |
|---|---|
| 1. Stage1 train | checkpoint, training plan, data hashes, batch identity, W&B/run status |
| 2. Stage1 internal eval | readout, query sensitivity, FVT distribution reports |
| 3. Stage2 train | checkpoint, training plan, Stage1 checkpoint identity, validation losses |
| 4. Stage2 internal eval | same diagnostic family as Stage2 D DeepStack golden |
| 5. CoreDev-2511 free benchmark | complete 2,511 rows, manifest verification, trigger/parse/append metrics |
| 6. CoreDev-2511 softforce benchmark | complete 2,511 rows, manifest verification, trigger/parse/append metrics |
| 7. Decision | promote, reject, or keep as side result/diagnostic |

## Automation Entry

Use `scripts/run_clean_ddeepstack_ablation_pipeline.py` to generate a resumable
clean-native driver for the post-Stage1 pipeline:

```bash
python scripts/run_clean_ddeepstack_ablation_pipeline.py \
  --branch-id mce_size3 \
  --timestamp 20260703_224753 \
  --stage1-checkpoint outputs/clean_training/qwen3_stage1_ddeepstack_size3_8gpu_20260703_201944/stage1_micro3/clean_training_execution/checkpoint_step_2000.pt \
  --steps stage2_smoke,stage2_train,bench_smoke,bench_free,bench_softforce \
  --write-driver
```

The current generated `mce_size3` driver is:
`outputs/clean_pipeline/mce_size3_20260703_224753/run_pipeline.sh`.

## Matrix CE Size Sweep

This sweep changes only the Matrix CE same-image grouping/readout size on top
of the golden D DeepStack setting. All other training and benchmark variables
should remain fixed.

Detailed per-benchmark results for this sweep are recorded in
[TGVF_ABLATION_BENCHMARK_RESULTS.md](TGVF_ABLATION_BENCHMARK_RESULTS.md).
Tables in this section are ordered by size from small to large so the trend is
easier to read. `size 1` is the Matrix-CE-off control.

| Branch | Matrix CE setting | Status | Stage1 | Stage1 internal | Stage2 | Stage2 internal | CoreDev free | CoreDev softforce | Decision / note |
|---|---|---|---|---|---|---|---|---|---|
| `mce_off_size1` | Matrix CE off / size `1` | Done | Done (`QUEUE-20260704-010949-mce-size2-size1-continuation`) | Done | Done | Optional / skipped | 35.84% | 36.17% | No-Matrix-CE control; lower Stage1 retrieval despite lower readout NLL |
| `mce_size2` | enabled, size `2` | Done | Done (`QUEUE-20260704-010949-mce-size2-size1-continuation`) | Done | Done | Optional / skipped | 35.79% | 35.40% | Pairwise same-image contrast degrades Stage1 retrieval and softforce transfer |
| `mce_size3` | enabled, size `3` | Done | Done (`EXP-20260703-201944-stage1-d-deepstack-size3-8gpu`) | Done (`DIAG-20260703-224753-stage1-d-deepstack-size3-internal-diagnostics`) | Done (`PIPE-20260703-224753-mce-size3-post-stage1-clean-driver`) | Optional / skipped | 35.72% | 38.42% | Strong Stage1 internal; best softforce score in the current size sweep |
| `mce_size4_golden` | enabled, size `4` | Done | Done | Done | Done | Done | 37.04% | 37.92% | Current golden experimental baseline |
| `mce_size5` | enabled, size `5` | Done | Done | Done | Done | Optional / skipped | 36.08% | 36.43% | Best Stage1 internal retrieval, but external benchmark underperforms golden |

### Stage1 Internal Sweep Summary

All rows below use the Stage1 internal diagnostics on 200 examples. Readout
NLL is `mean_nll_correct_D`; query metrics are retrieval top-1/top-2/MRR from
the query-sensitivity report.

| Branch | Matrix CE setting | Readout NLL | Query top-1 | Query top-2 | Query MRR | Note |
|---|---|---:|---:|---:|---:|---|
| `mce_off_size1` | Matrix CE off / size `1` | 1.119355 | 0.585 | 0.795 | 0.750333 | NLL alone is misleading; retrieval remains weak |
| `mce_size2` | enabled, size `2` | 1.347021 | 0.555 | 0.800 | 0.738083 | Retrieval drops sharply |
| `mce_size3` | enabled, size `3` | 1.272090 | 0.790 | 0.920 | 0.881250 | Also above golden on retrieval |
| `mce_size4_golden` | enabled, size `4` | 1.333408 | 0.770 | 0.915 | 0.869167 | Golden internal anchor |
| `mce_size5` | enabled, size `5` | 1.301865 | 0.820 | 0.950 | 0.900833 | Best Stage1 retrieval in this sweep |

Interpretation: size `5` and size `3` are healthy at the Stage1 internal
level, while size `2` and Matrix-CE-off lose the retrieval structure that this
diagnostic is meant to expose. The no-Matrix-CE branch has the lowest correct-D
NLL, but its weak query retrieval means that lower readout NLL is not sufficient
evidence of a better TGVF conditioning branch.

### Matrix CE CoreDev-2511 Result Snapshot

| Branch | Free acc | Free trigger | Softforce acc | Softforce trigger | Rows |
|---|---:|---:|---:|---:|---:|
| `mce_off_size1` | 35.84% | 26.48% | 36.17% | 42.73% | 2,511 |
| `mce_size2` | 35.79% | 30.86% | 35.40% | 44.05% | 2,511 |
| `mce_size3` | 35.72% | 29.07% | 38.42% | 45.52% | 2,511 |
| `mce_size4_golden` | 37.04% | 29.79% | 37.92% | 40.98% | 2,511 |
| `mce_size5` | 36.08% | 27.32% | 36.43% | 40.58% | 2,511 |

Per-mode source summary paths for all size-sweep branches are listed in the
source-artifact table of
[TGVF_ABLATION_BENCHMARK_RESULTS.md](TGVF_ABLATION_BENCHMARK_RESULTS.md).

## D DeepStack On/Off Ablation

This ablation is already completed at the external benchmark level. It compares
the current Stage2 line without D DeepStack against the D DeepStack branch while
holding the CoreDev-2511 benchmark identity fixed.

| Branch | D DeepStack | Status | Stage1/Stage2 train | Internal eval | CoreDev free | CoreDev softforce | Trigger free/soft | Decision |
|---|---|---|---|---|---:|---:|---:|---|
| `d_deepstack_off` | off | Done | Done | Reference diagnostics available | 35.53% | 35.98% | 31.22% / 45.24% | Superseded by D DeepStack on |
| `d_deepstack_on_golden` | on | Done | Done | Done | 37.04% | 37.92% | 29.79% / 40.98% | Current golden experimental baseline |

D DeepStack on improves CoreDev-2511 overall by `+1.51` points in free mode
and `+1.94` points in softforce mode over the comparable D-off Stage2 line.
Full per-benchmark results and source paths are recorded in
[TGVF_ABLATION_BENCHMARK_RESULTS.md](TGVF_ABLATION_BENCHMARK_RESULTS.md).

## Resolution Ablation

This ablation keeps the golden D DeepStack and Matrix CE size-4 setting, but
changes the image resolution from the golden `512` to `214`. The intended
comparison is a complete training/eval pipeline rerun, not only a benchmark-time
resolution change, unless explicitly marked as an eval-only diagnostic.

Detailed per-benchmark resolution results are recorded in
[TGVF_ABLATION_BENCHMARK_RESULTS.md](TGVF_ABLATION_BENCHMARK_RESULTS.md).

| Branch | Max image resolution | Status | Stage1 | Stage1 internal | Stage2 | Stage2 internal | CoreDev free | CoreDev softforce | Notes |
|---|---:|---|---|---|---|---|---|---|---|
| `resolution512_golden` | 512 | Done | Done | Done | Done | Done | 37.04% | 37.92% | Current golden experimental baseline |
| `resolution1024` | 1024 | Failed at Stage1 smoke | OOM before train | Not run | Not run | Not run | N/A | N/A | Golden micro-batch-4 Stage1 recipe does not fit at 1024 resolution on the available 180GB-class GPUs |
| `resolution1024_eval_only` | 1024 | Done / diagnostic | Not retrained | Not run | Not retrained | Not run | 39.76% | 39.58% | Eval-only diagnostic. Original@1024 is 32.62%, but the original output is unhealthy: 58.70% hit the 512-token cap and MathVerse hit rate is 96.40%. Requires a healthy output-budget rerun before final interpretation |
| `resolution1024_health2048_coredev350` | 1024 | Done / diagnostic | Not retrained | Not run | Not retrained | Not run | 47.41% | 48.54% | CoreDev-350 diagnostic with `max_answer_tokens/max_tokens=2048`. Original is 48.00%, but still unhealthy: 25.14% hit the 2048-token cap, including MathVerse 30/50 and MMMU-Pro 25/50. TGVF free/softforce both have 0/350 hit>=2048 |
| `original_health_params_coredev70` | 256/384/512 + 1024/nores max4096 | Done / diagnostic | Not retrained | Not run | Not retrained | Not run | N/A | N/A | Original/TGVF high-budget diagnostics on balanced CoreDev-70. Original remains output-unhealthy at res1024/nores max4096 with 14/70 and 15/70 hit>=4096. TGVF res1024 has 0/70 hit>=4096 and no malformed rows; TGVF nores also has 0/70 hit>=4096 but is not clean because HR triggered focus append OOM causes 5-6 malformed rows |
| `resolution214` | 214 | Done | Done | Done | Done | Done | 30.82% | 30.73% | Original@214 is 23.96%; lowering resolution hurts all three methods versus 512, but TGVF still beats original@214 by about +6.8 points |

## Data Scale Ablation

This curve should use the golden training/eval recipe and vary only the amount
of accepted teacher data. The 50k point is the current golden family. The 75k
and 100k points are placeholders until those data are generated.

| Branch | Data scale | Data status | Training status | Stage1 internal | Stage2 internal | CoreDev free | CoreDev softforce | Notes |
|---|---:|---|---|---|---|---:|---:|---|
| `data25k` | 25k | Generated deterministic subset | Done | Done | Optional / skipped | 36.27% | 35.45% | Stage2 train rows `25000`; Stage1 matched focus train rows `21323`; free is `-0.78` vs golden, softforce is `-2.48` |
| `data50k_golden` | 50k | Available | Done | Done | Done | 37.04% | 37.92% | Current golden point; Stage1 train rows are `39998`, Stage2 train rows are `46883` after filtering/splitting |
| `data75k` | 75k | Not generated | Blocked | Blocked | Blocked | TODO | TODO | Need data generation before training |
| `data100k` | 100k | Not generated | Blocked | Blocked | Blocked | TODO | TODO | Need data generation before training |

For the curve, report at minimum:

- Stage1 query top1/top2/MRR and readout wrong-same beat rate.
- Stage2 query top1/top2/MRR and readout wrong-same beat rate.
- CoreDev-2511 free/softforce overall and macro accuracy.
- Trigger, parse, append, malformed rates.
- Training cost: wall time, GPU count, global batch, optimizer steps.

## Global Batch Ablation

This ablation keeps the golden D DeepStack, Matrix CE size-4, 50k data, and
512-resolution recipe fixed, but changes the effective global batch. The first
branch uses the same sample-slot budget as golden by halving optimizer steps
when doubling global batch. Micro-batch size stays `4` so Matrix CE group size
does not change.

| Branch | Stage1 batch / steps | Stage2 batch / steps | Sample budget | Status | Stage1 internal | CoreDev free | CoreDev softforce | Notes |
|---|---|---|---|---|---|---:|---:|---|
| `gbs32_128_golden` | `32 / 2000` | `128 / 1200` | Anchor | Done | Done | 37.04% | 37.92% | Current golden experimental baseline |
| `bs2x_gbs64_256_same_samples` | `64 / 1000` | `256 / 600` | Same as golden | Done | Top1 82.5%, top2 97.5%, readout NLL 1.2766 | 35.13% | 36.09% | Worse than golden despite stronger Stage1 internal; trigger rises to 39.43% free / 58.18% softforce |
| `bs2x_stage1_stage2_gbs128_normal` | `64 / 1000` | `128 / 1200` | Same as golden for Stage2 | Done | Same Stage1 as bs2x: top1 82.5%, top2 97.5%, readout NLL 1.2766 | 35.29% | 36.61% | Better than the bs2x large-Stage2-batch branch, but still below golden by `-1.75` free / `-1.31` softforce overall points |

Detailed per-benchmark results for `bs2x_stage1_stage2_gbs128_normal` are
recorded in
[TGVF_ABLATION_BENCHMARK_RESULTS.md](TGVF_ABLATION_BENCHMARK_RESULTS.md).

## Immediate Queue

| Priority | Task | Why |
|---:|---|---|
| P1 | Review Matrix CE size sweep and choose whether to promote size `3`, size `4`, or size `5` as the next training default | Size `5` has best Stage1 internal retrieval, size `3` has best current softforce score, and size `4` remains the golden anchor |
| P1 | Decide whether to run complete CoreDev-2511 at `resolution=1024` / `max_tokens=2048` | CoreDev-350 shows original remains output-unhealthy even at 2048, while TGVF stays healthy. A complete CoreDev-2511 rerun is only useful if we want final high-resolution reporting rather than a diagnostic conclusion |
| P3 | Generate 75k/100k teacher data | Required before upper data-scale curve points can start |
