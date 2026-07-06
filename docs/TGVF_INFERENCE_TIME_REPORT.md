# TGVF Inference Time Report

Date: 2026-07-06

This document defines how we should measure and report TGVF inference-time
cost, especially the cost of a focus-triggered `reencode` path. It also records
the first small quantitative timing run used to estimate the order of
magnitude.

## Goal

The main question is:

> If TGVF actually re-encodes visual evidence after a focus decision, how much
> latency does that add compared with our current cached-feature TGVF path?

The answer should be reported at two levels:

- per triggered focus call, because reencode cost only exists when focus is
  actually triggered;
- averaged over all evaluated rows, because user-facing latency depends on the
  trigger rate.

## Definitions

`current no-reencode TGVF` means the current golden clean TGVF inference path:
the model emits a focus action, then D visual tokens are produced from cached
original-image visual features and TGVF adapters. It does not re-run the image
encoder for the focused target.

`reencode TGVF` means an ablation or implementation that, after focus is
triggered, performs an additional visual encoding step for the target evidence
before continuing answer generation. The exact implementation must be named in
the timing run, because crop re-encoding, full-image re-encoding, and
DeepStack-compatible encoder re-encoding are different latency surfaces.

`triggered rows` are rows where a focus action is emitted and accepted by the
runner. `all rows` include both triggered and non-triggered rows. Both views are
required.

## Primary Comparisons

The reencode cost should be compared against three baselines.

| Comparison | What it answers | Required control |
|---|---|---|
| `reencode TGVF` vs `current no-reencode TGVF` | How much cost we save by not re-running visual encoding after focus | Same rows, same checkpoint, same prompt mode, same token budget, same attention backend |
| `TGVF end-to-end` vs `original end-to-end` | What the user-facing latency overhead is compared with direct answering | Same rows, same max image resolution, same generation budget, same scoring policy |
| `reencode phase` vs `initial image encode/prefill` | Whether reencode is roughly as expensive as encoding the image the first time | Same image resolution/crop policy and same GPU/runtime environment |

The first comparison is the most important method comparison. The second is the
best product-facing latency comparison. The third explains the mechanism.

## Required Metrics

| Metric | Definition | Reported on |
|---|---|---|
| `end2end_sec_per_sample` | Total model-run wall time per row, excluding offline scoring unless explicitly stated | all rows |
| `focus_capture_sec` | Time spent generating or forcing the focus action before D append | all rows, plus triggered rows |
| `append_sec_no_reencode` | Time for current D-token construction and append from cached features | triggered rows |
| `reencode_sec_per_trigger` | Time for additional visual re-encoding after focus | triggered rows |
| `post_d_continue_sec` | Time for answer continuation after D append/reencode | triggered rows |
| `delta_sec_per_trigger` | `reencode_sec_per_trigger - append_sec_no_reencode` | triggered rows |
| `trigger_weighted_overhead_sec` | `trigger_rate * delta_sec_per_trigger` | all rows |
| `output_tokens_per_sec` | Generated output tokens divided by generation time | all rows, optional |
| `parse_rate` | Answer parse success rate | all rows |
| `trigger_rate` | Focus trigger rate | all rows |

Do not compare raw end-to-end latency without also reporting output token count.
Longer answers can dominate latency and hide the visual-side cost.

## Timing Breakdown

For each run, record the following phases when available:

| Phase | Description |
|---|---|
| `preprocess_sec` | CPU-side prompt/image preprocessing |
| `h2d_sec` | Tensor transfer to GPU |
| `first_pass_prefill_sec` | Initial prompt and image prefill before focus/action generation |
| `focus_generate_sec` | Focus-action generation time |
| `focus_parse_sec` | Focus-action parsing and validation |
| `d_build_or_append_sec` | Current cached-feature D construction and append |
| `reencode_preprocess_sec` | Target crop or reencode input preparation |
| `reencode_vision_sec` | Additional visual encoder forward |
| `reencode_merge_adapter_sec` | Merger/adapter projection for re-encoded visual features |
| `post_d_continue_sec` | Continuation generation after D is available |
| `decode_sec` | Token decoding to text |
| `batch_wall_sec` | Batch wall-clock time |

The current original runner already records coarse preprocessing, H2D,
generation, decode, and batch wall timing. The TGVF native path currently
records end-to-end wall time and rich append metadata, but a clean reencode
timing run should add explicit phase timers around focus capture, D append,
reencode, and post-D continuation before we claim detailed phase numbers.

## Recommended Timing Test

Use a deterministic CoreDev-2511 timing subset rather than the complete
2,511-row benchmark for timing estimates. The default should be small because
single-GPU per-sample timing is slow: a stopped 140-row run completed only the
`original` method and that stage alone took about `21.4 min`.

| Test | Rows | Purpose |
|---|---:|---|
| Cheap timing estimate | 7 rows, 1 per benchmark | First order-of-magnitude estimate across all benchmark families |
| Wider timing estimate | 35 to 70 rows, 5 to 10 per benchmark | Reduce idiosyncratic row variance if needed |
| Formal timing estimate | 140 rows, 20 per benchmark | Use only when we need a more stable table and can afford the runtime |

Run identity should match the current clean default unless the experiment is
explicitly labeled otherwise:

| Axis | Default value |
|---|---|
| Manifest family | CoreDev-2511 / `core_balanced_dev_2511_seed20260625` |
| Backend | clean native `tgvf_stage2_qwen3_native` |
| Attention | FlashAttention-2 |
| Max image resolution | 512 |
| Generation budget | unified `max_tokens=512` |
| Post-D continuation | `kv_cache` |
| DeepStack eval scope | `no_block` |
| Modes | original, TGVF free, TGVF softforce |

For timing stability, run a small warmup first and exclude warmup rows from the
reported aggregate. If we care about per-sample latency, use one GPU and a
fixed batch size. If we care about throughput, use the dynamic multi-GPU queue
and report rows/sec separately from per-row phase timing.

## Measured CoreDev-7 Result

Run:
`outputs/clean_timing/tgvf_inference_time_coredev7_20260706_112106`.

Identity:

- Stage2 checkpoint:
  `outputs/clean_training/qwen3_stage2_ddeepstack_norm01_from_stage1_ddeepstack_8gpu_20260703_005210/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
- Manifest:
  CoreDev-2511 / `core_balanced_dev_2511_seed20260625`, hash
  `a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`.
- Sample rule:
  first `1` manifest row per benchmark, `7` measured rows total; warmup `1`
  row per method excluded from aggregates.
- Runtime:
  single GPU `cuda:0`, FlashAttention-2, resolution `512`, unified
  `max_tokens=512`, DeepStack `no_block`, post-D continuation `kv_cache`.
- Timing variants:
  `cached_prewarm` records a synthetic original-image vision-feature prewarm
  separately and excludes it from row wall time; `reencode` forces a fresh
  post-focus Qwen3 vision tap before D construction.

Overall timing summary:

| Method | Rows | Trigger rate | Mean end2end sec | P50 end2end sec | P90 end2end sec | Mean output tokens | Parse rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 7 | n/a | 8.509 | 12.950 | 13.063 | 333.4 | n/a |
| cached force | 7 | 100.0% | 3.011 | 2.653 | 4.586 | 45.0 | 100.0% |
| reencode force | 7 | 100.0% | 3.126 | 2.959 | 4.796 | 45.0 | 100.0% |
| cached free | 7 | 42.9% | 5.212 | 3.138 | 10.643 | 78.6 | 85.7% |
| reencode free | 7 | 42.9% | 5.172 | 3.073 | 10.686 | 78.6 | 85.7% |
| cached softforce | 7 | 28.6% | 3.498 | 3.075 | 5.435 | 43.3 | 100.0% |
| reencode softforce | 7 | 28.6% | 3.477 | 3.172 | 5.458 | 43.3 | 100.0% |

Per-token timing summary:

These are aggregate ratios over the measured rows, not averages of per-row
ratios. `E2E output tok/s` divides reported output tokens by row wall time.
For TGVF, `action tok/s` divides action tokens by focus-capture time, and
`answer tok/s` divides answer tokens on triggered rows by post-D continuation
time.

| Method | Output tokens | E2E output tok/s | E2E sec/token | Action tok/s | Action sec/token | Answer tok/s | Answer sec/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 2334 | 39.186 | 0.026 | n/a | n/a | n/a | n/a |
| cached force | 315 | 14.944 | 0.067 | 21.580 | 0.046 | 24.600 | 0.041 |
| reencode force | 315 | 14.393 | 0.069 | 21.632 | 0.046 | 23.611 | 0.042 |
| cached free | 550 | 15.076 | 0.066 | 16.724 | 0.060 | 24.867 | 0.040 |
| reencode free | 550 | 15.193 | 0.066 | 16.756 | 0.060 | 24.801 | 0.040 |
| cached softforce | 303 | 12.373 | 0.081 | 17.175 | 0.058 | 24.049 | 0.042 |
| reencode softforce | 303 | 12.451 | 0.080 | 17.156 | 0.058 | 23.113 | 0.043 |

Triggered-row phase summary:

| Method | Triggered rows | Focus capture sec | Cached prewarm sec | Reencode tap sec | D build sec | D append sec | Post-D continue sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| cached force | 7 | 0.966 | 0.054 | n/a | 0.008 | 0.114 | 1.829 |
| reencode force | 7 | 0.964 | n/a | 0.064 | 0.053 | 0.122 | 1.906 |
| cached free | 3 | 3.631 | 0.027 | n/a | 0.006 | 0.060 | 5.509 |
| reencode free | 3 | 3.634 | n/a | 0.027 | 0.030 | 0.060 | 5.524 |
| cached softforce | 2 | 3.817 | 0.026 | n/a | 0.006 | 0.054 | 1.746 |
| reencode softforce | 2 | 3.772 | n/a | 0.025 | 0.028 | 0.055 | 1.817 |

Compact per-benchmark force timing, one selected row per benchmark:

| Benchmark | Original sec | Cached force sec | Reencode force sec | Reencode tap sec |
|---|---:|---:|---:|---:|
| vstar_bench | 4.093 | 2.442 | 2.556 | 0.059 |
| hr_bench_4k | 1.668 | 1.983 | 1.987 | 0.066 |
| blink | 13.065 | 2.653 | 2.959 | 0.208 |
| ocrbench_v2 | 1.713 | 2.015 | 1.714 | 0.042 |
| mmmu_pro | 12.950 | 2.809 | 3.051 | 0.024 |
| mathvista | 13.011 | 4.577 | 4.876 | 0.027 |
| mathverse | 13.062 | 4.599 | 4.742 | 0.022 |

Main reading:

- The fresh post-focus vision tap is small in this run: `0.064s` per trigger
  in force mode on average, with P50 `0.042s` and P90 `0.123s`.
- The cached-feature prewarm probe is the same order of magnitude:
  `0.054s` mean in force mode.
- End-to-end latency is dominated by language generation phases: focus capture
  and post-D continuation. The free-mode triggered rows spend about `5.5s` in
  post-D continuation, while the reencode tap is only about `0.027s`.
- Per-token timing shows the same thing: original direct generation is about
  `39.2` output tok/s on this subset, while TGVF post-D answer continuation is
  about `23-25` answer tok/s and action generation is about `17-22` action
  tok/s.
- Observed cached-vs-reencode end-to-end deltas are tiny relative to generation
  variance on this 7-row subset: force `+0.115s`, free `-0.040s`, softforce
  `-0.022s` mean row wall time.
- Therefore, for current settings, true reencode overhead looks like tens of
  milliseconds to about one tenth of a second per trigger, not a main latency
  bottleneck. A larger subset would refine the distribution, but this is enough
  to rule out a multi-second reencode penalty.

## Formal Timing Extension

There are no unfilled measured tables in this report. A larger future timing
run should add a new measured section instead of leaving placeholders in the
main text.

## Interpretation Rules

Use triggered-row timing to discuss the pure cost of reencoding. Use all-row
timing to discuss practical latency, because free mode may trigger much less
often than softforce.

If reencode increases `reencode_sec_per_trigger` but does not increase
`end2end_sec_per_sample` much, the reason is probably low trigger rate or
shorter post-D answers. If end-to-end latency increases but reencode phase is
small, the cause is likely longer generation, longer focus capture, batch queue
effects, or preprocessing.

A healthy report should separate:

- visual-side overhead: D append or reencode;
- language-side overhead: focus generation and post-D continuation;
- system overhead: preprocessing, H2D transfer, batching, and queue waiting.

Do not describe a timing result as a reencode penalty unless the phase timer
shows that the reencode phase, not generation length or scheduling, is the
dominant contributor.

Phase timers in this report use CUDA synchronization around local regions, but
some timers are nested. Treat them as local probes rather than an exactly
additive decomposition of row wall time.

## Open Implementation Work

The current measured run uses isolated timing instrumentation in
`scripts/run_tgvf_inference_time_test.py`, without changing the normal clean
benchmark runtime. If we later need a formal timing test, add or verify
phase-level timers in the native TGVF runtime:

- timer around focus capture/generation;
- timer around focus parsing;
- timer around cached-feature D construction and append;
- timer around any reencode preprocessing and vision encoder forward;
- timer around post-D continuation generation;
- generated-token counts for focus and continuation separately.

After any larger timing run, record exact run identity and output paths in
`docs/EXPERIMENT_LEDGER.md`, then fill the report tables above with measured
values.
