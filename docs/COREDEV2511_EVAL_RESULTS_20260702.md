# CoreDev-2511 Evaluation Results

Date: 2026-07-02

D-DeepStack addendum: 2026-07-03

Size-5 Matrix CE sampler ablation addendum: 2026-07-03

Scope: complete `core_balanced_dev_2511_seed20260625` / CoreDev-2511 manifest, 2,511 samples.

Methods shown:

- `Original`: Qwen3 original baseline with max answer tokens 512.
- `Stage2 Free`: Stage2 checkpoint, `tgvf_free`.
- `Stage2 Soft`: Stage2 checkpoint, `tgvf_softforce`.
- `Stage3 Last Free`: Stage3 last checkpoint, step200, `tgvf_free`.
- `Stage3 Last Soft`: Stage3 last checkpoint, step200, `tgvf_softforce`.
- `D-DeepStack Stage2 Free`: Stage2 D-DeepStack checkpoint, `tgvf_free`.
- `D-DeepStack Stage2 Soft`: Stage2 D-DeepStack checkpoint,
  `tgvf_softforce`.
- `D-DeepStack Size5 Free`: Stage2 D-DeepStack checkpoint derived from the
  size-5 Matrix CE sampler ablation, `tgvf_free`.
- `D-DeepStack Size5 Soft`: same size-5 Matrix CE sampler ablation,
  `tgvf_softforce`.

All TGVF rows use the clean native backend, FlashAttention-2, DeepStack
`no_block`, post-D `kv_cache`, scoring backend `auto`, max image resolution
512, and unified `max_tokens=512`. Deltas are percentage points versus
`Original`.

## Overall Summary

| Method | Overall | Delta vs Original | Macro Avg | Macro Delta | Trigger | Parse | Append |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 30.86% | +0.00 | 36.14% | +0.00 | - | 94.15% | - |
| Stage2 Free | 35.53% | +4.68 | 39.67% | +3.53 | 31.22% | 99.56% | 100.00% |
| Stage2 Soft | 35.98% | +5.12 | 39.90% | +3.77 | 45.24% | 99.64% | 100.00% |
| Stage3 Last Free | 36.40% | +5.54 | 40.57% | +4.44 | 30.78% | 99.60% | 100.00% |
| Stage3 Last Soft | 36.06% | +5.20 | 40.26% | +4.12 | 45.92% | 99.52% | 100.00% |
| D-DeepStack Stage2 Free | 37.04% | +6.19 | 41.16% | +5.02 | 29.79% | 99.84% | 100.00% |
| D-DeepStack Stage2 Soft | 37.92% | +7.07 | 42.33% | +6.19 | 40.98% | 99.48% | 100.00% |

Best overall and macro result in this table is `D-DeepStack Stage2 Soft`:
37.92% overall and 42.33% macro average.

## D-DeepStack Stage2 Update

The D-DeepStack Stage2 checkpoint was trained as a named ablation on top of the
authoritative Stage2 recipe. The CoreDev-2511 evaluation below uses the same
manifest, scoring backend, FlashAttention-2, DeepStack `no_block`, post-D
`kv_cache`, max image resolution 512, and unified `max_tokens=512` setting as
the current clean benchmark default. Deltas are percentage points versus
`Original`. Trigger is the focus-valid trigger rate for TGVF modes.

| Benchmark | n | Original Acc | D-Deep Free Acc | D-Deep Free Delta | D-Deep Free Trigger | D-Deep Soft Acc | D-Deep Soft Delta | D-Deep Soft Trigger |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 2511 | 30.86% | 37.04% | +6.19 | 29.79% | 37.92% | +7.07 | 40.98% |
| VStar | 191 | 53.93% | 52.36% | -1.57 | 11.52% | 50.79% | -3.14 | 58.64% |
| HR | 200 | 52.00% | 52.50% | +0.50 | 40.00% | 59.50% | +7.50 | 72.50% |
| BLINK | 420 | 46.43% | 56.90% | +10.48 | 7.86% | 55.48% | +9.05 | 23.33% |
| OCRBench-v2 | 600 | 20.46% | 25.02% | +4.56 | 22.83% | 24.54% | +4.08 | 36.17% |
| MMMU-Pro | 300 | 34.33% | 35.67% | +1.33 | 29.67% | 38.00% | +3.67 | 33.00% |
| MathVista | 300 | 41.00% | 49.67% | +8.67 | 20.33% | 49.00% | +8.00 | 23.67% |
| MathVerse | 500 | 4.80% | 16.00% | +11.20 | 65.20% | 19.00% | +14.20 | 57.40% |

Compared with the current original baseline, D-DeepStack Stage2 improves
CoreDev-2511 overall accuracy by `+6.19` points in free mode and `+7.07`
points in softforce mode. Excluding VStar, the weighted overall is `35.78%`
for D-DeepStack free and `36.86%` for D-DeepStack softforce over the remaining
2,320 rows.

## Size-5 Matrix CE Sampler Ablation

This is a named ablation, not a replacement for the current D-DeepStack Stage2
default result. The changed variable is the size-5 Matrix CE sampler branch
used to produce the Stage1 source checkpoint and the derived Stage2
D-DeepStack checkpoint. The CoreDev-2511 evaluation identity is otherwise held
fixed: same 2,511-row manifest, same sample-id order, clean native backend,
FlashAttention-2, DeepStack `no_block`, D-DeepStack enabled, post-D `kv_cache`,
scoring backend `auto`, max image resolution 512, and unified
`max_tokens=512`.

| Mode | n | Acc | Delta vs Original | Delta vs prior D-Deep | Macro Avg | Trigger | Parse | Append | Malformed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-DeepStack Size5 Free | 2511 | 36.08% | +5.22 | -0.96 | 40.41% | 27.32% | 99.60% | 100.00% | 0.00% |
| D-DeepStack Size5 Soft | 2511 | 36.43% | +5.57 | -1.50 | 40.55% | 40.58% | 99.52% | 100.00% | 0.00% |

The size-5 ablation remains above `Original`, but underperforms the prior
D-DeepStack Stage2 checkpoint on CoreDev-2511 overall accuracy in both modes.
Macro average also drops versus prior D-DeepStack: `-0.75` points for free and
`-1.78` points for softforce.

| Benchmark | n | Size5 Free Acc | Delta vs prior D-Deep Free | Size5 Free Trigger | Size5 Soft Acc | Delta vs prior D-Deep Soft | Size5 Soft Trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| VStar | 191 | 49.74% | -2.62 | 7.33% | 48.69% | -2.09 | 62.30% |
| HR | 200 | 55.00% | +2.50 | 38.00% | 57.50% | -2.00 | 71.00% |
| BLINK | 420 | 56.19% | -0.71 | 8.33% | 55.71% | +0.24 | 25.48% |
| OCRBench-v2 | 600 | 24.00% | -1.02 | 22.67% | 24.95% | +0.41 | 38.83% |
| MMMU-Pro | 300 | 35.67% | +0.00 | 24.33% | 34.33% | -3.67 | 28.33% |
| MathVista | 300 | 48.67% | -1.00 | 17.33% | 46.67% | -2.33 | 22.33% |
| MathVerse | 500 | 13.60% | -2.40 | 60.00% | 16.00% | -3.00 | 53.20% |

## Per-Benchmark Results

| Benchmark | Original Acc | Stage2 Free Acc | Stage2 Free Delta | Stage2 Free Trigger | Stage2 Soft Acc | Stage2 Soft Delta | Stage2 Soft Trigger | Stage3 Last Free Acc | Stage3 Last Free Delta | Stage3 Last Free Trigger | Stage3 Last Soft Acc | Stage3 Last Soft Delta | Stage3 Last Soft Trigger |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VStar | 53.93% | 49.21% | -4.71 | 17.28% | 49.21% | -4.71 | 63.35% | 49.74% | -4.19 | 15.71% | 50.79% | -3.14 | 62.30% |
| HR | 52.00% | 54.00% | +2.00 | 38.00% | 51.50% | -0.50 | 73.00% | 54.50% | +2.50 | 37.50% | 54.00% | +2.00 | 75.00% |
| BLINK | 46.43% | 55.00% | +8.57 | 8.81% | 56.19% | +9.76 | 25.71% | 56.43% | +10.00 | 8.33% | 56.19% | +9.76 | 27.14% |
| OCRBench-v2 | 20.46% | 23.88% | +3.42 | 23.83% | 24.41% | +3.94 | 43.67% | 24.15% | +3.69 | 23.83% | 24.23% | +3.76 | 42.33% |
| MMMU-Pro | 34.33% | 34.67% | +0.33 | 31.67% | 33.00% | -1.33 | 34.33% | 35.67% | +1.33 | 31.00% | 34.00% | -0.33 | 34.33% |
| MathVista | 41.00% | 46.33% | +5.33 | 21.00% | 50.00% | +9.00 | 28.67% | 48.33% | +7.33 | 19.33% | 48.00% | +7.00 | 30.67% |
| MathVerse | 4.80% | 14.60% | +9.80 | 67.40% | 15.00% | +10.20 | 62.00% | 15.20% | +10.40 | 67.80% | 14.60% | +9.80 | 64.20% |

## BLINK Radar

This radar chart uses the CoreDev-2511 BLINK subset only: 420 samples, 14
subtasks, 30 samples per subtask. It should not be read as the official BLINK
validation-set score over 1,901 samples.

![BLINK radar on CoreDev-2511 subset](assets/coredev2511_blink_radar_last_step.png)

## BLINK Subtask Values

| Subtask | n | Original | Stage2 Free | Stage2 Soft | Stage3 Last Free | Stage3 Last Soft |
|---|---:|---:|---:|---:|---:|---:|
| Visual Correspondence | 30 | 46.67% | 50.00% | 50.00% | 56.67% | 53.33% |
| Functional Correspondence | 30 | 16.67% | 23.33% | 23.33% | 23.33% | 26.67% |
| Semantic Correspondence | 30 | 40.00% | 46.67% | 36.67% | 46.67% | 40.00% |
| Visual Similarity | 30 | 46.67% | 93.33% | 90.00% | 96.67% | 96.67% |
| IQ Test | 30 | 0.00% | 13.33% | 26.67% | 13.33% | 26.67% |
| Forensic Detection | 30 | 26.67% | 56.67% | 50.00% | 53.33% | 50.00% |
| Counting | 30 | 66.67% | 63.33% | 70.00% | 66.67% | 73.33% |
| Object Localization | 30 | 56.67% | 66.67% | 66.67% | 70.00% | 66.67% |
| Art Style | 30 | 50.00% | 76.67% | 80.00% | 76.67% | 73.33% |
| Jigsaw | 30 | 70.00% | 33.33% | 43.33% | 33.33% | 40.00% |
| Multi-view Reasoning | 30 | 23.33% | 36.67% | 40.00% | 40.00% | 36.67% |
| Spatial Relation | 30 | 86.67% | 80.00% | 86.67% | 80.00% | 86.67% |
| Relative Depth | 30 | 66.67% | 80.00% | 70.00% | 80.00% | 66.67% |
| Relative Reflectance | 30 | 53.33% | 50.00% | 53.33% | 53.33% | 50.00% |

## Result Paths

| Method | Summary |
|---|---|
| Original | `outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_maxans512_20260628_0216_rescored_bboxfix_20260628_035759/merged/summary.json` |
| Stage2 Free | `outputs/clean_benchmarks/qwen3_stage2_norm01_coredev2511_dynamic_maxtok512_flash2_multigridfix_gpu0_7_20260701_235228/tgvf_free/summary.json` |
| Stage2 Soft | `outputs/clean_benchmarks/qwen3_stage2_norm01_coredev2511_dynamic_maxtok512_flash2_multigridfix_gpu0_7_20260702_011343/tgvf_softforce/summary.json` |
| Stage3 Last Free | `outputs/clean_benchmarks/qwen3_stage3_step200_coredev2511_dynamic_maxtok512_flash2_gpu0_7_20260702_023637/tgvf_free/summary.json` |
| Stage3 Last Soft | `outputs/clean_benchmarks/qwen3_stage3_step200_coredev2511_dynamic_maxtok512_flash2_gpu0_7_20260702_023637/tgvf_softforce/summary.json` |
| D-DeepStack Stage2 Free | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu1_7_20260703_084300/tgvf_free/summary.json` |
| D-DeepStack Stage2 Soft | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_7_20260703_084300/tgvf_softforce/summary.json` |
| D-DeepStack Size5 Free | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_size5_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_7_20260703_172325/tgvf_free/summary.json` |
| D-DeepStack Size5 Soft | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_size5_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_7_20260703_172325/tgvf_softforce/summary.json` |
