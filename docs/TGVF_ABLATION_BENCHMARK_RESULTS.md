# TGVF Ablation Benchmark Results

Date: 2026-07-05

This file records detailed external benchmark results for TGVF ablations. The
main ablation board should keep only compact overall summaries and link here
when per-benchmark metrics are needed.

## Shared CoreDev-2511 Evaluation Identity

Unless a section says otherwise, these rows use the same CoreDev-2511 external
evaluation identity:

| Axis | Value |
|---|---|
| Manifest | `core_balanced_dev_2511_seed20260625` |
| Manifest hash | `a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579` |
| Rows | 2,511 |
| Backend | clean native `tgvf_stage2_qwen3_native` |
| Attention | FlashAttention-2 |
| DeepStack eval scope | `no_block` |
| Post-D continuation | `kv_cache` |
| Scoring backend | `auto` |
| Max image resolution | 512 |
| Generation budget | unified `max_tokens=512` |
| Modes | `tgvf_free`, `tgvf_softforce` |

Metric names:

- `Acc`: scored accuracy.
- `Trigger`: focus trigger rate.
- `Parse`: answer parse rate.
- `Focus-valid`: valid focus action rate.
- `Append`: successful D append rate.
- `Malformed`: malformed focus action rate.

## D DeepStack On/Off Ablation

This ablation compares the current Stage2 line without D DeepStack against the
D DeepStack golden branch. Both use original-image DeepStack eval scope
`no_block`; the changed variable is whether D-token DeepStack branch features
are enabled (`d_features_enabled=false` vs `true`).

### Overall Summary

| Branch | D DeepStack | Mode | Rows | Acc | Delta vs off | Macro acc | Parse | Trigger | Append |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `d_deepstack_off` | off | free | 2,511 | 35.53% | anchor | 39.67% | 99.56% | 31.22% | 100.00% |
| `d_deepstack_off` | off | softforce | 2,511 | 35.98% | anchor | 39.90% | 99.64% | 45.24% | 100.00% |
| `d_deepstack_on_golden` | on | free | 2,511 | 37.04% | +1.51 | 41.16% | 99.84% | 29.79% | 100.00% |
| `d_deepstack_on_golden` | on | softforce | 2,511 | 37.92% | +1.94 | 42.33% | 99.48% | 40.98% | 100.00% |

D DeepStack improves CoreDev-2511 overall by `+1.51` points in free mode and
`+1.94` points in softforce mode.

### Accuracy Matrix

Free mode:

| Benchmark | n | D-off free | D-on free | Delta |
|---|---:|---:|---:|---:|
| VStar | 191 | 49.21% | 52.36% | +3.14 |
| HR | 200 | 54.00% | 52.50% | -1.50 |
| BLINK | 420 | 55.00% | 56.90% | +1.90 |
| OCRBench-v2 | 600 | 23.88% | 25.02% | +1.15 |
| MMMU-Pro | 300 | 34.67% | 35.67% | +1.00 |
| MathVista | 300 | 46.33% | 49.67% | +3.33 |
| MathVerse | 500 | 14.60% | 16.00% | +1.40 |

Softforce mode:

| Benchmark | n | D-off softforce | D-on softforce | Delta |
|---|---:|---:|---:|---:|
| VStar | 191 | 49.21% | 50.79% | +1.57 |
| HR | 200 | 51.50% | 59.50% | +8.00 |
| BLINK | 420 | 56.19% | 55.48% | -0.71 |
| OCRBench-v2 | 600 | 24.41% | 24.54% | +0.14 |
| MMMU-Pro | 300 | 33.00% | 38.00% | +5.00 |
| MathVista | 300 | 50.00% | 49.00% | -1.00 |
| MathVerse | 500 | 15.00% | 19.00% | +4.00 |

### Full Per-Benchmark Metrics

| Branch | Mode | Benchmark | n | Acc | Trigger | Parse | Focus-valid | Append | Malformed |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `d_deepstack_off` | free | VStar | 191 | 49.21% | 17.28% | 100.00% | 17.28% | 100.00% | 0.00% |
| `d_deepstack_off` | free | HR | 200 | 54.00% | 38.00% | 99.00% | 38.00% | 100.00% | 0.00% |
| `d_deepstack_off` | free | BLINK | 420 | 55.00% | 8.81% | 98.33% | 8.81% | 100.00% | 0.00% |
| `d_deepstack_off` | free | OCRBench-v2 | 600 | 23.88% | 23.83% | 99.67% | 23.83% | 100.00% | 0.00% |
| `d_deepstack_off` | free | MMMU-Pro | 300 | 34.67% | 31.67% | 100.00% | 31.67% | 100.00% | 0.00% |
| `d_deepstack_off` | free | MathVista | 300 | 46.33% | 21.00% | 100.00% | 21.00% | 100.00% | 0.00% |
| `d_deepstack_off` | free | MathVerse | 500 | 14.60% | 67.40% | 100.00% | 67.40% | 100.00% | 0.00% |
| `d_deepstack_off` | softforce | VStar | 191 | 49.21% | 63.35% | 100.00% | 63.35% | 100.00% | 0.00% |
| `d_deepstack_off` | softforce | HR | 200 | 51.50% | 73.00% | 99.00% | 73.00% | 100.00% | 0.00% |
| `d_deepstack_off` | softforce | BLINK | 420 | 56.19% | 25.71% | 98.33% | 25.71% | 100.00% | 0.00% |
| `d_deepstack_off` | softforce | OCRBench-v2 | 600 | 24.41% | 43.67% | 100.00% | 43.67% | 100.00% | 0.00% |
| `d_deepstack_off` | softforce | MMMU-Pro | 300 | 33.00% | 34.33% | 100.00% | 34.33% | 100.00% | 0.00% |
| `d_deepstack_off` | softforce | MathVista | 300 | 50.00% | 28.67% | 100.00% | 28.67% | 100.00% | 0.00% |
| `d_deepstack_off` | softforce | MathVerse | 500 | 15.00% | 62.00% | 100.00% | 62.00% | 100.00% | 0.00% |
| `d_deepstack_on_golden` | free | VStar | 191 | 52.36% | 11.52% | 100.00% | 11.52% | 100.00% | 0.00% |
| `d_deepstack_on_golden` | free | HR | 200 | 52.50% | 40.00% | 99.50% | 40.00% | 100.00% | 0.00% |
| `d_deepstack_on_golden` | free | BLINK | 420 | 56.90% | 7.86% | 99.52% | 7.86% | 100.00% | 0.00% |
| `d_deepstack_on_golden` | free | OCRBench-v2 | 600 | 25.02% | 22.83% | 99.83% | 22.83% | 100.00% | 0.00% |
| `d_deepstack_on_golden` | free | MMMU-Pro | 300 | 35.67% | 29.67% | 100.00% | 29.67% | 100.00% | 0.00% |
| `d_deepstack_on_golden` | free | MathVista | 300 | 49.67% | 20.33% | 100.00% | 20.33% | 100.00% | 0.00% |
| `d_deepstack_on_golden` | free | MathVerse | 500 | 16.00% | 65.20% | 100.00% | 65.20% | 100.00% | 0.00% |
| `d_deepstack_on_golden` | softforce | VStar | 191 | 50.79% | 58.64% | 100.00% | 58.64% | 100.00% | 0.00% |
| `d_deepstack_on_golden` | softforce | HR | 200 | 59.50% | 72.50% | 99.00% | 72.50% | 100.00% | 0.00% |
| `d_deepstack_on_golden` | softforce | BLINK | 420 | 55.48% | 23.33% | 97.62% | 23.33% | 100.00% | 0.00% |
| `d_deepstack_on_golden` | softforce | OCRBench-v2 | 600 | 24.54% | 36.17% | 99.83% | 36.17% | 100.00% | 0.00% |
| `d_deepstack_on_golden` | softforce | MMMU-Pro | 300 | 38.00% | 33.00% | 100.00% | 33.00% | 100.00% | 0.00% |
| `d_deepstack_on_golden` | softforce | MathVista | 300 | 49.00% | 23.67% | 100.00% | 23.67% | 100.00% | 0.00% |
| `d_deepstack_on_golden` | softforce | MathVerse | 500 | 19.00% | 57.40% | 100.00% | 57.40% | 100.00% | 0.00% |

### Source Artifacts

| Branch | Mode | Summary path |
|---|---|---|
| `d_deepstack_off` | free | `outputs/clean_benchmarks/qwen3_stage2_norm01_coredev2511_dynamic_maxtok512_flash2_multigridfix_gpu0_7_20260701_235228/tgvf_free/summary.json` |
| `d_deepstack_off` | softforce | `outputs/clean_benchmarks/qwen3_stage2_norm01_coredev2511_dynamic_maxtok512_flash2_multigridfix_gpu0_7_20260702_011343/tgvf_softforce/summary.json` |
| `d_deepstack_on_golden` | free | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu1_7_20260703_084300/tgvf_free/summary.json` |
| `d_deepstack_on_golden` | softforce | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_7_20260703_084300/tgvf_softforce/summary.json` |

## Matrix CE Size Sweep

This ablation changes the same-image Matrix CE grouping/readout size while
keeping the D DeepStack Stage1/Stage2 recipe and CoreDev-2511 evaluation
identity fixed. Rows are ordered by size from small to large. `size 1` is the
Matrix-CE-off control.

### Overall Summary

| Size | Branch | Free acc | Free trigger | Softforce acc | Softforce trigger | Rows |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `mce_off_size1` | 35.84% | 26.48% | 36.17% | 42.73% | 2,511 |
| 2 | `mce_size2` | 35.79% | 30.86% | 35.40% | 44.05% | 2,511 |
| 3 | `mce_size3` | 35.72% | 29.07% | 38.42% | 45.52% | 2,511 |
| 4 | `mce_size4_golden` | 37.04% | 29.79% | 37.92% | 40.98% | 2,511 |
| 5 | `mce_size5` | 36.08% | 27.32% | 36.43% | 40.58% | 2,511 |

### Accuracy Matrix

Free mode:

| Benchmark | n | size 1 | size 2 | size 3 | size 4 | size 5 |
|---|---:|---:|---:|---:|---:|---:|
| VStar | 191 | 50.26% | 49.21% | 50.26% | 52.36% | 49.74% |
| HR | 200 | 57.00% | 56.00% | 55.00% | 52.50% | 55.00% |
| BLINK | 420 | 54.76% | 55.00% | 55.24% | 56.90% | 56.19% |
| OCRBench-v2 | 600 | 24.82% | 24.80% | 23.17% | 25.02% | 24.00% |
| MMMU-Pro | 300 | 33.67% | 34.33% | 32.67% | 35.67% | 35.67% |
| MathVista | 300 | 45.33% | 47.67% | 48.67% | 49.67% | 48.67% |
| MathVerse | 500 | 14.80% | 13.40% | 15.20% | 16.00% | 13.60% |

Softforce mode:

| Benchmark | n | size 1 | size 2 | size 3 | size 4 | size 5 |
|---|---:|---:|---:|---:|---:|---:|
| VStar | 191 | 50.79% | 48.17% | 50.26% | 50.79% | 48.69% |
| HR | 200 | 56.00% | 59.00% | 59.50% | 59.50% | 57.50% |
| BLINK | 420 | 55.48% | 53.33% | 58.33% | 55.48% | 55.71% |
| OCRBench-v2 | 600 | 24.54% | 23.97% | 25.10% | 24.54% | 24.95% |
| MMMU-Pro | 300 | 31.67% | 31.00% | 36.00% | 38.00% | 34.33% |
| MathVista | 300 | 48.00% | 45.67% | 48.33% | 49.00% | 46.67% |
| MathVerse | 500 | 16.00% | 16.20% | 20.20% | 19.00% | 16.00% |

### Size 1: Matrix CE Off Control

Branch: `mce_off_size1`

| Mode | Benchmark | n | Acc | Trigger | Parse | Focus-valid | Append | Malformed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| free | VStar | 191 | 50.26% | 6.81% | 100.00% | 6.81% | 100.00% | 0.00% |
| free | HR | 200 | 57.00% | 37.00% | 98.50% | 37.00% | 100.00% | 0.00% |
| free | BLINK | 420 | 54.76% | 6.43% | 98.81% | 6.43% | 100.00% | 0.00% |
| free | OCRBench-v2 | 600 | 24.82% | 25.17% | 99.83% | 25.17% | 100.00% | 0.00% |
| free | MMMU-Pro | 300 | 33.67% | 24.67% | 100.00% | 24.67% | 100.00% | 0.00% |
| free | MathVista | 300 | 45.33% | 14.00% | 100.00% | 14.00% | 100.00% | 0.00% |
| free | MathVerse | 500 | 14.80% | 56.80% | 100.00% | 56.80% | 100.00% | 0.00% |
| softforce | VStar | 191 | 50.79% | 65.45% | 100.00% | 65.45% | 100.00% | 0.00% |
| softforce | HR | 200 | 56.00% | 77.00% | 98.00% | 77.00% | 100.00% | 0.00% |
| softforce | BLINK | 420 | 55.48% | 27.86% | 95.71% | 27.86% | 100.00% | 0.00% |
| softforce | OCRBench-v2 | 600 | 24.54% | 44.00% | 100.00% | 44.00% | 100.00% | 0.00% |
| softforce | MMMU-Pro | 300 | 31.67% | 27.67% | 100.00% | 27.67% | 100.00% | 0.00% |
| softforce | MathVista | 300 | 48.00% | 23.00% | 100.00% | 23.00% | 100.00% | 0.00% |
| softforce | MathVerse | 500 | 16.00% | 52.20% | 100.00% | 52.20% | 100.00% | 0.00% |

### Size 2

Branch: `mce_size2`

| Mode | Benchmark | n | Acc | Trigger | Parse | Focus-valid | Append | Malformed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| free | VStar | 191 | 49.21% | 16.23% | 100.00% | 16.23% | 100.00% | 0.00% |
| free | HR | 200 | 56.00% | 44.00% | 98.50% | 44.00% | 100.00% | 0.00% |
| free | BLINK | 420 | 55.00% | 9.05% | 99.29% | 9.05% | 100.00% | 0.00% |
| free | OCRBench-v2 | 600 | 24.80% | 23.83% | 99.67% | 23.83% | 100.00% | 0.00% |
| free | MMMU-Pro | 300 | 34.33% | 31.33% | 100.00% | 31.33% | 100.00% | 0.00% |
| free | MathVista | 300 | 47.67% | 21.00% | 100.00% | 21.00% | 100.00% | 0.00% |
| free | MathVerse | 500 | 13.40% | 63.60% | 100.00% | 63.60% | 100.00% | 0.00% |
| softforce | VStar | 191 | 48.17% | 68.59% | 100.00% | 68.59% | 100.00% | 0.00% |
| softforce | HR | 200 | 59.00% | 75.50% | 99.50% | 75.50% | 100.00% | 0.00% |
| softforce | BLINK | 420 | 53.33% | 31.19% | 97.62% | 31.19% | 100.00% | 0.00% |
| softforce | OCRBench-v2 | 600 | 23.97% | 36.67% | 100.00% | 36.67% | 100.00% | 0.00% |
| softforce | MMMU-Pro | 300 | 31.00% | 33.67% | 100.00% | 33.67% | 100.00% | 0.00% |
| softforce | MathVista | 300 | 45.67% | 23.67% | 100.00% | 23.67% | 100.00% | 0.00% |
| softforce | MathVerse | 500 | 16.20% | 60.20% | 100.00% | 60.20% | 100.00% | 0.00% |

### Size 3

Branch: `mce_size3`

| Mode | Benchmark | n | Acc | Trigger | Parse | Focus-valid | Append | Malformed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| free | VStar | 191 | 50.26% | 10.99% | 100.00% | 10.99% | 100.00% | 0.00% |
| free | HR | 200 | 55.00% | 42.50% | 99.50% | 42.50% | 100.00% | 0.00% |
| free | BLINK | 420 | 55.24% | 7.62% | 99.05% | 7.62% | 100.00% | 0.00% |
| free | OCRBench-v2 | 600 | 23.17% | 20.50% | 99.50% | 20.50% | 100.00% | 0.00% |
| free | MMMU-Pro | 300 | 32.67% | 29.00% | 100.00% | 29.00% | 100.00% | 0.00% |
| free | MathVista | 300 | 48.67% | 18.00% | 100.00% | 18.00% | 100.00% | 0.00% |
| free | MathVerse | 500 | 15.20% | 65.60% | 100.00% | 65.60% | 100.00% | 0.00% |
| softforce | VStar | 191 | 50.26% | 71.20% | 100.00% | 71.20% | 100.00% | 0.00% |
| softforce | HR | 200 | 59.50% | 76.50% | 100.00% | 76.50% | 100.00% | 0.00% |
| softforce | BLINK | 420 | 58.33% | 30.71% | 97.14% | 30.71% | 100.00% | 0.00% |
| softforce | OCRBench-v2 | 600 | 25.10% | 39.33% | 100.00% | 39.33% | 100.00% | 0.00% |
| softforce | MMMU-Pro | 300 | 36.00% | 34.33% | 100.00% | 34.33% | 100.00% | 0.00% |
| softforce | MathVista | 300 | 48.33% | 23.33% | 100.00% | 23.33% | 100.00% | 0.00% |
| softforce | MathVerse | 500 | 20.20% | 63.20% | 100.00% | 63.20% | 100.00% | 0.00% |

### Size 4: Golden

Branch: `mce_size4_golden`

| Mode | Benchmark | n | Acc | Trigger | Parse | Focus-valid | Append | Malformed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| free | VStar | 191 | 52.36% | 11.52% | 100.00% | 11.52% | 100.00% | 0.00% |
| free | HR | 200 | 52.50% | 40.00% | 99.50% | 40.00% | 100.00% | 0.00% |
| free | BLINK | 420 | 56.90% | 7.86% | 99.52% | 7.86% | 100.00% | 0.00% |
| free | OCRBench-v2 | 600 | 25.02% | 22.83% | 99.83% | 22.83% | 100.00% | 0.00% |
| free | MMMU-Pro | 300 | 35.67% | 29.67% | 100.00% | 29.67% | 100.00% | 0.00% |
| free | MathVista | 300 | 49.67% | 20.33% | 100.00% | 20.33% | 100.00% | 0.00% |
| free | MathVerse | 500 | 16.00% | 65.20% | 100.00% | 65.20% | 100.00% | 0.00% |
| softforce | VStar | 191 | 50.79% | 58.64% | 100.00% | 58.64% | 100.00% | 0.00% |
| softforce | HR | 200 | 59.50% | 72.50% | 99.00% | 72.50% | 100.00% | 0.00% |
| softforce | BLINK | 420 | 55.48% | 23.33% | 97.62% | 23.33% | 100.00% | 0.00% |
| softforce | OCRBench-v2 | 600 | 24.54% | 36.17% | 99.83% | 36.17% | 100.00% | 0.00% |
| softforce | MMMU-Pro | 300 | 38.00% | 33.00% | 100.00% | 33.00% | 100.00% | 0.00% |
| softforce | MathVista | 300 | 49.00% | 23.67% | 100.00% | 23.67% | 100.00% | 0.00% |
| softforce | MathVerse | 500 | 19.00% | 57.40% | 100.00% | 57.40% | 100.00% | 0.00% |

### Size 5

Branch: `mce_size5`

| Mode | Benchmark | n | Acc | Trigger | Parse | Focus-valid | Append | Malformed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| free | VStar | 191 | 49.74% | 7.33% | 100.00% | 7.33% | 100.00% | 0.00% |
| free | HR | 200 | 55.00% | 38.00% | 98.50% | 38.00% | 100.00% | 0.00% |
| free | BLINK | 420 | 56.19% | 8.33% | 98.33% | 8.33% | 100.00% | 0.00% |
| free | OCRBench-v2 | 600 | 24.00% | 22.67% | 100.00% | 22.67% | 100.00% | 0.00% |
| free | MMMU-Pro | 300 | 35.67% | 24.33% | 100.00% | 24.33% | 100.00% | 0.00% |
| free | MathVista | 300 | 48.67% | 17.33% | 100.00% | 17.33% | 100.00% | 0.00% |
| free | MathVerse | 500 | 13.60% | 60.00% | 100.00% | 60.00% | 100.00% | 0.00% |
| softforce | VStar | 191 | 48.69% | 62.30% | 100.00% | 62.30% | 100.00% | 0.00% |
| softforce | HR | 200 | 57.50% | 71.00% | 98.50% | 71.00% | 100.00% | 0.00% |
| softforce | BLINK | 420 | 55.71% | 25.48% | 97.86% | 25.48% | 100.00% | 0.00% |
| softforce | OCRBench-v2 | 600 | 24.95% | 38.83% | 100.00% | 38.83% | 100.00% | 0.00% |
| softforce | MMMU-Pro | 300 | 34.33% | 28.33% | 100.00% | 28.33% | 100.00% | 0.00% |
| softforce | MathVista | 300 | 46.67% | 22.33% | 100.00% | 22.33% | 100.00% | 0.00% |
| softforce | MathVerse | 500 | 16.00% | 53.20% | 100.00% | 53.20% | 100.00% | 0.00% |

### Source Artifacts

| Branch | Mode | Summary path |
|---|---|---|
| `mce_off_size1` | free | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_mce_off_size1_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_1_2_3_4_5_6_7_20260704_010949/tgvf_free/summary.json` |
| `mce_off_size1` | softforce | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_mce_off_size1_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_1_2_3_4_5_6_7_20260704_010949/tgvf_softforce/summary.json` |
| `mce_size2` | free | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_mce_size2_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_1_2_3_4_5_6_7_20260704_010949/tgvf_free/summary.json` |
| `mce_size2` | softforce | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_mce_size2_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_1_2_3_4_5_6_7_20260704_010949/tgvf_softforce/summary.json` |
| `mce_size3` | free | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_mce_size3_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_1_2_3_4_5_6_7_20260703_224753/tgvf_free/summary.json` |
| `mce_size3` | softforce | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_mce_size3_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_1_2_3_4_5_6_7_20260703_224753/tgvf_softforce/summary.json` |
| `mce_size4_golden` | free | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu1_7_20260703_084300/tgvf_free/summary.json` |
| `mce_size4_golden` | softforce | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_7_20260703_084300/tgvf_softforce/summary.json` |
| `mce_size5` | free | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_size5_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_7_20260703_172325/tgvf_free/summary.json` |
| `mce_size5` | softforce | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_size5_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_7_20260703_172325/tgvf_softforce/summary.json` |

## Global Batch Ablation

This branch reuses the completed `bs2x` Stage1 checkpoint trained with global
batch `64 / 1000` steps, but restores the golden Stage2 schedule
`128 / 1200`. It isolates whether the earlier `bs2x_gbs64_256_same_samples`
regression came mainly from the larger Stage2 global batch. CoreDev-2511
evaluation identity is unchanged from the shared setting above.

Branch: `bs2x_stage1_stage2_gbs128_normal`

### Overall Summary

| Mode | Rows | Acc | Delta vs golden | Macro acc | Macro delta vs golden | Parse | Trigger | Append | Malformed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| free | 2,511 | 35.29% | -1.75 | 39.70% | -1.46 | 99.52% | 29.91% | 100.00% | 0.00% |
| softforce | 2,511 | 36.61% | -1.31 | 40.63% | -1.70 | 99.32% | 45.08% | 100.00% | 0.00% |

Restoring the normal Stage2 batch schedule improves over the large-Stage2-batch
`bs2x_gbs64_256_same_samples` branch, but it still does not recover the current
golden external benchmark score. The stronger `bs2x` Stage1 internal retrieval
therefore does not directly translate to better CoreDev-2511 accuracy.

### Per-Benchmark Accuracy

Free mode:

| Benchmark | n | Acc | Golden acc | Delta vs golden | Trigger | Parse |
|---|---:|---:|---:|---:|---:|---:|
| VStar | 191 | 51.31% | 52.36% | -1.05 | 10.99% | 100.00% |
| HR | 200 | 53.50% | 52.50% | +1.00 | 42.50% | 98.00% |
| BLINK | 420 | 55.24% | 56.90% | -1.67 | 10.71% | 98.57% |
| OCRBench-v2 | 600 | 24.03% | 25.02% | -0.99 | 22.67% | 99.67% |
| MMMU-Pro | 300 | 34.67% | 35.67% | -1.00 | 31.33% | 100.00% |
| MathVista | 300 | 47.33% | 49.67% | -2.33 | 18.33% | 100.00% |
| MathVerse | 500 | 11.80% | 16.00% | -4.20 | 63.00% | 100.00% |

Softforce mode:

| Benchmark | n | Acc | Golden acc | Delta vs golden | Trigger | Parse |
|---|---:|---:|---:|---:|---:|---:|
| VStar | 191 | 49.21% | 50.79% | -1.57 | 71.73% | 100.00% |
| HR | 200 | 57.50% | 59.50% | -2.00 | 76.50% | 99.50% |
| BLINK | 420 | 54.05% | 55.48% | -1.43 | 31.19% | 96.19% |
| OCRBench-v2 | 600 | 25.72% | 24.54% | +1.18 | 39.67% | 100.00% |
| MMMU-Pro | 300 | 33.00% | 38.00% | -5.00 | 33.67% | 100.00% |
| MathVista | 300 | 47.33% | 49.00% | -1.67 | 26.33% | 100.00% |
| MathVerse | 500 | 17.60% | 19.00% | -1.40 | 58.60% | 100.00% |

Append success is `100.00%`, focus-valid equals trigger rate, and malformed
focus action rate is `0.00%` for every benchmark in both modes.

### Source Artifacts

| Branch | Mode | Summary path |
|---|---|---|
| `bs2x_stage1_stage2_gbs128_normal` | Stage1 checkpoint | `outputs/clean_training/qwen3_stage1_ddeepstack_bs2x_gbs64_same_samples_8gpu_20260705_045433/stage1_micro4/clean_training_execution/checkpoint_step_1000.pt` |
| `bs2x_stage1_stage2_gbs128_normal` | Stage2 checkpoint | `outputs/clean_training/qwen3_stage2_ddeepstack_norm01_from_stage1_ddeepstack_bs2x_stage1_stage2_gbs128_normal_8gpu_20260705_103502/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt` |
| `bs2x_stage1_stage2_gbs128_normal` | free | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_bs2x_stage1_stage2_gbs128_normal_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_1_2_3_4_5_6_7_20260705_103502/tgvf_free/summary.json` |
| `bs2x_stage1_stage2_gbs128_normal` | softforce | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_bs2x_stage1_stage2_gbs128_normal_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_1_2_3_4_5_6_7_20260705_103502/tgvf_softforce/summary.json` |

## Resolution Ablation

This ablation changes max image resolution from the golden `512` setting to
`214`. For TGVF, Stage1 training, Stage1 internal diagnostics, Stage2 training,
Stage2 internal diagnostics, and CoreDev-2511 evaluation all use resolution
`214`. Original@214 is an eval rerun at resolution `214`.

### Overall Summary

`Delta vs 512` compares each method against its own `512` anchor.

| Method | Resolution | Rows | Acc | Delta vs 512 | Macro acc | Parse | Trigger | Append |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 512 | 2,511 | 30.86% | anchor | 36.14% | 94.15% | 0.00% | n/a |
| Original | 214 | 2,511 | 23.96% | -6.89 | 27.44% | 92.59% | 0.00% | n/a |
| D DeepStack free | 512 | 2,511 | 37.04% | anchor | 41.16% | 99.84% | 29.79% | 100.00% |
| D DeepStack free | 214 | 2,511 | 30.82% | -6.22 | 33.69% | 99.84% | 23.06% | 100.00% |
| D DeepStack softforce | 512 | 2,511 | 37.92% | anchor | 42.33% | 99.48% | 40.98% | 100.00% |
| D DeepStack softforce | 214 | 2,511 | 30.73% | -7.20 | 33.90% | 99.36% | 36.48% | 100.00% |

Resolution `214` is clearly worse than `512` for this recipe. TGVF still beats
original@214 by `+6.86` points in free mode and `+6.76` points in softforce
mode, but the absolute score drop is large enough that `214` should remain an
efficiency ablation rather than a default setting.

### Per-Benchmark Accuracy At Resolution 214

| Benchmark | n | Original acc | Free acc | Free delta vs original | Softforce acc | Softforce delta vs original |
|---|---:|---:|---:|---:|---:|---:|
| VStar | 191 | 39.79% | 39.27% | -0.52 | 37.70% | -2.09 |
| HR | 200 | 38.00% | 37.50% | -0.50 | 42.50% | +4.50 |
| BLINK | 420 | 38.10% | 50.48% | +12.38 | 47.38% | +9.29 |
| OCRBench-v2 | 600 | 17.45% | 19.47% | +2.03 | 18.42% | +0.98 |
| MMMU-Pro | 300 | 22.67% | 34.33% | +11.67 | 34.33% | +11.67 |
| MathVista | 300 | 31.67% | 41.00% | +9.33 | 41.33% | +9.67 |
| MathVerse | 500 | 4.40% | 13.80% | +9.40 | 15.60% | +11.20 |

### Per-Benchmark Parse And Trigger At Resolution 214

Original has no focus trigger. TGVF append success is `100.00%` on every
benchmark, and malformed focus action rate is `0.00%` on every benchmark.

| Benchmark | Original parse | Free parse | Free trigger | Softforce parse | Softforce trigger |
|---|---:|---:|---:|---:|---:|
| VStar | 98.95% | 100.00% | 0.52% | 100.00% | 52.88% |
| HR | 88.50% | 99.50% | 22.50% | 98.50% | 70.00% |
| BLINK | 61.67% | 99.29% | 3.81% | 96.90% | 18.81% |
| OCRBench-v2 | 100.00% | 100.00% | 24.00% | 100.00% | 37.00% |
| MMMU-Pro | 100.00% | 100.00% | 20.33% | 100.00% | 24.33% |
| MathVista | 100.00% | 100.00% | 13.67% | 100.00% | 24.33% |
| MathVerse | 100.00% | 100.00% | 54.20% | 100.00% | 45.60% |

### Internal Diagnostics At Resolution 214

| Stage | Mean NLL correct D | Wrong-same beat rate | Query top1 | Query top2 | MRR | Mean diagonal gap |
|---|---:|---:|---:|---:|---:|---:|
| Stage1 | 1.3541 | 90.00% | 78.50% | 94.50% | 88.21% | +0.1638 |
| Stage2 | 1.6719 | 35.50% | 22.50% | 41.00% | 47.49% | -0.0283 |

Stage2 internal diagnostics completed and wrote all artifacts. The CLI exited
with a final stdout JSON serialization error for a Python `set`; this is a
reporting bug after artifact writeout, not a failed diagnostic run.

### Source Artifacts

| Branch | Mode | Summary path |
|---|---|---|
| `resolution512_golden` | original | `outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_maxans512_20260628_0216_rescored_bboxfix_20260628_035759/merged/summary.json` |
| `resolution512_golden` | free | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu1_7_20260703_084300/tgvf_free/summary.json` |
| `resolution512_golden` | softforce | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_7_20260703_084300/tgvf_softforce/summary.json` |
| `resolution214` | original | `outputs/clean_benchmarks/qwen3_original_resolution214_coredev2511_dynamic_maxans512_flash2_gpu4_5_6_7_20260704_141526/summary.json` |
| `resolution214` | free | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_resolution214_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_1_2_3_4_5_6_7_20260704_141526/tgvf_free/summary.json` |
| `resolution214` | softforce | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_resolution214_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_1_2_3_4_5_6_7_20260704_141526/tgvf_softforce/summary.json` |
| `resolution214` | Stage1 internal readout | `outputs/clean_training/qwen3_stage1_ddeepstack_resolution214_4gpu_20260704_141526/stage1_micro4/internal_diagnostics_step2000_20260704_141526/readout/readout_eval_report.json` |
| `resolution214` | Stage1 internal query | `outputs/clean_training/qwen3_stage1_ddeepstack_resolution214_4gpu_20260704_141526/stage1_micro4/internal_diagnostics_step2000_20260704_141526/query_sensitivity/query_sensitivity_report.json` |
| `resolution214` | Stage2 internal readout | `outputs/clean_training/qwen3_stage2_ddeepstack_norm01_from_stage1_ddeepstack_resolution214_8gpu_20260704_141526/stage2_micro4/internal_diagnostics_step1200_20260704_141526/readout/readout_eval_report.json` |
| `resolution214` | Stage2 internal query | `outputs/clean_training/qwen3_stage2_ddeepstack_norm01_from_stage1_ddeepstack_resolution214_8gpu_20260704_141526/stage2_micro4/internal_diagnostics_step1200_20260704_141526/query_sensitivity/query_sensitivity_report.json` |

## Eval-Only Resolution 1024 Diagnostic

This diagnostic changes only benchmark-time image resolution from `512` to
`1024`. It does not retrain Stage1 or Stage2. The generation budget remains
`512`, so the table should be read together with the output-token health
metrics below.

Identity caveat: the dynamic benchmark command omitted explicit
`--eval-family project_native_external`, so the summary metadata records
`eval_family=internal_diagnostic`. The manifest, row count/order, model,
backend, scorer, FlashAttention-2 setting, and resolution are the intended
complete CoreDev-2511 evaluation identity.

### Overall Summary

| Method | Resolution | Rows | Acc | Delta vs own 512 | Macro acc | Parse | Trigger | Append | Hit>=512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 512 | 2,511 | 30.86% | anchor | 36.14% | 94.15% | 0.00% | n/a | not recorded |
| Original | 1024 | 2,511 | 32.62% | +1.76 | 39.26% | 94.58% | 0.00% | n/a | 58.70% |
| D DeepStack free | 512 | 2,511 | 37.04% | anchor | 41.16% | 99.84% | 29.79% | 100.00% | not recorded |
| D DeepStack free | 1024 | 2,511 | 39.76% | +2.71 | 45.53% | 99.80% | 30.59% | 100.00% | 2.19% |
| D DeepStack softforce | 512 | 2,511 | 37.92% | anchor | 42.33% | 99.48% | 40.98% | 100.00% | not recorded |
| D DeepStack softforce | 1024 | 2,511 | 39.58% | +1.65 | 45.07% | 99.40% | 41.78% | 100.00% | 1.75% |

### Per-Benchmark Accuracy At Resolution 1024

| Benchmark | n | Original acc | Free acc | Free delta vs original | Softforce acc | Softforce delta vs original |
|---|---:|---:|---:|---:|---:|---:|
| VStar | 191 | 69.63% | 68.59% | -1.05 | 67.54% | -2.09 |
| HR | 200 | 62.00% | 62.50% | +0.50 | 61.50% | -0.50 |
| BLINK | 420 | 49.05% | 56.67% | +7.62 | 54.52% | +5.48 |
| OCRBench-v2 | 600 | 21.51% | 26.05% | +4.54 | 26.46% | +4.95 |
| MMMU-Pro | 300 | 31.67% | 37.33% | +5.67 | 37.67% | +6.00 |
| MathVista | 300 | 36.33% | 51.00% | +14.67 | 49.00% | +12.67 |
| MathVerse | 500 | 4.60% | 16.60% | +12.00 | 18.80% | +14.20 |

### Output-Token Health At Resolution 1024

| Benchmark | Original hit>=512 | Original acc on hit | Original acc on non-hit | Free hit>=512 | Softforce hit>=512 |
|---|---:|---:|---:|---:|---:|
| VStar | 31/191 (16.23%) | 54.84% | 72.50% | 0/191 (0.00%) | 0/191 (0.00%) |
| HR | 32/200 (16.00%) | 12.50% | 71.43% | 0/200 (0.00%) | 0/200 (0.00%) |
| BLINK | 170/420 (40.48%) | 14.71% | 72.40% | 0/420 (0.00%) | 0/420 (0.00%) |
| OCRBench-v2 | 370/600 (61.67%) | 27.03% | 47.39% | 50/600 (8.33%) | 44/600 (7.33%) |
| MMMU-Pro | 211/300 (70.33%) | 19.43% | 60.67% | 2/300 (0.67%) | 0/300 (0.00%) |
| MathVista | 178/300 (59.33%) | 15.17% | 67.21% | 1/300 (0.33%) | 0/300 (0.00%) |
| MathVerse | 482/500 (96.40%) | 4.36% | 11.11% | 2/500 (0.40%) | 0/500 (0.00%) |

Original@1024 is not a healthy output setting: `58.70%` of all rows hit the
`512` output-token cap, and MathVerse is almost completely capped
(`482/500`). TGVF does not show the same truncation failure under the same
budget, so the low MathVerse absolute score for TGVF is likely not explained by
the same output-token pathology. A healthy follow-up should rerun original and
TGVF with the same higher output budget, e.g. `max_image_resolution=1024` and
`max_answer_tokens/max_tokens=2048`, before drawing final conclusions about
MathVerse or original-vs-TGVF deltas at high resolution.

### Source Artifacts

| Branch | Mode | Summary path |
|---|---|---|
| `resolution1024_eval_only` | original | `outputs/clean_benchmarks/qwen3_original_evalres1024_coredev2511_dynamic_maxans512_flash2_gpu0_7_20260706_160153/summary.json` |
| `resolution1024_eval_only` | free | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_evalres1024_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_7_20260706_160153/tgvf_free/summary.json` |
| `resolution1024_eval_only` | softforce | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_evalres1024_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_7_20260706_160153/tgvf_softforce/summary.json` |

## Healthy-Budget Resolution 1024 Diagnostic

This follow-up keeps benchmark-time image resolution at `1024` but increases
the output budget to `2048`: original uses `--max-answer-tokens 2048`, while
TGVF uses unified `--max-tokens 2048`. It is a CoreDev-350 diagnostic
(`7` benchmarks x `50` rows), not a complete CoreDev-2511 table.

### Overall Summary

| Method | Rows | Acc | Macro acc | Delta vs original | Parse | Trigger | Hit>=512 | Hit>=2048 | Avg output tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 350 | 48.00% | 48.00% | anchor | 98.57% | 0.00% | 170/350 (48.6%) | 88/350 (25.1%) | 870.2 |
| D DeepStack free | 350 | 47.41% | 47.41% | -0.58 | 99.71% | 30.86% | 4/350 (1.1%) | 0/350 (0.0%) | 71.1 |
| D DeepStack softforce | 350 | 48.54% | 48.54% | +0.54 | 99.14% | 45.43% | 3/350 (0.9%) | 0/350 (0.0%) | 64.9 |

### Per-Benchmark Accuracy

| Benchmark | n | Original acc | Free acc | Free delta | Softforce acc | Softforce delta |
|---|---:|---:|---:|---:|---:|---:|
| VStar | 50 | 70.00% | 68.00% | -2.00 | 66.00% | -4.00 |
| HR | 50 | 68.00% | 68.00% | +0.00 | 68.00% | +0.00 |
| BLINK | 50 | 60.00% | 60.00% | +0.00 | 62.00% | +2.00 |
| OCRBench-v2 | 50 | 27.97% | 27.87% | -0.09 | 29.75% | +1.79 |
| MMMU-Pro | 50 | 36.00% | 34.00% | -2.00 | 48.00% | +12.00 |
| MathVista | 50 | 56.00% | 60.00% | +4.00 | 46.00% | -10.00 |
| MathVerse | 50 | 18.00% | 14.00% | -4.00 | 20.00% | +2.00 |

### Output-Token Health

| Benchmark | Original hit>=512 | Original hit>=2048 | Free hit>=512 | Free hit>=2048 | Softforce hit>=512 | Softforce hit>=2048 |
|---|---:|---:|---:|---:|---:|---:|
| VStar | 5/50 (10.0%) | 1/50 (2.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) |
| HR | 5/50 (10.0%) | 2/50 (4.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 1/50 (2.0%) | 0/50 (0.0%) |
| BLINK | 20/50 (40.0%) | 7/50 (14.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) |
| OCRBench-v2 | 33/50 (66.0%) | 12/50 (24.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) |
| MMMU-Pro | 35/50 (70.0%) | 25/50 (50.0%) | 3/50 (6.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) |
| MathVista | 24/50 (48.0%) | 11/50 (22.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) |
| MathVerse | 48/50 (96.0%) | 30/50 (60.0%) | 1/50 (2.0%) | 0/50 (0.0%) | 2/50 (4.0%) | 0/50 (0.0%) |

| Benchmark | Original avg tokens | Free avg tokens | Softforce avg tokens |
|---|---:|---:|---:|
| VStar | 223.8 | 25.9 | 21.9 |
| HR | 287.3 | 32.1 | 63.1 |
| BLINK | 667.7 | 33.4 | 33.3 |
| OCRBench-v2 | 1100.0 | 86.6 | 79.4 |
| MMMU-Pro | 1307.2 | 170.6 | 68.2 |
| MathVista | 801.6 | 65.8 | 58.4 |
| MathVerse | 1703.6 | 83.4 | 130.1 |

The `2048` output budget is still not a healthy original setting at
`max_image_resolution=1024`: original hits the new `2048` cap on `25.1%` of
CoreDev-350, including MathVerse `30/50` and MMMU-Pro `25/50`. TGVF does not
show the same pathology under the same resolution and budget: both free and
softforce have `0/350` rows hitting `2048`, and fewer than `1.2%` hit `512`.

Interpretation: the low MathVerse score for original at high resolution is
heavily confounded by long-output truncation. TGVF's MathVerse absolute score
remains low, but it is not explained by the same output-token failure; this
points more toward task difficulty, scorer/answer-form sensitivity, or model
reasoning behavior after focus rather than simple max-token truncation.

### Source Artifacts

| Branch | Mode | Summary path |
|---|---|---|
| `healthy_res1024_maxtok2048_coredev350` | original | `outputs/clean_benchmarks/qwen3_original_health_res1024_coredev350_dynamic_maxans2048_flash2_gpu0_7_20260706_214629/summary.json` |
| `healthy_res1024_maxtok2048_coredev350` | free | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_health_res1024_coredev350_dynamic_maxtok2048_flash2_ddeepstack_gpu0_7_20260706_214629/tgvf_free/summary.json` |
| `healthy_res1024_maxtok2048_coredev350` | softforce | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_health_res1024_coredev350_dynamic_maxtok2048_flash2_ddeepstack_gpu0_7_20260706_214629/tgvf_softforce/summary.json` |

## Original Res/Max Health Sweep

This diagnostic asks whether original Qwen3-thinking can be made naturally
output-healthy by changing only benchmark-time image resolution while keeping a
large answer budget. It uses a balanced CoreDev-70 subset (`7` benchmarks x
`10` rows) sampled from the CoreDev-350 diagnostic manifest, runs original only
on GPUs `0,1,2,3`, and keeps `max_answer_tokens=2048`. A candidate is considered
healthy only if `hit>=2048 <= 5%`, parse is at least `95%`, and the output
length is not dominated by long thinking loops.

### Overall Summary

| Max image res | Rows | Acc | Parse | Hit>=512 | Hit>=1024 | Hit>=1536 | Hit>=2048 | Avg output tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 70 | 36.51% | 100.00% | 40/70 (57.1%) | 29/70 (41.4%) | 23/70 (32.9%) | 20/70 (28.6%) | 975.0 |
| 384 | 70 | 35.32% | 98.57% | 36/70 (51.4%) | 30/70 (42.9%) | 27/70 (38.6%) | 19/70 (27.1%) | 981.7 |
| 512 | 70 | 40.01% | 97.14% | 36/70 (51.4%) | 30/70 (42.9%) | 21/70 (30.0%) | 21/70 (30.0%) | 911.9 |

### Per-Benchmark `hit>=2048`

| Benchmark | res256 | res384 | res512 |
|---|---:|---:|---:|
| VStar | 0/10 | 0/10 | 0/10 |
| HR | 0/10 | 1/10 | 1/10 |
| BLINK | 0/10 | 0/10 | 1/10 |
| OCRBench-v2 | 3/10 | 2/10 | 2/10 |
| MMMU-Pro | 6/10 | 6/10 | 6/10 |
| MathVista | 4/10 | 3/10 | 4/10 |
| MathVerse | 7/10 | 7/10 | 7/10 |

Conclusion: no normal output-healthy original setting was found using only
`max_image_resolution in {256,384,512}` and `max_answer_tokens=2048` under the
current original Qwen3 thinking prompt. Lowering resolution does not remove the
long-output failure: MathVerse remains `7/10` hit>=2048 at all three
resolutions, and MMMU-Pro remains `6/10`. A practical `512` answer-token cap is
a truncation guard, not natural health. A genuinely healthy original comparison
would require changing generation behavior, such as a no-thinking/direct-answer
mode or a validated stop/answer extraction policy; that was intentionally not
part of this sweep.

### Hit-Stress No-Resolution-Cap Follow-Up

This follow-up selects the `21` rows from the `res512/max_answer_tokens=2048`
CoreDev-70 run that already hit `2048`, then reruns original Qwen3-thinking with
`max_answer_tokens=4096`. It compares the normal `max_image_resolution=512`
setting against `max_image_resolution=0`, where `0` means the original backend
omits the project-side `max_pixels` image cap. The goal is to distinguish
legitimate long reasoning from unhealthy loops and to test whether removing the
image cap mitigates the loop.

| Image setting | Rows | Acc | Parse | Hit>=2048 | Hit>=4096 | Avg output tokens |
|---|---:|---:|---:|---:|---:|---:|
| `res512` | 21 | 28.59% | 20/21 (95.2%) | 18/21 (85.7%) | 13/21 (61.9%) | 3350.7 |
| `nores` | 21 | 28.61% | 19/21 (90.5%) | 17/21 (81.0%) | 14/21 (66.7%) | 3332.4 |

Per-benchmark `hit>=4096`:

| Benchmark | n | res512 | nores |
|---|---:|---:|---:|
| HR | 1 | 1/1 | 1/1 |
| BLINK | 1 | 0/1 | 1/1 |
| OCRBench-v2 | 2 | 0/2 | 0/2 |
| MMMU-Pro | 6 | 6/6 | 6/6 |
| MathVista | 4 | 2/4 | 2/4 |
| MathVerse | 7 | 4/7 | 4/7 |

Paired comparison: `17/21` rows have exactly the same output-token count under
`res512` and `nores`; two rows become shorter under `nores` and two become
longer. The net effect is not a mitigation: `hit>=4096` increases from `13/21`
to `14/21`, with the BLINK stress row becoming a new cap hit.

Loop inspection: the `>=4096` cap hits are overwhelmingly unhealthy loops, not
normal long answers. They show repeated self-correction, cyclic reconsideration
of the same evidence, or high repeated n-gram counts near the tail. This is
especially clear for MMMU-Pro, where `6/6` stress rows hit `4096` under both
image settings. Some cap-hit rows can still score correctly because a parseable
option appears somewhere in the output, but they are operationally unhealthy
because the model does not naturally terminate. Non-cap `>=2048` rows are mixed:
some are OCR transcription or long-but-terminating reasoning, so `>=2048` should
be treated as a warning threshold, while `>=4096` is strong evidence of the
dead-loop pathology.

Conclusion: removing the project-side original-image resolution cap does not
fix the original Qwen3-thinking loop. The failure is primarily a generation
behavior issue, not a simple image-resolution cap issue. Further healthy
original comparisons should test generation changes such as no-thinking/direct
answer or a validated stop/answer-extraction policy rather than raising the
token cap again.

### Balanced No-Resolution-Cap Prevalence

Because the hit-stress diagnostic intentionally selected prior cap-hit rows, a
balanced CoreDev-70 prevalence check was run with the same no project-side image
cap and `max_answer_tokens=4096`.

| Setting | Rows | Acc | Parse | Avg prompt len | Max prompt len | Avg output tokens | Hit>=2048 | Hit>=3072 | Hit>=4096 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `nores/max4096` | 70 | 53.98% | 68/70 (97.1%) | 3046.7 | 15985 | 1296.2 | 18/70 (25.7%) | 15/70 (21.4%) | 15/70 (21.4%) |

Per-benchmark:

| Benchmark | n | Acc | Avg prompt len | Avg output tokens | Hit>=2048 | Hit>=3072 | Hit>=4096 |
|---|---:|---:|---:|---:|---:|---:|---:|
| VStar | 10 | 90.00% | 3221.5 | 54.4 | 0/10 | 0/10 | 0/10 |
| HR | 10 | 80.00% | 15473.0 | 517.7 | 1/10 | 1/10 | 1/10 |
| BLINK | 10 | 50.00% | 980.9 | 590.8 | 1/10 | 1/10 | 1/10 |
| OCRBench-v2 | 10 | 37.87% | 629.2 | 874.2 | 0/10 | 0/10 | 0/10 |
| MMMU-Pro | 10 | 40.00% | 320.4 | 2677.8 | 6/10 | 6/10 | 6/10 |
| MathVista | 10 | 40.00% | 386.6 | 1410.8 | 3/10 | 2/10 | 2/10 |
| MathVerse | 10 | 40.00% | 315.6 | 2947.5 | 7/10 | 5/10 | 5/10 |

Interpretation: the stress-set rate was inflated by selection bias
(`14/21`, `66.7%` hit>=4096), but nores is still not healthy on the balanced
sample (`15/70`, `21.4%` hit>=4096). The pathology is concentrated in
MMMU-Pro and MathVerse, not globally caused by longer visual prompts. For
example, VStar has long nores prompts but `0/10` cap hits, while MMMU-Pro has
short prompts and `6/10` cap hits. `qwen_vl_utils` also confirms that nores is
not truly unlimited: omitting project-side `max_pixels` falls back to the
library's default maximum, equivalent to roughly a `4096 x 4096` square at the
local patch/merge factor.

Recommended labels for `max_answer_tokens=4096`: use `hit>=4096` as the
high-precision unhealthy label; use `hit>=3072` as a strong warning or early
screen; treat `hit>=2048` as a warning requiring inspection because it also
captures some long but terminating OCR/reasoning rows.

### Original/TGVF 1024 And Nores Max4096

This diagnostic uses the same balanced CoreDev-70 sample set and compares
original, TGVF free, and TGVF softforce under `max_image_resolution=1024` and
under no project-side image cap. Original uses `max_answer_tokens=4096`; TGVF
uses unified `max_tokens=4096`. The nores original row reuses the completed
balanced nores run above.

| Setting | Mode | Rows | Scored | Acc scored | Acc all rows | Parse | Trigger | Append | Malformed | Avg output tokens | Hit>=512 | Hit>=2048 | Hit>=3072 | Hit>=4096 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `res1024/max4096` | original | 70 | 70 | 52.55% | 52.55% | 98.57% | 0.00% | 0.00% | 0.00% | 1255.0 | 30/70 | 17/70 | 14/70 | 14/70 |
| `res1024/max4096` | TGVF free | 70 | 70 | 45.16% | 45.16% | 100.00% | 34.29% | 34.29% | 0.00% | 106.8 | 1/70 | 1/70 | 1/70 | 0/70 |
| `res1024/max4096` | TGVF softforce | 70 | 70 | 48.00% | 48.00% | 98.57% | 47.14% | 47.14% | 0.00% | 122.2 | 2/70 | 1/70 | 1/70 | 0/70 |
| `nores/max4096` | original | 70 | 70 | 53.98% | 53.98% | 97.14% | 0.00% | 0.00% | 0.00% | 1296.2 | 31/70 | 18/70 | 15/70 | 15/70 |
| `nores/max4096` | TGVF free | 70 | 65 | 45.55% | 42.30% | 92.86% | 32.86% | 25.71% | 7.14% | 105.8 | 1/70 | 1/70 | 1/70 | 0/70 |
| `nores/max4096` | TGVF softforce | 70 | 64 | 44.69% | 40.86% | 90.00% | 44.29% | 35.71% | 8.57% | 66.7 | 1/70 | 0/70 | 0/70 | 0/70 |

Per-benchmark `Acc scored`:

| Setting | Mode | VStar | HR | BLINK | OCRBench-v2 | MMMU-Pro | MathVista | MathVerse |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `res1024` | original | 80.00% | 70.00% | 60.00% | 37.87% | 40.00% | 40.00% | 40.00% |
| `res1024` | TGVF free | 80.00% | 80.00% | 60.00% | 26.10% | 20.00% | 30.00% | 20.00% |
| `res1024` | TGVF softforce | 90.00% | 80.00% | 50.00% | 25.98% | 40.00% | 30.00% | 20.00% |
| `nores` | original | 90.00% | 80.00% | 50.00% | 37.87% | 40.00% | 40.00% | 40.00% |
| `nores` | TGVF free | 90.00% | 80.00% | 70.00% | 26.10% | 20.00% | 30.00% | 20.00% |
| `nores` | TGVF softforce | 90.00% | 75.00% | 50.00% | 26.03% | 40.00% | 30.00% | 20.00% |

Interpretation: `res1024/max4096` is a valid high-budget diagnostic. Original
scores highest on this small CoreDev-70 slice but remains output-unhealthy:
`14/70` rows hit `4096`. TGVF free and softforce have `0/70` hit>=4096 and no
malformed rows at `res1024`, so they remain output-healthy but lower accuracy
on this reasoning-heavy slice. The nores TGVF rows are not a clean comparison:
triggered HR focus append can OOM at 180GB, producing `5` malformed rows for
free and `6` for softforce. For nores TGVF, `Acc all rows` is the safer
practical number until that OOM path is fixed.

### Source Artifacts

| Branch | Summary path |
|---|---|
| `original_health_params_coredev70_res256` | `outputs/clean_benchmarks/qwen3_original_healthparam_coredev70_res256_maxans2048_flash2_gpu0_3_20260706_230528/summary.json` |
| `original_health_params_coredev70_res384` | `outputs/clean_benchmarks/qwen3_original_healthparam_coredev70_res384_maxans2048_flash2_gpu0_3_20260706_230528/summary.json` |
| `original_health_params_coredev70_res512` | `outputs/clean_benchmarks/qwen3_original_healthparam_coredev70_res512_maxans2048_flash2_gpu0_3_20260706_230528/summary.json` |
| `original_hitstress_res512_max4096` | `outputs/clean_benchmarks/qwen3_original_hitstress_res512_maxans4096_flash2_gpu0_3_20260706_235519/summary.json` |
| `original_hitstress_nores_max4096` | `outputs/clean_benchmarks/qwen3_original_hitstress_nores_maxans4096_flash2_gpu0_3_20260706_235519/summary.json` |
| `original_nores_balanced_coredev70_max4096` | `outputs/clean_benchmarks/qwen3_original_nores_coredev70_maxans4096_flash2_gpu0_3_20260707_010022/summary.json` |
| `original_res1024_coredev70_max4096` | `outputs/clean_benchmarks/qwen3_original_res1024_coredev70_maxans4096_flash2_gpu0_1_2_3_5_6_7_20260707_012845/summary.json` |
| `tgvf_free_res1024_coredev70_max4096` | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_res1024_coredev70_maxtok4096_flash2_ddeepstack_gpu0_1_2_3_5_6_7_20260707_012845/tgvf_free/summary.json` |
| `tgvf_softforce_res1024_coredev70_max4096` | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_res1024_coredev70_maxtok4096_flash2_ddeepstack_gpu0_1_2_3_5_6_7_20260707_012845/tgvf_softforce/summary.json` |
| `tgvf_free_nores_coredev70_max4096` | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_nores_coredev70_maxtok4096_flash2_ddeepstack_gpu0_1_2_3_5_6_7_20260707_012845/tgvf_free/summary.json` |
| `tgvf_softforce_nores_coredev70_max4096` | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_nores_coredev70_maxtok4096_flash2_ddeepstack_gpu0_1_2_3_5_6_7_20260707_012845/tgvf_softforce/summary.json` |

## Data Scale Ablation

This ablation keeps the golden D DeepStack and Matrix CE size-4 recipe, but
changes the training data scale. The `data25k` subset uses exactly `25000`
Stage2 train rows selected from the clean 50k family with deterministic
stratified sampling. Stage1 train keeps rows whose `source_uid` matches the
selected Stage2 `v4_uid` set, yielding `21323` Stage1 focus-train rows. Eval
and benchmark identity are unchanged.

### Overall Summary

| Branch | Stage2 train rows | Stage1 train rows | Mode | Rows | Acc | Delta vs 50k | Macro acc | Parse | Trigger | Append |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `data25k` | 25,000 | 21,323 | free | 2,511 | 36.27% | -0.78 | 40.59% | 99.64% | 34.21% | 100.00% |
| `data25k` | 25,000 | 21,323 | softforce | 2,511 | 35.45% | -2.48 | 39.63% | 99.48% | 51.97% | 100.00% |
| `data50k_golden` | 46,883 | 39,998 | free | 2,511 | 37.04% | anchor | 41.16% | 99.84% | 29.79% | 100.00% |
| `data50k_golden` | 46,883 | 39,998 | softforce | 2,511 | 37.92% | anchor | 42.33% | 99.48% | 40.98% | 100.00% |

Free mode is fairly robust to the 25k subset, while softforce is more sensitive:
it triggers more often but loses accuracy versus the 50k golden branch.

### Per-Benchmark Accuracy

Free mode:

| Benchmark | n | data25k | 50k golden | Delta |
|---|---:|---:|---:|---:|
| VStar | 191 | 51.83% | 52.36% | -0.52 |
| HR | 200 | 57.50% | 52.50% | +5.00 |
| BLINK | 420 | 55.24% | 56.90% | -1.67 |
| OCRBench-v2 | 600 | 24.27% | 25.02% | -0.75 |
| MMMU-Pro | 300 | 32.00% | 35.67% | -3.67 |
| MathVista | 300 | 46.67% | 49.67% | -3.00 |
| MathVerse | 500 | 16.60% | 16.00% | +0.60 |

Softforce mode:

| Benchmark | n | data25k | 50k golden | Delta |
|---|---:|---:|---:|---:|
| VStar | 191 | 48.69% | 50.79% | -2.09 |
| HR | 200 | 57.00% | 59.50% | -2.50 |
| BLINK | 420 | 53.57% | 55.48% | -1.90 |
| OCRBench-v2 | 600 | 23.34% | 24.54% | -1.20 |
| MMMU-Pro | 300 | 32.33% | 38.00% | -5.67 |
| MathVista | 300 | 45.67% | 49.00% | -3.33 |
| MathVerse | 500 | 16.80% | 19.00% | -2.20 |

### Stage1 Internal Diagnostics

| Branch | Mean NLL correct D | Wrong-same beat rate | Query top1 | Query top2 | MRR | Mean diagonal gap |
|---|---:|---:|---:|---:|---:|---:|
| `data25k` | 1.2928 | 87.00% | 68.00% | 88.00% | 81.72% | +0.1067 |
| `data50k_golden` | 1.3334 | 90.50% | 77.00% | 91.50% | 86.92% | n/a |

The lower NLL for `data25k` should not be overread: retrieval and wrong-same
separation are weaker than the 50k golden anchor.

### Source Artifacts

| Branch | Mode | Summary path |
|---|---|---|
| `data25k` | subset manifest | `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_25k_clean_imend_from_50k_seed20260704/subset_manifest.json` |
| `data25k` | Stage1 internal readout | `outputs/clean_training/qwen3_stage1_ddeepstack_data25k_8gpu_20260704_202756/stage1_micro4/internal_diagnostics_step2000_20260704_202756/readout/readout_eval_report.json` |
| `data25k` | Stage1 internal query | `outputs/clean_training/qwen3_stage1_ddeepstack_data25k_8gpu_20260704_202756/stage1_micro4/internal_diagnostics_step2000_20260704_202756/query_sensitivity/query_sensitivity_report.json` |
| `data25k` | free | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_data25k_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_1_2_3_4_5_6_7_20260704_202756/tgvf_free/summary.json` |
| `data25k` | softforce | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_data25k_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_1_2_3_4_5_6_7_20260704_202756/tgvf_softforce/summary.json` |
| `data50k_golden` | free | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu1_7_20260703_084300/tgvf_free/summary.json` |
| `data50k_golden` | softforce | `outputs/clean_benchmarks/qwen3_stage2_ddeepstack_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_7_20260703_084300/tgvf_softforce/summary.json` |
