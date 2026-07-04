# TGVF Ablation Benchmark Results

Date: 2026-07-04

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
