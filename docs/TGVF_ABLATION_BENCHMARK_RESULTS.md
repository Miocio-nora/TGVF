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
