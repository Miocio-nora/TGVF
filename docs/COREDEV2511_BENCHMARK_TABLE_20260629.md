# CoreDev-2511 Benchmark Table, 2026-06-29

This table summarizes the clean Qwen3 CoreDev-2511 benchmark results, including
the full no-block rerun completed on 2026-06-29. Metrics are read from existing
`summary.json` files.

## CoreDev-2511 Identity

- Manifest id: `core_balanced_dev_2511_seed20260625`
- Manifest hash: `a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`
- Samples: `2511`
- Composition:

| Benchmark | n |
|---|---:|
| VStar | 191 |
| HRBench4K | 200 |
| BLINK | 420 |
| OCRBench v2 | 600 |
| MMMU-Pro | 300 |
| MathVista | 300 |
| MathVerse | 500 |

## Main CoreDev-2511 Results

| Run | Mode | Scope | Forward | n | Acc | Delta vs Original | Parse | Trigger | Focus | Append | Malformed | Output |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3 original final | `original` | `off` | `kv_cache` | 2511 | 30.86 | 0.00 | 94.15 | 0.00 | n/a | n/a | 0.00 | `outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_maxans512_20260628_0216_rescored_bboxfix_20260628_035759/merged` |
| TGVF free old | `tgvf_free` | `through_answer` | `no_kv_full_sequence` | 2511 | 32.07 | +1.21 | 98.73 | 18.64 | 18.64 | 96.79 | 0.60 | `outputs/clean_benchmarks/qwen3_stage2_norm01_free_coredev2511_deepstack512_recovery_4shard_20260628_0615/merged` |
| TGVF softforce old | `tgvf_softforce` | `through_answer` | `no_kv_full_sequence` | 2511 | 31.38 | +0.52 | 98.09 | 34.01 | 34.01 | 97.19 | 0.96 | `outputs/clean_benchmarks/qwen3_stage2_norm01_softforce_coredev2511_deepstack512_recovery_4shard_20260628_102713/merged` |
| TGVF free no-block | `tgvf_free` | `no_block` | `kv_cache` | 2511 | 32.64 | +1.79 | 97.37 | 18.64 | 18.64 | 89.96 | 1.87 | `outputs/clean_benchmarks/qwen3_stage2_norm01_coredev2511_kv_deepstack_no_block_4shard_20260629_201340/tgvf_free/merged` |
| TGVF softforce no-block | `tgvf_softforce` | `no_block` | `kv_cache` | 2511 | 32.65 | +1.80 | 95.10 | 34.01 | 34.01 | 88.41 | 3.94 | `outputs/clean_benchmarks/qwen3_stage2_norm01_coredev2511_kv_deepstack_no_block_4shard_20260629_201340/tgvf_softforce/merged` |

## Per-Benchmark CoreDev-2511 Results

| Benchmark | n | Original | Free old | Free no-block | Softforce old | Softforce no-block | Free trigger | Softforce trigger |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VStar | 191 | 53.93 | 49.74 | 49.74 | 48.17 | 50.26 | 16.75 | 63.87 |
| HRBench4K | 200 | 52.00 | 54.00 | 54.00 | 51.00 | 52.50 | 37.00 | 72.00 |
| BLINK | 420 | 46.43 | 55.48 | 55.56 | 53.57 | 54.05 | 8.10 | 23.33 |
| OCRBench v2 | 600 | 20.46 | 22.23 | 23.21 | 21.89 | 24.93 | 22.00 | 40.67 |
| MMMU-Pro | 300 | 34.33 | 28.19 | 28.42 | 27.76 | 28.67 | 14.33 | 18.00 |
| MathVista | 300 | 41.00 | 41.95 | 44.19 | 41.75 | 45.31 | 11.00 | 18.33 |
| MathVerse | 500 | 4.80 | 4.82 | 6.20 | 5.42 | 7.00 | 24.00 | 27.40 |
| Overall | 2511 | 30.86 | 32.07 | 32.64 | 31.38 | 32.65 | 18.64 | 34.01 |

## No-Block Delta Summary

| Comparison | Acc Delta | Parse Delta | Append Delta | Notes |
|---|---:|---:|---:|---|
| Free no-block vs free old | +0.57 | -1.35 | -6.84 | Same trigger rate. Accuracy gains mainly from OCRBench, MathVista, and MathVerse. |
| Softforce no-block vs softforce old | +1.27 | -2.99 | -8.78 | Same trigger rate. Strongest gains on OCRBench, MathVista, MathVerse, and VStar. |

The no-block full rerun is not a pure single-variable ablation against the old
CoreDev runs because the forward mode also changed from `no_kv_full_sequence`
to `kv_cache`. It is the current intended inference path.

## Latest HR200 No-Block Diagnostic

These are not full CoreDev-2511 runs. They use the HRBench4K 200-row slice
extracted from the CoreDev-2511 manifest:

- Manifest id: `diagnostic_hr_core200_from_coredev2511_seed20260625`
- Manifest hash: `705b29fc3c380d97a31947b54f3423f3c4da94da7fa79826c05ad8b2df297e76`
- Samples: `200`, all `hr_bench_4k_800`

| Run | Mode | Scope | Forward | n | Acc | Parse | Trigger | Focus | Append | Output |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Old CoreDev HR subset | `tgvf_free` | `through_answer` | `no_kv_full_sequence` | 200 | 54.00 | 97.50 | 37.00 | 37.00 | 100.00 | `outputs/clean_benchmarks/qwen3_stage2_norm01_free_coredev2511_deepstack512_recovery_4shard_20260628_0615/merged` |
| New HR200 no-block | `tgvf_free` | `no_block` | `kv_cache` | 200 | 54.00 | 97.50 | 37.00 | 37.00 | 100.00 | `outputs/clean_benchmarks/diagnostic_hr200_free_softforce_kv_deepstack_no_block_20260629/tgvf_free/merged` |
| Old CoreDev HR subset | `tgvf_softforce` | `through_answer` | `no_kv_full_sequence` | 200 | 51.00 | 99.50 | 72.00 | 72.00 | 100.00 | `outputs/clean_benchmarks/qwen3_stage2_norm01_softforce_coredev2511_deepstack512_recovery_4shard_20260628_102713/merged` |
| New HR200 no-block | `tgvf_softforce` | `no_block` | `kv_cache` | 200 | 52.50 | 99.50 | 72.00 | 72.00 | 100.00 | `outputs/clean_benchmarks/diagnostic_hr200_free_softforce_kv_deepstack_no_block_20260629/tgvf_softforce/merged` |

Interpretation:

- Free: HR200 no-block did not change accuracy versus the old CoreDev HR subset
  (`54.00 -> 54.00`).
- Softforce: HR200 no-block improved over the old CoreDev HR subset
  (`51.00 -> 52.50`, `+1.50`).
- The HR200 result is now superseded by the full CoreDev-2511 no-block rerun
  above, but it remains useful as a slice-level sanity check.

## Run Inventory

| Status | Run | Mode | n | Acc | Parse | Trigger | Scope | Forward | Notes |
|---|---|---|---:|---:|---:|---:|---|---|---|
| Superseded | `qwen3_original_coredev2511_4shard_scoringfix_20260628_010859` | `original` | 2511 | 19.37 | 84.03 | 0.00 | `off` | `kv_cache` | Earlier original baseline before max-answer and bbox/scoring fixes. |
| Superseded | `qwen3_original_coredev2511_4shard_maxans512_20260628_0216` | `original` | 2511 | 30.21 | 88.25 | 0.00 | `off` | `kv_cache` | Superseded by bboxfix rescore. |
| Main | `qwen3_original_coredev2511_4shard_maxans512_20260628_0216_rescored_bboxfix_20260628_035759` | `original` | 2511 | 30.86 | 94.15 | 0.00 | `off` | `kv_cache` | Current original CoreDev-2511 baseline. |
| Superseded | `qwen3_stage2_norm01_free_coredev2511_deepstack512_4shard_20260628_0407` | `tgvf_free` | 2511 | 28.86 | 29.39 | 4.78 | `through_answer` | `no_kv_full_sequence` | Pre-recovery run with severe parse/media issues. |
| Invalid | `qwen3_stage2_norm01_free_coredev2511_deepstack512_mediafix_4shard_20260628_0510` | `tgvf_free` | 2511 | n/a | 0.00 | 0.00 | `through_answer` | `no_kv_full_sequence` | Failed run; parse rate zero. |
| Invalid | `qwen3_stage2_norm01_free_coredev2511_deepstack512_mediafix_login_4shard_20260628_0520` | `tgvf_free` | 2511 | n/a | 0.00 | 0.00 | `through_answer` | `no_kv_full_sequence` | Failed run; parse rate zero. |
| Reference | `qwen3_stage2_norm01_free_coredev2511_deepstack512_recovery_4shard_20260628_0615` | `tgvf_free` | 2511 | 32.07 | 98.73 | 18.64 | `through_answer` | `no_kv_full_sequence` | Old free through-answer CoreDev-2511 result. |
| Reference | `qwen3_stage2_norm01_softforce_coredev2511_deepstack512_recovery_4shard_20260628_102713` | `tgvf_softforce` | 2511 | 31.38 | 98.09 | 34.01 | `through_answer` | `no_kv_full_sequence` | Old softforce through-answer CoreDev-2511 result. |
| Main | `qwen3_stage2_norm01_coredev2511_kv_deepstack_no_block_4shard_20260629_201340/tgvf_free` | `tgvf_free` | 2511 | 32.64 | 97.37 | 18.64 | `no_block` | `kv_cache` | Current free no-block CoreDev-2511 result. |
| Main | `qwen3_stage2_norm01_coredev2511_kv_deepstack_no_block_4shard_20260629_201340/tgvf_softforce` | `tgvf_softforce` | 2511 | 32.65 | 95.10 | 34.01 | `no_block` | `kv_cache` | Current softforce no-block CoreDev-2511 result. |
