# Stage2 Rank-16 Disentanglement Ablations

## Status

- Suite status: completed.
- Mainline status: unchanged.
- The authoritative recipe remains the 2026-07-03 golden D-DeepStack Stage2
  run.
- This suite fixes LoRA rank at 16 and separates the three changes bundled in
  the first narrow-LoRA experiment.

## Question

At fixed rank 16 and alpha 64, which of target-module narrowing, protocol-token
row-only training, and evidence down-weighting contributes useful benchmark
behavior, and which changes damage tool triggering or reasoning-heavy tasks?

## Experiment Matrix

| ID | LoRA targets | Protocol-token training | Evidence weight | Status |
|---|---|---|---:|---|
| R16-A rank-only | q/k/v/o + gate/up/down | full modules | 1.0 | Done |
| R16-B targets-only | q/v/o | full modules | 1.0 | Done |
| R16-C row-only | q/k/v/o + gate/up/down | row only | 1.0 | Done |
| R16-D evidence-0.2 | q/k/v/o + gate/up/down | full modules | 0.2 | Done |
| R16-H combined | q/v/o | row only | 0.2 | Done |

R16-H is the completed combined experiment documented in
`docs/STAGE2_NARROW_LORA_ABLATION.md`. It is not rerun by this suite.

## Fixed Training Identity

- Golden Stage1 checkpoint:
  `outputs/clean_training/qwen3_stage1_ddeepstack_norm01_wandb_4gpu_20260702_180906/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt`
- Stage1 checkpoint sha256:
  `b119379fc13a3eee1d19fb347bda729262592599218ea1ff88733ba142cb0c0b`
- Stage2 train data:
  `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl`
- Train rows / sha256: `46883` /
  `b5027e72dda7601073ddb8bc9cf1853ec564fa415a3e9cb7d1e684cc7c0d733b`
- Stage2 validation data:
  `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`
- Validation rows / sha256: `1002` /
  `3b719e1bd03a09741cc05dac3ed85e423a9b2c29209b07546792a6795cdbbfc5`
- Model and processor:
  `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`
- Batch: `8 GPUs * micro batch 4 * accumulation 4 = 128`.
- Steps / save / eval cadence: `1200 / 300 / 300`.
- Seed: `20260525`.
- Resolution / sequence length: `512 / 2048`.
- Training attention: `sdpa`, dtype `bfloat16`.
- DeepStack: enabled, original-image scope `through_answer`.
- D DeepStack: enabled, branch layers `[8,16,24]` from golden Stage1.
- Original-image mask: probability `0.75`, scope `through_answer`.
- LoRA dropout / scaling: `0.05`, alpha/rank `4`.
- Optimizer, learning rates, scheduler, all non-evidence span weights, TGVF,
  visual-token manifold weight, and target focus ratio match golden Stage2.
- Main runs use W&B online project `tgvf-clean-qwen3-deepstack`.

## Fixed Evaluation Identity

- Internal diagnostic data:
  `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl`
- Internal rows / sha256: `867` /
  `de61c731eb961825a77df587cd76c00eabfea75b5c6003096f3cc7f1a51dd82d`
- Internal tasks: readout 200, distribution 200, query groups 50.
- External manifest:
  `revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json`
- Manifest rows / file sha256: `2511` /
  `3a013b2bcc64316054d28239a3cea3f44211cadbfe19787be3b7f285620fa5c1`
- Internal manifest hash:
  `a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`
- Modes: `tgvf_free` and `tgvf_softforce`.
- Softforce prompt: `Use focus tool.`.
- Backend: `tgvf_stage2_qwen3_native`, FlashAttention-2, bfloat16.
- DeepStack: enabled, original-image scope `no_block`, D DeepStack enabled.
- Post-D continuation: `kv_cache`.
- Resolution / unified generation budget: `512 / 512`.
- Scoring backend: `auto`.

## Execution

Each experiment runs this gated sequence:

1. Write and compare the formal training plan.
2. Complete an 8-GPU one-step training smoke.
3. Train Stage2 for 1200 optimizer steps.
4. Complete a 4-row external benchmark smoke.
5. Run internal diagnostics on GPU0 while complete CoreDev-2511 free runs on
   GPUs1-7.
6. Run complete CoreDev-2511 softforce on GPUs0-7.
7. Count malformed rows as wrong in strict accuracy in addition to preserving
   the scorer-reported accuracy.

- Suite id: `stage2_r16_disentanglement_20260712_121814`.
- Driver: `scripts/run_stage2_r16_disentanglement.sh`.
- Suite output:
  `outputs/clean_ablation/stage2_r16_disentanglement_20260712_121814`.

### Plan And Smoke Gate

| ID | Main plan sha256 | LLM trainable | Smoke loss | Validation | Peak GB |
|---|---|---:|---:|---:|---:|
| R16-A | `faf25f744188c388f9a4e6b43db5e883a521f19207ddbda1cbbaef3a9f6ae384` | 1,286,152,192 | 4.2344 | 4.2500 | 68.59 |
| R16-B | `e747b172b839a6b2088bf67ddf432bbbb2e032e1dba8c0e0634d4cac42fd6c27` | 1,254,891,520 | 4.2344 | 4.2500 | 52.62 |
| R16-C | `3e1aaaa469261afffbf025392c30757cc2f5cc1f626348f85dd66d05484b621a` | 43,679,744 | 4.2500 | 4.2500 | 65.16 |
| R16-D | `9beb47791eb80961fe1404915bdf0b08dc8c6c60a00b2b0ae64a8a943c4b9158` | 1,286,152,192 | 4.6484 | 4.9688 | 68.60 |

All smokes completed one optimizer step from four micro steps, ran validation,
and saved a checkpoint whose state checks passed. TGVF trainable parameters
are fixed at 72,055,808 in all four runs. After normalizing artifact paths and
declared ablation fields, R16-A matches the golden plan and R16-B/C/D match
R16-A exactly.

- Formal queue started: `2026-07-12 12:33:24 JST`.
- Formal queue finished: `2026-07-13 02:32:44 JST`.
- Queue wall time: `13h 59m 20s`.
- tmux: `stage2_r16_disentanglement_20260712_121814`.
- W&B runs: R16-A `vkzcu54m`, R16-B `sqr3ckcw`, R16-C `0vq6jcoe`,
  R16-D `62h5chag`.

## Primary Metrics

- Reasoning macro: strict accuracy mean over MMMU-Pro, MathVista, MathVerse.
- Tool responsiveness: softforce trigger rate minus free trigger rate.
- Accuracy responsiveness: softforce accuracy minus free accuracy.
- Visual/OCR accuracy: VStar, HR-Bench 4K, BLINK, OCRBench v2.
- Internal: correct-D NLL, wrong-same discrimination, Top-1/Top-2, MRR,
  D/V-merge norm ratio, finite rate, collapse warning.
- Output health: overall/direct/triggered mean tokens, p50/p90/p95, hit-512,
  malformed rows, and append failures.

## Training Results

| ID | Train time | Final loss | Validation loss | W&B | Checkpoint |
|---|---:|---:|---:|---|---|
| R16-A | 2:45:51 | 0.8809 | 0.9219 | `vkzcu54m` | `fdf3b2b1...` |
| R16-B | 2:24:00 | 0.9893 | 1.0859 | `sqr3ckcw` | `a27ea300...` |
| R16-C | 2:42:04 | 0.8857 | 0.9219 | `0vq6jcoe` | `dc550559...` |
| R16-D | 2:42:41 | 0.8076 | 0.7891 | `62h5chag` | `cc505421...` |

Every main run completed 1200 optimizer steps and published a valid final
checkpoint. The checkpoint column shows the sha256 prefix.

## Internal Results

| ID | Correct-D NLL | Wrong-same | Top-1 | Top-2 | MRR | D/V norm |
|---|---:|---:|---:|---:|---:|---:|
| Golden | 1.6373 | 40.50% | 23.50% | 49.00% | 0.5004 | 2.1198 |
| R16-A | 1.6101 | 39.50% | 18.50% | 43.50% | 0.4631 | 2.1150 |
| R16-B | 1.5968 | 47.50% | 32.00% | 53.00% | 0.5527 | 2.1784 |
| R16-C | 1.6097 | 34.00% | 18.50% | 41.00% | 0.4586 | 2.0905 |
| R16-D | 1.6161 | 44.00% | 22.00% | 51.50% | 0.4993 | 2.1328 |
| R16-H | 1.5840 | 51.50% | 28.00% | 53.50% | 0.5396 | 2.1659 |

All variants had finite rate 1.0 and no collapse warning. R16-B has the best
internal query metrics among A-D while producing the worst external results;
internal retrieval is therefore not a sufficient model-selection metric.

## CoreDev-2511 Results

The table below uses strict all-row accuracy, counting malformed rows as wrong.

| ID | Free acc | Soft acc | Free trigger | Soft trigger | Trigger lift | Free mean tokens | Soft mean tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Golden | 37.04% | 37.92% | 29.79% | 40.98% | +11.19 pp | 72.77 | 69.56 |
| R16-A | 36.96% | 36.60% | 36.16% | 49.30% | +13.14 pp | 74.72 | 67.92 |
| R16-B | 35.51% | 34.73% | 29.11% | 31.94% | +2.83 pp | 74.15 | 70.01 |
| R16-C | **37.57%** | **37.11%** | 38.87% | 50.62% | +11.75 pp | 72.24 | 66.65 |
| R16-D | 37.55% | 36.88% | 34.65% | 46.44% | +11.79 pp | 71.32 | 66.23 |
| R16-H | 36.91% | 34.98% | 20.91% | 20.19% | -0.72 pp | 75.56 | 74.55 |

R16-B had 11 free and 12 softforce empty-generation failures with
`target_hidden_states must contain at least one token`. All were counted wrong
above. A/C/D had zero malformed rows; every variant had zero append failures.
All manifests contain the exact expected 2511 sample ids in the expected order.

## Main Effects Relative To R16-A

| Change | Free acc | Soft acc | Free reasoning macro | Soft reasoning macro | Free trigger | Soft trigger |
|---|---:|---:|---:|---:|---:|---:|
| q/v/o targets only | -1.45 pp | -1.87 pp | -2.69 pp | -4.80 pp | -7.05 pp | -17.36 pp |
| Protocol rows only | +0.62 pp | +0.51 pp | +0.69 pp | -1.13 pp | +2.71 pp | +1.31 pp |
| Evidence weight 0.2 | +0.59 pp | +0.28 pp | +0.84 pp | +0.09 pp | -1.51 pp | -2.87 pp |

Target narrowing is the only large, consistently negative main effect. Row-only
and evidence down-weighting each provide small aggregate improvements, but
neither restores reasoning-heavy performance.

## Reasoning Macro

Reasoning macro is the strict accuracy mean over MMMU-Pro, MathVista, and
MathVerse.

| ID | Free | Softforce | Soft minus free |
|---|---:|---:|---:|
| Golden | 33.78% | **35.33%** | +1.56 pp |
| R16-A | 33.20% | 33.22% | +0.02 pp |
| R16-B | 30.51% | 28.42% | -2.09 pp |
| R16-C | 33.89% | 32.09% | -1.80 pp |
| R16-D | **34.04%** | 33.31% | -0.73 pp |
| R16-H | 32.00% | 29.96% | -2.04 pp |

Rank 16 alone does not preserve the golden softforce reasoning gain. Evidence
0.2 is the strongest A-D result for free reasoning macro, but remains below
golden softforce and is not a general reasoning recovery.

## Per-Benchmark Accuracy

### Free

| ID | VStar | HR | BLINK | OCR | MMMU | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|
| Golden | 52.36% | 52.50% | 56.90% | 25.02% | 35.67% | 49.67% | 16.00% |
| R16-A | 52.36% | 54.50% | 56.19% | 25.16% | 36.00% | 47.00% | 16.60% |
| R16-B | 47.64% | 50.50% | **59.52%** | 24.44% | 33.33% | 44.00% | 14.20% |
| R16-C | 52.36% | **56.00%** | 55.95% | **25.91%** | 35.00% | 48.67% | **18.00%** |
| R16-D | 51.83% | 55.00% | 57.62% | 24.98% | 33.00% | **51.33%** | 17.80% |
| R16-H | **56.54%** | **56.00%** | 55.71% | 25.46% | 32.00% | 48.00% | 16.00% |

### Softforce

| ID | VStar | HR | BLINK | OCR | MMMU | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|
| Golden | 50.79% | **59.50%** | 55.48% | 24.54% | 38.00% | **49.00%** | **19.00%** |
| R16-A | 51.31% | 57.00% | 55.24% | 24.68% | **38.67%** | 47.00% | 14.00% |
| R16-B | 50.26% | 54.00% | 56.90% | **25.00%** | 29.00% | 44.67% | 11.60% |
| R16-C | 51.83% | 59.00% | **57.38%** | 24.99% | 29.67% | **49.00%** | 17.60% |
| R16-D | 51.83% | 55.50% | 55.24% | 24.84% | 35.33% | 47.00% | 17.60% |
| R16-H | **52.36%** | 52.00% | 54.52% | 24.55% | 29.67% | 46.00% | 14.20% |

## Output Health

### Free

| ID | Mean | Direct | Triggered | p50 | p90 | p95 | Hit 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Golden | 72.77 | 72.15 | 74.24 | 42 | 141 | 250 | 57 |
| R16-A | 74.72 | 74.32 | 75.42 | 43 | 142 | 262 | 53 |
| R16-B | 74.15 | 78.38 | 63.86 | 46 | 126 | 261 | 76 |
| R16-C | 72.24 | 73.21 | 70.71 | 43 | 136 | 254 | 51 |
| R16-D | 71.32 | 75.67 | 63.13 | 44 | 134 | 205 | 52 |
| R16-H | 75.56 | 79.32 | 61.33 | 47 | 132 | 241 | 76 |

### Softforce

| ID | Mean | Direct | Triggered | p50 | p90 | p95 | Hit 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Golden | 69.56 | 75.06 | 61.63 | 40 | 131 | 233 | 46 |
| R16-A | 67.92 | 74.06 | 61.59 | 41 | 133 | 215 | 33 |
| R16-B | 70.01 | 74.43 | 60.60 | 45 | 119 | 197 | 60 |
| R16-C | 66.65 | 75.20 | 58.30 | 39 | 125 | 224 | 42 |
| R16-D | 66.23 | 76.51 | 54.37 | 42 | 125 | 184 | 37 |
| R16-H | 74.55 | 77.32 | 63.56 | 47 | 129 | 234 | 73 |

## Conclusions

1. Rank 16 alone leaves free accuracy almost unchanged from golden
   (`-0.09` points), but softforce accuracy falls `1.32` points and the golden
   reasoning-macro lift disappears. Lower rank is not a reasoning-preservation
   solution by itself.
2. Restricting LoRA to q/v/o is clearly harmful: it lowers external accuracy,
   reasoning macro, softforce responsiveness, and output health despite the
   strongest internal Top-1/MRR. This explains a substantial part of the
   combined R16-H failure.
3. Protocol row-only training is a small aggregate positive relative to R16-A
   and uses far fewer trainable LLM parameters, but its softforce reasoning
   macro remains below A and golden. It is promising for efficiency and
   general accuracy, not demonstrated reasoning preservation.
4. Evidence weight 0.2 is also a small aggregate positive and gives the best
   A-D free reasoning macro, mainly through MathVista/MathVerse while MMMU
   declines. The effect is task-specific.
5. None of A-D restores original reasoning behavior. The proposed no-focus
   original-Qwen reasoning replay remains the most targeted low-cost follow-up.
6. Golden D-DeepStack Stage2 remains authoritative. R16-C and R16-D merit
   follow-up only as named ablations, with another seed before promotion.

## Decision Policy

- Do not change golden defaults from a single aggregate score.
- Attribute a main effect only when its one-variable experiment differs from
  R16-A under the same evaluation identity.
- Treat the combined R16-H result as an interaction endpoint, not evidence for
  any individual component.
- Repeat promising configurations with another seed before promotion.
