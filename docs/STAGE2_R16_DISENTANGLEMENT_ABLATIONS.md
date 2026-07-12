# Stage2 Rank-16 Disentanglement Ablations

## Status

- Suite status: running.
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
| R16-A rank-only | q/k/v/o + gate/up/down | full modules | 1.0 | Running |
| R16-B targets-only | q/v/o | full modules | 1.0 | Smoke passed; queued |
| R16-C row-only | q/k/v/o + gate/up/down | row only | 1.0 | Smoke passed; queued |
| R16-D evidence-0.2 | q/k/v/o + gate/up/down | full modules | 0.2 | Smoke passed; queued |
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
- tmux: `stage2_r16_disentanglement_20260712_121814`.
- R16-A W&B:
  `https://wandb.ai/mio_nora/tgvf-clean-qwen3-deepstack/runs/vkzcu54m`.

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
| R16-A | TBD | TBD | TBD | TBD | TBD |
| R16-B | TBD | TBD | TBD | TBD | TBD |
| R16-C | TBD | TBD | TBD | TBD | TBD |
| R16-D | TBD | TBD | TBD | TBD | TBD |

## Internal Results

| ID | Correct-D NLL | Wrong-same | Top-1 | Top-2 | MRR | D/V norm |
|---|---:|---:|---:|---:|---:|---:|
| Golden | 1.6373 | 40.50% | 23.50% | 49.00% | 0.5004 | 2.1198 |
| R16-A | TBD | TBD | TBD | TBD | TBD | TBD |
| R16-B | TBD | TBD | TBD | TBD | TBD | TBD |
| R16-C | TBD | TBD | TBD | TBD | TBD | TBD |
| R16-D | TBD | TBD | TBD | TBD | TBD | TBD |
| R16-H | 1.5840 | 51.50% | 28.00% | 53.50% | 0.5396 | 2.1659 |

## CoreDev-2511 Results

| ID | Free acc | Soft acc | Free trigger | Soft trigger | Trigger lift | Free mean tokens | Soft mean tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Golden | 37.04% | 37.92% | 29.79% | 40.98% | +11.19 pp | 72.77 | 69.56 |
| R16-A | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| R16-B | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| R16-C | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| R16-D | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| R16-H | 36.94% | 35.02% | 20.91% | 20.19% | -0.72 pp | 75.56 | 74.55 |

## Decision Policy

- Do not change golden defaults from a single aggregate score.
- Attribute a main effect only when its one-variable experiment differs from
  R16-A under the same evaluation identity.
- Treat the combined R16-H result as an interaction endpoint, not evidence for
  any individual component.
- Repeat promising configurations with another seed before promotion.
