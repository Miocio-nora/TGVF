# Authoritative Stage1/Stage2 Training Configuration

更新时间：2026-07-02

本文档记录当前 clean project 的 **最新权威 Stage1 -> Stage2 训练配置**。以后没有明确命名 ablation 时，Stage1/Stage2 训练默认值应与本文档保持一致。

权威配置来自这条已完成训练线：

- Stage1 run: `clean_qwen3_stage1_norm01_manifold0_4gpu_20260627_133252`
- Stage1 plan: `outputs/clean_training/qwen3_stage1_norm01_manifold0_4gpu_20260627_133252/stage1_micro4/training_plan.json`
- Stage1 checkpoint: `outputs/clean_training/qwen3_stage1_norm01_manifold0_4gpu_20260627_133252/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt`
- Stage1 checkpoint sha256: `ab6bd554cfb405208f13270c298f7fd0ab01305c83ca07bf8eca667cbe1632a1`
- Stage2 run: `clean_qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250`
- Stage2 plan: `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/training_plan.json`
- Stage2 checkpoint: `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`

## Scope

权威配置定义的是训练 recipe，不是固定机器配置：

- 保持 `global_batch`，但不固定 GPU id 或 GPU 数。
- 实际运行必须记录 `world_size * micro_batch * gradient_accumulation_steps`。
- 默认 `model_id` 保持为 `Qwen/Qwen3-VL-8B-Thinking`，权威 run 实际使用本地 mirror `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`。
- 权威训练 run 使用 `attn_implementation=sdpa`。FlashAttention-2 是当前外部 benchmark/eval 默认和训练加速候选，不自动改写这条训练 recipe。

## Stage1 Defaults

| Field | Authoritative value |
|---|---:|
| protocol | `protocol_c_tool_observation` |
| variant | `tgvf_v2_bidirectional` |
| max image resolution | `512` |
| max steps | `2000` |
| save every | `500` |
| global batch | `32` |
| token row mode | `row_only` |
| capture mode | `teacher_forced` |
| FVT position mode | `native_source_grid` |
| focus action im_end | `true` |
| mask original image after TGVF | `true` |
| same-image negative mode | `matrix_ce` |
| readout batch size | `4` |
| optimizer | AdamW |
| learning rate | `1e-4` |
| LR scheduler | `cosine` |
| warmup steps | `100` |
| min LR ratio | `0.1` |
| max grad norm | `1.0` |
| loss gen | `1.0` |
| visual token manifold loss | `0.0` |
| visual token norm loss | `0.1` |
| same-image negative loss | `1.0` |

权威 run 的 batch identity 是 `32 = 4 * 4 * 2`。默认 launcher 仍可在单进程下解析为 `32 = 1 * 1 * 32`；换设备数时必须保持 global batch，除非这是明确命名的 ablation。

Stage1 train data:

```text
data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl
sha256: c94a38b824b6603e555eed5ef3584c19cc903b76995d49c67ace36b18268443c
rows: 39998
```

## Stage2 Defaults

| Field | Authoritative value |
|---|---:|
| protocol | `protocol_c_tool_observation` |
| stage2 path | `fast_batched` |
| use Stage1 TGVF config | `true` |
| variant | `tgvf_v2_bidirectional` |
| max image resolution | `512` |
| max sequence length | `2048` |
| max steps | `1200` |
| save/eval every | `300` |
| global batch | `128` |
| target focus ratio | `0.8` |
| mask original image after TGVF | `true` |
| mask probability | `0.75` |
| mask scope | `through_answer` |
| DeepStack enabled | `true` |
| DeepStack original image scope | `through_answer` |
| D DeepStack-like features | `false` |
| LoRA rank/alpha/dropout/bias | `64 / 256 / 0.05 / none` |
| LoRA target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| optimizer | AdamW |
| lr_lora | `2e-5` |
| lr_tgvf | `5e-6` |
| lr_calibration | `1e-5` |
| LR scheduler | `cosine` |
| warmup steps | `100` |
| warmup ratio | `0.03` |
| min LR ratio | `0.1` |
| Adam betas | `[0.9, 0.95]` |
| Adam eps | `1e-8` |
| weight decay | `0.01` |
| max grad norm | `1.0` |
| visual token manifold loss | `0.0` |

Stage2 weighted span loss:

```text
evidence_state: 0.2
focus_target: 1.5
evidence: 1.0
value_span: 1.0
answer: 1.0
no_focus_evidence_state: 0.2
no_focus_answer: 1.0
```

权威 run 的 batch identity 是 `128 = 4 * 4 * 8`。默认 launcher 仍可在单进程下解析为 `128 = 1 * 1 * 128`；换设备数时必须保持 global batch，除非这是明确命名的 ablation。

Stage2 data and Stage1 checkpoint identity:

```text
stage1_checkpoint:
outputs/clean_training/qwen3_stage1_norm01_manifold0_4gpu_20260627_133252/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt
sha256: ab6bd554cfb405208f13270c298f7fd0ab01305c83ca07bf8eca667cbe1632a1

train:
data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl
sha256: b5027e72dda7601073ddb8bc9cf1853ec564fa415a3e9cb7d1e684cc7c0d733b
rows: 46883

val:
data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl
sha256: 3b719e1bd03a09741cc05dac3ed85e423a9b2c29209b07546792a6795cdbbfc5
rows: 1002
```

## D DeepStack Ablation

`D DeepStack` 是在上述权威配置基础上的 ablation，不是默认权威 recipe 的一部分。默认保持：

```text
Stage1 d_deepstack_enabled: false
Stage2 deepstack.d_features_enabled: false
```

打开该 ablation 时，D path 保持原有 frozen Qwen visual merger finalize 路径；额外为 D token 构造自己的 conditioned DeepStack branch features：

- branch layers 默认仍为 `8,16,24`。
- 每个 branch 使用独立的 `tgvf_v2_bidirectional` adapter，和主 D adapter 的操作形式一致。
- branch 输入来自同一次 vision forward 捕获的 cached pre-merge hidden states，不重新 encode 整张图。
- Stage1 readout 只把 D DeepStack features 注入 D token positions。
- Stage2 fast batched path 在最终 focus/action 序列里同时注入 original image DeepStack 和 D DeepStack，并按 token position 合并排序。
- Matrix CE 交换 negative D 时，D merge tokens 和 D DeepStack branch features 一起交换。
- Norm loss 在该 ablation 中同时监督主 D norm 和 D DeepStack branch norm；CE 与 matrix CE 目标不变。

Stage1 入口：

```text
--d-deepstack-enabled --d-deepstack-branch-layers 8,16,24
```

Stage2 入口：

```text
--deepstack-enabled --d-deepstack-enabled --fast-batched-stage2
```

## Default Sync

The clean defaults in code are expected to match this authority for configurable recipe values:

- `revisit_vlm_clean.cli.train_stage1 --print-defaults`
- `revisit_vlm_clean.cli.train_stage2 --print-defaults`
- `Stage1LaunchConfig`
- `Stage2LaunchConfig`

Paths that identify a concrete dataset, checkpoint, output directory, processor mirror, GPU list, or W&B run must still be supplied or recorded per run. They are not silently hardcoded into the launcher defaults.
