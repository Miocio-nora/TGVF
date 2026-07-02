# Pro Handoff: 当前 TGVF / Clean Project 讲解文档

更新时间：2026-07-02

本文档用于给已经落后当前工程状态的 Pro / 协作者快速对齐上下文。目标不是替代代码，而是把项目宗旨、当前 clean project 状态、最新 protocol、训练/评测入口、benchmark 体系、mask/DeepStack 语义，以及不能再犯的比较错误讲清楚。

当前权威实现入口在 `revisit_vlm_clean/`。历史 root project 仍可作为 reference，少数路径也可能通过明确标注的 diagnostic bridge 复用历史实现；但是**清理旧 project 不属于当前 clean project 的完成目标**。

当前权威 Stage1 -> Stage2 训练配置记录在
`docs/AUTHORITATIVE_STAGE1_STAGE2_CONFIG.md`。没有明确命名 ablation 时，
Stage1/Stage2 training launcher 默认值应与该文档保持一致。

## 1. 项目一句话宗旨

TGVF 的目标是让 VLM 在回答问题时能够判断是否需要局部视觉证据；如果需要，就生成一个 focus target，把这个 target 转成一段 learned visual tokens `D`，再把 `D` 作为视觉证据接回对话，让模型继续自然回答。

这不是简单 crop，也不是只靠 prompt 让模型看局部区域。当前主线机制是：

```text
问题 + 原图
  -> 模型生成 focus target
  -> TGVF/foveal module 生成 D visual tokens
  -> 对话中 append <|tgvf_start|> D <|tgvf_end|>
  -> 模型继续生成 evidence / answer
```

当前工程目标也不只是模型效果，而是让训练和评测**可复现、可审计、可比较**。任何训练/评测结果都必须绑定 exact checkpoint、processor、sample ids、protocol、parser/scorer、mask/DeepStack 状态、forward mode、GPU/batch 数学和输出目录。

## 2. 当前工程状态

当前 clean project 的角色：

- `revisit_vlm_clean/` 已经是主入口，不再只是 skeleton。
- 已覆盖 deterministic data transforms、Stage1/Stage2 training plan/executor、clean benchmark evaluation、ValKit handoff/execution、shard merge。
- 旧 root implementation 可以读作 reference。
- historical Stage2 bridge 只保留为明确的 diagnostic backend。
- 旧 project 的删除、归档、瘦身、重命名不属于当前目标，不能顺手做。

写本文档时的关键状态：

```text
branch: clean/tgvf-clean-project-20260625
baseline commit: 19a6bb2 document clean project current entrypoints
本地未跟踪资产: logs/, third_party/
```

`logs/` 和 `third_party/` 是本地资产，按当前边界不提交。

## 3. Clean Project 入口总览

脚本入口定义在 `revisit_vlm_clean/pyproject.toml`。

主入口：

```bash
tgvf_build_manifest
tgvf_generate_data
tgvf_eval_benchmark
tgvf_merge_benchmark
tgvf_eval_valkit
tgvf_train_stage1
tgvf_train_stage2
tgvf_train_stage1_executor
tgvf_train_stage2_executor
```

未安装 package 时，用：

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.manifest --list
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.generate_data --help
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.benchmark --help
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.valkit --help
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.train_stage1 --print-defaults
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.train_stage2 --print-defaults
```

## 4. 核心术语

`TGVF`：target-guided visual focus pipeline。

`focus target`：模型生成的局部视觉证据描述，用 protocol focus tags 包起来。

`D` / `FVT`：由 focus target 和图像上下文生成的 learned visual token span。当前主线里，D 是 v-merge-level visual tokens，不默认带自己的 DeepStack-like features。

`original image`：原始图像输入对应的常规视觉 tokens / visual keys。

`post-TGVF append`：focus target 生成后，把 D 接入对话：

```text
<|tgvf_start|>
[D visual tokens]
<|tgvf_end|>
```

然后模型自然继续写。当前 clean benchmark continuation 只保留：

```text
natural_continue
```

`DeepStack`：Qwen3-VL 原生的 original-image 多层视觉特征注入机制。它和普通 attention mask 不是一回事。当前主线 DeepStack 只作用于 original image features，不给 D 加 DeepStack-like features。

`no_block`：启用 original-image DeepStack，但 post-TGVF 后不屏蔽 original image access。

`through_answer`：post-TGVF 之后一直到 answer 都屏蔽 original image access。

`evidence_only`：D/evidence 阶段屏蔽 original image access，answer 阶段恢复。

## 5. 当前 Protocol Surface

clean project 当前支持：

```text
protocol_c_tool_observation
protocol_c_tool_observation_qwen2_no_think
```

主线是：

```text
protocol_c_tool_observation
```

Qwen2 诊断分支使用：

```text
protocol_c_tool_observation_qwen2_no_think
```

Qwen2 不应被当作 Qwen3 主线结论来源。

## 6. Protocol C Tool Observation

Qwen3 主线关键 tags：

```text
<think>
</think>
<|focus_start|>
<|focus_end|>
<|tgvf_start|>
<|tgvf_end|>
<|im_start|>tool
<|im_end|>
```

Qwen3 force mode 的 action prefix 形态：

```text
<think>
</think>
<|focus_start|>
```

有效 focus target 之后，tool observation wrapper 形态：

```text
<|im_start|>tool
<|tgvf_start|>
[D visual tokens]
<|tgvf_end|><|im_end|>
<|im_start|>assistant
```

然后模型自然继续生成。注意：

- `tgvf_free` 当前没有额外 prompt。
- `tgvf_force` 只加 protocol control prefix。
- `tgvf_softforce` 才会加配置里的短 soft-force prompt。

Qwen2 no-think variant 保持 tool-observation 结构，但 force action prefix 去掉 Qwen3 thinking wrapper：

```text
<|focus_start|>
```

## 7. Parser 语义

当前 clean action parser：

```text
revisit_vlm_clean.tgvf_protocol.parse_tgvf_action
```

它会抽取：

- `<|focus_start|>` 到 `<|focus_end|>` 中间的 focus target；
- protocol 对应的 answer；
- malformed reasons，例如 repeated focus tags、missing closing focus、empty focus、generic target、legacy tags in protocol C。

当前 benchmark answer parser/scorer identity：

```text
revisit_vlm_clean.scoring.parse_and_score:v3_external
revisit_vlm_clean.scoring.extract_choice_strict
```

不要混用旧 parser 或临时 parser 来解释新结果。

## 8. Eval Modes

clean benchmark eval modes：

```text
original
tgvf_free
tgvf_force
tgvf_softforce
```

含义：

- `original`：原始 Qwen 路径，不走 TGVF。
- `tgvf_free`：没有额外 prompt，模型自然决定是否触发 focus。
- `tgvf_force`：强制 focus action prefix，然后 append D 并继续。
- `tgvf_softforce`：加入配置的短 soft-force prompt，让 router 自然决定。

当前 clean continuation 只保留：

```text
natural_continue
```

forward modes 仍在配置面暴露：

```text
kv_cache
no_kv_full_sequence
```

但是不要再把结果简单解释成 “KV vs no-KV”。如果要比较 forward mode，必须控制 DeepStack、position ids、rope deltas、prompt/continuation、parser、sample identity。对 Qwen3 来说，DeepStack on/off 是一等变量。

## 9. Stage1 训练默认配置

Stage1 负责训练 focus visual token / readout 相关能力，即 focus target 到 D 的基础能力。

查看默认值：

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.train_stage1 --print-defaults
```

当前默认值：

```text
model_id: Qwen/Qwen3-VL-8B-Thinking
protocol: protocol_c_tool_observation
variant: tgvf_v2_bidirectional
max_image_resolution: 512
max_steps: 2000
save_every: 500
global_batch_size: 32
world_size: 1
micro_batch_size: 1
gradient_accumulation_steps: 32
token_row_mode: row_only
capture_mode: teacher_forced
fvt_position_mode: native_source_grid
focus_action_im_end: true
same_image_negative: matrix_ce
visual_token_manifold_loss: 0.0
visual_token_norm_loss: 0.1
lr_scheduler: cosine
warmup_steps: 100
min_lr_ratio: 0.1
max_grad_norm: 1.0
```

Stage1 clean 决策：

- 只保留 `row_only`。
- 只保留 `teacher_forced`，旧 `decode_loop` 不进 clean launcher。
- 只保留 `native_source_grid`。
- `matrix_ce` 是默认 same-image-negative / contrastive mode。
- `cyclic_margin` 仍是允许的命名 alternative，没有删除。

## 10. Stage2 训练默认配置

Stage2 负责完整对话行为，包括 focus/no-focus routing、focus target、evidence、value span、answer，并接入 Stage1 的 D 能力。

查看默认值：

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.train_stage2 --print-defaults
```

当前默认值：

```text
model_id: Qwen/Qwen3-VL-8B-Thinking
protocol: protocol_c_tool_observation
stage2_path: fast_batched
max_image_resolution: 512
max_steps: 1200
global_batch_size: 128
world_size: 1
micro_batch_size: 1
gradient_accumulation_steps: 128
target_focus_ratio: 0.8
mask_original_image_after_tgvf: true
mask_original_image_after_tgvf_prob: 0.75
mask_original_image_after_tgvf_scope: through_answer
deepstack_enabled: true
deepstack_original_image_scope: through_answer
deepstack_d_features_enabled: false
deepstack_supported: true
lora_rank: 64
lora_alpha: 256
lora_dropout: 0.05
lora_bias: none
lr_lora: 2e-5
lr_tgvf: 5e-6
lr_calibration: 1e-5
lr_scheduler: cosine
warmup_ratio: 0.03
warmup_steps: 100
min_lr_ratio: 0.1
adam_betas: [0.9, 0.95]
adam_eps: 1e-8
weight_decay: 0.01
max_grad_norm: 1.0
loss_visual_token_manifold: 0.0
```

当前 Stage2 weighted span loss 默认：

```text
evidence_state: 0.2
focus_target: 1.5
evidence: 1.0
value_span: 1.0
answer: 1.0
no_focus_evidence_state: 0.2
no_focus_answer: 1.0
```

当前 Stage2 LoRA targets：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Stage2 重要边界：

- Stage2 必须绑定 exact Stage1 checkpoint。
- clean benchmark runner 当前只支持 `d_condition=correct_D`。
- clean benchmark runner 当前只支持 `force_prefix_mode=target_hint`。
- `mask_original_image_after_tgvf_prob` 和 mask scope 必须按实验显式记录；
  当前权威 Stage2 训练默认是 mask prob `0.75`。
- Stage2 training 默认启用 DeepStack，scope 是 `through_answer`。
- 当前 benchmark/inference 主线默认是 `DeepStack enabled +
  original_image_scope=no_block + post_tgvf_forward_mode=kv_cache`。
- 旧 `through_answer + no_kv_full_sequence` CoreDev 结果降为
  side/reference，不再作为主表默认。
- `evidence_only` 仍支持，但必须作为独立 setting，不要和 no-block 或
  through-answer 混比。

## 11. DeepStack 和 Mask 语义

DeepStack 的默认值按 surface 区分：

- Stage2 training 默认开启，`original_image_scope=through_answer`，与当前权威
  Stage2 训练线一致。
- benchmark/inference 默认开启，`original_image_scope=no_block`，与
  2026-07-02 CoreDev-2511 retest 一致。
- 其它 surface 必须显式记录 DeepStack state，不要靠口头默认。

启用 DeepStack 时：

- 从 Qwen3 原生 image feature output 捕获 original-image DeepStack features；
- 把这些 features 带入 post-TGVF append path；
- `no_block`：启用 original-image DeepStack，但 post-D 原图 visual keys
  继续可见；
- `through_answer`：through answer 都屏蔽 original-image DeepStack；
- `evidence_only`：D/evidence 阶段屏蔽，answer boundary 之后恢复；
- D 仍然是 v-merge-level visual-token span；
- D DeepStack-like features 不是 clean default，只能作为明确命名 ablation 加。

当前支持状态：

- Clean Qwen3 Stage2 training 支持 enabled DeepStack 的 `through_answer` 和 `evidence_only`。
- Clean Qwen3 Stage2 benchmark eval 支持 full-sequence/KV DeepStack 的
  `no_block`、`through_answer` 和 `evidence_only`。
- 当前 Qwen3 benchmark/inference 默认使用 `no_block + kv_cache`。
- Legacy bridge DeepStack 会被拒绝，不能默默当成有效结果。

## 12. Benchmark Sets

当前 clean core populations：

```text
vstar_test_questions_191: VStar, n=191
hr_bench_4k_800: HR-Bench-4K, n=800
blink_val_all_subtasks_1901: BLINK all subtasks, n=1901
ocrbench_v2_data_test_10000: OCRBench-v2, n=10000
mmmu_pro_standard10_test_1730: MMMU-Pro standard 10 options, n=1730
mathvista_testmini_1000: MathVista testmini, n=1000
mathverse_testmini_3940: MathVerse testmini, n=3940
```

当前 clean subsets：

```text
CoreSmoke-256: 快速代码/parser/scorer/DeepStack 字段 smoke，不用于效果结论
CoreDev-2511: 主 fast experimental comparison subset；当前主表使用
no-block + kv-cache 的 original/free/softforce 对比
CoreFull-19562: 最终 full clean image-core confirmation
```

还有一些 diagnostic VStar subsets，只用于 runner/backend validation，不用于 benchmark-scale effect claims。

关键 BLINK 警告：

- 历史 “BLINK full n=120” 指的是 BLINK Counting only：
  `snapshot/Counting/val-00000-of-00001.parquet` 全部 120 行。
- 当前 clean BLINK full 指 BLINK all subtasks，n=1901。
- 当前随机/stratified 120 行不是旧 Counting-120，不能直接比较。

## 13. Scoring

Scoring 在所有 rows 产生之后统一执行。这是有意设计，因为 OCRBench-v2 等 official scorers 需要 batch-level prediction files。

scoring backend：

```text
auto
official
project
```

当前 official-compatible / official 支持：

- BLINK：official-compatible multiple-choice exact match。
- HR-Bench-4K：official-compatible multiple-choice scorer。
- OCRBench-v2：本地 `official_code` 存在时走 official batch scorer。
- MMMU-Pro：本地 `official_code` 存在时走 official batch scorer。
- MathVista：本地 `official_code` 存在时走 official batch scorer；当前 clean path 使用 disabled-LLM mode。
- MathVerse：本地 `official_code` 存在时走 official batch scorer；当前 clean path 使用 disabled-LLM mode。

如果指定 `scoring_backend=official` 但缺少对应本地 official tool，clean scorer 会 fail fast。

## 14. Data Generation

Data generation 是 clean 的一等入口，不是 legacy bridge。

入口：

```bash
tgvf_generate_data
```

当前 deterministic transforms：

```text
v4_to_protocol_c
v4_to_stage1_protocol_c_focus
choice_to_open_answer
clean_imend
```

每个 generated split 应记录：

- source path/hash；
- transform；
- protocol；
- schema；
- split/hash；
- field weights；
- mask policy；
- output file identities。

heavy teacher trajectory generation 仍留在 clean tree 外，只有明确需要重新生成 traces 时再推进。

## 15. Training Execution Model

clean training 分两层。

第一层：launch planner

```bash
tgvf_train_stage1 --write-plan ...
tgvf_train_stage2 --write-plan ...
```

这些命令写 auditable plan artifacts 和 commands，不偷偷启动长训练。

第二层：executor

```bash
tgvf_train_stage1_executor --plan /path/to/training_plan.json --preflight-only
tgvf_train_stage2_executor --plan /path/to/training_plan.json --preflight-only
tgvf_train_stage2_executor --plan /path/to/training_plan.json --prepare-execution
tgvf_train_stage2_executor --plan /path/to/training_plan.json --prepare-execution --audit-runtime
tgvf_train_stage2_executor --plan /path/to/training_plan.json --prepare-execution --audit-runtime --audit-launch-readiness
tgvf_train_stage2_executor --plan /path/to/training_plan.json --launch-training
```

executor mode 必须显式选择。没有 `--preflight-only`、`--prepare-execution`、`--audit-runtime` 或 `--launch-training` 时，executor fail fast，不会启动 partial / historical training path。

distributed launch 规则：

- `world_size=1`：使用 `python -m ... --launch-training`。
- `world_size>1`：使用 `torchrun --nproc-per-node <world_size>`。
- global batch 必须等于 `world_size * micro_batch_size * gradient_accumulation_steps`。
- GPU 数变化时，默认应自动调整 micro/accum，保持 global batch 不变，除非用户明确批准改变 global batch。

## 16. Benchmark Execution Model

clean benchmark runner：

```bash
tgvf_eval_benchmark
```

支持：

- manifest identity validation；
- materialized sample output；
- rendered prompt/media/control rows；
- dry-run rows；
- original Qwen backend；
- clean-native Qwen3 Stage2 backend；
- diagnostic historical Stage2 bridge；
- shard execution and merge。

Stage2 backend names：

```text
tgvf_stage2_qwen3
tgvf_stage2_qwen3_native
tgvf_stage2_qwen3_legacy
```

含义：

- `tgvf_stage2_qwen3`：generic clean Stage2 name，会 resolve 到 native。
- `tgvf_stage2_qwen3_native`：明确的 final clean-native backend。
- `tgvf_stage2_qwen3_legacy`：diagnostic bridge only，要求 `eval_family=internal_diagnostic`。

clean external benchmark entrypoint 只用于：

```text
project_native_external
internal_diagnostic
```

ValKit 必须用：

```bash
tgvf_eval_valkit
```

## 17. ValKit / VLMEvalKit

ValKit 入口：

```bash
tgvf_eval_valkit
```

它写 clean ValKit plan 和 preflight report。显式 `--execute` 时调用：

```text
<valkit-root>/run.py
```

它不调用 historical shell wrappers。

execution gating：

- 缺 `--valkit-root` 或 `--valkit-model-name`：
  `execute_permitted=false`；
- configured root 且有 `run.py` 和 model key：
  `execute_permitted=true`；
- prepare/preflight 不会启动 ValKit；
- 只有显式 `--execute` 会启动 ValKit。

## 18. Reproducibility Rules

启动任何真实训练或评测前，必须从文件证明 experiment identity。不要相信记忆、脚本名、或者 “full” 这种词。

必须记录/核验：

- exact checkpoint 和 processor；
- exact git commit 和 worktree state；
- exact benchmark source files；
- exact sample count 和 sample ids 或 deterministic source rule；
- exact prompt/continuation mode；
- exact eval mode 和 scoring backend；
- exact DeepStack state 和 mask scope；
- exact GPU/world-size/micro-batch/accumulation；
- output directory 和 run_config/rows/summary files。

使用 `docs/EXPERIMENT_LEDGER.md` 作为实验台账：

- launch 前：planned run、baseline、intended diff、command、GPUs；
- launch 后：tmux/session 和 start time；
- finish 后：metrics、output paths、elapsed time、conclusion；
- 跑错时：标记为 side result 或 invalid for baseline，不能静默混进主表。

如果和旧结果比较，要检查旧输出：

```text
run_config.txt
merged_summary.json
merged_rows.jsonl
shard row files
logs if needed
```

## 19. 常见错误

不要不解析 sample identity 就说某个结果是 “full”。

不要把旧 BLINK Counting-120 和当前 BLINK all-subtasks 结果比较。

不要把结果简单解释成 “KV vs no-KV”。必须控制 DeepStack、position ids、rope deltas、prompt/continuation、parser、sample identity。

不要把 Qwen2 diagnostic behavior 当成 Qwen3 mainline behavior。

不要说 `tgvf_free` 有额外 prompt。当前 clean `tgvf_free` 没有额外 prompt。

不要把 legacy Stage2 bridge 当 clean benchmark backend。它是 diagnostic-only。

不要在 clean project 工作中顺手删除或裁剪旧 project。那是单独任务，需要单独 plan 和用户确认。

不要只看 aggregate accuracy 就下机制结论。必须同时看 rows、trigger rate、focus validity、append success、parse rate、scoring backend。

## 20. 当前 Sanity Status

当前 clean project 已通过：

```bash
PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests
```

写本文档时结果：

```text
129 passed, 2 warnings
```

warnings 是 checkpoint runtime audit 测试中依赖层的 `SwigPy*` deprecation warnings，不是 clean test failure。

## 21. 推荐 Pro 阅读顺序

1. `docs/PRO_HANDOFF_CURRENT_PROJECT_GUIDE.md`
2. `revisit_vlm_clean/README.md`
3. `revisit_vlm_clean/docs/IMPLEMENTATION_PLAN.md`
4. `revisit_vlm_clean/src/revisit_vlm_clean/defaults.py`
5. `revisit_vlm_clean/src/revisit_vlm_clean/schema.py`
6. `revisit_vlm_clean/src/revisit_vlm_clean/tgvf_protocol.py`
7. `revisit_vlm_clean/src/revisit_vlm_clean/rendering.py`
8. `revisit_vlm_clean/src/revisit_vlm_clean/training_plan.py`
9. `revisit_vlm_clean/src/revisit_vlm_clean/runner.py`
10. 新实验或新比较前读 `docs/EXPERIMENT_LEDGER.md`

## 22. 如果 Pro 被要求开跑

先问清或从文件证明：

- 哪个 exact checkpoint？
- 哪个 exact processor？
- 哪个 exact subset 或 manifest hash？
- 哪个 mode：original/free/force/softforce？
- 哪个 forward mode？
- DeepStack enabled 还是 disabled？
- 哪个 mask scope？
- 哪个 scoring backend？
- 哪些 GPU，以及 global batch 数学？
- 这是新结果、复现、还是 side diagnostic？

这些都解析清楚之后，先写 ledger/preflight，再开 tmux 或 torchrun。
