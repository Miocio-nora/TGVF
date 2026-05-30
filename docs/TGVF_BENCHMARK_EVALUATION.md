# TGVF Benchmark Evaluation Harness v0

本文档说明 TGVF benchmark-only evaluation harness 的评测设定、运行方式和输出格式。

这个 harness 只用于 benchmark 评测。它不负责：

- 模型训练
- teacher-guide 数据生成
- benchmark 数据下载
- benchmark 数据集创建
- Qwen LLM finetuning
- 新 FVT module 开发
- crop / oracle crop baseline

## 代码入口

Harness 代码位于：

```text
src/tgvf_eval/
```

主要命令：

```text
python -m tgvf_eval.run
python -m tgvf_eval.run_suite
python -m tgvf_eval.profile
```

核心文件：

```text
src/tgvf_eval/adapters.py          benchmark adapter 和 registry
src/tgvf_eval/config.py            method / trigger / video mode 配置
src/tgvf_eval/model_runner.py      direct、prompt-only、module TGVF 推理路径
src/tgvf_eval/results.py           输出目录、resume、summary
src/tgvf_eval/video_foveation.py   video foveation mode 状态机
src/tgvf_eval/run.py               单 benchmark runner
src/tgvf_eval/run_suite.py         suite runner
src/tgvf_eval/profile.py           小样本 profiler
```

## 数据路径

默认 benchmark root：

```text
/home/dredvpn009/Flash_Storage/datasets/benchmarks
```

默认 official / collected tools root：

```text
/home/dredvpn009/Flash_Storage/datasets/benchmarks/_tools
```

默认项目输出目录：

```text
eval_outputs/tgvf_benchmarks/
```

一个完整 run 会写入：

```text
eval_outputs/tgvf_benchmarks/runs/<run_id>/
```

## 支持的 Benchmarks

当前 registry 包含：

```text
vstar_bench
hr_bench_4k
ocrbench_v2
blink
mmmu_pro
mathvista
mathverse
ovo_bench
```

StreamingBench 已从本项目移除，当前不纳入 registry、suite 或评测计划。

每个 benchmark 通过 adapter 统一暴露以下能力：

- 加载本地样本
- 根据 tier 做确定性采样
- 构建 prompt
- 解析 raw output
- fallback scoring
- 记录 official tool path / scorer metadata
- 写 per-sample 评测结果

v0 会优先解析和记录 official tool 路径。但如果某个 benchmark 的 official scorer invocation 还没有在 adapter 中明确接好，报告会写：

```text
official_tool_used: false
scorer_name: fallback
```

这样做是为了避免没有真正调用 official scorer，却在报告里误标为 official score。

## Evaluation Methods

### direct_qwen

直接 Qwen2-VL baseline。

```text
trigger_mode = direct
tgvf_mode    = none
```

不触发 foveation，不追加 FVT。

### tgvf_prompt_only_force

Prompt-only forced foveation。

```text
trigger_mode = force
tgvf_mode    = prompt_only
```

模型被要求先输出：

```text
<|foveate|>target<|/foveate|>
```

然后 controller 从当前 generation state 继续。这个模式不需要 TGVF checkpoint，主要用于诊断“强制 target selection”是否有帮助。

### tgvf_prompt_only_free

Prompt-only free triggering。

```text
trigger_mode = free
tgvf_mode    = prompt_only
```

模型可以直接回答，也可以输出 foveation span。报告会记录：

```text
trigger_rate
accuracy_triggered_only
accuracy_not_triggered
```

这是最接近“无 LLM finetuning 条件下 prompt-only 使用方式”的模式。

### tgvf_module_force

Structural / module TGVF with forced target generation。

```text
trigger_mode = force
tgvf_mode    = module
```

流程：

```text
image/question
  -> 强制生成 <|foveate|>target<|/foveate|>
  -> 捕获 target hidden states H_q
  -> 从 H_q 和 V_pre 生成 FVT D
  -> bracketed FVT append
  -> 从 KV cache continuation answer
```

### tgvf_module_free

Structural / module TGVF with free trigger。

```text
trigger_mode = free
tgvf_mode    = module
```

这是主要 end-to-end TGVF 评测模式。模型如果触发 foveation，就走 FVT append；如果没有触发，则把直接输出当作最终答案。

### Diagnostic Controls

小样本 sanity check 可用：

```text
tgvf_no_D
tgvf_random_D
tgvf_wrong_D
```

这些 control 不建议默认跑全量 benchmark。它们用于判断提升是否来自真实 FVT evidence，而不是 prompt/token effect。

## Trigger Mode 和 TGVF Mode

支持：

```text
--trigger-mode direct
--trigger-mode force
--trigger-mode free
```

含义：

```text
direct: 不请求 foveation
force:  必须先生成一次 foveation request
free:   模型自行决定是否 foveate
```

支持：

```text
--tgvf-mode none
--tgvf-mode prompt_only
--tgvf-mode module
```

含义：

```text
none:        direct baseline
prompt_only: 只使用 foveation prompting protocol，不要求 trained FVT module
module:      使用当前 TGVF module / checkpoint 生成并追加 FVT
```

一般不需要手动传 `--trigger-mode` 和 `--tgvf-mode`，因为 `--method` 会自动推导。只有做特殊对照时再显式覆盖。

## CoT 设置

默认关闭：

```bash
--cot false
```

原因：

- CoT 会增加推理成本
- CoT 会改变 benchmark comparability
- TGVF 的主要假设是 repeated visual revisiting，不是 textual chain-of-thought
- 当前 LLM 没有针对 foveation protocol 做 finetuning

开启：

```bash
--cot true
```

报告中会记录：

```text
cot_enabled: true
cot_prompt_source: minimal
```

CoT 和 non-CoT 结果不要混在同一个 score table 里比较。

## Foveation Count

默认：

```bash
--max-foveations 1
```

v0 默认只跑单次 foveation，原因是：

- repeated revisiting 成本可能快速膨胀
- 当前 prompt-only LLM 没有针对多轮 foveation finetune
- 单次 foveation 是最清晰的第一版 benchmark 设置

诊断时可以跑：

```bash
--max-foveations 2
```

更适合 HR-Bench FCP、多步视觉搜索、视频任务等。

## Image Budget

默认：

```bash
--image-budget mid
```

支持：

```text
low
mid
high
```

同一组比较中，`direct_qwen` 和 TGVF 方法必须使用同样的 raw image budget。不要给 TGVF 额外分辨率，除非报告里明确说明。

## Video Settings

视频 benchmark 默认按 tier 设置 frame count：

```text
light   -> 8 frames
medium  -> 16 frames
full    -> 32 frames
```

可显式覆盖：

```bash
--video-nframes 4
--video-nframes 8
--video-nframes 16
--video-nframes 32
```

## Video Foveation Modes

### off

不做 video foveation update。

```bash
--video-foveation-mode off
```

### per_frame_reencode

每个选中 frame / frame chunk 都重新生成或刷新 target/H_q。

```bash
--video-foveation-mode per_frame_reencode
```

这是直接、但更贵的 video revisit baseline。

### nextframe_encode_no_reencode

复用 frame `t` 生成的 revisit instruction / H_q 到后续 frame。

```bash
--video-foveation-mode nextframe_encode_no_reencode
--nextframe-reuse-ttl 1
```

概念流程：

```text
frame t:
  生成 target，捕获 H_q_t

frame t+1:
  不重新让 LLM 生成 target
  复用 H_q_t
  提取 V_pre_{t+1}
  生成 D_{t+1} = FovealModule(H_q_t, V_pre_{t+1})
  append frame-specific FVT
```

TTL：

```bash
--nextframe-reuse-ttl K
```

含义：

```text
K=1: frame t 的 H_q 只复用到 frame t+1
K=2: frame t 的 H_q 复用到 frame t+1 和 t+2
```

v0 只实现 TTL-based refresh，不做复杂 tracking / scene cut 检测。

报告字段：

```text
num_reencode_events
num_reuse_encode_events
reuse_rate
avg_foveation_events_per_query
```

## Sampling Tiers

支持：

```bash
--tier light
--tier medium
--tier full
```

也可以显式限制样本数：

```bash
--limit 50
```

如果传了 `--limit`，它会覆盖 tier 默认样本数。

默认 seed：

```bash
--seed 20260525
```

Light tier：

```text
vstar_bench     50
hr_bench_4k     50
ocrbench_v2     500
blink           280
mmmu_pro        300
mathvista       200
mathverse       300
ovo_bench       90
```

Medium tier：

```text
vstar_bench     191
hr_bench_4k     200
ocrbench_v2     2000
blink           700
mmmu_pro        900
mathvista       1000
mathverse       1200
ovo_bench       450
```

Full tier 会尽量使用本地 adapter 能确定的全部样本。MathVista 默认按 offline testmini 风格处理。

## 输出目录和文件

一个 run 输出到：

```text
eval_outputs/tgvf_benchmarks/runs/<run_id>/
```

目录结构：

```text
config.yaml
predictions/
raw_outputs/
scores/
reports/
logs/
artifacts/
```

Per-sample prediction 文件：

```text
predictions/<benchmark>__<method>.jsonl
```

Raw output 文件：

```text
raw_outputs/<benchmark>__<method>.jsonl
```

Score 文件：

```text
scores/<benchmark>__<method>.scores.json
```

Summary 文件：

```text
reports/<benchmark>__<method>.summary.json
```

## Per-Sample 字段

每个样本行包含：

```text
benchmark
sample_id
method
trigger_mode
tgvf_mode
cot_enabled
cot_prompt_source
image_budget
video_nframes
video_foveation_mode
nextframe_reuse_ttl
question
media
choices
raw_output
parsed_answer
gold_answer
score
metadata
triggered
foveation_target
num_foveations
second_full_forward_used
wall_time_sec
visual_token_count
output_tokens
official_tool_used
official_tool_path
scorer_name
prompt_source
error
```

重点字段：

```text
triggered:
  是否触发 foveation

foveation_target:
  模型请求 revisit 的 target 文本

second_full_forward_used:
  module TGVF 路径必须为 false

official_tool_used:
  是否实际调用 official scorer

scorer_name:
  official 或 fallback
```

## Summary 字段

Summary report 包含：

```text
run_id
benchmark
tier
method
num_samples
score
trigger_rate
accuracy_triggered_only
accuracy_not_triggered
avg_num_foveations
avg_wall_time_sec
p50_wall_time_sec
p90_wall_time_sec
avg_visual_tokens
cot_enabled
second_full_forward_used
scoring
```

`scoring` 中会记录 official / fallback scorer metadata。

## Resume

默认启用 resume。

Runner 会读取：

```text
predictions/<benchmark>__<method>.jsonl
```

并跳过已经存在的 `sample_id`。

禁用 resume：

```bash
--no-resume
```

如果要重新跑同一个 run ID，建议使用新的 `--run-id`，或者先明确处理旧输出，避免新旧结果混在一个 JSONL 中。

## 单 Benchmark 使用示例

### Direct Qwen baseline

```bash
python -m tgvf_eval.run   --benchmark vstar_bench   --tier light   --method direct_qwen   --model-path Qwen/Qwen2-VL-2B-Instruct   --benchmark-root /home/dredvpn009/Flash_Storage/datasets/benchmarks   --tools-root /home/dredvpn009/Flash_Storage/datasets/benchmarks/_tools   --output-root eval_outputs/tgvf_benchmarks
```

### Prompt-only free mode

```bash
python -m tgvf_eval.run   --benchmark vstar_bench   --tier light   --method tgvf_prompt_only_free   --cot false   --max-foveations 1
```

### TGVF module free mode

```bash
python -m tgvf_eval.run   --benchmark hr_bench_4k   --tier light   --method tgvf_module_free   --tgvf-checkpoint outputs/tgvf_fvt/20k_foveal_ablation_v2_20260527_124521/target_slot_foveal_cross_merger/checkpoint_step_5176.pt   --tgvf-variant target_slot_foveal_cross_merger   --trigger-mode free   --cot false   --image-budget mid
```

### TGVF module force mode

```bash
python -m tgvf_eval.run   --benchmark hr_bench_4k   --tier light   --method tgvf_module_force   --tgvf-checkpoint <checkpoint.pt>   --tgvf-variant target_slot_foveal_cross_merger   --cot false   --image-budget mid
```

### Diagnostic controls

```bash
python -m tgvf_eval.run   --benchmark ocrbench_v2   --tier light   --limit 50   --method tgvf_no_D   --tgvf-checkpoint <checkpoint.pt>   --tgvf-variant target_slot_foveal_cross_merger
```

```bash
python -m tgvf_eval.run   --benchmark ocrbench_v2   --tier light   --limit 50   --method tgvf_random_D   --tgvf-checkpoint <checkpoint.pt>   --tgvf-variant target_slot_foveal_cross_merger
```

```bash
python -m tgvf_eval.run   --benchmark ocrbench_v2   --tier light   --limit 50   --method tgvf_wrong_D   --tgvf-checkpoint <checkpoint.pt>   --tgvf-variant target_slot_foveal_cross_merger
```

## Video 使用示例

### OVO-Bench per-frame reencode

```bash
python -m tgvf_eval.run   --benchmark ovo_bench   --tier light   --method tgvf_module_free   --tgvf-checkpoint <checkpoint.pt>   --tgvf-variant target_slot_foveal_cross_merger   --video-nframes 8   --video-foveation-mode per_frame_reencode   --max-foveations 1
```

### OVO-Bench next-frame no-reencode

```bash
python -m tgvf_eval.run   --benchmark ovo_bench   --tier light   --method tgvf_module_free   --tgvf-checkpoint <checkpoint.pt>   --tgvf-variant target_slot_foveal_cross_merger   --video-nframes 8   --video-foveation-mode nextframe_encode_no_reencode   --nextframe-reuse-ttl 1   --max-foveations 1
```

## Dry-Run Smoke Test

`--dry-run` 不加载 Qwen 权重，用于检查：

- adapter 能否加载样本
- deterministic sampling 是否工作
- parser / scorer 是否工作
- 输出目录是否正确
- resume 文件是否正确

示例：

```bash
python -m tgvf_eval.run   --benchmark vstar_bench   --tier light   --limit 3   --method direct_qwen   --dry-run   --run-id smoke_tgvf_eval_v0   --output-root /tmp/tgvf_eval_smoke
```

输出示例：

```text
/tmp/tgvf_eval_smoke/runs/smoke_tgvf_eval_v0/
```

## Suites

### core_light_image

Benchmarks：

```text
vstar_bench
hr_bench_4k
ocrbench_v2
blink
mmmu_pro
mathvista
```

推荐 methods：

```text
direct_qwen
tgvf_prompt_only_free
tgvf_module_free
tgvf_module_force
```

命令：

```bash
python -m tgvf_eval.run_suite   --suite core_light_image   --methods direct_qwen,tgvf_prompt_only_free,tgvf_module_free,tgvf_module_force   --cot false
```

### core_medium_image

用途：light tier 有信号之后，用于内部技术判断。

```bash
python -m tgvf_eval.run_suite   --suite core_medium_image   --methods direct_qwen,tgvf_prompt_only_free,tgvf_module_free,tgvf_module_force   --cot false
```

### video_light

Benchmarks：

```text
ovo_bench
```

推荐比较：

```text
direct_qwen
tgvf_module_free + per_frame_reencode
tgvf_module_free + nextframe_encode_no_reencode
```

命令：

```bash
python -m tgvf_eval.run_suite   --suite video_light   --methods direct_qwen,tgvf_module_free   --cot false   --video-nframes 8   --nextframe-reuse-ttl 1
```

### math_light

Benchmarks：

```text
mathvista
mathverse
```

命令：

```bash
python -m tgvf_eval.run_suite   --suite math_light   --methods direct_qwen,tgvf_prompt_only_free,tgvf_module_free,tgvf_module_force   --cot false
```

## Profiling

跑 medium / full 之前，建议先用 50 samples profile：

```bash
python -m tgvf_eval.profile   --benchmarks vstar_bench,hr_bench_4k,ocrbench_v2   --methods direct_qwen,tgvf_module_free,tgvf_module_force   --limit 50   --model-path Qwen/Qwen2-VL-2B-Instruct   --tgvf-checkpoint <checkpoint.pt>   --tgvf-variant target_slot_foveal_cross_merger
```

Profile 主要看：

```text
avg_wall_time_sec
p50_wall_time_sec
p90_wall_time_sec
trigger_rate
avg_num_foveations
avg_visual_tokens
```

用这些估算 full run 的总耗时和成本。

## Scoring 和 Parsing

Multiple-choice parser 会优先解析常见格式：

```text
Answer: C
The answer is (B).
option D
A
```

如果出现多个候选字母，会优先取常见 answer marker 后面的第一个合法选项。

Open-answer fallback scorer 使用 normalized exact match。

对 OCR、math、video benchmark，最终报告应优先使用 official scorer。若 official scorer 未接入或不可用，报告必须明确保留：

```text
official_tool_used: false
scorer_name: fallback
```

## No Second Full Forward 约束

TGVF module mode 必须满足：

- 不在 foveation 后重新跑完整 original image/question prompt
- 不重新生成完整 prompt + foveation span
- 不修改旧 KV cache entries
- 不替换已缓存的原始 visual tokens
- 只允许 capture、FVT generation、bracketed FVT append、cache continuation

有效 module TGVF run 中：

```text
second_full_forward_used: false
```

对 video `nextframe_encode_no_reencode`：

- 不为 reused next frame 重新生成 LLM target
- 复用上一帧 / chunk 的 `H_q`
- TTL 到期后再 refresh

## 推荐评测顺序

第一轮：

```text
suite: core_light_image
methods:
  direct_qwen
  tgvf_prompt_only_free
  tgvf_module_free
  tgvf_module_force
settings:
  cot=false
  max_foveations=1
  image_budget=mid
```

第二轮：

```text
suite: video_light
methods:
  direct_qwen
  tgvf_module_free + per_frame_reencode
  tgvf_module_free + nextframe_encode_no_reencode
settings:
  cot=false
  video_nframes=8
  nextframe_reuse_ttl=1
  max_foveations=1
```

第三轮：

```text
suite: core_medium_image
```

只有当 light tier 显示有用信号，并且 profiler 显示成本可接受时，再跑 medium / full。

## 当前 v0 注意事项

- Benchmark adapters 已经提供统一接口，但不同 benchmark 的 official scorer 仍需要逐个 adapter 接入。
- 当前 fallback scoring 适合 smoke / dev，不应直接替代 official benchmark reporting。
- `--dry-run` 只测试 harness，不代表模型性能。
- TGVF 不一定在 tiny OCR、小物体定位、高分辨率显式搜索等 crop-friendly 任务上强于 crop/oracle crop。
- 本 harness 的主要目标是比较同 visual budget 下 TGVF 相对 direct Qwen 的整体表现、触发可靠性、成本和 video reuse tradeoff。


## Strict Guideline Checks

当前 v0 实现遵守以下关键约束：

- `tgvf_module_force` / `tgvf_prompt_only_force` 只硬约束 foveation markers；target 文本由模型生成，不再固定成通用 target。
- module TGVF append 后会追加 continuation instruction：`The newly provided visual tokens...`。
- module TGVF continuation 从已有 generation state / KV cache 继续，报告字段 `second_full_forward_used` 必须为 `false`。
- prediction JSONL 会写 `debug_metadata`，用于检查 `hard_force_trigger_used`、`continuation_instruction_appended`、`cache_preserved`。
- `--image-budget low|mid|high` 会映射到 Qwen content 的 `max_pixels`。
- 视频路径会通过 PyAV 按 `--video-nframes` 抽帧，作为多图输入送入 Qwen，避免依赖缺失的 `torchvision.io.read_video`。
- 真实 video foveation update modes 还没有接入 `QwenTGVFModelRunner` 时会显式报 `NotImplementedError`，不会把 mock state machine 结果伪装成真实 benchmark 结果。

## GPU4 Light Debug 命令

指定 GPU4、当前 20k checkpoint、跑 core light image 的 `tgvf_module_force` limit=1 debug：

```bash
CUDA_VISIBLE_DEVICES=4 python -m tgvf_eval.run_suite \
  --suite core_light_image \
  --methods tgvf_module_force \
  --limit 1 \
  --model-path Qwen/Qwen2-VL-2B-Instruct \
  --tgvf-checkpoint outputs/tgvf_fvt/20k_4gpu_ablation_20k_ablation_v1/token_direct_three_loss/checkpoint_step_5176.pt \
  --tgvf-variant token_direct \
  --cot false \
  --image-budget mid \
  --max-foveations 1 \
  --device cuda:0 \
  --run-id strict_guideline_core_light_image_limit1_gpu4_v2 \
  --no-resume
```

单跑 HR-Bench base64 image smoke：

```bash
CUDA_VISIBLE_DEVICES=4 python -m tgvf_eval.run \
  --benchmark hr_bench_4k \
  --tier light \
  --limit 1 \
  --method tgvf_module_force \
  --model-path Qwen/Qwen2-VL-2B-Instruct \
  --tgvf-checkpoint outputs/tgvf_fvt/20k_4gpu_ablation_20k_ablation_v1/token_direct_three_loss/checkpoint_step_5176.pt \
  --tgvf-variant token_direct \
  --cot false \
  --image-budget mid \
  --max-foveations 1 \
  --device cuda:0 \
  --run-id strict_guideline_hr_base64_smoke_gpu4 \
  --no-resume
```

单跑 OVO video frame fallback smoke：

```bash
CUDA_VISIBLE_DEVICES=4 python -m tgvf_eval.run \
  --benchmark ovo_bench \
  --tier light \
  --limit 1 \
  --method tgvf_module_force \
  --model-path Qwen/Qwen2-VL-2B-Instruct \
  --tgvf-checkpoint outputs/tgvf_fvt/20k_4gpu_ablation_20k_ablation_v1/token_direct_three_loss/checkpoint_step_5176.pt \
  --tgvf-variant token_direct \
  --cot false \
  --image-budget mid \
  --max-foveations 1 \
  --device cuda:0 \
  --run-id strict_guideline_ovo_smoke_gpu4_v3 \
  --no-resume
```

## 当前已知限制

- 多数 benchmark adapter 仍使用 fallback scorer；如果 official scorer invocation 尚未接好，报告会明确写 `official_tool_used=false`。
- `per_frame_reencode` 和 `nextframe_encode_no_reencode` 的真实 runner 接线仍未完成；当前只有状态机和单测，不会作为真实模型结果输出。
