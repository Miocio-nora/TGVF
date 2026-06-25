# TGVF RL Dataset v0

This note records the first unified RL-dataset export for the current TGVF
project state.

## Builder

```bash
python scripts/build_tgvf_rl_dataset.py
```

Default output:

```text
data/tgvf_rl/v0/
```

The builder is deterministic over its input JSONL files and writes SHA-256 hashes
into `manifest.json`.

## Output Files

```text
data/tgvf_rl/v0/teacher_sft_seed.train.jsonl
data/tgvf_rl/v0/teacher_sft_seed.test.jsonl
data/tgvf_rl/v0/rl_prompt_seed.heldout.jsonl
data/tgvf_rl/v0/benchmark_rollout_eval.jsonl
data/tgvf_rl/v0/manifest.json
```

Current counts:

```text
teacher_sft_seed.train.jsonl     46,883
teacher_sft_seed.test.jsonl       1,002
rl_prompt_seed.heldout.jsonl      1,002
benchmark_rollout_eval.jsonl      6,774
```

## Teacher Seed Source

Teacher seed rows come from the current open-answer Protocol C Stage2 split:

```text
data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl
data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl
```

Train split distribution:

```text
focus rows      39,655
no-focus rows    7,228

visual_genome   19,625
docvqa           9,150
textvqa          7,240
textocr          6,652
chartqa          4,216
```

The source file has already converted multiple-choice teacher answers into
short-text/open answers while preserving original choices under
`metadata.choice_to_open_answer`.

## RL Prompt Source

RL prompt rows are intentionally separated from SFT train rows. The default RL
prompt seed is the heldout/test teacher split:

```text
data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl
```

The builder writes:

```text
data/tgvf_rl/v0/rl_prompt_seed.heldout.jsonl
```

It also checks that `teacher_sft_seed.train.jsonl` and
`rl_prompt_seed.heldout.jsonl` have no overlapping `source.source_id` values.
If overlap is detected, the build fails.

## Rollout Eval Source

Rollout rows come from the 2026-06-23 old-mask/correct-Stage1 benchmark
diagnostic:

```text
outputs/full_benchmarks_oldmaskprob075_throughanswer_correct_stage1_4gpu_20260623_005426
```

Included benchmark rows:

```text
hr_bench_4k   1,600
ocrbench_v2   4,934
blink           240
```

Rollout rows contain model outputs, parsed answers, scores, focus validity,
trigger decisions, and capture errors. The existing rollout files do not include
image paths, so `benchmark_rollout_eval.jsonl` is a reward/eval diagnostic pool,
not a complete training pool by itself.

## Unified Schema

Every row uses:

```text
schema_version
dataset_role
sample_id
source
media
prompt
supervision
reward
metadata
```

`teacher_sft_seed` rows have image paths and teacher trajectory supervision but
no rollout reward. `rl_prompt_seed` rows have image paths and heldout teacher
references; they are intended as prompt sources for fresh policy rollouts.
`benchmark_rollout_eval` rows have reward components and candidate outputs but
no media paths.

## Intended Use

Recommended first RL flow:

1. Use `teacher_sft_seed.train.jsonl` only for supervised seed / SFT warmup.
2. Use `rl_prompt_seed.heldout.jsonl` as the initial RL prompt source.
3. Generate fresh policy rollouts against those heldout prompts, preserving
   image paths and generation config.
4. Score rollouts with answer correctness, parse stability, focus validity,
   target quality, and evidence/answer format rewards.
5. Keep `benchmark_rollout_eval.jsonl` as a diagnostic or reward calibration
   set unless a later builder reconstructs benchmark media paths.

Do not mix benchmark-derived rows into final RL training without an explicit
leakage-control decision.

The heldout RL prompt seed is only 1,002 rows, so it is enough for wiring and
small RL diagnostics. A larger RL run should generate a new teacher/prompt pool
from images excluded from SFT training and then pass it through the same disjoint
source-ID check.
