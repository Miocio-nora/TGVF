---
name: tgvf-experiment-workflow
description: Use when working in this TGVF/revisit_vlm repository on training, evaluation, benchmark comparisons, ablations, checkpoint selection, experiment logging, or project cleanup. Enforces reproducible experiment identity before running commands.
---

# TGVF Experiment Workflow

This repository is fragile because code, adapters, prompts, checkpoints, and benchmark surfaces have all changed over time. Do not trust memory, script names, or words like "full" until they are resolved to concrete artifacts.

## Non-Negotiable Rule

Before launching any training or evaluation, prove the experiment identity from files. If it cannot be proven, stop and ask the user.

Experiment identity means:

- exact checkpoint and processor;
- exact code path and current git/worktree state;
- exact benchmark source files;
- exact sample count and sample ids or deterministic source rule;
- exact prompt/continuation mode;
- exact evaluation mode and scoring backend;
- exact GPU/world-size/micro-batch/accumulation relationship for training.

## Required Preflight

1. Read `docs/EXPERIMENT_LEDGER.md`.
2. Find the baseline entry or old result being compared.
3. Inspect the old result files, not just the summary:
   - `run_config.txt`;
   - `merged_summary.json`;
   - `merged_rows.jsonl` or shard row files;
   - logs if needed.
4. If comparing to an old run, inspect the old adapter/code behavior with `git show <commit>:<file>` when relevant.
5. Print a preflight summary before launch:
   - baseline run and output path;
   - planned checkpoint/processor;
   - planned benchmark source;
   - planned n;
   - sample rule and overlap with old rows when applicable;
   - planned modes;
   - changed variables and variables held fixed.
6. If any row of the preflight is uncertain, ask the user before launching.

## Dataset Identity Rules

Never use `tier`, `full`, or `limit` as the only dataset definition.

Use this hierarchy:

1. Exact `merged_rows.jsonl` sample ids from the old run.
2. Exact source file rule from the old adapter, confirmed by `git show`.
3. Current adapter deterministic sampling, only when no old sample set exists.

For BLINK specifically:

- Historical "BLINK full n=120" in older tables means `snapshot/Counting/val-00000-of-00001.parquet`, all 120 rows.
- Current BLINK full after adapter expansion means all subtasks, 1901 rows.
- A stratified `limit=120` under the current adapter is a different experiment and must not be compared as the old full table.

## Launch Gate

Do not launch if any of these are false:

- checkpoint exists;
- processor exists when expected;
- benchmark root exists;
- resolved sample count matches the baseline when doing a comparable rerun;
- sample overlap is 100 percent when rerunning the same table;
- `docs/EXPERIMENT_LEDGER.md` has a `PLANNED` or `RUNNING` entry;
- command writes `run_config.txt`;
- mode names are unambiguous.

If a new command-line option or adapter behavior is needed, ask first unless it is purely diagnostic and will not be used as the main comparable run.

## Logging Requirements

Every run must update `docs/EXPERIMENT_LEDGER.md`:

- before launch: status, question, baseline, intended diff, paths, command, GPUs;
- after launch: tmux/session and start time;
- after finish: metrics, output paths, elapsed time, conclusion;
- if invalid: mark as `SIDE_RESULT` or `INVALID_FOR_<baseline>` and explain why.

If an accidental run used the wrong sample set, do not delete it silently. Record it as a side result and exclude it from comparison tables.

## Training Rules

Before Stage1/Stage2 training:

- bind the exact Stage1 checkpoint/processor and Stage2 train/test data;
- record freeze/trainable modules;
- record global batch as `world_size * micro_batch * grad_accum`;
- keep global batch constant when changing GPU count unless the user approves;
- record mask behavior, mask probability, mask scope, token-row mode, manifold loss weight, D norm statistics, and visual merger training state;
- verify whether KV cache is used for training efficiency only, separately from eval generation semantics.

## Evaluation Rules

Post-D evaluation defaults:

- Prefer no-KV full-sequence continuation for correctness-path post-D evaluation unless explicitly studying KV cache behavior.
- Label any KV result as KV-continuation.
- Keep `post_tgvf_continuation`, `question_suffix`, `max_image_resolution`, and parser behavior fixed when comparing.

For external benchmark tables:

- compare only rows with the same sample set;
- include n, accuracy, parse rate, trigger rate, focus-valid rate, scoring backend, and output path;
- inspect row examples before giving mechanism conclusions.

## Cleanup Rules

Do not normalize or delete project assets until:

- current worktree is archived by a user-approved commit/branch/tag;
- outputs to preserve are listed;
- obsolete scripts are mapped to replacement scripts;
- the user chooses which experiment families remain first-class.

When unclear, ask the user to choose between 2-3 concrete options.

