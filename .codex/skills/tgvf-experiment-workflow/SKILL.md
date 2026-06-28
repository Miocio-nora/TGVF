---
name: tgvf-experiment-workflow
description: Use when working in this TGVF/revisit_vlm repository on training, evaluation, benchmark comparisons, ablations, checkpoint selection, experiment logging, or project cleanup. Enforces reproducible experiment identity before running commands.
---

# TGVF Experiment Workflow

This repository is fragile because code, adapters, prompts, checkpoints, and benchmark surfaces have all changed over time. Do not trust memory, script names, or words like "full" until they are resolved to concrete artifacts.

## General Work Rule

Start every non-trivial task with a macro plan before execution. The plan must
name:

- the actual objective;
- the repository/worktree strategy;
- what will be changed now;
- what will not be touched;
- unresolved decisions that require user confirmation;
- how the result will be verified.

Do not improvise the plan after starting implementation. If the task scope
changes, stop, restate the updated macro plan, and wait for user confirmation
when the change affects code, experiments, cleanup, benchmark identity, or
reported conclusions.

User confirmation of a direction is not the same as permission to edit every
related file. Translate confirmation into a concrete next action and state it
before touching files.

## Non-Negotiable Rule

Before launching any training or evaluation, prove the experiment identity from files. If it cannot be proven, stop and ask the user.

Experiment identity means:

- exact checkpoint and processor;
- exact code path and current git/worktree state;
- exact benchmark source files;
- exact sample count and sample ids or deterministic source rule;
- exact prompt/continuation mode;
- exact evaluation mode and scoring backend;
- exact DeepStack state for original-image features and any post-TGVF mask scope;
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
- record DeepStack support/state separately from attention masks:
  - clean Qwen3 must support DeepStack training semantics, but the default is
    DeepStack off unless the run explicitly enables it;
  - if DeepStack is enabled, original-image DeepStack injection after D must
    follow the same scope as original-image visual-key masking;
  - `through_answer` means original-image DeepStack is blocked from D/evidence
    through answer;
  - `evidence_only` means original-image DeepStack is blocked for D/evidence
    and restored for answer;
  - D remains a v-merge-level visual-token span by default; do not add
    DeepStack-like D features unless it is a named ablation;
- verify whether KV cache is used for training efficiency only, separately from eval generation semantics.

## Evaluation Rules

Post-D evaluation identity:

- Record the forward mode, but do not explain a result as "KV vs no-KV" unless
  DeepStack state, position ids, rope deltas, prompt/continuation, and parser are
  controlled.
- For Qwen3, treat DeepStack on/off as a first-class eval variable. If a manual
  full-sequence path disables DeepStack and a native path enables it, the result
  difference must be reported as DeepStack on/off, not as cache behavior.
- Keep `post_tgvf_continuation`, `question_suffix`, `max_image_resolution`, and parser behavior fixed when comparing.

For external benchmark tables:

- compare only rows with the same sample set;
- include n, accuracy, parse rate, trigger rate, focus-valid rate, scoring backend, and output path;
- when a method table is intended to compare TGVF behavior, include original,
  `tgvf_free`, and `tgvf_softforce` columns by default, and report trigger rate
  for each mode in the same table;
- inspect row examples before giving mechanism conclusions.

## Cleanup Rules

Cleanup must start with a macro plan, not local edits. Before changing or deleting code for project normalization, first present the user with:

- the intended repository strategy: archive-only, in-place cleanup branch, new clean project/tree, or a staged combination;
- the exact role of the current branch;
- a whitelist of files/modules/scripts/protocols that will enter the clean project;
- a separate archive/remove list;
- a list of unresolved decisions that will not be touched yet.

Do not infer that "this option should not enter the clean project" means "delete it from the current repo now." Treat those as different actions. Deletion from the working tree requires explicit confirmation that the current branch is the implementation cleanup branch, not merely a planning or audit branch.

Do not normalize or delete project assets until:

- current worktree is archived by a user-approved commit/branch/tag;
- outputs to preserve are listed;
- obsolete scripts are mapped to replacement scripts;
- the user chooses which experiment families remain first-class.

When unclear, ask the user to choose between 2-3 concrete options.

If the user is still confirming the whitelist, do not edit executable code. At most update planning docs or the skill itself, and clearly say that no implementation cleanup has started.
