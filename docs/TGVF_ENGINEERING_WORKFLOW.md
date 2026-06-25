# TGVF Engineering Workflow

This document is the operating procedure for this repository. It exists because experiment identity has drifted across adapters, scripts, prompts, checkpoints, and benchmark definitions.

## Core Principle

No experiment is defined by a short phrase such as `full`, `latest ckpt`, `free`, or `same as before`.

An experiment is defined by concrete files, code version, resolved samples, prompts, and generated outputs.

## Preflight Checklist

Before any run, write or update an entry in `docs/EXPERIMENT_LEDGER.md` and verify:

- Baseline output path.
- Baseline checkpoint and processor.
- New checkpoint and processor.
- Code commit plus dirty worktree status.
- Benchmark root and actual annotation files.
- Resolved sample count.
- Resolved sample ids or old adapter source rule.
- Mode: original, free, softforce, force, teacher-forced, or protocol eval.
- Prompt suffix and post-TGVF continuation.
- Post-D forward mode.
- DeepStack state for original-image features, including post-TGVF mask scope if
  DeepStack is enabled.
- Scoring backend and parser.
- GPU set and sharding.

The run should not start if this checklist cannot be filled.

## Comparable Rerun Protocol

When the user asks for "same table", "same full setting", or "compare with previous":

1. Read the old `summary` and old `merged_rows.jsonl`.
2. Read old `run_config.txt`.
3. If adapter behavior may have changed, inspect the old code with `git show`.
4. Resolve whether the old sample set came from:
   - an exact file;
   - a deterministic sample;
   - a broken/legacy adapter behavior.
5. Reproduce that old sample set by the old source rule when possible.
6. Only use new filtering interfaces after asking the user.

Example: the historical BLINK table with `n=120` came from old `BlinkAdapter.preferred_files = ("snapshot/Counting/val-00000-of-00001.parquet",)`. It is not equivalent to current full BLINK `n=1901` or current stratified `limit=120`.

## Run Output Contract

Every run directory must contain:

- `run_config.txt` with all resolved paths and knobs.
- Raw row files.
- Merged rows.
- Merged summary.
- A table JSON when producing user-facing comparisons.
- Logs for each shard.

## Reporting Contract

Report these fields for benchmark rows:

- model/checkpoint;
- benchmark and exact sample identity;
- mode;
- n;
- accuracy;
- answer parse rate;
- trigger rate;
- focus-valid rate;
- post-D forward mode;
- output path;
- whether the result is comparable to the requested baseline.

Never merge a side result into a comparison table without marking it.

## Code Change Rules

Do not add a new interface to solve a comparison problem until old behavior is understood.

Preferred order:

1. Reuse old interface semantics.
2. Recreate old source layout in a temporary run root.
3. Add a minimal explicit option only after user approval.

If a new option is added, document:

- why old interfaces are insufficient;
- whether it changes experimental identity;
- tests or smoke checks;
- how it appears in `run_config.txt`.

## Training Contract

Training runs must record:

- Stage1 checkpoint and processor.
- Stage2 checkpoint/output.
- Train and validation JSONL.
- Frozen and trainable modules.
- Global batch formula.
- Mask mode, probability, and scope.
- DeepStack support/state:
  - clean Qwen3 must support DeepStack training semantics;
  - default training setting is DeepStack off unless explicitly enabled;
  - when enabled, original-image DeepStack injection after D follows the same
    scope as original-image visual-key masking;
  - `through_answer` blocks original-image DeepStack from D/evidence through
    answer;
  - `evidence_only` blocks original-image DeepStack for D/evidence and restores
    it for answer;
  - D remains a v-merge-level visual-token span by default, with no
    DeepStack-like D features unless named as an ablation.
- Protocol token-row mode.
- Manifold loss weight and D norm diagnostics.
- Whether the visual merger is trained.

Changing GPU count is allowed only if the global batch is kept constant or the change is explicitly part of the experiment.

## Eval Contract

Post-D eval must record forward mode, but do not interpret a result as a
KV/no-KV effect unless DeepStack state is controlled. For Qwen3, DeepStack
on/off is a first-class variable. If one path uses native image forward with
DeepStack and another path uses manual `inputs_embeds` without DeepStack, report
the comparison as DeepStack on/off rather than cache behavior.

The eval report must state:

- `post_tgvf_forward_mode`;
- `post_tgvf_continuation`;
- `deepstack_enabled`;
- `deepstack_original_image_scope` when applicable;
- `question_suffix`;
- parser/scoring backend;
- max image resolution;
- max action/answer tokens.

## Failure Handling

If a launched run is discovered to be invalid:

1. Stop it if continuing wastes GPU time.
2. Mark the ledger entry as `SIDE_RESULT` or `INVALID_FOR_<baseline>`.
3. Explain the mismatch.
4. Preserve enough output to debug the mistake.
5. Do not quote its metric as comparable.
