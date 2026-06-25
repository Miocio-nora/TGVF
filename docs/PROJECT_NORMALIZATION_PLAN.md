# Project Normalization Plan

The repository needs a cleanup pass, but cleanup must be staged. The current worktree contains many uncommitted changes and several experimental branches of behavior.

## Current State

Recent git anchors:

- `25d4451 add probabilistic evidence-only stage2 image masking`
- `eccffc9 snapshot tgvf v3 protocol c experiments`

Current worktree is dirty with changes across docs, eval scripts, training scripts, model code, tests, and many untracked scripts/outputs pointers. This should be treated as valuable experimental state until archived.

## Phase 0: Archive Before Cleanup

Do not delete, move, or normalize files until the user chooses an archive strategy.

Options:

1. **Archive commit on current branch**
   - Commit all current tracked and selected untracked files as a single "experiment snapshot".
   - Best for preserving exact state quickly.
   - Risk: commits messy files into main history.

2. **Archive branch**
   - Create a branch such as `archive/tgvf-experiments-20260625`.
   - Commit the snapshot there, then clean main branch separately.
   - Best balance for safety.

3. **Patch bundle only**
   - Save `git diff`, untracked file manifest, and selected scripts/docs to `backups/`.
   - Least invasive.
   - Risk: harder to reproduce than a commit.

Recommended: option 2.

## Phase 1: Define First-Class Experiment Families

Before deleting or consolidating scripts, choose which families remain first-class:

- Qwen3 protocol C row-only open-answer, 20260619 baseline.
- Qwen3 full-mask baseline, 20260621.
- Qwen3 oldmask075 correct-stage1, 20260623.
- Qwen2 no-think protocol branch.
- VPT/Qwen2 compatibility branch.
- RL/stage3 branch.

Everything else should become archived, deprecated, or deleted only after mapping to one of these families.

## Phase 2: Normalize Run Interfaces

Create one canonical launcher per job type:

- Stage1 train/eval.
- Stage2 train/eval.
- Protocol eval.
- External benchmark eval.
- Chat/manual inference.
- Diagnostic probes.

Each launcher should write `run_config.txt` and resolved sample metadata. Old scripts should become thin wrappers or move to an archive folder.

## Phase 3: Normalize Evaluation Data Identity

Add a resolved-samples artifact to every benchmark run:

- `resolved_samples.jsonl` with sample id, source file, subtask, gold, and media count.
- `sample_summary.json` with counts by benchmark/subtask/source file.

Comparison scripts should refuse to compare if sample sets differ unless explicitly marked as cross-sample comparison.

## Phase 4: Normalize Experiment Tables

Keep a small set of generated tables:

- `reports/qwen3_benchmark_table.json`
- `reports/qwen3_protocol_eval_table.json`
- `reports/qwen2_diagnostics_table.json`
- `reports/side_results.json`

Each table row should include output path and comparability status.

## Phase 5: Reduce Surface Area

Candidates for cleanup after archive:

- One-off launch scripts that duplicate canonical launchers.
- Abandoned probe scripts whose conclusions are already in `docs/EXPERIMENT_LEDGER.md`.
- Old docs that duplicate newer workflow docs.
- Output-summary scratch files in repo root.

Do not remove `third_party/`, old benchmark outputs, or checkpoint paths without explicit user approval.

## Decisions Needed From User

1. Archive strategy:
   - current branch commit;
   - archive branch commit;
   - patch bundle only.

2. Which Qwen3 branch is the canonical default:
   - 20260619 open-answer row-only;
   - 20260621 full-mask;
   - 20260623 oldmask075 correct-stage1;
   - another checkpoint.

3. Whether Qwen2 remains active or becomes side/diagnostic only.

4. Whether VPT remains active or is archived as a baseline reference.

5. Whether RL/stage3 files are first-class now or should be parked until Qwen3 supervised path is stable.

## User Decisions Recorded 2026-06-25

- Archive strategy:
  - Use archive branch commit first.
  - Strongly consider a clean new project/tree after archival, because the current repository has too many historical switches and ambiguous launch paths.
- Qwen3 supervised baseline:
  - Use `20260619 row-only open-answer` as the historical baseline anchor.
  - For the next actual Qwen3 checkpoint, train a new clean run rather than treating an existing experimental checkpoint as canonical.
  - Mask mode preference: old mask behavior is the intended supervised direction unless a new ablation explicitly changes it.
- Qwen2:
  - Demote to diagnostic side branch.
- VPT:
  - Archive as baseline reference.
- RL/stage3:
  - Keep active and advance in parallel with the supervised Qwen3 cleanup.
- Main concern:
  - The repository has too many branches/options/protocol variants.
  - Unknown or unused switches must be removed or quarantined, not kept as default-visible knobs.

## Recommended New Clean Project

Create a clean vNext tree after archival. The clean project should not inherit every historical flag.

Recommended shape:

```text
revisit_vlm_clean/
  docs/
    EXPERIMENT_LEDGER.md
    ENGINEERING_WORKFLOW.md
    ACTIVE_BASELINES.md
  src/
    tgvf/
      protocol.py
      train_stage1.py
      train_stage2.py
      eval_protocol.py
      eval_benchmarks.py
      chat.py
  scripts/
    train_stage1_qwen3.sh
    train_stage2_qwen3_oldmask.sh
    eval_qwen3_protocol.sh
    eval_qwen3_benchmarks.sh
    build_rl_dataset.sh
  archive_refs/
    README.md
```

The clean tree should contain only:

- Qwen3 supervised mainline.
- RL/stage3 active path.
- Diagnostic references for Qwen2 and VPT, not active default scripts.
- One canonical benchmark runner.
- One canonical protocol evaluator.
- One canonical chat runner.

## Option Surface To Collapse

### Protocols

Current problem:

- Multiple protocol variants exist (`legacy_v3_tags`, `protocol_c_thinking_special`, `protocol_c_tool_observation`, `protocol_c_tool_observation_qwen2_no_think`, protocol D/E).
- Some are historical or diagnostic but still appear in shared choices.

Recommendation:

- Active default: `protocol_c_tool_observation` for Qwen3.
- Diagnostic only: `protocol_c_tool_observation_qwen2_no_think`.
- Archive/remove from active launchers: legacy tags, protocol D, protocol E, older protocol C variants unless an active experiment requires them.

### Mask Modes

Current problem:

- `full_mask`, row-only, evidence-only, through-answer, mask probability, open-answer variants are mixed into scripts.

Recommendation:

- Active supervised default:
  - row-only/open-answer Stage1 baseline anchor from 20260619;
  - new Stage2 trained with old mask behavior.
- Explicit ablations only:
  - `full_mask`;
  - `mask_original_image_after_tgvf_prob`;
  - `evidence_only`.
- Remove these from default launchers; keep them only in archived ablation launchers or config files.

### Eval Forward Modes

Current problem:

- KV continuation and no-KV full-sequence can produce different semantics.

Recommendation:

- Default correctness-path eval: `no_kv_full_sequence`.
- KV continuation: explicit ablation/debug only.
- Every result table must label forward mode.

### Benchmark Identity

Current problem:

- `full` changed meaning after adapter changes.

Recommendation:

- Default runners must write `resolved_samples.jsonl`.
- Comparison runners must check sample identity against the baseline.
- Historical BLINK `n=120` must be labeled `BLINK Counting historical n=120`, not simply `BLINK full`.

### Launch Scripts

Current problem:

- Many one-off scripts encode stale defaults.

Recommendation:

- Keep one canonical runner per active path.
- Move one-off scripts into an archive folder after the archive commit.
- Do not expose old experimental flags in the canonical runner; use named config presets instead.
