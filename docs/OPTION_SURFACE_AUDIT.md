# Option Surface Audit

Status: draft inventory, not a deletion plan.

This document lists option families that currently make the repository hard to reason about. Nothing here should be deleted until it is classified and confirmed.

Generated helper artifact:

- `reports/argparse_option_inventory.json` contains a raw inventory of Python `argparse.add_argument(...)` options from `scripts/` and `eval/`.

## Current User Decisions

- Archive branch commit is done at `ce028f9` on `archive/tgvf-experiments-20260625`.
- A clean new project/tree is preferred after archival.
- Qwen3 supervised baseline anchor: `20260619 row-only open-answer`.
- Qwen3 next canonical checkpoint should be newly trained cleanly.
- Preferred supervised mask direction: old mask behavior.
- Qwen2: diagnostic side branch.
- VPT: archived baseline reference.
- RL/stage3: active in parallel.
- Unknown/unused switches must be verified with the user before deletion.

## Protocol Options

Observed active choices:

- `legacy_v3_tags`
- `protocol_c_thinking_special`
- `protocol_c_tool_observation`
- `protocol_c_tool_observation_qwen2_no_think`
- `protocol_d_qwen_tool`
- `protocol_e_action_evidence_special`

Proposed classification:

- **Active mainline candidate**: `protocol_c_tool_observation`
  - Qwen3 supervised path.
- **Diagnostic**: `protocol_c_tool_observation_qwen2_no_think`
  - Qwen2 side branch only.
- **Archive or unknown**:
  - `legacy_v3_tags`
  - `protocol_c_thinking_special`
  - `protocol_d_qwen_tool`
  - `protocol_e_action_evidence_special`

Questions for user:

1. Is `protocol_c_tool_observation` the only Qwen3 protocol that should remain in the clean mainline?
2. Should `protocol_c_thinking_special` be archived, or is it still needed for any current checkpoint?
3. Are protocol D/E dead, or should one be retained as a future tool-calling branch?

## Stage1 Options

Observed knobs:

- `--protocol-token-row-mode row_only|full_mask`
- `--capture-mode teacher_forced|decode_loop`
- `--focus-action-im-end`
- `--encoder-adapter-type bidirectional|bidirectional_film_aggressive`
- `--encoder-adapter-layers`
- `--encoder-adapter-share-weights`
- `--encoder-reencode-deepstack-compatible`
- `--train-reencode-vision-branch`
- `--same-image-negative-mode cyclic_margin|matrix_ce`
- `--loss-visual-token-manifold`
- `--mask-original-image-after-tgvf`
- `--fvt-position-mode native_source_grid|inherit_source_visual_positions`

Proposed classification:

- **Active mainline candidate**:
  - `row_only`
  - `teacher_forced`
  - `focus_action_im_end=true` for Protocol C tool-observation checkpoints
  - `bidirectional` adapter
  - `matrix_ce`
  - `native_source_grid`
  - frozen Qwen backbone
- **Ablation/unknown**:
  - `full_mask`
  - `decode_loop`
  - `bidirectional_film_aggressive`
  - adapter sharing
  - reencode/deepstack branch
  - `cyclic_margin`
  - `inherit_source_visual_positions`

Questions for user:

1. For the new clean Qwen3 training, should Stage1 be row-only only?
2. Should full-mask be preserved only as archived ablation?
3. Is encoder reencode still a live idea, or can it be archived?
4. Should `cyclic_margin` be removed from clean mainline in favor of `matrix_ce` only?

## Stage2 Options

Observed knobs:

- `--mask-original-image-after-tgvf`
- `--mask-original-image-after-tgvf-prob`
- `--mask-original-image-after-tgvf-scope evidence_only|through_answer`
- `--fast-batched-stage2`
- weighted span losses:
  - evidence_state
  - focus_target
  - evidence
  - value_span
  - answer
  - no_focus_evidence_state
  - no_focus_answer
- `--loss-visual-token-manifold`
- LoRA rank/alpha/dropout/targets
- `--use-stage1-tgvf-config`
- `--fvt-position-mode`

Proposed classification:

- **Active mainline candidate**:
  - old mask behavior / through-answer semantics, if confirmed;
  - `mask_original_image_after_tgvf_prob=1.0`;
  - `fast_batched_stage2=true`;
  - existing weighted span loss defaults;
  - use Stage1 config;
  - LoRA q/k/v/o/gate/up/down with current rank/alpha unless user chooses otherwise.
- **Ablation/unknown**:
  - `evidence_only`
  - probabilistic mask ratios such as 0.5/0.75
  - Stage2 manifold loss weight
  - alternative fvt position mode

Questions for user:

1. When you say old mask, should clean Stage2 default be `through_answer` with probability 1.0?
2. Should `mask_original_image_after_tgvf_prob` be removed from canonical launcher and kept only in ablation configs?
3. Do we raise Stage2 manifold loss in the new clean run, or first reproduce old behavior exactly?
4. Are the weighted span loss defaults now fixed, or should they remain configurable in mainline?

## Evaluation Options

Observed knobs:

- `--eval-mode force|free|both`
- `--post-tgvf-continuation natural_continue|answer_only|evidence_then_answer|think_then_answer`
- `--post-tgvf-forward-mode no_kv_full_sequence|kv_cache`
- `--question-suffix`, especially `use focus tool`
- `--force-prefix-mode` variants
- `--append-prefill-mode kv_append|full_sequence`
- `--mask-original-visual-keys-after-tgvf`
- `--include-stage2-direct`
- benchmark `tier` and `limit`

Proposed classification:

- **Active mainline candidate**:
  - free and softforce external benchmarks;
  - no-KV full-sequence for post-D correctness path;
  - `natural_continue` for historical external benchmark comparability unless a new table defines otherwise;
  - explicit resolved sample metadata for every run.
- **Diagnostic**:
  - force correct_D;
  - teacher-forced post-TGVF;
  - KV continuation;
  - mask original visual keys after TGVF;
  - append-prefill ablations.
- **Archive/unknown**:
  - many force-prefix variants;
  - `answer_only`/`think_then_answer` outside Qwen2 diagnostic or protocol eval;
  - `include-stage2-direct` inside TGVF external benchmark runner.

Questions for user:

1. For clean Qwen3 benchmark tables, should the official modes be only `free` and `softforce`?
2. Should `force` move to protocol diagnostics only, not external benchmark tables?
3. Should `post_tgvf_continuation=natural_continue` remain benchmark default for Qwen3?
4. Should `evidence_then_answer` be reserved for protocol eval and Qwen2 diagnostics?
5. Should KV continuation be removed from canonical eval runners and kept only in probe scripts?

## Benchmark Identity Options

Problem:

- `tier=full` does not define a stable sample set across code changes.
- BLINK historical `n=120` means old Counting-only adapter behavior.
- Current BLINK full means 1901 all-subtask rows.

Proposed clean rule:

- Every benchmark run writes:
  - `resolved_samples.jsonl`
  - `sample_summary.json`
  - `run_config.txt`
- Comparison scripts refuse to compare when sample sets differ unless explicitly marked cross-sample.

Questions for user:

1. Should historical benchmark tables be preserved exactly, including old adapter quirks?
2. Should new clean tables switch to current full BLINK 1901, while old BLINK Counting 120 becomes an archived legacy metric?
3. Which benchmark sample sets should be first-class in clean project:
   - VStar full;
   - HR full;
   - OCR full;
   - BLINK Counting historical 120;
   - BLINK all-subtasks 1901;
   - sampled multi-benchmark table?

## Chat Options

Observed knobs:

- chat profiles;
- force/free/direct;
- stateful KV;
- toolobs action stop `im_end|focus_end`;
- post-TGVF continuation choices;
- max foveations;
- raw/focus display flags.

Proposed classification:

- **Active mainline**:
  - one Qwen3 chat profile tied to the clean canonical checkpoint;
  - no-KV correctness path where relevant;
  - max image resolution 512.
- **Diagnostic/archive**:
  - Qwen2 profile;
  - older checkpoint profiles;
  - stateful KV;
  - old action stop modes.

Questions for user:

1. Should clean chat support only Qwen3 canonical profile initially?
2. Should Qwen2 chat be moved to diagnostics?
3. Is stateful KV still useful, or too risky given continuation mismatch?

## Script Surface

Many one-off launchers exist. Proposed clean categories:

- `scripts/train_stage1_qwen3.sh`
- `scripts/train_stage2_qwen3_oldmask.sh`
- `scripts/eval_qwen3_protocol.sh`
- `scripts/eval_qwen3_benchmarks.sh`
- `scripts/chat_qwen3_tgvf.sh`
- `scripts/build_rl_dataset.sh`
- `scripts/eval_diagnostic_qwen2.sh`
- `scripts/eval_reference_vpt.sh`

Everything else should become:

- archived launcher;
- diagnostic script;
- deleted only after confirmation.

Questions for user:

1. Should old launchers remain in `archive_scripts/` in the clean project, or only in the archive branch?
2. Do you want shell launchers or config-driven Python launchers as the clean default?
3. Should wandb be mandatory in clean training scripts?

