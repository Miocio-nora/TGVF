# Clean Project Whitelist

This document is the control surface for building the clean TGVF project/tree.
It is a planning document, not an implementation diff. Do not delete or rewrite
executable code from this repository just because an item is marked out here.

## Repository Strategy

- Keep the current historical project as archive/reference.
- Use the cleanup branch for planning, audits, and small verification only unless
  the user explicitly confirms it as an implementation cleanup branch.
- Build the clean project/tree from a whitelist after the whitelist is confirmed.
- Unresolved items must not be removed, rewritten, or silently defaulted.

## Confirmed In

### Data Generation Mainline

Data generation is a first-class clean-project surface, not a side branch. The
deterministic dataset builders are already relatively clean, so the clean
project should preserve and wrap that path rather than rewrite it as part of
the current execution cleanup.

Boundary decision: data generation is not the main cleanup risk. Treat the
deterministic transform path as clean-native code that needs identity records,
tests, and schema/version discipline, not as a legacy bridge to be replaced
before training/eval cleanup can proceed.

- Keep the clean `tgvf_generate_data` entry point.
- Keep deterministic, local transforms as clean-native code:
  - `choice_to_open_answer`;
  - `clean_imend`;
  - V4 teacher rows to Protocol-C Stage1 focus data;
  - V4 teacher rows to Protocol-C Stage2 conversation data.
- Each generated dataset must record:
  - clean data-generation output schema version;
  - source manifest path/hash or source run id;
  - prompt/schema/protocol version;
  - transform names and parameters;
  - split policy and split hashes;
  - generated output file identities: path, SHA-256, byte size, and line count;
  - field/span weights;
  - focus/no-focus rules;
  - mask behavior, mask probability, and mask scope;
  - `im_end` policy.
- Heavy teacher trajectory generation should remain available as the upstream
  trace-production path, but it is not part of the immediate clean-native
  execution port. Existing generated runs can be consumed by explicit
  path/hash identity, and teacher-trace regeneration should only be ported when
  we intentionally regenerate traces.
- Do not mix generated-data identity with training launch defaults. Training
  launchers must consume generated JSONL by explicit path/hash and print that
  identity before launch.

### Main Qwen3 Protocol

- `protocol_c_tool_observation`.

### Stage1 Mainline

- Model family: Qwen3-VL-8B-Thinking mainline.
- Training data:
  - train: `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl`
  - eval: `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl`
- Protocol: `protocol_c_tool_observation`.
- Focus action termination: `focus_action_im_end=true`.
- TGVF module variant: `tgvf_v2_bidirectional`.
- Number of D tokens: dynamic/source-image visual token count, i.e.
  `num_foveated_tokens=none` for this variant.
- Protocol token row training mode: `row_only`.
- Capture mode: `teacher_forced`.
- FVT position mode: `native_source_grid`.
- Stage1 readout original-image context:
  - replace original image placeholder embeddings with the true Qwen merged visual
    tokens `V_merge`;
  - append D as a native Qwen visual span;
  - compute real Qwen3 M-RoPE position ids over the full trajectory.
- Stage1 readout attention mask: `mask_original_image_after_tgvf=true`, using
  weak-strict original-image-key blocking after the TGVF append point.
- Trainable modules:
  - TGVF module;
  - Protocol C input/output token rows via row-only wrappers.
- Frozen modules:
  - Qwen language/vision backbone;
  - Qwen visual merger, used only through the frozen merger finalize path.
- Clean launch plans must emit this module policy in machine-readable form,
  including frozen merger state, training `use_cache=false`, and the requirement
  to print trainable parameter names before launch.
- Clean launch plans may keep historical script commands as references only.
  The final training entrypoint must be the clean-native command/status channel,
  not `legacy_reference_command.sh`.
- Any `legacy_reference_command` artifact in a clean launch plan must be
  `executable=false`; it is an audit record, not a runnable fallback.
- Clean executors may write `clean_training_execution_bundle.json` as the
  executor-owned handoff artifact; until the trainer loop is ported it must
  record `will_launch_training=false`.
- Prepare-execution must also write `dataset_runtime_identity.json` and
  `first_batch_identity.json` by actually scanning the training JSONL; malformed
  rows or missing required clean training fields must fail before launch.
- Prepare-execution must write `checkpoint_contract.json`; Stage1 records its
  required output checkpoint keys, while Stage2 must load and validate the
  Stage1 checkpoint before accepting the handoff.
- Prepare-execution must write `optimizer_groups.json` from clean plan identity,
  including Stage1 `tgvf_module` / `protocol_c_token_rows` groups and Stage2
  `llm_lora` / `tgvf_refiner` / `fvt_calibration` groups.
- Runtime audit must validate the execution bundle plus prepared artifacts and
  write `clean_training_runtime_audit.json`,
  `clean_training_runtime_audit_status.json`, and `trainable_parameters.json`.
  Until the real model is loaded, `trainable_parameters.json` must be marked as
  `pending_model_load_not_actual_parameter_audit` and must not be treated as an
  actual trainable-parameter audit.
- Explicit model-parameter audit may load model components and write
  `status=actual_model_parameter_audit`, but it still must not launch optimizer
  steps or set `will_launch_training=true`.
- Explicit optimizer audit may write `optimizer_runtime.json` after constructing
  AdamW and LambdaLR from actual loaded modules. `optimizer_groups.json` is only
  the plan contract; `optimizer_runtime.json` is the real construction evidence.
  This still must not run backward, optimizer steps, scheduler steps, or
  checkpoint saving.
- Explicit checkpoint audit may write `checkpoint_runtime.json` plus a local
  `checkpoint_runtime_probe.pt` after saving and reloading the clean checkpoint
  schema from loaded modules and optimizer/scheduler state. The probe is audit
  evidence only, not a publishable training checkpoint, and it still must not run
  backward, optimizer steps, or scheduler steps.
- Explicit training-step audit may write `training_step_runtime.json` after a
  no-backward forward probe. Stage1 may validate readout context, M-RoPE
  position ids, matrix-CE mode, and manifold-loss wiring. Stage2 may validate
  fast batched path, weighted-span loss wiring, and original-image mask scope.
  This still must not run backward, optimizer steps, scheduler steps, or
  checkpoint saving.
- Explicit optimizer-step audit may write `optimizer_step_runtime.json` after
  one bounded backward, gradient clipping, `optimizer.step`, `scheduler.step`,
  and post-step `zero_grad` probe from the clean training-step loss. It is
  first-step wiring evidence only and must not enter an epoch loop, perform
  gradient accumulation, publish a checkpoint, or set `will_launch_training=true`.
- Explicit trainer-loop audit may write `trainer_loop_runtime.json` after one
  bounded gradient-accumulation probe using the planned
  `gradient_accumulation_steps`. It may validate micro-step backward
  accumulation and one optimizer/scheduler step, but it still must not enter the
  full epoch loop, publish checkpoints, or set `will_launch_training=true`.
- Explicit checkpoint-publish audit may write
  `training_checkpoint_publish_runtime.json` plus a local
  `training_checkpoint_publish_probe_step_1.pt` after the bounded trainer-loop
  probe. It may validate post-loop checkpoint keys, state parity, optimizer and
  scheduler reload, and historical step counters, but it still must not launch
  the full training run.
- Explicit checkpoint-resume audit may write
  `training_checkpoint_resume_runtime.json` after reloading a fresh
  model/optimizer/scheduler stack from the published checkpoint probe. It may
  validate model-state parity, optimizer/scheduler restore, protocol-token rows
  when required, and step counters, but it still must not continue into the full
  training run.
- Explicit cadence audit may write `training_cadence_runtime.json` after
  resolving `max_steps`, checkpoint-save steps, and Stage2 eval steps from the
  clean plan. It is static launch-contract evidence only and must not load the
  model or launch training.
- Explicit launch-readiness audit may write `training_launch_readiness.json`.
  It must summarize the existing runtime launch gates, checkpoint-resume probe,
  cadence probe, artifact statuses, and DeepStack state without introducing a
  parallel gate implementation. A clean contract may be marked ready for the
  trainer loop only when every required gate is identity-validated and no
  unsupported runtime feature remains;
  `will_launch_training` and `launch_permitted` must remain `false`.
- `clean_prepare_execution_command.sh` is the runnable clean handoff command;
  `clean_training_command.sh` is executable for supported single-process and
  torchrun distributed clean launches. Plans that require Stage2 DeepStack
  training injection must keep the command non-executable and record the
  blocking reason.
- Explicit clean launch may write `single_process_training_runtime.json`,
  `distributed_rank_N_training_runtime.json`,
  `clean_training_launch_result.json`, `clean_training_launch_status.json`, and
  `checkpoint_step_N.pt` checkpoints. Distributed torchrun launch must shard
  train samples by rank, set local-rank device maps, average optimizer gradients
  before clipping, and keep checkpoint/result writing on rank 0. Stage2
  `val_file` runs rank-0 no-backward validation at the planned `eval_every`
  cadence and records `validation_records`.
- Single-process launch must use deterministic sample cursors instead of
  replaying the fixed audit probe batch. Stage1 may group same-image samples
  for matrix CE; Stage2 train cursor follows `target_focus_ratio` when both
  focus/no-focus rows exist; validation cursor cycles sequentially. Runtime
  output must record cursor summaries and per-step `sample_trace` evidence.
- Loss defaults:
  - generation/readout LM loss: `1.0`;
  - visual token manifold: `0.1`;
  - same-image negative: `1.0`;
  - contrastive alignment: `0.0`.
- Same-image negative loss default: `matrix_ce`.
- Keep `cyclic_margin` available as an active diagnostic/ablation path; do not
  archive it as dead code.
- Optimizer/schedule defaults:
  - optimizer: AdamW;
  - learning rate: `1e-4`;
  - scheduler: `cosine`;
  - warmup steps: `100`;
  - min LR ratio: `0.1`;
  - max grad norm: `1.0`.
- Batch/default run scale:
  - target global batch: `32`;
  - default launcher: `4 GPUs * batch_size 4 * grad_accum 2`;
  - if GPU count changes, adjust micro-batch and accumulation to preserve global
    batch unless explicitly changing batch is the ablation.
- Default max image resolution: `512`.
- Default training length: `2000` steps.

### Stage2 Mainline

- Model family: Qwen3-VL-8B-Thinking mainline.
- Stage1 dependency:
  - start from the clean Stage1 mainline checkpoint, not a pre-2026-06-24
    Stage1 checkpoint with old readout context behavior;
  - load Stage1 `tgvf_module`;
  - restore Protocol C token rows from Stage1;
  - use the Stage1 TGVF config rather than hand-overriding the variant.
- Training data:
  - train: `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl`
  - eval: `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`
- Protocol: `protocol_c_tool_observation`.
- Focus action / tool observation behavior:
  - `focus_action_im_end=true`;
  - tool-observation action stop at `<|im_end|>`;
  - no duplicate leading `<|im_end|>` before the TGVF observation append.
- Data style: open-answer Stage2 data. Do not use old multiple-choice-only
  answer formatting as the clean mainline.
- TGVF module variant: inherited from clean Stage1, currently
  `tgvf_v2_bidirectional`.
- Number of D tokens: inherited from clean Stage1, currently dynamic/source-image
  visual token count, i.e. `num_foveated_tokens=none`.
- FVT position mode: `native_source_grid`.
- Default max image resolution: `512`.
- Default max sequence length: `2048`.
- Stage2 implementation path: fast batched Stage2 only. The slow path can remain
  as an archive/debug path, but it is not the clean benchmark-training path.
- Trainable modules:
  - Qwen LoRA adapters;
  - TGVF module continued from Stage1;
  - Protocol C token rows restored from Stage1 and preserved in Stage2
    checkpoints.
- Stage2 token-row implementation caveat:
  - the current PEFT path uses `modules_to_save=["embed_tokens", "lm_head"]`
    for token-row protocols;
  - the clean launcher must print trainable parameter names before launch;
  - do not silently switch to a different token-row implementation without
    making it an explicit ablation.
- Frozen modules:
  - base Qwen weights outside LoRA / protocol-token saved modules;
  - Qwen vision encoder;
  - Qwen visual merger, used through the frozen merger finalize path.
- Clean launch plans must emit this module policy in machine-readable form,
  including the PEFT token-row implementation caveat, frozen merger state,
  training `use_cache=false`, gradient checkpointing state, and the requirement
  to print trainable parameter names before launch.
- Clean launch plans may keep historical script commands as references only.
  The final training entrypoint must be the clean-native command/status channel,
  not `legacy_reference_command.sh`.
- Any `legacy_reference_command` artifact in a clean launch plan must be
  `executable=false`; it is an audit record, not a runnable fallback.
- Clean executors may write `clean_training_execution_bundle.json` as the
  executor-owned handoff artifact; until the trainer loop is ported it must
  record `will_launch_training=false`.
- Prepare-execution must also write `dataset_runtime_identity.json` and
  `first_batch_identity.json` by actually scanning the training JSONL; malformed
  rows or missing required clean training fields must fail before launch.
- Prepare-execution must write `checkpoint_contract.json`; Stage1 records its
  required output checkpoint keys, while Stage2 must load and validate the
  Stage1 checkpoint before accepting the handoff.
- Prepare-execution must write `optimizer_groups.json` from clean plan identity,
  including Stage1 `tgvf_module` / `protocol_c_token_rows` groups and Stage2
  `llm_lora` / `tgvf_refiner` / `fvt_calibration` groups.
- Runtime audit must validate the execution bundle plus prepared artifacts and
  write `clean_training_runtime_audit.json`,
  `clean_training_runtime_audit_status.json`, and `trainable_parameters.json`.
  Until the real model is loaded, `trainable_parameters.json` must be marked as
  `pending_model_load_not_actual_parameter_audit` and must not be treated as an
  actual trainable-parameter audit.
- Explicit model-parameter audit may load model components and write
  `status=actual_model_parameter_audit`, but it still must not launch optimizer
  steps or set `will_launch_training=true`.
- Explicit optimizer audit may write `optimizer_runtime.json` after constructing
  AdamW and LambdaLR from actual loaded modules. `optimizer_groups.json` is only
  the plan contract; `optimizer_runtime.json` is the real construction evidence.
  This still must not run backward, optimizer steps, scheduler steps, or
  checkpoint saving.
- Explicit checkpoint audit may write `checkpoint_runtime.json` plus a local
  `checkpoint_runtime_probe.pt` after saving and reloading the clean checkpoint
  schema from loaded modules and optimizer/scheduler state. The probe is audit
  evidence only, not a publishable training checkpoint, and it still must not run
  backward, optimizer steps, or scheduler steps.
- Explicit training-step audit may write `training_step_runtime.json` after a
  no-backward forward probe. Stage1 may validate readout context, M-RoPE
  position ids, matrix-CE mode, and manifold-loss wiring. Stage2 may validate
  fast batched path, weighted-span loss wiring, and original-image mask scope.
  This still must not run backward, optimizer steps, scheduler steps, or
  checkpoint saving.
- Explicit optimizer-step audit may write `optimizer_step_runtime.json` after
  one bounded backward, gradient clipping, `optimizer.step`, `scheduler.step`,
  and post-step `zero_grad` probe from the clean training-step loss. It is
  first-step wiring evidence only and must not enter an epoch loop, perform
  gradient accumulation, publish a checkpoint, or set `will_launch_training=true`.
- Explicit trainer-loop audit may write `trainer_loop_runtime.json` after one
  bounded gradient-accumulation probe using the planned
  `gradient_accumulation_steps`. It may validate micro-step backward
  accumulation and one optimizer/scheduler step, but it still must not enter the
  full epoch loop, publish checkpoints, or set `will_launch_training=true`.
- Explicit checkpoint-publish audit may write
  `training_checkpoint_publish_runtime.json` plus a local
  `training_checkpoint_publish_probe_step_1.pt` after the bounded trainer-loop
  probe. It may validate post-loop checkpoint keys, state parity, optimizer and
  scheduler reload, and historical step counters, but it still must not launch
  the full training run.
- Explicit checkpoint-resume audit may write
  `training_checkpoint_resume_runtime.json` after reloading a fresh
  model/optimizer/scheduler stack from the published checkpoint probe. It may
  validate model-state parity, optimizer/scheduler restore, protocol-token rows
  when required, and step counters, but it still must not continue into the full
  training run.
- Explicit cadence audit may write `training_cadence_runtime.json` after
  resolving `max_steps`, checkpoint-save steps, and Stage2 eval steps from the
  clean plan. It is static launch-contract evidence only and must not load the
  model or launch training.
- Explicit launch-readiness audit may write `training_launch_readiness.json`.
  It must summarize the existing runtime launch gates, checkpoint-resume probe,
  cadence probe, artifact statuses, and DeepStack state without introducing a
  parallel gate implementation. A clean contract may be marked ready for the
  trainer loop only when every required gate is identity-validated and no
  unsupported runtime feature remains;
  `will_launch_training` and `launch_permitted` must remain `false`.
- `clean_prepare_execution_command.sh` is the runnable clean handoff command;
  `clean_training_command.sh` is executable for supported single-process and
  torchrun distributed clean launches. Plans that require Stage2 DeepStack
  training injection must keep the command non-executable and record the
  blocking reason.
- Explicit clean launch may write `single_process_training_runtime.json`,
  `distributed_rank_N_training_runtime.json`,
  `clean_training_launch_result.json`, `clean_training_launch_status.json`, and
  `checkpoint_step_N.pt` checkpoints. Distributed torchrun launch must shard
  train samples by rank, set local-rank device maps, average optimizer gradients
  before clipping, and keep checkpoint/result writing on rank 0. Stage2
  `val_file` runs rank-0 no-backward validation at the planned `eval_every`
  cadence and records `validation_records`.
- Single-process launch must use deterministic sample cursors instead of
  replaying the fixed audit probe batch. Stage2 train cursor follows
  `target_focus_ratio` when both focus/no-focus rows exist; validation cursor
  cycles sequentially. Runtime output must record cursor summaries and per-step
  `sample_trace` evidence.
- Focus/no-focus sampling:
  - `target_focus_ratio=0.8`.
- Batch/default run scale:
  - target global batch: `128`;
  - default launcher: `4 GPUs * batch_size 16 * grad_accum 2`;
  - if GPU count changes, adjust micro-batch and accumulation to preserve global
    batch unless explicitly changing batch is the ablation.
- Optimizer/schedule defaults:
  - optimizer: AdamW;
  - LoRA learning rate: `2e-5`;
  - TGVF learning rate: `5e-6`;
  - TGVF calibration learning rate: `1e-5`;
  - Adam betas: `(0.9, 0.95)`;
  - Adam epsilon: `1e-8`;
  - weight decay: `0.01`;
  - scheduler: `cosine`;
  - warmup steps: `100`;
  - min LR ratio: `0.1`;
  - max grad norm: `1.0`.
- Training efficiency defaults:
  - gradient checkpointing: enabled;
  - training `use_cache`: disabled.
- Default training length:
  - max steps: `1200`;
  - save/eval interval: `300`;
  - internal eval max samples: `128`.

### Stage2 Loss Weights

Use the legacy weighted-span defaults unless the user explicitly changes them:

- `evidence_state: 0.2`
- `focus_target: 1.5`
- `evidence: 1.0`
- `value_span: 1.0`
- `answer: 1.0`
- `no_focus_evidence_state: 0.2`
- `no_focus_answer: 1.0`
- Stage2 visual token manifold loss: `0.0`.
- Stage2 same-image negative / matrix CE: disabled.
- Stage2 contrastive alignment: disabled.

### Stage2 Mask Scope

Clean mainline:

- `mask_original_image_after_tgvf=true`
- `mask_original_image_after_tgvf_prob=1.0`
- `mask_original_image_after_tgvf_scope=through_answer`

Meaning:

- after D is appended, post-TGVF query tokens are blocked from attending to
  original image visual keys through the answer;
- this is the deterministic expansion of the old behavior used by important
  early baselines such as the 20260619 open-answer row-only run.

Retain `evidence_only` only as an explicitly named ablation/reference setting:

- `evidence_only` blocks original image visual keys during evidence/readout and
  reopens original image access for answer tokens;
- existing `evidence_only` results are mixed with other changes, so they do not
  override the clean mainline default.

### Eval Continuation Modes

Use only `natural_continue` for the clean main benchmark path.

Current status:

- Historical best 512 TGVF VStar used `natural_continue`.
- `natural_continue` means appending D and then allowing the model to continue
  naturally after `<|tgvf_end|>`, with no extra evidence/answer prefix.
- Do not include `answer_only`, `evidence_then_answer`, or
  `think_then_answer` in the clean benchmark runner default surface.
- If a structured continuation is ever needed again, treat it as an archived
  diagnostic script or a separately named probe, not as a clean benchmark mode.
- Clean eval must still label the continuation mode explicitly so old results
  cannot be mixed with new ones.

### Clean Benchmark Populations

Clean benchmark names must bind to explicit source files and sample counts.
Historical labels such as `full`, `medium`, and `limit=120` are not valid
population identities.

Core clean image benchmark populations:

| Clean population id | Source files | n | Status |
| --- | --- | ---: | --- |
| `vstar_test_questions_191` | `vstar_bench/snapshot/test_questions.jsonl` | 191 | In clean core. |
| `hr_bench_4k_800` | `hr_bench_4k/snapshot/hr_bench_4k.parquet` | 800 | In clean core. Ignore duplicate TSV unless explicitly auditing parity. |
| `blink_val_all_subtasks_1901` | all 14 `blink/snapshot/*/val-*.parquet` files | 1901 | In clean core. This replaces historical Counting-120. |
| `ocrbench_v2_data_test_10000` | `ocrbench_v2/snapshot/data/test-00000-of-00004.parquet` through `test-00003-of-00004.parquet` | 10000 | In clean core. Use aggregate `data/` shards, not duplicated task directories. |
| `mmmu_pro_standard10_test_1730` | both `mmmu_pro/snapshot/standard (10 options)/test-*.parquet` shards | 1730 | In clean core. |
| `mathvista_testmini_1000` | `mathvista/snapshot/data/testmini-00000-of-00001-725687bf7a18d64b.parquet` | 1000 | In clean core. |
| `mathverse_testmini_3940` | `mathverse/snapshot/testmini.json` | 3940 | In clean core. |

Reference / optional populations:

| Population id | Source files | n | Status |
| --- | --- | ---: | --- |
| `blink_test_all_subtasks_hidden_1906` | all 14 `blink/snapshot/*/test-*.parquet` files | 1906 | Not a clean scored default because local labels are not the scored validation labels. |
| `ocrbench_v2_en_test_7400` | `ocrbench_v2/snapshot/EN/test-*.parquet` | 7400 | Optional subset/reporting slice. |
| `ocrbench_v2_cn_test_2600` | `ocrbench_v2/snapshot/CN/test-*.parquet` | 2600 | Optional subset/reporting slice. |
| `mmmu_pro_standard4_test_1730` | both `standard (4 options)` shards | 1730 | Optional ablation/reference. |
| `mmmu_pro_vision_test_1730` | all four `vision/test-*.parquet` shards | 1730 | Optional ablation/reference. |
| `mathvista_test_5141` | both `mathvista/snapshot/data/test-*.parquet` shards | 5141 | Not a clean scored default until answer/scorer availability is confirmed. |
| `ovo_bench_new_1640` | `ovo_bench/data/ovo_bench_new.json` | 1640 | Video-side benchmark, not part of the clean image core. |

Clean runner requirements:

- each adapter exposes a population id, not just a benchmark name;
- each run writes the exact source-file manifest;
- when a run supplies `--subset-id` plus `--manifest-path`, the manifest
  `manifest_id` must equal the subset id;
- when a run supplies `--population-id` plus `--manifest-path`, every manifest
  sample must come from that single population id;
- each row writes a stable sample id including subtask/shard where needed;
- shard merge verifies the merged row count equals the declared population
  count;
- subsets must be named as subsets and cannot reuse the clean full population
  id.

Clean subset policy:

- Subsets are fixed manifests, not ad hoc `limit` values.
- A subset result must never be labeled `full`.
- Each subset manifest records the source population id, seed, stratification
  fields, selected stable sample ids, and per-stratum counts.
- If a stratum has fewer rows than requested, take all available rows and record
  that fact in the manifest.

Clean subsets:

| Subset id | n | Purpose | Status |
| --- | ---: | --- | --- |
| `core_smoke_256_seed20260625` | 256 | Fast code/parser/scorer/DeepStack field smoke. Not for effect conclusions. | In clean project. |
| `core_balanced_dev_2511_seed20260625` | 2511 | Main fast experimental comparison subset with balanced benchmark/task coverage. Human short name: `CoreDev-2511`. | In clean project. |
| `core_full_19562` | 19562 | Final full clean image-core confirmation. | In clean project. |

`CoreDev-2511` / `core_balanced_dev_2511_seed20260625` allocation:

| Population id | subset n | Stratification rule |
| --- | ---: | --- |
| `vstar_test_questions_191` | 191 | full population |
| `hr_bench_4k_800` | 200 | balance `category`, `cycle_category`, and answer |
| `blink_val_all_subtasks_1901` | 420 | 14 subtasks, 30 rows per subtask |
| `ocrbench_v2_data_test_10000` | 600 | 30 `type` values, 20 rows per type |
| `mmmu_pro_standard10_test_1730` | 300 | 30 subjects, 10 rows per subject |
| `mathvista_testmini_1000` | 300 | balance `question_type` and `answer_type` |
| `mathverse_testmini_3940` | 500 | 5 `problem_version` values, 100 rows each, balanced across `question_type` where possible |

The clean runner must load this subset from a manifest, for example:

```text
benchmark_manifests/core_balanced_dev_2511_seed20260625.json
```

It must not reconstruct the subset from a bare `--limit 2511`.

### Protocol / im_end Behavior

The clean Qwen3 mainline uses:

- `tgvf_protocol=protocol_c_tool_observation`
- `focus_action_im_end=true` for Stage1 teacher-forced action capture
- tool-observation action stop at `<|im_end|>` for inference/eval/chat

Meaning:

- the focus action is not complete at `<|focus_end|>` alone;
- for the im_end-trained tool-observation path, the generated action should end
  with `<|im_end|>`;
- append logic must avoid duplicating a leading `<|im_end|>` before the TGVF
  observation when the captured action already ended with `<|im_end|>`.

Old focus-end-only behavior exists for old checkpoints, but it is archive /
compatibility material and should not enter the clean main benchmark runner.

### Baseline / Side Branch Status

- Qwen3 remains the mainline.
- RL / Stage3 remains active and should continue in parallel.
- VPT is retained as baseline reference, not mainline training code.
- Qwen2 is diagnostic side-branch material, not Qwen3 clean mainline.

## Confirmed Out

### Protocols

- Do not include non-`protocol_c_tool_observation` protocols in the Qwen3 clean
  mainline unless explicitly re-confirmed.
- Do not include the old focus-end-only `protocol_c_tool_observation` action
  stop behavior in the clean mainline.

### Stage1

- `full_mask` protocol-token-row mode.
- `decode_loop` Stage1 capture mode.
- `inherit_source_visual_positions` FVT position mode.
- Encoder reencode / deepstack / `train-reencode-vision-branch`.
- `tgvf_encoder_bidir_8_16_24` as a clean mainline variant.
- `bidirectional_film_aggressive`.
- encoder adapter layer choices as a clean-project option surface.
- encoder adapter sharing as a clean-project option surface.
- separate trainable reencode Qwen visual branch.

### Stage2 Mask Probability

- Do not expose stochastic partial masking as a clean-project option.
- If the existing implementation still has `mask_original_image_after_tgvf_prob`,
  the clean mainline fixes it to `1.0`.
- Values other than `1.0` are archive/ablation-only and must be named in the run
  identity.

### Benchmarks / Sample Sets

- Do not preserve the historical BLINK Counting-120 rule as a clean benchmark
  standard. It was an accidental/narrow subset and should not define future
  benchmark identity.
- Do not keep `blink_counting_all_120` as a clean benchmark target. It can be
  mentioned only as a historical invalid/side comparison when explaining old
  results.

## Unresolved

These items need code-backed audit and user confirmation before any cleanup
implementation.

### Eval Forward Mode

Forward mode must be recorded, but clean conclusions must not be framed as
KV/no-KV effects unless DeepStack state and other inputs are controlled.

Resolved rule:

- `post_tgvf_forward_mode` remains an explicit result field.
- DeepStack on/off is a first-class eval variable.
- If one path enables native Qwen3 DeepStack and another path disables it through
  a manual `inputs_embeds` path, report the comparison as DeepStack on/off, not
  as cache behavior.

Still required for implementation:

- expose DeepStack state in canonical `run_config.json` and merged summaries;
- real model execution with `deepstack.enabled=true` must fail fast until
  original-image DeepStack injection/masking semantics are implemented;
- the fail-fast path must expose a structured `deepstack_execution_plan`
  recording scope-specific masking/restoration semantics and remaining blockers;
- keep prompt/continuation/parser/sample identity fixed when comparing forward
  paths;
- add a smoke check that native no-DeepStack and manual no-DeepStack are
  equivalent for a controlled sample.

### Eval Entry Points

The current repo has multiple eval surfaces. They should not all enter the
clean project as peer entry points.

Resolved clean direction:

- Use one clean benchmark runner for external benchmark tables:
  `tgvf_eval_benchmark`.
- Keep old `src/tgvf_eval/*` as archive/reference. Its useful adapter,
  sampling, scoring, and result-writing pieces may be reused as libraries, but
  the old `src/tgvf_eval/run.py` method surface is not the clean runner.
- Do not delete old eval code until a clean runner reproduces a known baseline
  with the same sample identity.
- Clean benchmark continuation is `natural_continue` only.
- Preserve three eval families in the clean project:
  - Stage1/Stage2 non-benchmark diagnostics;
  - project-native external benchmark evaluation;
  - ValKit/VLMEvalKit evaluation.
- Do not collapse project-native external benchmarks and ValKit into one
  canonical path yet. ValKit is comparatively authoritative, but the project
  runner is easier to instrument and must be kept until equivalence, sample
  identity, and parser/scorer behavior are proven.

Clean eval families:

- `internal_diagnostic`: Stage1 readout, Stage2 protocol, D/no-D/random-D,
  query sensitivity, and similar probes. These are not benchmark-table runners.
- `project_native_external`: the main clean benchmark runner for experimental
  tables, with TGVF-specific fields such as trigger, focus, D condition,
  parser identity, and DeepStack state.
- `valkit`: first-class external validation family. It remains separate from
  project-native external results unless sample identity, parser/scorer, and
  output semantics are proven equivalent.

The clean benchmark runner must support:

- modes: `original`, `tgvf_free`, `tgvf_force`, `tgvf_softforce`;
- sharded execution and deterministic merge;
- sample manifest / sample identity export;
- parser/scorer identity export;
- DeepStack state export;
- DeepStack execution guard: identity-only dry/materialize/render may record
  DeepStack state, but model execution must not silently run without the
  requested DeepStack behavior;
- max image resolution export;
- exact checkpoint and processor export;
- row outputs, merged summary, and canonical `run_config.json`.

Clean output schema:

Each clean benchmark run writes:

```text
run_config.json
rows.jsonl
summary.json
sample_manifest.json
logs/
```

For sharded runs:

```text
shards/shard_<index>/run_config.json
shards/shard_<index>/rows.jsonl
shards/shard_<index>/summary.json
merged/rows.jsonl
merged/summary.json
```

`run_config.json` required fields:

- run id, start time, code commit, dirty worktree status;
- checkpoint path, processor path/id, model id;
- eval family: `internal_diagnostic`, `project_native_external`, or `valkit`;
- mode: `original`, `tgvf_free`, `tgvf_force`, or `tgvf_softforce`;
- benchmark population id, subset id if any, manifest path, manifest hash;
- benchmark source-file manifest;
- max image resolution, max action tokens, max answer tokens;
- TGVF protocol, continuation mode, forward mode;
- DeepStack enabled/state/scope;
- prompt suffix / softforce prompt text;
- parser identity and scorer identity;
- official scorer status;
- GPU/shard identity.

`rows.jsonl` required fields:

- sample id, benchmark, population id, subset id, source file, shard id;
- question, choices, gold answer, benchmark metadata;
- method/mode, D condition, trigger policy;
- raw output, final output, parsed answer;
- score, answer parse success, malformed flag;
- trigger focus decision, focus valid, focus target;
- append success, D shape, continuation metadata;
- parser identity, scorer identity, official scorer status;
- DeepStack state and forward mode.
- DeepStack execution evidence:
  - requested DeepStack state/scope;
  - whether the requested state is executable;
  - FVT append path and position mode;
  - whether FVT append actually used DeepStack visual features;
  - any DeepStack caution/blocking note.

`summary.json` required fields:

- n rows and n scored;
- accuracy, answer parse rate, malformed rate;
- trigger rate, focus-valid rate, append-success rate;
- by benchmark, by population id, by method/mode, by D condition;
- prediction counts and gold counts for choice benchmarks;
- parser/scorer identity and official scorer status;
- manifest path/hash and merged row-count verification;
- DeepStack execution summary, including reported FVT rows and rows whose FVT
  append actually used DeepStack visual features;
- comparability flags, including whether the run is clean-core, subset,
  side-result, or invalid for a named baseline.

Text `run_config.txt` can be emitted for readability, but `run_config.json` is
the canonical machine-readable identity.

Current code-backed classification:

- Clean benchmark runner source material:
  - `eval/eval_v3_mmmu_force.py`
  - clean merge helper: `tgvf_merge_benchmark`
  - historical merge reference:
    `scripts/merge_tgvf_v3_external_benchmark_shards.py`
- Benchmark-specific archive/reference scripts:
  - `eval/eval_v3_vstar_force.py`
  - `eval/eval_qwen3_mmmu_base_direct.py`
  - `eval/eval_qwen3_vstar_base_direct.py`
- Stage2/protocol internal diagnostics:
  - `eval/eval_v3_stage2_protocol.py`
  - `eval/eval_v3_stage2_action_ab.py`
  - `scripts/probe_qwen2_post_d.py`
- Stage1/FVT diagnostics:
  - `eval/eval_v3_readout.py`
  - `eval/eval_v3_query_sensitivity.py`
  - `eval/eval_v3_fvt_distribution.py`
- Old generic benchmark framework:
  - `src/tgvf_eval/run.py`
  - `src/tgvf_eval/run_suite.py`
  - `src/tgvf_eval/force_ablation.py`
- Third-party wrapper:
  - `scripts/run_vlmevalkit_tgvf.sh`
  - `scripts/run_vstar_valkit_*.sh`
- Manual/chat diagnostics:
  - `scripts/chat_tgvf_v3_stage2.py`
  - `scripts/chat_tgvf_v3.py`
  - `scripts/chat_qwen3_vl.py`
- Baseline/reference side branches:
  - `scripts/eval_vpt_qwen2_blink_full_512.py`
  - Qwen2-specific probe/eval scripts.

Clean-project target:

- One external benchmark CLI should cover original/free/force/softforce,
  sharded execution, benchmark adapters, parser/scorer identity, sample
  identity, `natural_continue`, explicit forward mode, and DeepStack state.
- Internal Stage2 protocol and Stage1/FVT diagnostics should remain separate
  probe CLIs, not benchmark-table runners.
- The historical Stage2 bridge backend may remain only as an
  `internal_diagnostic` backend. It must be rejected for
  `project_native_external` benchmark runs.
- VLMEvalKit should remain a first-class preserved eval family, but not silently
  interchangeable with project-native external benchmark results unless its
  runner emits the same identity fields and sample definitions.
- `tgvf_eval_benchmark` is not the ValKit runner and must reject
  `eval_family=valkit`.
- ValKit's clean surface is `tgvf_eval_valkit`. It writes identity, preflight,
  and prepare-execution handoff artifacts without launching by default. Real
  ValKit execution requires explicit `--execute`, `--valkit-root`, and
  `--valkit-model-name`, and must call `<valkit-root>/run.py` directly rather
  than historical wrapper scripts.

Clean prompt policy:

- `original`: no extra prompt.
- `tgvf_free`: no extra prompt.
- `tgvf_force`: force focus-action triggering.
- `tgvf_softforce`: only one explicitly configured soft prompt, such as
  `use focus tool`; the exact text must be written to `run_config.json`.
- Free mode must never silently include a prompt suffix.

Clean parser/scorer rule:

- Split model-output parsing from benchmark scoring.
- Use the current stable V3 external benchmark parser/scorer path as the clean
  mainline:

```text
model_output_parser = revisit_vlm_clean.scoring.parse_and_score:v3_external
source = revisit_vlm_clean.scoring
choice_parser = revisit_vlm_clean.scoring.extract_choice_strict
scoring_backend = auto
official_scorer_source = src/tgvf_eval/official_tools.py
```

- The clean runner records protocol/focus/TGVF state separately on each row, and
  `model_output_parser` / `choice_parser` identify the answer parsing surface.
- `benchmark_scorer` computes accuracy. Use `scoring_backend=auto` by default:
  official / official-compatible scorer when wired, project fallback otherwise.
- If fallback scoring is used, output must include:

```text
scorer_name=project_choice_exact_match or project_open_exact_match
official_tool_used=false
```

- Fallback scores must not be reported as official benchmark scores.
- Do not use the old generic `src/tgvf_eval/parsing.py` parser as the clean main
  parser. It can remain archive/reference material.
- Do not keep VStar's older standalone `eval/eval_v3_vstar_force.py::extract_choice`
  as a peer clean parser. VStar should go through the unified V3 external path.

### Benchmark Taxonomy / Naming

Current problem:

- Historical names such as `full`, `medium`, and `limit=120` are not reliable
  experiment identities.
- Historical "BLINK full n=120" actually meant the Counting parquet only, not
  all current BLINK subtasks.
- Current BLINK `full` can mean all subtasks and a much larger sample count.
- Some launchers use benchmark category names that describe implementation
  convenience rather than the benchmark population being evaluated.
- This issue is not BLINK-specific. Current adapter code also has multi-shard
  population risks for OCRBench v2, MMMU-Pro, MathVista, and MathVerse.

Clean-project rule:

- Benchmark names must encode the population, not just the adapter tier.
- A result table must report both a human label and a concrete sample identity.
- Do not call a run `full` unless it evaluates the complete defined population
  for that clean benchmark label.
- Narrow subsets must be named as subsets, for example:
  - `blink_all_subtasks_full`
  - `blink_all_subtasks_sample300_seed42`
- External benchmark and ValKit results must use comparable labels only after
  their sample populations and scorers are confirmed equivalent.

Code-backed audit from the current project-native adapters:

| Benchmark | Current adapter/load behavior | Clean status |
| --- | --- | --- |
| `vstar_bench` | Uses `snapshot/test_questions.jsonl`, 191 rows. | Clean core population: `vstar_test_questions_191`. |
| `hr_bench_4k` | Uses `snapshot/hr_bench_4k.parquet`, 800 rows. TSV duplicate also exists. | Clean core population: `hr_bench_4k_800`; avoid double-counting TSV. |
| `blink` | Current adapter loads all validation subtasks, 1901 rows. Historical old run used Counting only, 120 rows. | Clean core population: `blink_val_all_subtasks_1901`. Counting-120 is historical only. |
| `ocrbench_v2` | Current preferred file is only `snapshot/EN/test-00000-of-00003.parquet`, 2467 rows, while local aggregate `data/` has four shards / 10000 rows. Task directories duplicate broader groupings and total 30000 rows if naively combined. | Clean core population: `ocrbench_v2_data_test_10000`. EN/CN are optional reporting slices. |
| `mmmu_pro` | Current preferred file is only `snapshot/standard (10 options)/test-00000-of-00002.parquet`, 865 rows. Local standard-10 has two shards, 1730 rows; standard-4 and vision variants also exist. | Clean core population: `mmmu_pro_standard10_test_1730`; standard-4 and vision are optional ablations. |
| `mathvista` | Current adapter uses `testmini` parquet, 1000 rows. Local snapshot also has `test` shards and `annot_testmini.json`. | Clean core population: `mathvista_testmini_1000`; `mathvista_test_5141` requires scorer/answer confirmation. |
| `mathverse` | Current adapter uses `testmini.json`, 3940 rows. Text-only variants also exist. | Clean core population: `mathverse_testmini_3940`. |
| `ovo_bench` | Uses `data/ovo_bench_new.json`, 1640 rows; official-code copy duplicates it. | Keep as video-side reference, not clean image core. |

Clean-project consequence:

- Old benchmark numbers should be treated as historical/side results unless the
  exact sample identity is reconstructed.
- The clean benchmark suite should be re-measured from explicit populations,
  not by trying to preserve old ambiguous `full`/`medium` meanings.
- Before running new clean measurements, each benchmark adapter must expose a
  stable population id and concrete source-file manifest.

### Parser / Scorer Surface

Resolved clean mainline:

- Use `revisit_vlm_clean.scoring.parse_and_score` as the clean external
  benchmark model-output parser/scorer dispatcher. It is the clean port of the
  stable V3 external path.
- Use `revisit_vlm_clean.scoring.extract_choice_strict` as the strict
  multiple-choice parser before scorer fallback.
- Use `src/tgvf_eval/official_tools.py` for official/official-compatible scorer
  wrappers under `scoring_backend=auto`.
- Use `tgvf_merge_benchmark` for clean deterministic shard merge. The historical
  `scripts/merge_tgvf_v3_external_benchmark_shards.py` remains only a summary
  field reference: accuracy, answer parse rate, trigger rate, focus-valid rate,
  prediction counts, and gold counts.

Archive/reference:

- `src/tgvf_eval/parsing.py` simple parser;
- `eval/eval_v3_mmmu_force.py` parser/scorer implementation as source code; it
  remains only a reference for the clean port;
- VStar standalone `extract_choice` in `eval/eval_v3_vstar_force.py`;
- any benchmark-specific parser not routed through the V3 external dispatcher.

Required implementation cleanup:

- move the stable V3 external parser/scorer code into a shared clean module
  instead of leaving it embedded in `eval_v3_mmmu_force.py`;
- add parser identity and scorer identity to `run_config.json`, row metadata,
  and merged summaries;
- keep fallback scorer clearly labeled.

### Stage1 Decision

Resolved for clean Qwen3 mainline:

- Use `tgvf_v2_bidirectional`, not encoder-reencode variants.
- Use `focus_action_im_end=true`.
- Use `row_only` token-row training.
- Use `teacher_forced` capture.
- Use `native_source_grid` position ids.
- Use Stage1 readout with original image embeddings replaced by true `V_merge`.
- Use `mask_original_image_after_tgvf=true` for Stage1 readout.
- Use `matrix_ce` as the default same-image negative objective.
- Keep `cyclic_margin` only as an explicit diagnostic/ablation objective.
- Use `loss_visual_token_manifold=0.1` by default.
- Treat old Stage1 checkpoints trained before the Stage1 readout
  original-image-embedding fix as a different setting, even if their CLI
  hyperparameters match.

Archived / not clean mainline:

- old non-im_end/focus-end-only behavior;
- full tensor gradient-mask token-row training (`full_mask`);
- decode-loop capture;
- inherited source visual positions;
- encoder reencode/deepstack/two-branch family;
- aggressive FiLM encoder adapters;
- adapter sharing/layer-choice option surface;
- trainable reencode vision branch.

### Stage2 Decision

Resolved for clean Qwen3 mainline:

- use the fast batched Stage2 path;
- use LoRA target modules:
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`;
- use LoRA rank `64`, alpha `256`, dropout `0.05`, bias `none`;
- enable gradient checkpointing for training;
- keep training `use_cache=false`;
- use `target_focus_ratio=0.8`;
- keep the legacy focus/no-focus weighted-span losses listed above;
- do not enable Stage2 visual manifold, matrix CE, same-image negative, or
  contrastive losses by default;
- use deterministic original-image-key masking through answer:
  `mask_original_image_after_tgvf=true`,
  `mask_original_image_after_tgvf_prob=1.0`,
  `mask_original_image_after_tgvf_scope=through_answer`.

Archive / ablation only:

- stochastic mask probability values other than `1.0`;
- `evidence_only` as the Stage2 default;
- no-mask Stage2 as the mainline;
- slow Stage2 as the benchmark-training path;
- changing LoRA rank/targets/dropout/LRs without naming the ablation;
- Stage2 manifold regularization greater than `0.0`;
- using raw script defaults such as `batch_size=1, grad_accum=16` as the clean
  launcher setting.

### Benchmark Set

Resolved clean image core:

- VSTAR: `vstar_test_questions_191`.
- HR-Bench 4K: `hr_bench_4k_800`.
- BLINK: `blink_val_all_subtasks_1901`.
- OCRBench v2: `ocrbench_v2_data_test_10000`.
- MMMU-Pro: `mmmu_pro_standard10_test_1730`.
- MathVista: `mathvista_testmini_1000`.
- MathVerse: `mathverse_testmini_3940`.

Resolved clean image subsets:

- `core_smoke_256_seed20260625`: smoke only, not for effect conclusions.
- `CoreDev-2511` / `core_balanced_dev_2511_seed20260625`: default fast dev
  comparison subset.
- `core_full_19562`: final full clean image-core confirmation.

Side/reference:

- OVO-Bench: video-side reference, not image core.
- StreamingBench: local deployment is `dry_run`; not clean core.
- COR benchmark: not present in the current local benchmark root; unresolved
  until the dataset path/source is provided or rediscovered.
- ValKit benchmark suite definitions remain first-class but separate from
  project-native populations until sample/scorer parity is proven.

## Implementation Rule

No executable cleanup should happen from this document until the user explicitly
approves the affected section. Approval of an item as "Confirmed Out" means it
will not enter the clean project; it does not automatically authorize deletion
from the historical project.
