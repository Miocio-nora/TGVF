# Revisit VLM Clean

Clean TGVF project skeleton.

This tree is intentionally separate from the historical implementation in the
repository root. The first implementation slice defines interfaces, schemas,
CLI entrypoints, and tests only. Heavy training/evaluation behavior is ported in
later phases after each contract is validated.

## Defaults

- Main model family: Qwen3-VL-8B-Thinking.
- Main protocol: `protocol_c_tool_observation`.
- Main benchmark continuation: `natural_continue`.
- Main parser/scorer identity:
  `revisit_vlm_clean.scoring.parse_and_score:v3_external`.
- Main fast dev subset: `CoreDev-2511`.
- DeepStack support: supported by schema, default disabled.
- Benchmark root and scoring backend are recorded in `run_config.json` and
  `run_config.txt`.
- `official` scoring currently supports official-compatible multiple-choice
  parsing/scoring for BLINK and HR-Bench-4K, plus official batch scorers for
  OCRBench-v2, MMMU-Pro, MathVista, and MathVerse when their local
  `official_code` trees are present. MathVista and MathVerse currently use the
  default disabled-LLM path only.

## Entry Points

```bash
tgvf_build_manifest --list
tgvf_generate_data --help
tgvf_eval_benchmark --help
tgvf_eval_valkit --help
tgvf_train_stage1 --print-defaults
tgvf_train_stage2 --print-defaults
tgvf_train_stage1_executor --help
tgvf_train_stage2_executor --help
```

The benchmark entrypoint can now validate identity, build/materialize fixed
manifests, render model input rows for smoke checks, and execute the dry-run or
original-Qwen backend. It can also execute the diagnostic Qwen3 Stage2 TGVF
bridge backend for path-backed samples; full benchmark claims still require
explicit manifest and ledger identity. This entrypoint is for
`project_native_external` and `internal_diagnostic` eval families only; ValKit
must use a separate runner surface.

The ValKit entrypoint is separate:

```bash
tgvf_eval_valkit --run-id valkit_preflight --checkpoint-path /path/to/model.pt --output-dir /tmp/valkit --benchmark vstar --preflight-only
tgvf_eval_valkit --run-id valkit_preflight --checkpoint-path /path/to/model.pt --output-dir /tmp/valkit --benchmark vstar --prepare-execution
tgvf_eval_valkit --run-id valkit_run --checkpoint-path /path/to/model.pt --output-dir /tmp/valkit --benchmark vstar --valkit-root third_party/VLMEvalKit --valkit-model-name clean_tgvf_qwen3 --execute
```

It writes a clean ValKit plan/preflight report plus an optional
prepare-execution bundle. With explicit `--execute`, it calls
`<valkit-root>/run.py` directly and records the launch command, stdout, stderr,
return code, and execution status. It refuses to call historical shell wrappers.

Scoring is applied after all rows are produced. This is intentional: some
official scorers, including OCRBench-v2, need batch-level prediction files
rather than one isolated row at a time.

The data-generation entrypoint records source files, hashes, protocol,
transform, field weights, mask policy, and output intent. It can currently
execute the deterministic `v4_to_protocol_c`,
`v4_to_stage1_protocol_c_focus`, `choice_to_open_answer`, and `clean_imend`
transforms. Executed transforms write output file identities and split hashes
beside the generated JSONL. This path is treated as a clean first-class asset
rather than a legacy bridge. Because these deterministic transforms are already
the clean part of the data path, the current cleanup preserves and wraps them
instead of rewriting them. Teacher trajectory generation still remains outside
the clean tree until regeneration is intentionally needed.

The training entrypoints currently produce auditable launch plans with
dataset/checkpoint hashes, batch math, mask policy, weighted losses, DeepStack
state, a clean-native executor status, and a separate temporary legacy
reference command. They also write `clean_prepare_execution_command.sh`, which
runs the clean executor handoff and produces execution-bundle artifacts without
starting training. `clean_training_command.sh` remains commented because the
trainer loop itself is not ported yet.

The planned clean training modules are importable:

```bash
python -m revisit_vlm_clean.training.stage1_executor --plan /path/to/training_plan.json --preflight-only
python -m revisit_vlm_clean.training.stage2_executor --plan /path/to/training_plan.json --preflight-only
tgvf_train_stage1_executor --plan /path/to/training_plan.json --preflight-only
tgvf_train_stage2_executor --plan /path/to/training_plan.json --preflight-only
tgvf_train_stage1_executor --plan /path/to/training_plan.json --prepare-execution
tgvf_train_stage2_executor --plan /path/to/training_plan.json --prepare-execution
tgvf_train_stage2_executor --plan /path/to/training_plan.json --prepare-execution --audit-runtime
tgvf_train_stage2_executor --plan /path/to/training_plan.json --prepare-execution --audit-runtime --audit-model-parameters
tgvf_train_stage2_executor --plan /path/to/training_plan.json --prepare-execution --audit-runtime --audit-optimizer
tgvf_train_stage2_executor --plan /path/to/training_plan.json --prepare-execution --audit-runtime --audit-checkpoint
tgvf_train_stage2_executor --plan /path/to/training_plan.json --prepare-execution --audit-runtime --audit-training-step
tgvf_train_stage2_executor --plan /path/to/training_plan.json --prepare-execution --audit-runtime --audit-optimizer-step
tgvf_train_stage2_executor --plan /path/to/training_plan.json --prepare-execution --audit-runtime --audit-trainer-loop
tgvf_train_stage2_executor --plan /path/to/training_plan.json --prepare-execution --audit-runtime --audit-checkpoint-publish
tgvf_train_stage2_executor --plan /path/to/training_plan.json --prepare-execution --audit-runtime --audit-checkpoint-resume
tgvf_train_stage2_executor --plan /path/to/training_plan.json --prepare-execution --audit-runtime --audit-cadence
```

Without `--preflight-only`, `--prepare-execution`, or `--audit-runtime`, these
executors fail fast instead of launching a partial or legacy training path.
Preflight writes a JSON report next to the plan by default, or to
`--preflight-report` when that path is provided.
`--prepare-execution` writes executor-owned
`clean_training_execution_bundle.json`, `clean_training_execution_status.json`,
`dataset_runtime_identity.json`, `first_batch_identity.json`,
`checkpoint_contract.json`, `optimizer_groups.json`, and a text summary without
launching training. These artifacts are the clean handoff surface for the
future trainer loop; they still record `will_launch_training=false` until that
loop is ported. Stage2 prepare-execution loads and validates the Stage1
checkpoint contract before the handoff is accepted.
`--audit-runtime` validates the execution bundle plus runtime artifacts and
writes `clean_training_runtime_audit.json`,
`clean_training_runtime_audit_status.json`, and `trainable_parameters.json`.
That trainable-parameter artifact is explicitly marked
`pending_model_load_not_actual_parameter_audit`; the real trainer loop must
overwrite it after loading the model and before the first optimizer step.
With explicit `--audit-model-parameters`, the runtime audit loads the planned
Stage1/Stage2 model components and writes an actual trainable/frozen parameter
audit instead. This is an expensive model-load check and still does not run
optimizer steps or mark `will_launch_training=true`.
With explicit `--audit-optimizer`, the runtime audit also writes
`optimizer_runtime.json` after constructing the planned AdamW optimizer and
LambdaLR scheduler from the loaded modules. `optimizer_groups.json` remains the
plan contract; `optimizer_runtime.json` is the actual construction evidence.
This still does not call `backward`, `optimizer.step`, `scheduler.step`, or
save a checkpoint.
With explicit `--audit-checkpoint`, the runtime audit also writes
`checkpoint_runtime.json` and a local `checkpoint_runtime_probe.pt` after
saving and reloading the clean checkpoint schema from loaded modules plus
optimizer/scheduler state. This is checkpoint contract evidence, not a training
checkpoint, and it still does not call `backward`, `optimizer.step`, or
`scheduler.step`.
With explicit `--audit-training-step`, runtime audit writes
`training_step_runtime.json` after running a no-backward forward probe. Stage1
records readout-context, M-RoPE position-id, matrix-CE, and manifold-loss
evidence. Stage2 records fast batched path, weighted-span loss, and
original-image mask-scope evidence. This still does not call `backward`,
`optimizer.step`, `scheduler.step`, or checkpoint save.
With explicit `--audit-optimizer-step`, runtime audit also writes
`optimizer_step_runtime.json` after one bounded backward, gradient clipping,
`optimizer.step`, `scheduler.step`, and post-step `zero_grad` probe from the
clean training-step loss. It implies model-parameter, optimizer, and
training-step runtime audits. This is first-step wiring evidence only: it still
does not enter an epoch loop, perform gradient accumulation, publish a
checkpoint, or set `will_launch_training=true`.
With explicit `--audit-trainer-loop`, runtime audit also writes
`trainer_loop_runtime.json` after one bounded gradient-accumulation probe using
the planned `gradient_accumulation_steps`. It proves micro-step backward
accumulation plus one optimizer/scheduler step from the clean training-step
loss. This is loop-order evidence only: it still does not enter the full epoch
loop, publish checkpoints, or set `will_launch_training=true`.
With explicit `--audit-checkpoint-publish`, runtime audit also writes
`training_checkpoint_publish_runtime.json` and a local
`training_checkpoint_publish_probe_step_1.pt` after the bounded trainer-loop
probe. The checkpoint uses historical clean step semantics: Stage1 records
`global_step=optimizer_step=1`; Stage2 records `global_step=1` and
`micro_step=gradient_accumulation_steps`. This proves post-loop checkpoint
publish/load wiring only; it still does not launch the full training run.
With explicit `--audit-checkpoint-resume`, runtime audit also writes
`training_checkpoint_resume_runtime.json` after reloading a fresh
model/optimizer/scheduler stack from the published checkpoint probe. This
proves clean resume wiring and step-counter restoration only; it still does not
continue into the full training run.
With explicit `--audit-cadence`, runtime audit writes
`training_cadence_runtime.json` after resolving `max_steps`, checkpoint-save
steps, and Stage2 eval steps from the plan. This is static launch-contract
evidence only and does not load the model.

## Fixed Manifests

Committed benchmark manifests:

```text
benchmark_manifests/core_smoke_256_seed20260625.json
benchmark_manifests/core_balanced_dev_2511_seed20260625.json
benchmark_manifests/core_full_19562.json
benchmark_manifests/diagnostic_vstar_first_1_20260626.json
benchmark_manifests/diagnostic_vstar_core_smoke_first_8_20260626.json
benchmark_manifests/diagnostic_vstar_core_smoke_32_20260626.json
benchmark_manifests/diagnostic_vstar_softforce_trigger_row10_20260626.json
```

`CoreDev-2511` is the default fast development comparison subset. Manifest
generation is deterministic and tested by count/hash.
Diagnostic manifests are for runner validation only, not benchmark reporting.

## Render Semantics

- `original`: question/media only, no TGVF controller.
- `tgvf_free`: question/media only; no extra prompt text.
- `tgvf_force`: question/media only plus a protocol control prefix.
- `tgvf_softforce`: question/media plus the configured short prompt text.

## Executable Backends

- `dry_run`: validates rows, parser, scorer, and summaries without loading a
  model.
- `qwen3_original`: runs `mode=original` only. Use a small
  `--max-answer-tokens` for smoke checks.
- `tgvf_stage2_qwen3_legacy`: diagnostic bridge to the historical Stage2
  evaluator for `tgvf_force`, `tgvf_free`, and `tgvf_softforce` on path-backed
  image samples. It is gated to `eval_family=internal_diagnostic` so it cannot
  silently enter clean benchmark tables.
- `tgvf_stage2_qwen3`: generic clean Stage2 backend name. It resolves to the
  clean-native backend.
- `tgvf_stage2_qwen3_native`: explicit clean-native backend name. It now owns a
  native engine and does not use the legacy evaluator class. Force/free/
  softforce control flow, lazy checkpoint/model loading, D construction, visual
  D append, and post-TGVF continuation are ported into the clean tree. A
  one-sample Qwen3 force-path GPU smoke has passed, and one-sample
  free/softforce no-trigger GPU smokes have passed. A trigger-positive
  softforce smoke also matches the legacy bridge on the same fixed sample. An
  8-row fixed VStar softforce manifest now matches the legacy bridge exactly at
  row-output level. No-KV validation is still required before benchmark claims.

## Clean-Native Exit Criteria

The final clean project must not depend on historical evaluator or launcher
entrypoints for first-class workflows. The current Stage2 bridge is allowed only
as a diagnostic compatibility layer until the native runner replaces it.
The current training launchers may write temporary historical reference
commands, but those commands are not final clean-native execution paths.

Data generation is a first-class clean surface, but the cleanup preserves the
already-clean deterministic path instead of rewriting it. The clean
data-generation path records source manifests, protocol identity, field/span
weights, mask behavior, split hashes, and Stage1/Stage2 dataset identities
beside the generated JSONL artifacts. The deterministic Stage1 focus and Stage2
conversation builders are clean-native transforms; heavy teacher trajectory
generation remains an upstream asset until regeneration is intentionally needed.
