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
- Main parser/scorer identity: `v3_external_parse_and_score`.
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
tgvf_train_stage1 --print-defaults
tgvf_train_stage2 --print-defaults
tgvf_train_stage1_executor --help
tgvf_train_stage2_executor --help
```

The benchmark entrypoint can now validate identity, build/materialize fixed
manifests, render model input rows for smoke checks, and execute the dry-run or
original-Qwen backend. It can also execute the diagnostic Qwen3 Stage2 TGVF
bridge backend for path-backed samples; full benchmark claims still require
explicit manifest and ledger identity.

Scoring is applied after all rows are produced. This is intentional: some
official scorers, including OCRBench-v2, need batch-level prediction files
rather than one isolated row at a time.

The data-generation entrypoint records source files, hashes, protocol,
transform, field weights, mask policy, and output intent. It can currently
execute the deterministic `v4_to_protocol_c`,
`v4_to_stage1_protocol_c_focus`, `choice_to_open_answer`, and `clean_imend`
transforms. Executed transforms write output file identities and split hashes
beside the generated JSONL. This path is treated as a clean first-class asset
rather than a legacy bridge. Teacher trajectory generation still remains
outside the clean tree.

The training entrypoints currently produce auditable launch plans with
dataset/checkpoint hashes, batch math, mask policy, weighted losses, DeepStack
state, a clean-native executor status, and a separate temporary legacy
reference command. They also write a commented `clean_training_command.sh`
showing the intended final clean entrypoint, but that command is explicitly
not executable until the native training executors are ported. They do not
start training jobs yet.

The planned clean training modules are importable:

```bash
python -m revisit_vlm_clean.training.stage1_executor --plan /path/to/training_plan.json --preflight-only
python -m revisit_vlm_clean.training.stage2_executor --plan /path/to/training_plan.json --preflight-only
tgvf_train_stage1_executor --plan /path/to/training_plan.json --preflight-only
tgvf_train_stage2_executor --plan /path/to/training_plan.json --preflight-only
```

Without `--preflight-only`, these executors fail fast instead of launching a
partial or legacy training path. Preflight writes a JSON report next to the
plan by default, or to `--preflight-report` when that path is provided.

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
  image samples. It preserves clean manifest/render/output identity while the
  native clean TGVF runner is still being ported.
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

Data generation is expected to enter the clean tree as a first-class pipeline,
not as an opaque historical script call. The clean data-generation path should
record source manifests, protocol identity, field/span weights, mask behavior,
split hashes, and Stage1/Stage2 dataset identities beside the generated JSONL
artifacts. The deterministic Stage1 focus and Stage2 conversation builders are
now clean-native transforms; heavy teacher trajectory generation still remains
outside the clean tree.
