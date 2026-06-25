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

## Entry Points

```bash
tgvf_build_manifest --list
tgvf_eval_benchmark --help
tgvf_train_stage1 --print-defaults
tgvf_train_stage2 --print-defaults
```

The benchmark entrypoint can now validate identity, build/materialize fixed
manifests, render model input rows for smoke checks, and execute the dry-run or
original-Qwen backend. It does not run TGVF benchmark inference yet.

## Fixed Manifests

Committed benchmark manifests:

```text
benchmark_manifests/core_smoke_256_seed20260625.json
benchmark_manifests/core_balanced_dev_2511_seed20260625.json
benchmark_manifests/core_full_19562.json
```

`CoreDev-2511` is the default fast development comparison subset. Manifest
generation is deterministic and tested by count/hash.

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
