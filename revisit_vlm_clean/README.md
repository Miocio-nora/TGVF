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

The entrypoints are stubs in this phase. They parse and validate identity, but
they do not run model training or benchmark inference yet.
