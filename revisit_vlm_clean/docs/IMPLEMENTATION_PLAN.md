# Clean TGVF Implementation Plan

## Phase 0: Archive Gate

Completed before this skeleton was created:

- archive branch: `archive/tgvf-clean-start-20260625`
- archive commit: `03dd657 archive tgvf clean project planning state`
- excluded local assets: `logs/`, `logs/tgvf_v4_teacher_50k.pid`,
  `third_party/VLMEvalKit/`

## Phase 1: Skeleton

Completed in commit `20a598c`.

This phase created:

- top-level clean mini-project `revisit_vlm_clean/`;
- schema/dataclass contracts;
- benchmark population and subset constants;
- manifest interfaces;
- CLI stubs;
- basic unit tests.

It does not port heavy training/evaluation behavior.

## Phase 2: Manifest Generation

Implemented after the skeleton:

- deterministic manifest generator;
- committed `CoreSmoke-256`, `CoreDev-2511`, and `CoreFull-19562` manifests;
- manifest count/hash tests;
- source population count validation against local benchmark files.

Manifest hashes:

```text
core_smoke_256_seed20260625:
  7da4963199c7d75baee224e52049625129a0f9335d156dd84efd652b2df0c036
core_balanced_dev_2511_seed20260625:
  a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579
core_full_19562:
  1b2942590ff4eada644b51507acfde461f6049e87a96486d061d71a3f1de0352
```

## Phase 3: Parser/Scorer Skeleton

Implemented:

- shared strict choice parser;
- open-answer extraction;
- project exact-match scorer;
- explicit `NotImplementedError` for official scorer execution until wrappers
  are ported.

## Phase 4: Output Schema Smoke

Implemented:

- benchmark CLI can write `run_config.json`, `rows.jsonl`, `summary.json`, and
  `sample_manifest.json` without model inference for schema validation.

## Later Phases

1. Full benchmark runner with model inference.
2. Official scorer wrapper port.
3. Stage1/Stage2 launchers.
4. DeepStack training/eval support.
