# Clean TGVF Implementation Plan

## Phase 0: Archive Gate

Completed before this skeleton was created:

- archive branch: `archive/tgvf-clean-start-20260625`
- archive commit: `03dd657 archive tgvf clean project planning state`
- excluded local assets: `logs/`, `logs/tgvf_v4_teacher_50k.pid`,
  `third_party/VLMEvalKit/`

## Phase 1: Skeleton

This phase creates:

- top-level clean mini-project `revisit_vlm_clean/`;
- schema/dataclass contracts;
- benchmark population and subset constants;
- manifest interfaces;
- CLI stubs;
- basic unit tests.

It does not port heavy training/evaluation behavior.

## Later Phases

1. Manifest generation for full populations and `CoreDev-2511`.
2. Shared parser/scorer module ported from the stable V3 external path.
3. Clean benchmark runner.
4. Stage1/Stage2 launchers.
5. DeepStack training/eval support.
