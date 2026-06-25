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

## Phase 5: Manifest-Backed Sample Materialization

Implemented in commit `f2f52f2`.

- clean runner can materialize fixed manifest rows back into benchmark samples;
- source identity comes from `source_file + metadata.row_index`, not tier/limit;
- materialized smoke output writes `materialized_rows.jsonl` without image bytes;
- embedded-image datasets keep payload metadata separate from prompt/render rows.

Smoke command:

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.benchmark \
  --run-id materialize_smoke \
  --checkpoint-path outputs/checkpoint.pt \
  --mode tgvf_force \
  --post-tgvf-forward-mode kv_cache \
  --manifest-path revisit_vlm_clean/benchmark_manifests/core_smoke_256_seed20260625.json \
  --manifest-hash 7da4963199c7d75baee224e52049625129a0f9335d156dd84efd652b2df0c036 \
  --subset-id core_smoke_256_seed20260625 \
  --benchmark-root /home/dredvpn009/Flash_Storage/datasets/benchmarks \
  --output-dir /tmp/tgvf_clean_materialize_smoke \
  --materialize-samples
```

Validated on 2026-06-25: 256 rows, expected benchmark distribution, no zero-media
samples.

## Phase 6: Rendered Input Smoke

Implemented after sample materialization.

- clean render rows are written to `rendered_inputs.jsonl`;
- `original` and `tgvf_free` do not add extra prompt text;
- `tgvf_force` keeps the user prompt unchanged and records a protocol control
  prefix;
- `tgvf_softforce` appends only the configured short prompt text;
- current supported protocols are `protocol_c_tool_observation` and
  `protocol_c_tool_observation_qwen2_no_think`.

Validated on 2026-06-25 with `CoreSmoke-256`:

- `tgvf_free`: 256/256 require TGVF controller, no force prefix, no softforce text;
- `tgvf_force`: 256/256 have force control prefix, prompt unchanged;
- `tgvf_softforce`: 256/256 append `Use focus tool.`, no force prefix.

## Phase 7: Executable Runner Skeleton

Implemented in commit `f240a7f`.

- benchmark CLI supports `--execute`;
- `dry_run` backend executes the full manifest -> materialize -> render ->
  parse/score -> rows/summary path without model loading;
- `qwen3_original` backend loads a Qwen2-VL/Qwen3-VL model and runs only
  `mode=original`;
- TGVF modes are intentionally not silently downgraded to direct generation.

Validated on 2026-06-25:

```text
CoreSmoke-256 dry_run:
  n_rows=256
  n_scored=256
  accuracy=1.0
  answer_parse_rate=1.0
  malformed_rate=0.0

1-row Qwen3-VL-2B original smoke:
  model=/nvmesv/dredvpn009/models/hf/Qwen3-VL-2B-Thinking
  sample=vstar first CoreSmoke row
  accuracy=1.0
  output_tokens=97
  wall_time_sec=117.0
```

For future real-model smoke, pass a small `--max-answer-tokens` explicitly.

## Phase 8: Stage2 Runtime Identity Gate

Implemented in commit `844c54e`.

- clean protocol parser supports:
  - `protocol_c_tool_observation`;
  - `protocol_c_tool_observation_qwen2_no_think`;
- Stage2 runtime config validates:
  - Stage2 checkpoint path;
  - Stage2 eval JSONL path;
  - protocol;
  - `d_condition=correct_D`;
  - `force_prefix_mode=target_hint`;
- benchmark CLI supports `--validate-stage2-runtime` without loading the model
  or running inference.

Validated on 2026-06-25 with:

```text
checkpoint:
  outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/checkpoint_step_1200.pt
  global_step=1200
  protocol=protocol_c_tool_observation
  processor_id=outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617/train/processor_step_2000

eval_jsonl:
  data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl
  n_rows=1002
  need_focus=857
  no_focus=145
```

## Phase 9: Stage2 Backend Legacy Bridge

Implemented in commit `cbacf10`.

- `tgvf_stage2_qwen3` backend is wired to the historical
  `Stage2ProtocolEvaluator`;
- clean runner still owns manifest/sample/render/rows/summary identity;
- legacy bridge maps clean modes as:
  - `tgvf_force` -> `force_end2end`;
  - `tgvf_free` and `tgvf_softforce` -> free-router style execution;
- legacy bridge maps forward modes as:
  - `kv_cache` -> legacy `append_prefill_mode=kv_cache`;
  - `no_kv_full_sequence` -> legacy `append_prefill_mode=full_sequence`;
- only `correct_D`, `target_hint`, `native_source_grid`, path-backed images are
  currently allowed.

Validated with real one-row TGVF execution in Phase 10.

Important follow-up from validation:

- the bridge must not pass clean `choices` into legacy `prompt_question`, because
  clean benchmark questions are already rendered with answer choices and the
  legacy sample object otherwise appends a second `Choices:` block;
- answer-choice metadata is preserved under `metadata.choices` while clean
  runner scoring still uses the original clean `BenchmarkSample.choices`.

## Phase 10: Real One-Row Stage2 Smoke

Implemented on 2026-06-26.

- added `run_config.txt` beside JSON output files so launch identity is readable
  without ad hoc parsing;
- benchmark CLI now records git commit and tracked dirty-worktree state in
  `RunConfig`;
- benchmark CLI exposes `--tgvf-protocol` explicitly;
- committed deterministic manifest
  `benchmark_manifests/diagnostic_vstar_first_1_20260626.json`;
- ran `tgvf_stage2_qwen3` on one path-backed VStar row with the historical
  20260617 row-only Stage2 checkpoint.

Smoke result:

```text
output:
  outputs/clean_smokes/stage2_tgvf_force_vstar1_20260626_002759
manifest:
  diagnostic_vstar_first_1_20260626
manifest_hash:
  851e301ea0730ee90086c83565135fa5c5fcc5d33b962fd074444d85241d6995
mode:
  tgvf_force
backend:
  tgvf_stage2_qwen3
n_rows:
  1
accuracy:
  1.0
trigger_rate:
  1.0
focus_valid_rate:
  1.0
append_success_rate:
  1.0
malformed_rate:
  0.0
```

This validates executable wiring only. It is not a benchmark comparison or
evidence about focus-target quality.

## Phase 11: Real Multi-Row Stage2 Smoke

Implemented on 2026-06-26.

- committed deterministic manifest
  `benchmark_manifests/diagnostic_vstar_core_smoke_first_8_20260626.json`;
- manifest rule:
  first eight VStar samples from committed `CoreSmoke-256`;
- verified materialization/render/dry execution before launch:
  - 8 rows materialized;
  - all image paths exist;
  - force control prefix is stable;
  - no `Choices:` duplication in rendered prompts;
- ran `tgvf_stage2_qwen3` on all 8 rows with the same 20260617 row-only Stage2
  checkpoint used in Phase 10.

Smoke result:

```text
output:
  outputs/clean_smokes/stage2_tgvf_force_vstar8_20260626_004135
manifest:
  diagnostic_vstar_core_smoke_first_8_20260626
manifest_hash:
  3a6020b145c1543be3fc8a5541c17d00b4fc5d27aeddbecb90a10417d25c84f4
mode:
  tgvf_force
backend:
  tgvf_stage2_qwen3
n_rows:
  8
accuracy:
  0.625
trigger_rate:
  1.0
focus_valid_rate:
  1.0
append_success_rate:
  1.0
malformed_rate:
  0.0
```

This validates multi-row bridge stability. It is still not a benchmark
comparison. The legacy debug field `target_answer_leakage_flag` fired on all 8
rows, so these rows should not be used for focus-target quality claims.

## Later Phases

1. Official scorer wrapper parity against historical project/official scoring.
2. Larger clean path-backed subset smoke before CoreDev-scale runs.
3. Stage1/Stage2 launchers.
4. DeepStack training/eval support.
