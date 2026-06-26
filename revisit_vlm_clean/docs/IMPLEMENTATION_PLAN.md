# Clean TGVF Implementation Plan

## Phase 0: Archive Gate

Completed before this skeleton was created:

- archive branch: `archive/tgvf-clean-start-20260625`
- archive commit: `03dd657 archive tgvf clean project planning state`
- excluded local assets: `logs/`, `logs/tgvf_v4_teacher_50k.pid`,
  `third_party/VLMEvalKit/`

## Current Boundary: Data Generation

Data generation is treated as clean-native, not as a legacy bridge.

- keep `tgvf_generate_data` in the final clean project;
- preserve the deterministic transforms already ported into
  `revisit_vlm_clean.data_generation`:
  - `v4_to_protocol_c`;
  - `v4_to_stage1_protocol_c_focus`;
  - `choice_to_open_answer`;
  - `clean_imend`;
- require source path/hash, transform, protocol, schema, split/hash, field
  weights, mask policy, and output file identities for every generated split;
- do not rewrite this path as part of the current training/eval cleanup;
- keep heavy teacher trajectory generation as an upstream trace-production path
  until we intentionally regenerate traces.

The active cleanup risk is training/eval execution identity, not deterministic
data conversion.

## Current Boundary: Historical Project Cleanup

Historical project cleanup is not part of the active clean-project goal.

- the root historical implementation may be read as a reference for behavior,
  interfaces, and parity checks;
- deleting, pruning, archiving, or normalizing historical scripts is out of
  scope for this goal;
- clean-native progress is measured by the `revisit_vlm_clean/` entrypoints,
  contracts, tests, and auditable artifacts;
- any future historical-project cleanup must be planned as a separate task with
  its own whitelist and confirmation gate.

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
- explicit scorer-backend separation. At this early phase, unsupported
  `official` scorer execution failed fast until benchmark-specific wrappers
  were ported in later phases.

## Phase 4: Output Schema Smoke

Implemented:

- benchmark CLI can write `run_config.json`, `rows.jsonl`, `summary.json`,
  `sample_manifest.json`, and `benchmark_sources.json` without model inference
  for schema validation.

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

- `tgvf_stage2_qwen3` backend name was initially wired to the historical
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

## Phase 12: Benchmark Identity and Scorer Backend Wiring

Implemented after the multi-row smoke.

- `RunConfig` now records `benchmark_root`, so benchmark source identity is
  visible in both `run_config.json` and `run_config.txt`;
- benchmark CLI exposes `--scoring-backend auto|project|official`;
- explicit `project` and `official` scoring disable fallback in
  `ParserScorerIdentity`, while `auto` keeps fallback enabled;
- runner scoring now uses `config.parser_scorer.scoring_backend` instead of a
  hard-coded project scorer;
- per-row outputs include scorer metadata:
  - `scorer_name`;
  - `official_tool_used`;
  - `official_compatible`.

Historical scorer state at the end of Phase 12:

- `official` scoring was intentionally unavailable in the clean runner and
  failed fast instead of silently falling back. This early state was superseded
  by Phases 13-18, which added official-compatible choice scoring and the
  current batch official scorer wrappers.

## Phase 13: Official-Compatible Choice Scorer Parity

Implemented after Phase 12.

- clean scorer now supports the historical official-compatible multiple-choice
  path for:
  - `blink` with scorer name `official_blink_exact_match`;
  - `hr_bench_4k` with scorer name `official_compatible_hrbench4k_mc`;
- `auto` scoring uses this official-compatible path for those benchmarks;
- explicit `official` scoring uses this path for those benchmarks and still
  fails fast for unsupported benchmarks;
- project scoring remains separate and can differ from official-compatible
  choice parsing, because project scoring allows option-text fallback while the
  historical official-compatible helper primarily extracts letters.

Validation:

- tests dynamically load `src/tgvf_eval/official_tools.py` and compare clean
  `extract_choice_official_compatible` against historical `_extract_choice`;
- tests compare clean BLINK official-compatible score against historical
  `_score_choice`;
- runner-level dry backend test verifies that `benchmark=blink` under `auto`
  records `official_blink_exact_match`.

Historical scorer gap at the end of Phase 13, now superseded by Phases 16-18:

- MMMU-Pro, OCRBench-v2, MathVista, and MathVerse official wrappers had not yet
  been added to the clean runner.

## Phase 14: Larger Path-Backed Stage2 Smoke

Implemented on 2026-06-26.

- committed deterministic manifest
  `benchmark_manifests/diagnostic_vstar_core_smoke_32_20260626.json`;
- manifest rule:
  all 32 VStar samples from committed `CoreSmoke-256`;
- preflight verified:
  - 32 rows materialized;
  - all image paths exist;
  - no `Choices:` duplication in rendered prompts;
  - dry execution `accuracy=1.0`;
- ran `tgvf_stage2_qwen3` on all 32 rows with the same 20260617 row-only
  Stage2 checkpoint used in Phase 10/11.

Smoke result:

```text
output:
  outputs/clean_smokes/stage2_tgvf_force_vstar32_20260626_010550
manifest:
  diagnostic_vstar_core_smoke_32_20260626
manifest_hash:
  d18563b8d2c1392295e80f7e3a8c4453cb7f725aebbba9b714fa58822be8ace7
mode:
  tgvf_force
backend:
  tgvf_stage2_qwen3
n_rows:
  32
accuracy:
  0.4375
trigger_rate:
  1.0
focus_valid_rate:
  1.0
append_success_rate:
  1.0
malformed_rate:
  0.0
```

This validates larger path-backed bridge stability only. It is not a benchmark
comparison. Legacy `target_answer_leakage_flag` fired on 30/32 rows.

## Phase 15: Batch Scoring Hook

Implemented after Phase 14.

- runner now collects model output rows first and then applies a single
  `score_output_rows(...)` pass;
- row-level project and official-compatible choice scoring behavior is preserved;
- error rows are skipped by the scoring pass and keep `score=None`;
- this creates the required integration point for future official scorers that
  need to score a whole row set, such as OCRBench-v2, MMMU-Pro, MathVista, and
  MathVerse.

Validation:

- clean tests cover BLINK auto scoring through the runner;
- clean tests cover batch scoring and error-row skip behavior;
- `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests`
  passed with 44 tests.

## Phase 16: OCRBench-v2 Official Batch Scorer Parity

Implemented after Phase 15.

- clean sample materialization now preserves scorer-critical metadata from
  source records, including `answers`, `type`, `eval`, `precision`, `pid`,
  `query`, and related benchmark identifiers;
- `score_output_rows(...)` now receives `benchmark_root` from the clean runner,
  so batch scorers can resolve local official-code paths without relying on
  global state;
- OCRBench-v2 official scoring is wired through the local official
  `OCRBench_v2/eval_scripts/eval.py::process_predictions` entrypoint when
  `benchmark_root/ocrbench_v2/official_code` exists;
- OCRBench-v2 predictions are first normalized with the same final-answer
  extraction semantics used by the historical wrapper before being passed to
  the official scorer;
- per-row outputs now include `official_tool_path` when an official scorer is
  used.

Validation:

- unit tests preserve OCRBench-v2 metadata through clean materialization;
- scorer tests compare clean OCRBench-v2 behavior against the historical
  `src/tgvf_eval/official_tools.py::OCRBenchV2OfficialScorer` on a fixed fake
  official scorer tree;
- runner-level dry backend test verifies that
  `RunConfig.benchmark_root -> score_output_rows -> official_tool_path` is
  connected;
- explicit `official` OCRBench-v2 scoring fails fast if local official code is
  not present;
- `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests`
  passed with 48 tests.

## Phase 17: MMMU-Pro Official Batch Scorer Parity

Implemented after Phase 16.

- clean official scoring now resolves the local MMMU-Pro official entrypoint at
  `benchmark_root/mmmu_pro/official_code/mmmu-pro/evaluate.py`;
- the batch scorer uses the official helper functions:
  - `get_multi_choice_info`;
  - `parse_multi_choice_response`;
  - `eval_multi_choice`;
- parsing uses deterministic seeding per row before calling the official parser,
  matching the historical wrapper's behavior for ambiguous predictions;
- per-row outputs record `scorer_name=official_mmmu_pro`,
  `official_tool_used=true`, and `official_tool_path`;
- explicit `official` MMMU-Pro scoring fails fast when the local official code
  is missing.

Validation:

- scorer tests compare clean MMMU-Pro behavior against the historical
  `src/tgvf_eval/official_tools.py::MMMUProOfficialScorer` on a fixed fake
  official scorer tree;
- runner-level dry backend test verifies that clean `RunConfig.benchmark_root`
  reaches the MMMU-Pro official scorer path;
- `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests`
  passed with 51 tests.

## Phase 18: MathVista and MathVerse Official Scorer Parity

Implemented after Phase 17.

- clean official scoring now resolves:
  - MathVista:
    `benchmark_root/mathvista/official_code/evaluation/calculate_score.py`;
  - MathVerse:
    `benchmark_root/mathverse/official_code/evaluation/score_answer_s2.py`;
- MathVista is ported for the default disabled-LLM path:
  - extract final answer from model output;
  - normalize multiple-choice, integer, float, list, and text answers using the
    same local rules as the historical wrapper;
  - write per-row `prediction`, `score`, `official_tool_path`, and
    `llm_judge_used=false`;
- MathVerse is ported for the default disabled-LLM path:
  - parse the local multiple-choice/final-answer prediction;
  - score with the same local choice scorer as the historical wrapper;
  - write per-row `official_tool_path` and `llm_judge_used=false`;
- explicit `official` scoring fails fast when the required local official code
  path is missing.

Validation:

- scorer tests compare clean MathVista and MathVerse behavior against the
  historical `src/tgvf_eval/official_tools.py` wrappers with
  `official_llm_mode=disabled`;
- runner-level dry backend tests verify that clean `RunConfig.benchmark_root`
  reaches both official scorer paths;
- `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests`
  passed with 57 tests.

Current scorer status after Phase 18:

- row-level official-compatible scoring is ported for BLINK and HR-Bench-4K;
- batch official scorers are ported for OCRBench-v2, MMMU-Pro, MathVista, and
  MathVerse when their local `official_code` trees are present;
- MathVista and MathVerse are intentionally limited to the disabled-LLM local
  path;
- unsupported explicit `official` scoring still fails fast rather than falling
  back to project scoring.

## Clean-Native Exit Criteria

- `tgvf_stage2_qwen3_legacy` is a diagnostic bridge only. The final clean
  project must replace it with a native clean Stage2 runner before first-class
  benchmark or training claims rely on the clean tree.
- `tgvf_stage2_qwen3` is the generic clean Stage2 backend name and must resolve
  to the native clean backend. Legacy use must request
  `tgvf_stage2_qwen3_legacy` explicitly.
- Data generation is a first-class clean-project surface, but the cleanup should
  preserve and wrap the already-clean deterministic path rather than rewrite it:
  - fixed source manifests and sample ids;
  - Stage1 target/description/D dataset generation;
  - Stage2 protocol conversation generation;
  - field/span weights, mask behavior, `im_end` policy, no-focus rules, and
    split hashes recorded beside generated JSONL artifacts.
- Heavy teacher/focus trajectory generation remains an upstream trace-production
  asset until regeneration is intentionally needed; clean training consumes
  existing generated runs by explicit path/hash identity.
- Stage1/Stage2 launchers must consume those clean generated datasets by
  explicit path/hash identity.

## Phase 19: Data-Generation Identity Layer

Implemented after Phase 18.

- added `tgvf_generate_data` clean CLI;
- added data-generation identity schema for:
  - teacher trajectory generation;
  - Stage1 Protocol-C focus data;
  - Stage2 Protocol-C conversation data;
  - split cleaning;
  - choice-to-open-answer conversion;
- the initial CLI supports `--dry-run` and `--write-plan` only;
- plan files include:
  - `data_generation_config.json`;
  - `data_generation_config.txt`;
  - `input_files.json`;
  - `data_generation_report.json`;
- input files are identified by existence, size, SHA-256, and line count;
- protocol, transform, source manifest path/hash, source run id, split policy,
  field weights, and mask policy are recorded before any heavy generator is
  ported or launched;
- this intentionally does not generate training JSONL yet.

Validation:

- CLI tests cover dry-run planning and written plan artifacts with temporary
  JSONL inputs;
- `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests`
  passed with 59 tests.

## Phase 20: Deterministic Data Transform Execution

Implemented after Phase 19.

- `tgvf_generate_data --execute` now runs two clean-native deterministic
  transforms:
  - `choice_to_open_answer`;
  - `clean_imend`;
- `choice_to_open_answer` ports the historical Stage2 choice-to-open conversion:
  - strips answer choices from the prompt;
  - converts `choices` to an empty list;
  - stores the original choice metadata under `metadata.choice_to_open_answer`;
  - updates `answer`, `short_answer`, `answer_format`, `value_span_text`, and
    focus-step value spans;
- `clean_imend` ports the historical polluted-text filter for Protocol-C split
  rows;
- execution writes transformed JSONL files under the clean output directory,
  while preserving the Phase 19 identity artifacts and adding:
  - `generated_files.json`;
  - `transform_report.json`;
  - `generated_data_written=true` in `data_generation_report.json`;
- unported transforms still fail fast instead of silently dispatching to old
  scripts.

Validation:

- tests execute both transforms on tiny temporary JSONL inputs;
- `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests`
  passed with 61 tests.

## Phase 21: V4 Teacher to Protocol-C Stage2 Builder

Implemented after Phase 20.

- `tgvf_generate_data --execute --transform v4_to_protocol_c` now ports the
  deterministic historical `build_tgvf_v4_stage2_protocol_c.py` behavior;
- supported V4 item types:
  - `single_refocus` -> `single_focus`;
  - `multi_refocus` -> `multi_focus` with exactly two focus steps;
  - `no_refocus_continue` and `no_refocus_answer` -> `direct_answer`;
- output rows preserve the Stage2 compatibility schema:
  - `schema_version=tgvf_teacher_schema_v4_stage2_compat`;
  - image/source/question/choice/answer fields;
  - target, evidence description, target cues, leakage risk, evidence state;
  - pre/post focus think spans where available;
- transform reports include raw/written counts, focus/no-focus counts, and
  focus ratios.

Validation:

- tests execute the transform on tiny `single_refocus` and `no_refocus_answer`
  V4 teacher rows;
- `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests`
  passed with 62 tests.

## Phase 22: V4 Teacher to Stage1 Protocol-C Focus Builder

Implemented after Phase 21.

- `tgvf_generate_data --execute --transform v4_to_stage1_protocol_c_focus`
  now ports the deterministic Stage1 focus-row builder into the clean data
  generation entrypoint;
- supported V4 item types:
  - `single_refocus` -> one Stage1 focus row;
  - `multi_refocus` -> one Stage1 focus row per focus step;
  - `no_refocus_continue` and `no_refocus_answer` -> skipped, because Stage1
    focus training consumes focus rows only;
- output rows preserve the historical Stage1 compatibility contract:
  - `schema_version=tgvf_teacher_schema_v4_stage1_compat`;
  - `need_focus=true`;
  - `evidence_state=need_local_visual_evidence`;
  - `trajectory_type=single_focus`;
  - required loader fields `image`, `question`, `target`,
    `evidence_description`;
  - source trace, answer/choice metadata, focus-step index, target cues,
    leakage risk, evidence type, confidence, and source uid;
- transform reports include source row counts, converted focus rows, skipped
  no-focus rows, converted rows by source item type, and `unique_images` over
  written focus rows.

Validation:

- tests execute the transform on a tiny `multi_refocus` V4 teacher row and
  verify that two Stage1 focus rows are written while a no-focus row is skipped;
- temporary execution against the historical 50k split matched the old Stage1
  split reports:

```text
tgvf_v4_teacher_50k.train.jsonl:
  converted_focus_rows=40021
  unique_images=9561
tgvf_v4_teacher_1k.test.jsonl:
  converted_focus_rows=867
  unique_images=203
```

- full clean test suite passed:
  `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests`
  -> 63 tests.

## Phase 23: Clean Training Launch Plans

Implemented after Phase 22.

- added `training_plan.py` as the shared contract for Stage1/Stage2 training
  launch identity;
- `tgvf_train_stage1` and `tgvf_train_stage2` now support:
  - `--print-defaults`;
  - `--dry-run`;
  - `--write-plan`;
- launch plans bind:
  - exact train/val JSONL and Stage1 checkpoint file identities;
  - model, processor, protocol, dtype, max resolution, and sequence limits;
  - global batch math as
    `world_size * micro_batch_size * gradient_accumulation_steps`;
  - Stage1 clean constraints: `teacher_forced`, `row_only`,
    `native_source_grid`;
  - machine-readable trainable/frozen module policy, including frozen Qwen
    visual merger usage and training `use_cache=false`;
  - Stage2 mask policy, weighted span losses, target focus ratio, and
    DeepStack state;
  - git commit and tracked dirty-worktree state;
  - a temporary historical reference command for auditability;
- the historical reference command is explicitly marked as
  `temporary_legacy_reference_not_final_clean_native`;
- launch plans now also include a machine-readable `clean_native_training`
  status so the final clean-native executor gap is explicit and cannot be
  confused with the historical reference command;
- if Stage2 DeepStack training semantics are enabled, the launcher records the
  intended state but marks the historical command non-executable, because the
  old Stage2 script has no DeepStack training controls.

Artifacts written by `--write-plan`:

```text
training_plan.json
training_plan.txt
dataset_identity.json
clean_native_training_status.json
legacy_reference_command.sh
```

Validation:

- tests cover Stage1 launch-plan writing, automatic accumulation-step
  resolution, Stage2 train/val/checkpoint identity, mask/span-loss command
  mapping, and the DeepStack legacy-command gate;
- targeted CLI tests passed:
  `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests/test_cli.py -q`
  -> 13 tests.
- targeted ruff passed for the new launcher/planning code and touched CLI tests;
- full clean test suite passed:
  `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests`
  -> 66 tests.

## Phase 24: Stage2 Backend Identity Split

Implemented after Phase 23.

- introduced explicit Stage2 backend identities:
  - `tgvf_stage2_qwen3_legacy`: diagnostic bridge to the historical
    `Stage2ProtocolEvaluator`;
  - `tgvf_stage2_qwen3_native`: reserved final clean-native backend name;
  - `tgvf_stage2_qwen3`: deprecated alias resolving to
    `tgvf_stage2_qwen3_legacy`;
- superseded by Phase 42: `tgvf_stage2_qwen3` now resolves to
  `tgvf_stage2_qwen3_native`;
- `BackendConfig.to_dict()` now records:
  - requested backend;
  - resolved backend;
  - whether the requested backend was a deprecated alias;
- executed row outputs now include `resolved_runner_backend` and
  `runner_backend_deprecated_alias`;
- benchmark CLI backend choices now expose both explicit Stage2 names and the
  deprecated compatibility alias;
- native Stage2 backend currently validates runtime identity then fails fast,
  because clean-native capture/append has not yet been ported.

This phase does not claim native Stage2 execution is complete. It prevents the
legacy bridge from looking like the final backend and gives the remaining port a
stable target name.

Validation:

- runner backend tests cover explicit legacy backend construction, deprecated
  alias resolution, `BackendConfig` resolved-backend metadata, and native
  backend fast-fail behavior;
- targeted runner/backend tests passed:
  `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests/test_runner_backend.py revisit_vlm_clean/tests/test_legacy_stage2_adapter.py revisit_vlm_clean/tests/test_stage2_runtime.py revisit_vlm_clean/tests/test_benchmark_data.py -q`
  -> 20 tests;
- targeted ruff passed for runner, benchmark CLI, runner backend tests, and the
  touched benchmark-data test;
- full clean test suite passed:
  `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests`
  -> 68 tests.

## Phase 25: Stage2 Native Engine Boundary

Implemented after Phase 24.

- added `revisit_vlm_clean.stage2_native.NativeStage2Engine` as the owner of
  the final clean-native Stage2 runtime contract;
- `tgvf_stage2_qwen3_native` now prepares through this native engine instead of
  failing immediately in the runner;
- native prepare records:
  - Stage2 runtime config;
  - run mode/forward/protocol/deepstack/parser identity;
  - checkpoint file hash;
  - eval JSONL identity;
  - backend dtype/device options;
- native execution remains explicitly unported. Calling the backend produces a
  structured row error with `native_stage2_execution_ported=false` instead of
  silently falling back to the legacy bridge;
- tests assert that the native module does not import the historical Stage2
  evaluator module.

This phase still does not claim clean-native Stage2 capture, D construction, or
post-TGVF continuation are complete. It creates the place where that code must
land.

Data generation remains a clean first-class asset in this tree. The deterministic
Stage2/Stage1 transforms should be preserved; only heavy teacher trajectory
generation remains to be ported when we need to regenerate teacher traces rather
than consume existing committed/generated runs.

Validation:

- targeted runner/backend tests passed:
  `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests/test_runner_backend.py -q`
  -> 6 tests;
- targeted ruff passed for runner, native Stage2 engine, and runner backend
  tests;
- full clean test suite passed:
  `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests`
  -> 71 tests.

## Phase 26: Stage2 Native Force/Free Control Flow

Implemented after Phase 25.

- expanded `NativeStage2Engine` from identity-only into the owner of the native
  Stage2 execution path;
- added clean `NativeStage2Sample` conversion from benchmark samples/rendered
  prompts, keeping choices in the already-rendered prompt and avoiding duplicate
  choice text;
- ported the native force/free/softforce control flow into the clean tree:
  - force mode captures a forced focus action, builds D, appends visual D, and
    continues naturally after TGVF;
  - free/softforce mode captures the router output, decides whether focus was
    triggered, and either returns direct output or runs the same post-D path;
  - post-D execution selects KV append or full-sequence prefill from the clean
    `ForwardMode`;
- added lazy heavy runtime loading:
  - checkpoint protocol validation;
  - Qwen3-VL + processor loading;
  - protocol-token setup;
  - LoRA load validation;
  - frozen foveal module construction from checkpoint config;
- ported visual-D append and full-sequence prefill logic into the clean engine,
  reusing lower-level Qwen3/TGVF primitives but not the historical
  `Stage2ProtocolEvaluator` class;
- runner errors now report native runtime failures as row errors with native
  identity debug metadata.

This phase does not launch a GPU smoke benchmark and therefore does not prove
the native backend is benchmark-ready. It proves the control-flow ownership has
moved into the clean project.

Validation:

- targeted runner/backend tests passed:
  `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests/test_runner_backend.py -q`
  -> 9 tests;
- targeted ruff passed for runner, native Stage2 engine, and runner backend
  tests;
- full clean test suite passed:
  `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests`
  -> 74 tests.

## Phase 27: Stage2 Native Force GPU Smoke

Implemented after Phase 26.

- registered committed diagnostic VStar manifests as clean subsets so the CLI
  can execute them by subset id:
  - `diagnostic_vstar_first_1_20260626`;
  - `diagnostic_vstar_core_smoke_first_8_20260626`;
  - `diagnostic_vstar_core_smoke_32_20260626`;
- launched one clean-native Qwen3 Stage2 force-path smoke on physical GPU 2:
  - backend: `tgvf_stage2_qwen3_native`;
  - checkpoint: `BASE-20260619-open-answer-rowonly` Stage2 checkpoint;
  - manifest: `diagnostic_vstar_first_1_20260626`;
  - mode: `tgvf_force`;
  - forward mode: `kv_cache`;
  - max image resolution: 512;
  - DeepStack: off;
  - output:
    `outputs/clean_native_smoke/qwen3_stage2_native_vstar1_force_20260626_030329`;
- confirmed the run produced one scored row with no row error:
  - accuracy: 1.0;
  - answer parse rate: 1.0;
  - trigger rate: 1.0;
  - focus valid rate: 1.0;
  - append success rate: 1.0;
  - FVT shape: `[234, 4096]`;
  - append path: `clean_native_qwen3_visual_special_tokens_embedding_replace`.

This is still a runtime smoke, not a benchmark result. It proves the
clean-native forced post-D path can execute a real checkpoint on a real image.
At the end of this phase, free/softforce trigger behavior, no-KV full-sequence
behavior, and equivalence/differences versus the legacy bridge remained future
checks. Later phases supersede this early smoke limitation.

Validation:

- Stage2 runtime identity validation passed:
  - checkpoint global step: 1200;
  - checkpoint protocol: `protocol_c_tool_observation`;
  - Stage2 eval JSONL rows: 1002;
  - need_focus/no_focus: 857/145;
- processor note:
  - checkpoint config points to the 20260619 Stage1 processor;
  - `diff -qr` confirmed that directory is identical to the Stage2
    `processor_step_1200` directory used for the smoke;
- full clean test suite passed before launch:
  `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests`
  -> 75 tests.

## Phase 28: Stage2 Native Free/Softforce No-Trigger Smoke

Implemented after Phase 27.

- launched two additional clean-native Qwen3 Stage2 smokes on the same fixed
  one-row VStar diagnostic manifest:
  - `tgvf_free`;
  - `tgvf_softforce` with prompt text `Use focus tool.`;
- held fixed:
  - checkpoint: `BASE-20260619-open-answer-rowonly` Stage2 checkpoint;
  - manifest: `diagnostic_vstar_first_1_20260626`;
  - forward mode: `kv_cache`;
  - protocol: `protocol_c_tool_observation`;
  - max image resolution: 512;
  - DeepStack: off;
- outputs:
  - `outputs/clean_native_smoke/qwen3_stage2_native_vstar1_free_20260626_031442`;
  - `outputs/clean_native_smoke/qwen3_stage2_native_vstar1_softforce_20260626_031442`;
- confirmed both runs produced one scored row with no row error:
  - free accuracy: 1.0;
  - softforce accuracy: 1.0;
  - answer parse rate: 1.0 for both;
  - trigger rate: 0.0 for both;
  - append success rate: null for both because focus was not triggered.

This phase validates clean-native free/softforce runtime and parser/scorer
output on the no-trigger branch. It does not validate trigger-positive
free/softforce post-D behavior.

## Phase 29: Stage2 Native Trigger-Positive Legacy Comparison

Implemented after Phase 28.

- added a fixed trigger-positive VStar diagnostic manifest:
  - `diagnostic_vstar_softforce_trigger_row10_20260626`;
  - manifest hash:
    `6421ea792f5db73555c895a7b28a4bacfb8eefa6046b1d35080b17b5081cfb44`;
  - sample:
    `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/10_000010`;
  - selection evidence: historical ckpt19 VStar softforce row triggered focus
    with `append_success=true`;
- ran clean-native softforce on the trigger-positive manifest:
  - output:
    `outputs/clean_native_smoke/qwen3_stage2_native_vstar1_trigger_softforce_20260626_032556`;
  - trigger rate: 1.0;
  - focus valid rate: 1.0;
  - append success rate: 1.0;
  - parsed answer: `C`;
  - score: 1.0;
- ran the same manifest through the diagnostic legacy bridge:
  - backend: `tgvf_stage2_qwen3_legacy`;
  - output:
    `outputs/clean_native_smoke/qwen3_stage2_legacy_vstar1_trigger_softforce_20260626_032925`;
  - trigger/focus/append/answer/score matched clean-native;
  - focus target and raw output matched clean-native exactly.

This phase proves the clean-native triggered softforce post-D path aligns with
the historical bridge on one fixed real sample. It is still diagnostic; a small
fixed manifest is the next step before benchmark-scale claims.

## Phase 30: Stage2 Native VStar-8 Legacy Comparison

Implemented after Phase 29.

- ran clean-native and legacy bridge on the same fixed 8-row VStar manifest:
  - manifest: `diagnostic_vstar_core_smoke_first_8_20260626`;
  - manifest hash:
    `3a6020b145c1543be3fc8a5541c17d00b4fc5d27aeddbecb90a10417d25c84f4`;
  - mode: `tgvf_softforce`;
  - prompt text: `Use focus tool.`;
  - forward mode: `kv_cache`;
  - checkpoint: `BASE-20260619-open-answer-rowonly` Stage2 checkpoint;
- outputs:
  - native:
    `outputs/clean_native_smoke/qwen3_stage2_native_vstar8_softforce_20260626_033554`;
  - legacy:
    `outputs/clean_native_smoke/qwen3_stage2_legacy_vstar8_softforce_20260626_033554`;
- aggregate metrics matched exactly:
  - n: 8;
  - accuracy: 0.625;
  - answer parse rate: 1.0;
  - trigger rate: 0.25;
  - focus valid rate: 0.25;
  - append success rate: 1.0 over triggered rows;
  - malformed rate: 0.0;
- per-row outputs matched exactly for:
  - trigger decision;
  - focus validity;
  - append success;
  - parsed answer;
  - score;
  - focus target;
  - raw output.

This phase validates clean-native Stage2 softforce against the legacy bridge on
a small mixed manifest with both triggered and no-trigger rows. It is still not
a benchmark claim. At the end of this phase, no-KV and clean benchmark subset
boundaries remained pending; Phase 31 addresses the fixed-manifest no-KV
diagnostic comparison.

## Phase 31: Stage2 Native No-KV Legacy Comparison

Implemented after Phase 30.

- ran clean-native and legacy bridge on the same fixed 8-row VStar manifest:
  - manifest: `diagnostic_vstar_core_smoke_first_8_20260626`;
  - manifest hash:
    `3a6020b145c1543be3fc8a5541c17d00b4fc5d27aeddbecb90a10417d25c84f4`;
  - mode: `tgvf_softforce`;
  - prompt text: `Use focus tool.`;
  - forward mode: `no_kv_full_sequence`;
  - checkpoint: `BASE-20260619-open-answer-rowonly` Stage2 checkpoint;
- outputs:
  - native:
    `outputs/clean_native_smoke/qwen3_stage2_native_vstar8_softforce_nokv_20260626_034311`;
  - legacy:
    `outputs/clean_native_smoke/qwen3_stage2_legacy_vstar8_softforce_nokv_20260626_034311`;
- aggregate metrics matched exactly:
  - n: 8;
  - accuracy: 0.625;
  - answer parse rate: 1.0;
  - trigger rate: 0.25;
  - focus valid rate: 0.25;
  - append success rate: 1.0 over triggered rows;
  - malformed rate: 0.0;
- per-row outputs matched exactly for:
  - trigger decision;
  - focus validity;
  - append success;
  - parsed answer;
  - score;
  - focus target;
  - raw output.

Compared with the Phase 30 KV comparison, aggregate metrics, parsed answers,
and scores were unchanged. One triggered row changed only explanatory wording.
This validates clean-native no-KV/full-sequence parity with the legacy bridge on
the fixed diagnostic manifest; it is still not a benchmark-scale effect claim.

## Phase 32: Benchmark Manifest Identity Guard

Implemented after Phase 31.

- verified that rebuilding committed clean image-core manifests from the current
  benchmark root reproduces committed counts, stable hashes, and sample ids:
  - `core_smoke_256_seed20260625`: n=256,
    `7da4963199c7d75baee224e52049625129a0f9335d156dd84efd652b2df0c036`;
  - `core_balanced_dev_2511_seed20260625`: n=2511,
    `a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`;
  - `core_full_19562`: n=19562,
    `1b2942590ff4eada644b51507acfde461f6049e87a96486d061d71a3f1de0352`;
- added benchmark CLI manifest identity validation:
  - when `--subset-id` is used with `--manifest-path`, the manifest payload
    `manifest_id` must exactly equal the requested subset id;
  - when `--population-id` is used with `--manifest-path`, the manifest samples
    must contain exactly that one population id;
- updated the manifest CLI help text so `--build` no longer claims to be
  unimplemented.

This prevents a run from recording one clean subset/population name while
actually materializing a different manifest.

## Phase 33: DeepStack Execution and Training Status Guards

Implemented after Phase 32.

- real Stage2 benchmark backends now reject `deepstack.enabled=true` during
  backend prepare:
  - `tgvf_stage2_qwen3_native`;
  - `tgvf_stage2_qwen3_legacy`;
- dry/materialize/render paths may still record DeepStack identity, but real
  model execution cannot silently claim DeepStack behavior until original-image
  DeepStack injection/masking is ported;
- Stage1/Stage2 training plans now write:
  - `clean_native_training` inside `training_plan.json`;
  - `clean_native_training_status.json`;
- the clean-native training status records:
  - final clean-native executor is not implemented yet;
  - legacy reference command is not final;
  - blockers for native dataloader/runtime execution, checkpoint parity, and
    trainable-parameter audit;
  - for Stage2 DeepStack plans, the additional blocker that DeepStack
    original-image injection/masking is specified but not implemented by a clean
    executor.

This phase does not complete DeepStack execution support. It removes a silent
false-positive path and makes the remaining final-clean training gap explicit.

## Phase 34: Row-Level Eval Identity Fields

Implemented after Phase 33.

- clean executed benchmark rows now include per-row identity fields that were
  previously only available in run config or summary:
  - `eval_family`;
  - `tgvf_protocol`;
  - `post_tgvf_continuation`;
  - `post_tgvf_forward_mode`;
  - `deepstack`;
  - `parser_scorer`;
  - `d_condition`;
- this applies to dry-run and real backends through the shared row-construction
  path;
- tests assert these fields on a dry-run executed row.

This closes part of the clean output-schema gap: row-level analysis no longer
has to recover forward mode, parser/scorer, or DeepStack state from side files.

## Phase 35: Run-Level Output Schema Identity

Implemented after Phase 34.

- `RunConfig` now includes:
  - `output_schema_version="clean_benchmark_run_v1"`;
  - `started_at`;
- benchmark CLI fills `started_at` with a UTC ISO timestamp when constructing
  run config;
- `run_config.txt` also prints `output_schema_version` and `started_at`;
- tests assert CLI-produced executed outputs contain these fields.

This closes another clean output-schema gap: new benchmark outputs now carry
the run start time and schema version in the canonical run config.

## Phase 36: Unsharded Run Shard Identity

Implemented after Phase 35.

- `RunConfig` now includes explicit shard identity:
  - `num_shards=1`;
  - `shard_index=0`;
- validation rejects invalid shard identities;
- executed rows include `num_shards` and `shard_index`;
- `run_config.txt` prints both fields.

This does not implement real sharded execution or deterministic merge yet. It
does make the unsharded case explicit in the same schema that future shard
outputs will use.

## Phase 37: Deterministic Benchmark Shard Selection

Implemented after Phase 36.

- benchmark CLI now accepts:
  - `--num-shards`;
  - `--shard-index`;
- materialize/render/execute paths apply deterministic source-manifest-order
  modulo sharding before sample materialization;
- `--manifest-hash` continues to validate the source full manifest before
  sharding;
- shard `sample_manifest.json` records:
  - shard manifest id/hash;
  - source manifest id/hash;
  - `num_shards`;
  - `shard_index`;
  - source and selected sample counts;
  - selection rule `source_manifest_order_modulo`;
- run config and rows carry the shard identity.

This implements deterministic shard selection. Deterministic shard merge is
implemented in the next phase.

## Phase 38: Deterministic Benchmark Shard Merge

Implemented after Phase 37.

- added `benchmark_merge.py` and CLI `tgvf_merge_benchmark`;
- merge input is a set of shard output directories containing:
  - `run_config.json`;
  - `sample_manifest.json`;
  - `rows.jsonl`;
  - `summary.json`;
- merge validates:
  - shard indices cover `0..num_shards-1`;
  - shard run configs agree on `num_shards`;
  - shard manifests agree on source manifest hash;
  - each shard row order matches its shard sample manifest;
  - merged row ids match the reconstructed merged sample manifest;
- merged output writes:
  - `rows.jsonl`;
  - `summary.json`;
  - `run_config.json`;
  - `sample_manifest.json`;
  - `merge_metadata.json`;
- rows and samples are restored using the same source-manifest-order modulo
  rule used by shard selection;
- tests run two dry-run shards and merge them in reversed directory order to
  prove deterministic ordering.

## Phase 39: Data-Generation Output Identity

Implemented after Phase 38.

- data generation remains a clean-native first-class surface;
- `DataGenerationConfig` now records
  `output_schema_version=clean_data_generation_v1`;
- `tgvf_generate_data --execute` now records generated output identities:
  - output path;
  - byte size;
  - SHA-256;
  - line count;
- `data_generation_report.json` now includes:
  - `output_files`;
  - `split_hashes`;
  - generated file count, total generated lines, and total generated bytes;
- `data_generation_config.txt` is rewritten after execution so it no longer
  reports `identity_only=true` for a run that actually wrote transformed data;
- transform semantics are unchanged. This phase only strengthens reproducible
  generated-data identity.

## Phase 40: Clean Package Lint Gate

Implemented after Phase 39.

- normalized import order, typing imports, and line wrapping across
  `revisit_vlm_clean/src/revisit_vlm_clean` and `revisit_vlm_clean/tests`;
- no benchmark, training, scoring, data-generation, or Stage2 control-flow
  semantics were intentionally changed;
- `ruff check revisit_vlm_clean/src/revisit_vlm_clean revisit_vlm_clean/tests`
  now passes;
- full clean tests still pass.

## Phase 41: Training Plan Clean-Command Separation

Implemented after Phase 40.

- training plans now carry
  `training_plan_schema_version=clean_training_plan_v1`;
- Stage1 and Stage2 plan outputs now write both:
  - `clean_training_command.sh`;
  - `legacy_reference_command.sh`;
- `clean_training_command.sh` is the intended final clean-native entrypoint
  shape, but it is commented and marked not executable until the native
  Stage1/Stage2 training executors are ported;
- legacy command payloads now record `final_clean_native=false`;
- clean command payloads record `final_clean_native=true`,
  `planned_entrypoint`, `status=clean_native_executor_not_ported`, and a
  clear unavailable reason;
- this does not launch training. It removes ambiguity: historical scripts are
  preserved only as references, not as the clean mainline.

## Phase 42: Stage2 Generic Backend Resolves to Native

Implemented after Phase 41.

- `tgvf_stage2_qwen3` now resolves to `tgvf_stage2_qwen3_native`;
- `tgvf_stage2_qwen3_legacy` remains available only as an explicit diagnostic
  bridge name;
- backend identity output now records:
  - `stage2_generic_alias`;
  - `alias_target`;
  - `deprecated_alias=false`;
- tests assert the generic backend constructs the clean-native backend rather
  than the historical bridge.

## Phase 43: Clean Training Executor Preflight Entrypoints

Implemented after Phase 42.

- added importable clean training modules:
  - `revisit_vlm_clean.training.stage1_executor`;
  - `revisit_vlm_clean.training.stage2_executor`;
- both modules accept `--plan training_plan.json`;
- both modules support `--preflight-only` to validate:
  - training plan schema and stage;
  - batch math;
  - dataset/checkpoint file identities;
  - module policy;
  - clean command identity;
  - legacy command is not final clean-native;
- without `--preflight-only`, the modules fail fast and do not launch training;
- this makes the planned clean entrypoints real and testable while preserving
  the rule that incomplete native training must not silently fall back to old
  scripts.

## Phase 44: Training Preflight Report Artifacts

Implemented after Phase 43.

- added package scripts:
  - `tgvf_train_stage1_executor`;
  - `tgvf_train_stage2_executor`;
- executor preflight now writes a JSON report:
  - default path:
    `<plan-dir>/<stage>_training_preflight_report.json`;
  - override path: `--preflight-report`;
- tests assert report creation for Stage1 and custom report path behavior for
  Stage2;
- this keeps training launch fail-safe while making executor validation
  auditable as a file artifact, not just terminal output.

## Phase 45: Legacy Stage2 Backend Diagnostic Gate

Implemented after Phase 44.

- benchmark CLI now exposes explicit `--eval-family`;
- `tgvf_stage2_qwen3_legacy` can only be constructed when
  `eval_family=internal_diagnostic`;
- clean project-native external benchmark runs must use
  `tgvf_stage2_qwen3` or `tgvf_stage2_qwen3_native`;
- row/run identity still records requested and resolved backend names;
- tests assert:
  - default benchmark dry-run remains `project_native_external`;
  - explicit diagnostic eval family is accepted;
  - legacy Stage2 backend rejects non-diagnostic configs;
  - legacy Stage2 backend remains available for explicit diagnostic configs.

## Phase 46: ValKit Eval-Family Separation

Implemented after Phase 45.

- `tgvf_eval_benchmark` now accepts only:
  - `project_native_external`;
  - `internal_diagnostic`;
- `valkit` remains in the shared schema as a preserved eval family, but it is
  rejected by the project-native benchmark CLI;
- tests assert that `--eval-family valkit` fails on `tgvf_eval_benchmark`;
- this prevents ValKit results from being silently produced through the
  project-native runner with incompatible sample/scoring semantics.

## Phase 47: ValKit Preflight Surface

Implemented after Phase 46.

- added `tgvf_eval_valkit` as the clean ValKit/VLMEvalKit entrypoint;
- added `valkit_plan_schema_version=clean_valkit_plan_v1`;
- ValKit preflight records:
  - checkpoint identity;
  - optional Stage2 checkpoint identity for TGVF modes;
  - benchmark names;
  - mode, protocol, forward mode, max image resolution;
  - ValKit root/work-dir identity fields;
  - runner status and `legacy_shell_wrapper_allowed=false`;
- `--write-plan` and `--preflight-only` write:
  - `valkit_plan.json`;
  - `valkit_plan.txt`;
  - `valkit_preflight_report.json`;
- ValKit execution is not launched by preflight/write-plan. The clean surface
  requires explicit `--execute` plus `--valkit-root` and `--valkit-model-name`
  before it can call ValKit.

## Phase 48: Clean Parser/Scorer Identity Finalization

Implemented after Phase 47.

- the clean external benchmark parser/scorer source is now explicitly
  `revisit_vlm_clean.scoring`, not the historical `eval/` script;
- default parser identities are:
  - `model_output_parser=revisit_vlm_clean.scoring.parse_and_score:v3_external`;
  - `choice_parser=revisit_vlm_clean.scoring.extract_choice_strict`;
- `run_config.json`, row `parser_scorer`, and `summary.json` preserve these
  identities so future benchmark tables cannot silently mix old parser labels
  with clean rows;
- the historical `eval/eval_v3_mmmu_force.py` parser/scorer path remains a
  reference for the port, not the clean implementation source.

## Phase 49: Non-Executable Legacy Training References

Implemented after Phase 48.

- Stage1/Stage2 launch plans still preserve historical training command text as
  an audit reference;
- `legacy_reference_command.executable=false` is now required for clean plans;
- generated `legacy_reference_command.sh` comments the historical command out
  with a clear `not executable` reason;
- executor preflight rejects any plan that marks the legacy reference command
  executable;
- the final training path remains the clean-native executor command/status
  surface. Full clean-native training execution is still a later phase.

## Phase 50: Stage1/Stage2 Clean Launcher Default Alignment

Implemented after Phase 49.

- Stage1 clean launcher defaults now match the confirmed mainline:
  - `focus_action_im_end=true`;
  - `lr_scheduler=cosine`;
  - `warmup_steps=100`;
  - `min_lr_ratio=0.1`;
- Stage1 legacy reference command text records those clean defaults instead of
  inherited raw-script defaults;
- Stage2 clean launcher now exposes and defaults `warmup_steps=100`, while still
  recording `warmup_ratio=0.03` as an explicit secondary scheduler field;
- Stage2 legacy reference command passes `--warmup-steps 100`, so the effective
  warmup is not silently derived as `ceil(max_steps * warmup_ratio)`;
- tests assert the default print surface, launch-plan optimizer identity, and
  generated command text.

## Phase 51: Stage2 LoRA and Optimizer Identity

Implemented after Phase 50.

- Stage2 launch plans now record the clean LoRA config explicitly:
  - target modules:
    `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`;
  - rank `64`, alpha `256`, dropout `0.05`, bias `none`;
- Stage2 optimizer identity now records:
  - optimizer name `adamw`;
  - Adam betas `[0.9, 0.95]`;
  - epsilon `1e-8`;
  - weight decay `0.01`;
  - max grad norm `1.0`;
- Stage2 legacy reference command text passes these values explicitly instead
  of relying on raw-script defaults;
- Stage1 optimizer identity now records `max_grad_norm=1.0` and passes it in
  the reference command.

## Phase 52: Stage1 Readout Context Identity

Implemented after Phase 51.

- Stage1 launch plans now include `readout_context` as a machine-readable field;
- the readout context records:
  - original image placeholder embeddings are replaced with Qwen `V_merge`;
  - D is appended as a native Qwen visual span;
  - position ids are real Qwen3 M-RoPE over the full trajectory;
  - `fvt_position_mode=native_source_grid`;
  - D token count is dynamic/source-image visual token count;
  - visual merger path remains frozen finalize;
  - Stage1 readout attention mask uses weak-strict original-image-key blocking
    after the TGVF append point;
- executor preflight rejects Stage1 plans missing these readout-context
  identities.

## Phase 53: Clean Training Execution Bundle

Implemented after Phase 52.

- Stage1/Stage2 clean executors now support `--prepare-execution`;
- this writes executor-owned artifacts:
  - `clean_training_execution_bundle.json`;
  - `clean_training_execution_status.json`;
  - `clean_training_execution_bundle.txt`;
- the bundle records:
  - plan path and SHA-256 identity;
  - stage, run id, output dir, git identity;
  - model/protocol/dataset/batch/training/module/loss/optimizer identity;
  - Stage1 readout context;
  - Stage2 mask policy, DeepStack state, and LoRA identity;
  - clean executor handoff status;
  - `legacy_reference_allowed=false`;
- `--prepare-execution` still records `will_launch_training=false` and
  `trainer_loop_ported=false`. It is the stable clean handoff surface for the
  future trainer loop, not a fake training launch.

## Phase 54: Structured DeepStack Execution Plan

Implemented after Phase 53.

- Stage2 benchmark backends still reject `deepstack.enabled=true` for real
  execution;
- the rejection now includes a structured `deepstack_execution_plan` instead of
  only a generic error string;
- the plan records:
  - backend;
  - `execution_supported=false`;
  - original-image DeepStack injection source;
  - scope-specific masking semantics;
  - `through_answer` means no answer restoration;
  - `evidence_only` means restore original-image DeepStack for answer;
  - D DeepStack-like features remain disabled and not required for the current
    mainline;
  - the remaining blocking items before real DeepStack execution can be enabled.

## Phase 55: Prepare-Execution Command Handoff

Implemented after Phase 54.

- Stage1/Stage2 launch plans now emit `clean_prepare_execution_command`;
- generated plans write `clean_prepare_execution_command.sh`;
- this command is executable in the sense that it is not commented out and calls:
  `python -m revisit_vlm_clean.training.<stage>_executor --plan ... --prepare-execution`;
- `clean_native_training.status` is now
  `handoff_supported_trainer_loop_not_ported`;
- `clean_native_training.prepare_execution_supported=true` records that the
  clean executor handoff is implemented;
- `clean_training_command.sh` remains commented and non-executable because the
  trainer loop still has not been ported.

## Phase 56: ValKit Prepare-Execution Handoff

Implemented after Phase 55.

- `tgvf_eval_valkit` now supports `--prepare-execution`;
- `write_valkit_plan` emits `valkit_prepare_execution_command.sh`;
- `--prepare-execution` writes:
  - `valkit_execution_bundle.json`;
  - `valkit_execution_status.json`;
  - `valkit_execution_bundle.txt`;
- the bundle records plan SHA-256, checkpoint identity, benchmark names, mode,
  protocol, forward mode, ValKit root/work-dir identity, and runner handoff
  state;
- the handoff still records `will_launch_valkit=false`,
  `valkit_runtime_ported=false`, and `legacy_shell_wrapper_allowed=false`.
  Real ValKit execution is performed only by explicit `--execute`.

## Phase 63: Clean ValKit Runtime Execution

Implemented after Phase 62.

- `tgvf_eval_valkit` now supports explicit `--execute`;
- execution requires:
  - `--valkit-root`, pointing at a directory containing `run.py`;
  - `--valkit-model-name`, the model key registered in VLMEvalKit config;
- the clean runner builds a direct ValKit command:
  `python <valkit-root>/run.py --data ... --model ... --work-dir ... --mode ...`;
- the command is owned by the clean runner and does not call
  `scripts/run_vlmevalkit_tgvf.sh` or other historical shell wrappers;
- execution writes:
  - `valkit_launch_command.sh`;
  - `valkit_execution_result.json`;
  - `valkit_stdout.log`;
  - `valkit_stderr.log`;
  - updated `valkit_execution_bundle.json`;
  - updated `valkit_execution_status.json`;
- tests execute a fake ValKit `run.py` to prove the clean subprocess path works
  without launching a heavyweight benchmark;
- ValKit results remain a separate `eval_family=valkit` surface and are not
  silently comparable with project-native benchmark rows.

## Phase 64: DeepStack Execution Evidence in Benchmark Rows

Implemented after Phase 63.

- clean benchmark rows now include a first-class `deepstack_execution` field
  separate from the raw `debug_metadata` blob;
- each row records:
  - requested DeepStack state/scope;
  - whether that requested state is executable by the current clean runner;
  - runner backend and resolved backend;
  - FVT append path and FVT position mode;
  - whether FVT append actually used DeepStack visual features;
  - any DeepStack caution emitted by the Stage2 native append path;
- clean Stage2 native append debug now promotes `uses_deepstack_for_fvt` and
  `deepstack_caution` from append metadata into the row debug payload;
- executed `summary.json` and merged shard `summary.json` now include
  `deepstack_execution` aggregation, so benchmark tables cannot silently mix
  DeepStack-enabled and no-DeepStack FVT execution;
- this phase does not implement DeepStack feature injection. Current clean
  Stage2 FVT append still reports `uses_deepstack_for_fvt=false`; requests with
  `deepstack.enabled=true` remain rejected before model execution.

## Phase 57: Lazy Legacy Stage2 Bridge Isolation

Implemented after Phase 56.

- the clean benchmark runner no longer imports `legacy_stage2_adapter` at module
  import time;
- the historical Stage2 bridge is loaded only inside the explicit
  `tgvf_stage2_qwen3_legacy` diagnostic backend;
- generic `tgvf_stage2_qwen3` and explicit `tgvf_stage2_qwen3_native` continue
  to resolve to the clean-native engine;
- tests guard against reintroducing a top-level legacy adapter import in the
  runner.

## Phase 58: Trainer Runtime Contract in Execution Bundle

Implemented after Phase 57.

- Stage1/Stage2 `--prepare-execution` bundles now include
  `trainer_runtime_contract`;
- the contract records the future clean executor launch function, launch
  permission state, required launch gates, required runtime artifacts, and
  before-first-optimizer-step audit requirements;
- Stage1 gates include Qwen `V_merge` readout context, real Qwen3 M-RoPE
  position ids, and matrix-CE/manifold loss parity;
- Stage2 gates include Stage1 checkpoint restoration, LoRA attachment,
  fast-batched Stage2, weighted-span losses, original-image mask scope, and
  DeepStack scope when enabled;
- execution status now reports `trainer_runtime_contract_status=not_ported`.

## Phase 59: Dataset Runtime Artifacts for Training Handoff

Implemented after Phase 58.

- Stage1/Stage2 `--prepare-execution` now scans the training JSONL instead of
  only copying file identities from the plan;
- prepare-execution writes:
  - `dataset_runtime_identity.json`;
  - `first_batch_identity.json`;
- `dataset_runtime_identity.json` records required field checks, line count,
  focus/no-focus counts when present, answer-format counts, and source-dataset
  counts;
- `first_batch_identity.json` records the requested global batch size,
  materialized first-batch size, stable row digests, and row-level identity
  fields;
- malformed JSONL, empty training data, or missing required clean training
  fields now fail during prepare-execution before any future trainer loop can
  launch.

## Phase 60: Stage2 Checkpoint Contract Validation

Implemented after Phase 59.

- Stage1/Stage2 `--prepare-execution` now writes `checkpoint_contract.json`;
- Stage1 records that no input checkpoint is required and lists required output
  checkpoint keys for the future clean trainer;
- Stage2 prepare-execution now actually `torch.load`s the Stage1 checkpoint
  before handoff;
- Stage2 rejects checkpoints that are not loadable mappings, are missing a
  non-empty `tgvf_module`, have a protocol mismatch, or are missing Protocol-C
  token rows for Protocol-C training;
- the contract records checkpoint keys, global step, selected config identity,
  TGVF state-dict summary, and Protocol-C token-row summary;
- execution status now reports `checkpoint_contract_status=validated` for a
  valid Stage2 handoff.

## Phase 61: Optimizer Group Runtime Artifact

Implemented after Phase 60.

- Stage1/Stage2 `--prepare-execution` now writes `optimizer_groups.json`;
- Stage1 records the clean optimizer groups corresponding to the historical
  Stage1 script:
  - `tgvf_module`;
  - `protocol_c_token_rows`;
- Stage1 also records that AdamW weight decay is inherited from the historical
  `torch.optim.AdamW` default of `0.01`;
- Stage2 records the three historical fast Stage2 optimizer groups:
  - `llm_lora` with `lr_lora`;
  - `tgvf_refiner` with `lr_tgvf`;
  - `fvt_calibration` with `lr_calibration`;
- the artifact records optimizer name, betas, epsilon, weight decay, max grad
  norm, scheduler identity, warmup, and min-LR ratio;
- execution status now reports `optimizer_groups_status=validated`.

## Phase 62: Training Runtime Audit Gate

Implemented after Phase 61.

- Stage1/Stage2 executors now support `--audit-runtime`;
- the audit validates `clean_training_execution_bundle.json` plus the prepared
  runtime artifacts:
  - `dataset_runtime_identity.json`;
  - `first_batch_identity.json`;
  - `checkpoint_contract.json`;
  - `optimizer_groups.json`;
- the audit writes:
  - `clean_training_runtime_audit.json`;
  - `clean_training_runtime_audit_status.json`;
  - `clean_training_runtime_audit.txt`;
  - `trainable_parameters.json`;
- `trainable_parameters.json` is deliberately a blocking placeholder with
  `status=pending_model_load_not_actual_parameter_audit`; it records expected
  trainable/frozen module policy but must be overwritten by the real trainer
  after model load and before the first optimizer step;
- launch gates are split into identity-validated gates and
  `pending_real_trainer_loop` gates, so checkpoint-save, model-load, protocol
  rows, Stage1 readout execution, Stage2 LoRA attachment, weighted losses, mask
  scope, and DeepStack semantics are not falsely reported as implemented;
- the runtime audit still records `will_launch_training=false` and
  `training_runtime_ported=false`.

## Phase 65: Actual Model Parameter Audit Gate

Implemented after Phase 64.

- Stage1/Stage2 executors now support explicit `--audit-model-parameters`
  together with `--audit-runtime`;
- default runtime audit remains lightweight and writes
  `status=pending_model_load_not_actual_parameter_audit`;
- with `--audit-model-parameters`, the executor loads the planned training
  model components and writes `trainable_parameters.json` with:
  - `status=actual_model_parameter_audit`;
  - trainable and frozen parameter names;
  - per-module trainable/frozen tensor counts;
  - trainable/frozen numel totals;
  - loader identity and expected trainable/frozen module policy;
- the Stage1 loader uses the clean Stage1 plan plus Qwen/TGVF primitives to
  freeze Qwen, enable protocol token-row training, infer dimensions, and build
  the TGVF module;
- the Stage2 loader uses the clean Stage2 plan plus the Stage1 checkpoint to
  attach LoRA, restore Protocol-C token rows, load Stage1 TGVF weights, and
  build the TGVF module for audit;
- runtime launch gates now mark model-load, protocol-token, cache-disable, and
  trainable-parameter audit gates as `identity_validated` only when an actual
  parameter audit was written;
- this phase still does not run optimizer steps, weighted losses, checkpoint
  save/load parity, or DeepStack training execution. `will_launch_training`
  remains `false`.

## Phase 66: Actual Optimizer/Scheduler Audit Gate

Implemented after Phase 65.

- Stage1/Stage2 executors now support explicit `--audit-optimizer` together
  with `--audit-runtime`;
- `--audit-optimizer` loads the planned model components when needed, reuses the
  same loaded modules for the trainable-parameter audit, and writes
  `optimizer_runtime.json`;
- `optimizer_runtime.json` records:
  - `status=actual_optimizer_scheduler_audit`;
  - planned optimizer group names from `optimizer_groups.json`;
  - constructed optimizer group names;
  - empty planned groups;
  - per-group parameter names, tensor counts, numel, LR, and weight decay;
  - AdamW betas/epsilon/weight decay and LambdaLR scheduler identity;
- `optimizer_groups.json` remains only a plan contract. Runtime launch gates now
  mark `construct_optimizer_and_scheduler_from_plan` as `identity_validated`
  only when `optimizer_runtime.json` proves actual construction;
- this phase still does not run backward, optimizer steps, scheduler steps,
  weighted losses, checkpoint save/load parity, or DeepStack training execution.
  `will_launch_training` remains `false`.

## Phase 67: Actual Checkpoint Save/Load Audit Gate

Implemented after Phase 66.

- Stage1/Stage2 executors now support explicit `--audit-checkpoint` together
  with `--audit-runtime`;
- `--audit-checkpoint` loads model components when needed and implicitly
  constructs optimizer/scheduler runtime state, because the clean checkpoint
  schema must include optimizer and scheduler state dicts;
- the audit writes:
  - `checkpoint_runtime.json`;
  - local probe file `checkpoint_runtime_probe.pt`;
- Stage1 checkpoint probes include:
  - `tgvf_module`;
  - `config`;
  - `global_step`;
  - `optimizer_step`;
  - `optimizer`;
  - `scheduler`;
  - `protocol_c_token_rows` when Protocol-C rows are required;
- Stage2 checkpoint probes include:
  - `qwen_lora`;
  - `tgvf_module`;
  - `config`;
  - `global_step`;
  - `micro_step`;
  - `optimizer`;
  - `scheduler`;
- runtime launch gates now mark `save_checkpoint_with_clean_contract` as
  `identity_validated` only when the probe is saved, loaded, required keys are
  present, state-dict key/shape parity passes, and optimizer/scheduler states
  can be reloaded;
- this phase still does not run backward, optimizer steps, scheduler steps,
  weighted losses, or DeepStack training execution. `will_launch_training`
  remains `false`.

## Phase 68: Stage1/Stage2 No-Backward Training-Step Audit Gate

Implemented after Phase 67.

- Stage1/Stage2 executors now support explicit `--audit-training-step` together
  with `--audit-runtime`;
- the audit loads model components when needed and writes
  `training_step_runtime.json`;
- Stage1 real path runs a forward-only probe through `v3_stage1_training_step`
  using:
  - the first usable Stage1 train samples, preferring same-image groups when
    available for matrix-CE coverage;
  - the clean Stage1 loss config;
  - the clean Stage1 `native_source_grid` position mode;
  - the clean Stage1 original-image mask behavior;
  - the clean Stage1 Protocol-C identity;
- Stage2 real path runs a forward-only probe through
  `v3_stage2_batched_training_step` using:
  - the first usable Stage2 train samples;
  - the clean Stage2 weighted-span loss config;
  - the clean Stage2 `native_source_grid` position mode;
  - the clean Stage2 original-image mask policy;
  - the clean Stage2 Protocol-C identity;
- runtime launch gates now mark the following Stage2 gates as
  `identity_validated` only when the probe payload proves them:
  - `use_fast_batched_stage2_path`;
  - `apply_weighted_span_losses_from_plan`;
  - `apply_original_image_mask_scope_from_plan`;
- runtime launch gates now mark the following Stage1 gates as
  `identity_validated` only when the probe payload proves them:
  - `stage1_readout_context_uses_qwen_v_merge`;
  - `stage1_position_ids_use_real_qwen3_mrope`;
  - `stage1_matrix_ce_and_manifold_losses_match_plan`;
- this phase still does not run backward, optimizer steps, scheduler steps,
  publish training checkpoints, or implement DeepStack training execution.
  `will_launch_training` remains `false`.

## Phase 69: Bounded Optimizer-Step Runtime Audit Gate

Implemented after Phase 68.

- Stage1/Stage2 executors now support explicit `--audit-optimizer-step`
  together with `--audit-runtime`;
- `--audit-optimizer-step` implies:
  - actual model-parameter audit;
  - actual optimizer/scheduler construction;
  - actual training-step forward probe;
- the audit writes `optimizer_step_runtime.json` after one bounded probe that:
  - starts from the clean training-step loss tensor;
  - calls `backward`;
  - clips gradients with the planned `max_grad_norm` when present;
  - calls `optimizer.step`;
  - calls `scheduler.step`;
  - calls post-step `zero_grad`;
- runtime launch gates now mark
  `run_backward_optimizer_scheduler_step_from_plan` as `identity_validated`
  only when the probe proves backward, optimizer step, and scheduler step ran
  without publishing a checkpoint or entering a training loop;
- this phase still does not enter an epoch loop, perform gradient accumulation,
  publish training checkpoints, or implement DeepStack training execution.
  `will_launch_training` remains `false`.

## Phase 70: Bounded Gradient-Accumulation Trainer-Loop Audit Gate

Implemented after Phase 69.

- Stage1/Stage2 executors now support explicit `--audit-trainer-loop` together
  with `--audit-runtime`;
- `--audit-trainer-loop` implies:
  - actual model-parameter audit;
  - actual optimizer/scheduler construction;
  - actual training-step forward probe;
- the audit writes `trainer_loop_runtime.json` after one bounded
  gradient-accumulation probe that:
  - uses the planned `gradient_accumulation_steps`;
  - reruns the clean training-step forward for each micro-step;
  - scales each micro-step loss before `backward`;
  - accumulates gradients across micro-steps;
  - clips gradients with the planned `max_grad_norm` when present;
  - calls one `optimizer.step`;
  - calls one `scheduler.step`;
  - calls post-step `zero_grad`;
- runtime launch gates now mark
  `run_gradient_accumulation_loop_from_plan` as `identity_validated` only when
  the probe proves the planned number of micro-steps, backward calls, and one
  optimizer/scheduler step without publishing a checkpoint or launching a full
  training run;
- a passing trainer-loop audit also satisfies
  `run_backward_optimizer_scheduler_step_from_plan`, because it contains that
  ordered step inside the accumulation loop;
- this phase still does not enter the full epoch loop, publish training
  checkpoints, enable `clean_training_command.sh`, or implement DeepStack
  training execution. `will_launch_training` remains `false`.

## Phase 71: Post-Loop Checkpoint Publish Runtime Audit Gate

Implemented after Phase 70.

- Stage1/Stage2 executors now support explicit `--audit-checkpoint-publish`
  together with `--audit-runtime`;
- `--audit-checkpoint-publish` implies:
  - actual model-parameter audit;
  - actual optimizer/scheduler construction;
  - actual training-step forward probe;
  - bounded gradient-accumulation trainer-loop probe;
- the audit writes:
  - `training_checkpoint_publish_runtime.json`;
  - local probe file `training_checkpoint_publish_probe_step_1.pt`;
- the checkpoint publish probe is saved after the bounded trainer-loop update,
  then reloaded and validated for:
  - required Stage1/Stage2 checkpoint keys;
  - TGVF and Qwen-LoRA state-dict key/shape parity;
  - optimizer and scheduler state reload;
  - historical step counters:
    - Stage1: `global_step=1`, `optimizer_step=1`;
    - Stage2: `global_step=1`,
      `micro_step=gradient_accumulation_steps`;
  - Protocol-C token rows when Stage1 requires them;
- runtime launch gates now mark
  `publish_training_checkpoint_after_trainer_loop` as `identity_validated`
  only when the post-loop checkpoint publish probe is saved, loaded, state
  checked, and step checked without entering the full training run;
- a passing checkpoint-publish audit also satisfies
  `save_checkpoint_with_clean_contract`, because it is stronger than the static
  checkpoint save/load audit;
- this phase still does not enable `clean_training_command.sh`, run the full
  max-step training loop, implement resume, or implement DeepStack training
  execution. `will_launch_training` remains `false`.

## Phase 72: Clean Checkpoint Resume Runtime Audit Gate

Implemented after Phase 71.

- Stage1/Stage2 executors now support explicit `--audit-checkpoint-resume`
  together with `--audit-runtime`;
- `--audit-checkpoint-resume` implies:
  - actual model-parameter audit;
  - actual optimizer/scheduler construction;
  - actual training-step forward probe;
  - bounded trainer-loop probe;
  - post-loop checkpoint publish probe;
- the audit writes `training_checkpoint_resume_runtime.json` after:
  - loading a fresh model/module stack through the clean loader;
  - constructing a fresh AdamW/LambdaLR stack from the clean optimizer plan;
  - loading TGVF and Stage2 Qwen-LoRA state from the published checkpoint;
  - restoring Stage1 Protocol-C token rows when present and required;
  - loading optimizer and scheduler state dicts;
  - validating resumed model-state parity and historical step counters;
- runtime launch gates now mark `resume_training_from_clean_checkpoint` as
  `identity_validated` only when the fresh stack successfully restores model,
  optimizer, scheduler, protocol-row, and step-counter state without entering
  the full training run;
- this phase still does not enable `clean_training_command.sh`, continue past
  the resume probe, run the full max-step loop, or implement DeepStack training
  execution. `will_launch_training` remains `false`.

## Phase 73: Trainer Save/Eval Cadence Runtime Audit Gate

Implemented after Phase 72.

- Stage1/Stage2 executors now support explicit `--audit-cadence` together with
  `--audit-runtime`;
- the audit writes `training_cadence_runtime.json` after resolving from the
  clean plan:
  - `max_steps`;
  - checkpoint save cadence from `save_every`;
  - final checkpoint save at `max_steps`;
  - Stage2 eval cadence from `eval_every` only when a validation file is
    present;
  - final Stage2 eval at `max_steps` when eval is enabled;
- runtime launch gates now mark `validate_training_cadence_from_plan` as
  `identity_validated` only when checkpoint cadence ends at `max_steps` and,
  when enabled, eval cadence also ends at `max_steps`;
- this phase does not load the model, enter the training loop, write model
  checkpoints, enable `clean_training_command.sh`, or implement DeepStack
  training execution. `will_launch_training` remains `false`.

## Phase 74: Clean Training Launch Readiness Audit

Implemented after Phase 73.

- Stage1/Stage2 executors now support explicit `--audit-launch-readiness`
  together with `--audit-runtime`;
- this flag implies the strongest non-launch runtime probes that are needed for
  readiness:
  - checkpoint publish/resume audit;
  - optimizer/scheduler construction;
  - bounded training-step and gradient-accumulation trainer-loop probes;
  - save/eval cadence audit;
- the audit writes `training_launch_readiness.json`, summarizing the existing
  runtime launch gates rather than maintaining a second gate implementation:
  - total required gates;
  - identity-validated gates;
  - pending gates;
  - unknown gates;
  - expected non-launch blocker;
  - unexpected blockers;
  - artifact statuses;
  - DeepStack enabled/scope/gate status;
- Stage2 default `deepstack.enabled=false` now satisfies
  `apply_deepstack_training_scope_when_enabled`; an explicitly enabled
  DeepStack plan must still wait for the actual training injection/masking path;
- when every required gate is identity-validated, readiness reports either
  `status=launch_contract_ready_trainer_loop_disabled` for blocked runtimes or
  `status=launch_contract_ready_explicit_launch_required` when the clean
  single-process launch path is available;
- this phase does not enter the full `max_steps` loop, enable
  `clean_training_command.sh`, or launch training. `will_launch_training`
  remains `false`.

## Phase 75: Clean Single-Process Training Launch

Implemented after Phase 74.

- Stage1/Stage2 executors now support explicit `--launch-training`;
- launch first runs the normal clean prepare-execution path, then starts the
  clean single-process trainer loop from the execution bundle;
- launch writes:
  - `single_process_training_runtime.json`;
  - `clean_training_launch_result.json`;
  - `clean_training_launch_status.json`;
  - `checkpoint_step_N.pt` checkpoints according to `save_every` and final
    `max_steps`;
- the launch loop reuses the clean executor primitives already covered by the
  runtime audits:
  - model/module loader;
  - trainable/frozen parameter audit;
  - optimizer/scheduler construction;
  - training-step forward;
  - gradient accumulation;
  - gradient clipping;
  - optimizer/scheduler step;
  - clean checkpoint save/load validation;
- `clean_training_command.sh` is executable only when the plan is currently
  launchable by the clean executor:
  - `world_size=1`;
  - Stage2 DeepStack training disabled;
- Stage2 `val_file` is supported in the single-process launch path:
  - validation runs with no backward pass;
  - validation follows `eval_every` plus final `max_steps` cadence;
  - validation output is recorded in `validation_records`;
- plans that require DDP/multi-process training or Stage2 DeepStack training
  remain valid for prepare/audit but their
  `clean_training_command.sh` is non-executable and records the blocking
  reason;
- this phase does not port DDP, distributed sampler/state, or DeepStack
  original-image injection/masking during training.

## Phase 76: Clean Single-Process Dataset Cursor

Implemented after Phase 75.

- single-process launch now reads train and validation data through deterministic
  cursors instead of repeatedly replaying the fixed audit probe samples;
- Stage1 launch uses same-image groups when available so matrix-CE-compatible
  batches can contain related examples;
- Stage2 train launch uses a deterministic `target_focus_ratio` cursor when
  both focus and no-focus rows exist, preserving the clean focus/no-focus
  sampling contract in the real loop;
- Stage2 in-training validation uses a separate sequential validation cursor,
  so repeated validation steps advance through the validation file instead of
  reusing the first row;
- `single_process_training_runtime.json` now records `train_cursor`,
  `validation_cursor`, and per-micro-step/per-validation `sample_trace`
  evidence with sample indices and hashed image/question identities;
- audit-only probes retain their fixed probe-batch behavior unless launch passes
  explicit cursor samples.

## Phase 77: Clean Torchrun Distributed Training Launch

Implemented after Phase 76.

- Stage1/Stage2 launch plans now treat `world_size>1` as a supported clean
  `distributed_torchrun` runtime when Stage2 DeepStack training is disabled;
- `clean_training_command.sh` is executable for distributed plans and uses
  `torchrun --nproc-per-node <world_size> -m revisit_vlm_clean.training...`;
- direct `python -m ... --launch-training` for a distributed plan fails before
  model load unless torchrun provides matching `WORLD_SIZE`;
- distributed launch initializes a torch distributed process group, sets the
  local CUDA device when available, and injects the local-rank `device_map`
  into the clean model loader;
- each rank writes its own runtime evidence, while rank 0 owns
  `clean_training_launch_result.json`, `clean_training_launch_status.json`,
  checkpoint publication, and Stage2 validation records;
- train cursors are rank-sharded and preserve original sample indices in
  `sample_trace`;
- gradients in optimizer param groups are all-reduced and averaged before
  gradient clipping, matching the legacy Stage2 manual averaging behavior while
  keeping checkpoint state dicts unwrapped;
- Stage2 DeepStack training remains blocked until original-image DeepStack
  injection/masking is ported. This historical blocker is superseded by
  Phase 101.

## Phase 78: Structured DeepStack Training Contract

Implemented after Phase 77.

- Stage2 launch plans now include a structured `deepstack_training_plan`;
- prepare-execution writes `deepstack_training_plan.json` as a required Stage2
  runtime artifact;
- disabled DeepStack is recorded as a validated no-op and does not block clean
  launch;
- enabled DeepStack records:
  - requested DeepStack scope;
  - original-image DeepStack blocking/restoration semantics for
    `through_answer` and `evidence_only`;
  - the fact that D remains a v-merge-level visual-token span rather than a
    DeepStack-like feature path;
  - current unimplemented training blockers for native Qwen3 original-image
    DeepStack injection and post-D scope masking/restoration;
- clean launch rejects enabled Stage2 DeepStack from the prepared artifact
  blockers, not from an unstructured status string;
- runtime audit and launch-readiness include the same structured DeepStack
  blocker evidence;
- launch-readiness now requires every required gate to be
  `identity_validated`; blocked gates cannot accidentally count as ready.

This phase does not implement actual Stage2 DeepStack feature injection or
masking. It makes the unsupported state explicit, machine-readable, and
preserved across plan, prepare, runtime audit, and launch.

## Phase 79: Benchmark Source Manifest Artifact

Implemented after Phase 78.

- each clean benchmark output mode now writes `benchmark_sources.json`;
- `benchmark_sources.json` records schema version, benchmark root,
  manifest/source-manifest hashes, every source file used by the run, resolved
  path, byte size, SHA-256, existence status, benchmark ids, population ids,
  sample counts, and row-index evidence;
- `run_config.json` and `summary.json` include a compact
  `benchmark_source_manifest` reference with artifact path, sample/source-file
  counts, all-files-exist status, manifest hashes, and a hash of the source-file
  list;
- shard merge now requires each shard's `benchmark_sources.json`, validates
  source-file identity consistency, and writes a merged
  `benchmark_sources.json`;
- this phase does not change parser/scorer behavior, model execution, or sample
  selection semantics.

## Phase 80: Benchmark Summary Breakdowns

Implemented after Phase 79.

- executed benchmark outputs and merged shard outputs now include
  `result_breakdowns` in `summary.json`;
- `result_breakdowns` records:
  - metrics by benchmark;
  - metrics by population id;
  - metrics by method/mode;
  - metrics by D condition;
  - prediction and gold counts for choice rows;
  - official-scorer status counts and scorer-name counts;
- the same helper is used by normal execution and shard merge, so merged tables
  keep the same summary contract as single-shard outputs;
- this phase does not change row scoring, parser behavior, sample selection, or
  model execution.

## Phase 81: Benchmark Manifest Verification Summary

Implemented after Phase 80.

- executed benchmark outputs and merged shard outputs now include
  `manifest_verification` in `summary.json`;
- `manifest_verification` records manifest id/hash, source-manifest id/hash,
  row count, sample-manifest count, row/sample id order match, missing/extra row
  sample ids, source-manifest sample-count match, source-file existence status,
  shard identity, and merge metadata when present;
- normal execution and shard merge use the same helper, so merged summaries
  prove row-count/order restoration rather than relying on side artifacts only;
- this phase does not change row scoring, parser behavior, sample selection, or
  model execution.

## Phase 82: Benchmark Comparability Flags

Implemented after Phase 81.

- executed benchmark outputs and merged shard outputs now include
  `comparability` in `summary.json`;
- `comparability` records clean-core population identity, subset/full/population
  scope, expected-n matches where known, diagnostic-subset status, side-result
  and invalid-for-baseline flags, merge status, and a stable comparison scope;
- `side_result` and `invalid_for_baseline` default to `false` until ledger or
  baseline-specific launch logic marks them otherwise, so the summary schema is
  ready for comparable reruns without relying on prose;
- this phase does not change row scoring, parser behavior, sample selection, or
  model execution.

## Phase 83: Benchmark Row Contract Fields

Implemented after Phase 82.

- clean benchmark rows now explicitly include `final_output`,
  `trigger_policy`, `d_shape`, and `continuation_metadata`;
- `trigger_policy` records original/free/force/softforce routing semantics in a
  structured field instead of requiring downstream code to infer from mode;
- `continuation_metadata` records continuation mode, forward mode, Stage2 append
  forward mode when present, backend identity, token/time metadata, append
  status, and full-sequence continuation flags when available;
- `d_shape` is normalized from clean/native or legacy debug metadata when D/FVT
  was appended, and is `null` for direct/no-D rows;
- this phase does not change row scoring, parser behavior, sample selection, or
  model execution.

## Phase 84: Merged Summary Identity Parity

Implemented after Phase 83.

- merged shard `summary.json` now preserves the same core identity fields as
  single-run executed summaries:
  - parser/scorer identity;
  - DeepStack state;
  - post-TGVF forward mode;
  - post-TGVF continuation mode;
  - eval family;
  - TGVF protocol;
  - runner backend summary;
- merge validates that rows agree on the single-valued identity fields instead
  of silently emitting a mixed-identity merged table;
- runner backend identity is summarized by backend/resolved-backend counts and
  alias/deprecated-alias row counts;
- this phase does not change row scoring, parser behavior, sample selection, or
  model execution.

## Phase 85: Run Config Execution Backend Identity

Implemented after Phase 84.

- executed benchmark `run_config.json` now records `execution_backend`, including
  runner backend, resolved backend, dtype, device, device map, attention
  implementation, trust-remote-code flag, alias status, and Stage2 runtime
  identity when present;
- merged benchmark `run_config.json` now records a merged execution-backend
  summary with source shard backend hashes/counts and runner backend counts;
- `run_config.txt` mirrors `execution_backend` for readability, but
  `run_config.json` remains canonical;
- this phase does not change row scoring, parser behavior, sample selection, or
  model execution.

## Phase 86: Clean Training Launch Result Identity

Implemented after Phase 85.

- clean training launch results now record plan identity, git/worktree identity,
  dataset train/validation identity, first-batch digest, input checkpoint
  contract, final checkpoint identity, and runtime artifact file identities;
- clean training launch status mirrors the compact identity fields needed to
  identify a produced checkpoint without opening the full launch result;
- this phase does not change training data order, losses, optimizer behavior,
  checkpoint contents, validation cadence, or model execution.

## Phase 87: ValKit Execution Result Identity

Implemented after Phase 86.

- explicit ValKit `--execute` results now record plan hash, git/worktree
  identity, checkpoint identities, benchmark/mode/protocol identity, ValKit
  root/run.py identity, work-dir identity, launch-command identity, and
  stdout/stderr file identities;
- `valkit_execution_status.json` mirrors the execution identity and result/log
  identities so a ValKit run can be audited without relying on prose;
- this phase does not change the ValKit command argv, benchmark selection,
  scoring behavior, or subprocess execution semantics.

## Phase 88: Shared DeepStack Scope Contract

Implemented after Phase 87.

- added a shared `clean_deepstack_scope_contract_v1` helper used by both clean
  benchmark DeepStack execution plans and Stage2 training DeepStack plans;
- `through_answer` and `evidence_only` now share one machine-readable mapping
  for original-image DeepStack blocking, answer restoration, D feature policy,
  and the v-merge-level FVT path;
- Stage2 training preflight validates that `deepstack_training_plan` carries a
  matching scope contract, so train/eval DeepStack semantics cannot silently
  diverge;
- this phase does not enable DeepStack injection yet. The remaining blocker is
  the actual Qwen3 original-image DeepStack feature injection plus scope-mask
  execution path.

## Phase 89: Disabled DeepStack No-Op Plan Semantics

Implemented after Phase 88.

- `build_deepstack_execution_plan(...)` now distinguishes the default disabled
  DeepStack state from enabled-but-unimplemented DeepStack execution:
  - disabled DeepStack records `execution_supported=true`,
    `status=disabled_noop`, and no blockers;
  - enabled DeepStack still records `execution_supported=false`,
    `status=not_ported`, and the original-image injection/masking blockers;
- this keeps benchmark/training DeepStack identity from treating the default
  no-op state as an execution blocker;
- this phase does not enable original-image DeepStack injection or D
  DeepStack-like features.

## Phase 90: Merged Benchmark Run Config Text Identity

Implemented after Phase 89.

- deterministic shard merge now writes `run_config.txt` beside the merged
  `run_config.json`;
- the text artifact includes the merged run id, manifest hash, source manifest
  reference, parser/scorer identity, DeepStack state, and merged execution
  backend summary;
- merge outputs now return the `run_config_txt` path so callers can verify the
  same launch-gate identity artifact shape used by executed benchmark runs;
- this phase does not change shard validation, row ordering, scoring, or
  summary metrics.

## Phase 91: Benchmark Shard Run-Config Identity Gate

Implemented after Phase 90.

- deterministic shard merge now validates semantic run-config identity across
  all shards before rows are interleaved;
- shard-specific fields such as `run_id`, `started_at`, `shard_index`, and each
  shard manifest hash may differ, but benchmark semantics must match:
  checkpoint, model/processor, eval family, mode, protocol, continuation,
  forward mode, parser/scorer, DeepStack state, max resolution/token limits,
  prompt suffixes, benchmark root, git state, and backend configuration;
- backend `device` and `device_map` are ignored for this semantic comparison so
  equivalent shards can run on different GPUs;
- merge fails fast with the mismatched field name instead of producing a table
  that silently mixes incompatible shard settings;
- this phase does not change source-manifest order restoration, row scoring, or
  summary metric computation.

## Phase 92: Benchmark Shard Row Identity Gate

Implemented after Phase 91.

- deterministic shard merge now validates every row against its shard
  `run_config.json` and sample-manifest row before any merged table is written;
- row fields must match the authoritative shard identity for benchmark,
  population, source file, shard index, eval family, mode, protocol,
  continuation, forward mode, parser/scorer, DeepStack, runner backend, and
  Stage2 `d_condition`;
- this prevents a manually edited or partially stale `rows.jsonl` from being
  merged into a clean benchmark result whose `run_config.json` says something
  different;
- mismatch errors report the sample id and exact row field;
- this phase does not change row ordering, scoring, summary aggregation, or
  source-manifest identity.

## Phase 93: Benchmark Shard Nested Row Identity Gate

Implemented after Phase 92.

- deterministic shard merge now validates the row-level nested identity blocks
  that are used in benchmark tables and mechanism analysis:
  - `trigger_policy`;
  - `continuation_metadata`;
  - `deepstack_execution`;
- only identity fields are checked. Runtime result fields such as token counts,
  wall time, trigger decisions, append success, FVT path, and DeepStack caution
  remain per-row observations and are not forced to fixed values;
- this catches stale or manually edited row metadata even when the top-level row
  fields still match the shard `run_config.json`;
- this phase does not change model execution, scoring, row ordering, or summary
  metric computation.

## Phase 94: Merged Benchmark Shard Artifact Identity

Implemented after Phase 93.

- deterministic shard merge now requires each input shard to contain
  `run_config.txt` in addition to the JSON artifacts;
- `merge_metadata.json` records per-shard artifact identities for:
  - `run_config.json`;
  - `run_config.txt`;
  - `rows.jsonl`;
  - `summary.json`;
  - `sample_manifest.json`;
  - `benchmark_sources.json`;
- each artifact identity includes path, existence, byte size, SHA-256, and line
  count, so a merged benchmark result can be traced back to the exact shard
  files that were merged;
- this phase does not change shard validation semantics, row ordering, scoring,
  or metric aggregation.

## Phase 95: README Runtime Status Alignment

Implemented after Phase 94.

- README executable-backend status now reflects the current clean-native Stage2
  diagnostic evidence:
  - fixed VStar-8 parity with the legacy bridge has been recorded for both
    `kv_cache` and `no_kv_full_sequence`;
  - this remains diagnostic parity, not a benchmark-scale effect claim;
- README benchmark merge description now records the current merge identity
  gates and input shard artifact hashes;
- this phase changes documentation only. It does not add runtime results, change
  model execution, alter scoring, or change merge semantics.

## Phase 96: ValKit Plan Artifact Identity

Implemented after Phase 95.

- ValKit prepare-execution and explicit execution bundles now record identities
  for the concrete plan artifacts written beside the run:
  - `valkit_plan.json`;
  - `valkit_plan.txt`;
  - `valkit_preflight_report.json`;
  - `valkit_prepare_execution_command.sh`;
- `valkit_execution_status.json` mirrors those identities so an execution
  directory can be audited back to the exact handoff plan files, not only to the
  in-memory `plan_sha256`;
- tests assert that both handoff-only and fake-runtime `--execute` paths expose
  the plan artifact identities;
- this phase does not change the ValKit command argv, benchmark selection,
  scoring behavior, subprocess execution, or data generation.

## Phase 97: Benchmark Backend Role Identity

Implemented after Phase 96.

- benchmark backend identity now records whether the selected backend is a
  final clean backend or only a diagnostic bridge:
  - `final_clean_backend`;
  - `diagnostic_bridge`;
  - `requires_internal_diagnostic_family`;
- row outputs mirror the same role with `runner_backend_final_clean` and
  `runner_backend_diagnostic_bridge`;
- shard merge validates these row-level role fields against each shard
  `run_config.json` and reports aggregate `final_clean_backend_rows` and
  `diagnostic_bridge_rows` in merged runner-backend summaries;
- this makes the historical Stage2 compatibility bridge auditable in output
  artifacts without changing backend selection, model execution, scoring, or
  benchmark sampling.

## Phase 98: Training Plan Artifact Identity

Implemented after Phase 97.

- clean training execution bundles now record identities for the concrete
  plan-side artifacts written by `tgvf_train_stage1/2 --write-plan`:
  - `training_plan.json`;
  - `training_plan.txt`;
  - `dataset_identity.json`;
  - `clean_native_training_status.json`;
  - `clean_prepare_execution_command.sh`;
  - `clean_training_command.sh`;
  - `legacy_reference_command.sh`;
- `clean_training_execution_status.json` mirrors those identities, and explicit
  clean launch result/status artifacts preserve them after the training loop;
- this keeps plan handoff identity separate from runtime artifact identity:
  plan artifacts prove what was intended to launch, runtime artifacts prove what
  the executor audited or produced;
- this phase does not change training data order, losses, optimizer behavior,
  checkpoint contents, DeepStack execution, or launch semantics.

## Phase 99: DeepStack Runtime Hook Contract

Implemented after Phase 98.

- added shared `clean_deepstack_runtime_hooks_v1` metadata under the existing
  DeepStack scope contract;
- the hook contract breaks enabled DeepStack support into concrete runtime
  pieces:
  - `capture_original_image_deepstack_features`;
  - `carry_original_image_deepstack_through_post_tgvf_append`;
  - `apply_post_tgvf_deepstack_scope_mask`;
  - `restore_deepstack_for_answer_when_scope_requires`;
- benchmark DeepStack execution plans and Stage2 training plans now use the
  same hook contract as their blocker source, so train/eval cannot silently
  disagree about what remains unported;
- Stage2 training preflight validates the hook contract schema, surface,
  enabled state, scope, and blocker list before accepting a plan;
- this phase does not enable DeepStack execution yet. It turns the remaining
  blocker into a hook-level checklist for the next implementation slices.

## Phase 100: DeepStack Eval Runtime Slice

Implemented after Phase 99.

- added clean Qwen3 DeepStack runtime primitives:
  - capture native original-image `deepstack_features` from
    `get_image_features`;
  - build `visual_pos_masks` over original-image tokens only;
  - keep D as v-merge-level visual tokens, without D DeepStack-like features;
  - build 4D original-image key-block masks for full-sequence prefill and
    cached one-token continuation;
- clean-native Stage2 full-sequence through-answer append can now inject
  original-image DeepStack features into Qwen3's text model by calling the
  native `language_model` surface with `visual_pos_masks` and
  `deepstack_visual_embeds`;
- the corresponding benchmark execution plan reports
  `supported_full_sequence_through_answer` only for:
  - backend `tgvf_stage2_qwen3_native`;
  - `post_tgvf_forward_mode=no_kv_full_sequence`;
  - `deepstack.original_image_scope=through_answer`;
- legacy bridge execution, cache-continuation DeepStack execution, and
  `evidence_only` answer-stage restoration still failed fast at this phase
  rather than claiming unsupported behavior.

## Phase 101: DeepStack Stage2 Training Runtime Slice

Implemented after Phase 100.

- clean Stage2 training plans no longer block `deepstack.enabled=true` when the
  requested scope is represented by the clean Qwen3 training runtime hooks;
- `deepstack_scope_contract` and `deepstack_training_plan` now share the same
  hook implementation evidence, so the top-level plan and nested scope contract
  cannot disagree about whether capture/carry/mask/restore hooks are ported;
- the clean Stage2 executor passes the plan's DeepStack state into the fast
  Stage2 training step probe and launch path;
- the Stage2 focus training runtime now:
  - captures native Qwen3 original-image `deepstack_features` from the same
    image feature pass used to attach visual embeddings;
  - splits those features per sample and carries them through focus-first and
    focus-final manual `inputs_embeds` forwards;
  - calls the native Qwen3 `language_model` surface with `visual_pos_masks` and
    `deepstack_visual_embeds` when DeepStack is enabled;
  - applies original-image DeepStack only to original image token positions,
    not to D/FVT tokens;
  - relies on the existing post-TGVF attention-mask scope so `through_answer`
    blocks original-image DeepStack through answer and `evidence_only` restores
    it for answer tokens;
- D remains a v-merge-level visual-token span. D DeepStack-like features remain
  disabled unless introduced as a named ablation;
- the historical Stage2 reference command remains non-executable for DeepStack
  plans because it has no equivalent DeepStack controls. This is a diagnostic
  reference boundary, not a final clean-native path.

## Phase 102: DeepStack Eval Evidence-Only Answer Restore

Implemented after Phase 101.

- clean-native Qwen3 Stage2 benchmark eval now supports enabled DeepStack for:
  - backend `tgvf_stage2_qwen3_native`;
  - `post_tgvf_forward_mode=no_kv_full_sequence`;
  - `deepstack.original_image_scope=through_answer` or `evidence_only`;
- `build_deepstack_execution_plan` now uses one hook contract for both the
  top-level plan and nested scope contract, so supported eval DeepStack states
  cannot report ported and blocked simultaneously;
- `evidence_only` uses the same full-sequence prefill as `through_answer`:
  native Qwen3 original-image DeepStack features are injected through
  `visual_pos_masks` and `deepstack_visual_embeds`, and the prefill blocks
  original-image keys over the post-TGVF append/evidence prefix;
- continuation remains token-by-token. For `evidence_only`, each generated
  token uses the blocked original-image key mask until the protocol-specific
  answer boundary is complete:
  - `</think>` for thinking/tool-observation protocols;
  - `<|evidence_end|>` for evidence-tag protocols;
  - `<ANSWER>` for legacy tags;
- after that boundary, answer tokens use normal 2D attention, restoring access
  to original-image DeepStack-injected keys;
- D remains a v-merge-level visual-token span. D DeepStack-like features remain
  disabled unless introduced as a named ablation;
- legacy bridge DeepStack and cache-continuation DeepStack remain rejected
  before rows. They are not final clean-native paths.

## Phase 103: Launch Readiness Contract Alignment

Implemented after Phase 102.

- `training_launch_readiness.json` now distinguishes two facts that were
  previously conflated:
  - the readiness audit artifact never launches training and always records
    `will_launch_training=false`;
  - if every required launch gate is identity-validated and the clean trainer
    loop is ported, the artifact records `launch_permitted=true`;
- ready clean plans now report:
  - `status=launch_contract_ready_explicit_launch_required`;
  - `launch_permitted=true`;
  - `launch_disabled_reason=null`;
- blocked plans still report `launch_permitted=false` with a concrete disabled
  reason;
- the executor help text now describes the current clean launch surface:
  single-process and torchrun-distributed launches are selected from the plan's
  batch identity, and Stage2 validation is supported when `val_file` exists;
- invoking the executor without an explicit mode now reports that an execution
  mode is required, rather than saying clean-native training is unimplemented.

## Later Phases

1. Keep data generation first-class:
   - deterministic Stage1/Stage2 transforms stay in `tgvf_generate_data` and
     are preserved rather than rewritten in the current execution cleanup;
   - heavy teacher trajectory generation is ported only when regeneration is
     needed, with source manifest/hash and prompt/schema identity recorded;
   - this boundary was re-confirmed on 2026-06-26: data generation is currently
     considered clean enough, so pruning work should focus on training/eval
     execution surfaces unless a concrete data-transform bug is found.
