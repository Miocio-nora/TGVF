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

Current limitation:

- `official` scoring is still intentionally not ported in the clean runner and
  fails fast instead of silently falling back. The next scorer phase must compare
  clean rows against historical `src/tgvf_eval/official_tools.py` behavior on
  fixed rows before any benchmark table claims.

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

Remaining scorer gap at the end of Phase 13:

- MMMU-Pro, OCRBench-v2, MathVista, and MathVerse official wrappers are still
  not ported into the clean runner.

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

## Clean-Native Exit Criteria

- `tgvf_stage2_qwen3_legacy` is a diagnostic bridge only. The final clean
  project must replace it with a native clean Stage2 runner before first-class
  benchmark or training claims rely on the clean tree.
- `tgvf_stage2_qwen3` is a deprecated alias for the legacy bridge, not a final
  backend.
- Data generation must be part of the clean project, because it is already a
  natural pure pipeline:
  - fixed source manifests and sample ids;
  - teacher/focus trajectory generation;
  - Stage1 target/description/D dataset generation;
  - Stage2 protocol conversation generation;
  - field/span weights, mask behavior, `im_end` policy, no-focus rules, and
    split hashes recorded beside generated JSONL artifacts.
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
  - Stage2 mask policy, weighted span losses, target focus ratio, and
    DeepStack state;
  - git commit and tracked dirty-worktree state;
  - a temporary historical reference command for auditability;
- the historical reference command is explicitly marked as
  `temporary_legacy_reference_not_final_clean_native`;
- if Stage2 DeepStack training semantics are enabled, the launcher records the
  intended state but marks the historical command non-executable, because the
  old Stage2 script has no DeepStack training controls.

Artifacts written by `--write-plan`:

```text
training_plan.json
training_plan.txt
dataset_identity.json
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
It does not yet validate free/softforce trigger behavior, no-KV full-sequence
behavior, or equivalence/differences versus the legacy bridge.

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
a benchmark claim; no-KV and clean benchmark subset boundaries remain pending.

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

## Later Phases

1. Replace training launch plans with clean-native training execution.
2. Clean benchmark subset boundaries for CoreSmoke/CoreDev execution.
3. Full DeepStack training/eval execution support.
4. Keep data generation first-class:
   - deterministic Stage1/Stage2 transforms stay in `tgvf_generate_data`;
   - heavy teacher trajectory generation is ported only when regeneration is
     needed, with source manifest/hash and prompt/schema identity recorded.
