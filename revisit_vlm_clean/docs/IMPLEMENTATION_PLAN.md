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

## Later Phases

1. TGVF Stage2 controller inference path.
2. Official scorer wrapper port.
3. Stage1/Stage2 launchers.
4. DeepStack training/eval support.
