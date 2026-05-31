# TGVF-v3 Teacher 50k Dataset Generation Record

Date: 2026-05-31

This document records how the first TGVF-v3 teacher dataset was generated, what files were produced, and the observed quality of the 50k run.

## Scope

This run covers teacher-data construction only.

It does not implement or run:

- TGVF projector/refiner training
- Stage 1 or Stage 2 dataloaders beyond JSONL row output
- benchmark evaluation
- crop/zoom tools
- multi-foveation trajectories

The generated rows are trajectory-ready textual supervision for:

```text
<EVIDENCE_STATE>
<FOCUS>
<TGVF>
<EVIDENCE>
<ANSWER>
```

The teacher does not generate visual embeddings. It generates question, target, evidence, answer, metadata, and focus/no-focus labels.

## Code path

Primary implementation:

- `src/tgvf_data/generate_teacher.py`

Relevant behavior added for v3:

- default prompt version: `tgvf_v3_teacher_trajectory_visual_cue_v1`
- default schema version: `tgvf_teacher_schema_v3`
- v3 image-level schema with `focus_items` and `direct_items`
- legacy old-JSON normalization into the shared v3 row format
- visual-cue-aware target filtering
- answer-leakage filtering for target and question
- sensitive personal identifier filtering
- flattened accepted/rejected JSONL output
- report generation for style, cue, evidence type, rejection reason, and image group-size distributions
- version-aware ledger/resume behavior

Old prompt/schema versions remain selectable for compatibility.

## Main output paths

Run directory:

```text
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k
```

Important files:

```text
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/config.yaml
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/prompt.txt
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/schema.json
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/selected_images.input.jsonl
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/raw_responses/responses.jsonl
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/raw_responses/errors.jsonl
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/parsed/image_level_items.raw.jsonl
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/parsed/image_level_items.validated.jsonl
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/final/tgvf_teacher_items.accepted.jsonl
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/final/tgvf_teacher_items.rejected.jsonl
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/reports/generation_summary.json
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/reports/quality_filter_report.json
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/reports/cost_usage_report.json
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k/reports/run_state.json
```

Global append-only ledger:

```text
data/tgvf_teacher/generated/teacher_generation_ledger.jsonl
```

At the time of this record, the ledger had 24,135 lines, with 9,260 entries matching `tgvf_v3_teacher_50k`.

## Configuration used

The completed 50k run used:

```yaml
teacher_run_id: tgvf_v3_teacher_50k
selection_path: data/tgvf_teacher/preparation/selections/selected_images_v3_50k_v0.jsonl
project_root: /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm
openai:
  model: gpt-5.4
  image_detail: original
  fallback_detail: high
  temperature: 0.2
  max_output_tokens: 3000
  timeout_seconds: 120
  max_retries: 5
  allow_json_mode_fallback: false
generation:
  target_accepted_samples: 50000
  min_items_per_image: 1
  max_items_per_image: 6
  confidence_threshold: 0.75
  seed: 20260525
  prompt_version: tgvf_v3_teacher_trajectory_visual_cue_v1
  schema_version: tgvf_teacher_schema_v3
  strict_visual_cue_schema: false
  allow_prompt_schema_mismatch: false
image_backend: sync_base64
```

The selected-image file had 10,000 candidate images. The run stopped after 9,200 image calls because it reached the accepted-sample target.

## Reproduction commands

Set the API key in the environment before running. Do not put the key into a committed script or document.

```bash
export OPENAI_API_KEY="..."
```

Build the 50k v3 image selection:

```bash
PYTHONPATH=src python -m tgvf_data.prepare select \
  --manifest data/tgvf_teacher/preparation/image_pool_manifest.latest.jsonl \
  --target-samples 50000 \
  --selected-images 10000 \
  --output data/tgvf_teacher/preparation/selections/selected_images_v3_50k_v0.jsonl \
  --selection-run-id selection_v3_50k_v0 \
  --planned-teacher-run-id tgvf_v3_teacher_50k \
  --seed 20260525 \
  --allow-used-images
```

Run the 50k generation:

```bash
PYTHONPATH=src python -m tgvf_data.generate_teacher resume-sync \
  --selection data/tgvf_teacher/preparation/selections/selected_images_v3_50k_v0.jsonl \
  --project-root /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm \
  --run-id tgvf_v3_teacher_50k \
  --target-accepted-samples 50000 \
  --model gpt-5.4 \
  --detail original \
  --prompt-version tgvf_v3_teacher_trajectory_visual_cue_v1 \
  --schema-version tgvf_teacher_schema_v3 \
  --max-output-tokens 3000 \
  --timeout-seconds 120 \
  --max-retries 5 \
  --confidence-threshold 0.75 \
  --seed 20260525 \
  --concurrency 8 \
  --wandb-mode disabled
```

Normal resume behavior is to rerun the same command. Do not use `--allow-regenerate` for normal continuation, because the default path skips already completed images and continues from existing outputs.

Use `--allow-regenerate` only when intentionally regenerating images for a new experiment.

## Resume and duplicate behavior

The generation path is designed to be interruptible.

The important state is:

- global ledger: `data/tgvf_teacher/generated/teacher_generation_ledger.jsonl`
- run-local raw responses: `raw_responses/responses.jsonl`
- run-local errors: `raw_responses/errors.jsonl`
- run-local state/report: `reports/run_state.json`
- accepted/rejected JSONL outputs under `final/`

The v3 path uses prompt/schema/model-aware skipping so that old v2 outputs do not incorrectly block v3 generation for the same image. This mattered during smoke testing: without version-aware skip logic, previous teacher runs caused v3 smoke calls to be skipped.

## 50k result summary

Final accepted rows:

```text
accepted: 50022
rejected: 1074
raw items: 51096
image calls: 9200
successful teacher calls: 9191
failed calls: 9
images with zero accepted items: 29
average accepted items per image: 5.44
```

Final files:

```text
50022  final/tgvf_teacher_items.accepted.jsonl
 1074  final/tgvf_teacher_items.rejected.jsonl
```

Token usage:

```text
input tokens: 43,232,303
output tokens: 9,549,292
total tokens: 52,781,595
tokens per image: 5,737.13
tokens per accepted item: 1,055.17
```

## Trajectory distribution

```text
single_focus: 35542
direct_answer: 14480
```

Interpretation:

- focus rows are 71.1% of accepted rows
- direct/no-focus rows are 28.9% of accepted rows
- there are no multi-TGVF rows in this dataset

This means the current dataset has multiple targets per same image, but each flattened training row is still a single-focus trajectory. Same-question multi-step TGVF is not represented.

## Source mix

Accepted row source mix:

```text
visual_genome: 20011
docvqa: 10008
textvqa: 7665
textocr: 7337
chartqa: 5001
```

Source profile mix:

```text
natural_image: 20011
scene_text: 15002
document: 10008
chart: 5001
```

This matches the intended rough 50k source balance:

```text
visual_genome: 20000
textvqa_textocr: 15000
docvqa: 10000
chartqa: 5000
```

## Target style distribution

```text
semantic: 17286
visual_cue: 5916
mixed: 12340
none: 14480
unknown: 0
```

For focus-only rows:

```text
semantic: 17286
visual_cue: 5916
mixed: 12340
focus total: 35542
```

Interpretation:

- explicit `visual_cue` rows are 11.8% of all rows and 16.6% of focus rows
- `mixed` rows are 24.7% of all rows and 34.7% of focus rows
- many `mixed` rows contain useful visual-cue information even if the teacher did not label them as pure `visual_cue`
- the visual-cue label ratio is below the original rough aim of 30%, so downstream sampling or a stricter prompt can be used if a visual-cue-heavy subset is needed

## Target cue distribution

Most common cues:

```text
location: 34106
nearby_anchor: 22487
text_like: 17726
region_type: 15692
size: 10569
color: 8966
object_part: 8686
shape: 5913
chart_anchor: 3748
number_like: 3762
table_anchor: 1618
pattern: 1573
relation: 1868
texture: 514
material: 397
```

The distribution is consistent with TGVF-v3 target design: targets are mostly local, spatially anchored, and inspectable.

## Evidence type distribution

```text
ocr_text: 12954
other: 8940
attribute: 7172
document_field: 5156
counting: 3327
chart_value: 3008
object_part: 3077
spatial_relation: 1785
state_action: 1741
table_cell: 1343
logo_symbol: 1031
texture_material: 488
```

The dataset is OCR/document-heavy enough for local readout training, while still retaining natural-image attribute/object-part coverage.

## Answer format distribution

```text
short_text: 36625
numeric: 7015
boolean: 4600
date: 1307
free_text: 418
multiple_choice: 57
```

Most rows are short-answer style, which is appropriate for early Stage 1/Stage 2 readout and trajectory training.

## Confidence and difficulty

Confidence:

```text
0.95_1.00: 37386
0.90_0.95: 8063
0.80_0.90: 4417
0.75_0.80: 156
```

Visual difficulty:

```text
clear: 40880
medium: 9087
hard: 55
```

Interpretation:

- the run is conservative; most accepted samples are high-confidence and clear
- hard examples are rare
- this is acceptable for the first v3 50k pilot, but a later hard/control pass should be added if robustness is the goal

## Leakage and rejection behavior

Accepted target leakage risk:

```text
none: 19706
low: 30316
```

Rejected rows:

```text
total rejected: 1074
```

Top rejection reasons:

```text
sensitive_personal_identifier: 318
short_answer_in_question: 242
short_answer_in_target: 223
most_answer_tokens_in_target: 204
duplicate_existing_item_content_hash: 115
target_leakage_risk_too_high: 60
duplicate_target: 38
strict_answer_value_in_target: 13
strict_answer_value_in_question: 12
visually_generic_target: 11
duplicate_evidence_description: 11
uncertain_evidence_description: 7
```

This is the intended behavior: rows that leak the answer into the target or question are rejected instead of silently entering the training set.

## Image group-size histogram

Accepted item count per image:

```text
1 item: 4 images
2 items: 25 images
3 items: 172 images
4 items: 780 images
5 items: 3483 images
6 items: 4182 images
7 items: 535 images
8 items: 10 images
```

Same-image multi-target groups are present and large enough to support later query-sensitivity diagnostics and same-image negative evaluation.

## Quality assessment

The 50k dataset is usable as the first TGVF-v3 teacher pilot.

Strong points:

- strict v3 schema is used by default
- old JSON compatibility is preserved
- targets are generally local and visually locatable
- answer leakage filters are active and catch many bad rows
- same-image multi-target grouping is strong
- OCR, document, chart, and natural-image sources are all represented
- focus/no-focus behavior is mixed in the same dataset
- raw, parsed, accepted, rejected, and report outputs are all retained for auditability

Known limitations:

- direct/no-focus ratio is high at 28.9%, compared with the original rough 50k target of about 15%
- explicit `visual_cue` target style is lower than desired, although many `mixed` targets include cue-like pointers
- hard/control rows are not present in this run
- there is no multi-foveation trajectory; all focus rows are `single_focus`
- v3 validation generation has not been run yet in the current output directory

Practical recommendation:

- use this as the first 50k v3 training pilot
- if Stage 2 over-learns direct answering, downsample direct rows or construct a focus-heavier training mix
- if visual-cue behavior is weak, oversample `visual_cue` and `mixed` rows or run a second visual-cue-biased generation pass
- add a separate hard/control generation pass instead of forcing hard examples into this clean pilot

## Train/validation status

Current completed v3 train run:

```text
data/tgvf_teacher/generated/runs/tgvf_v3_teacher_50k
```

Current v3 train selection:

```text
data/tgvf_teacher/preparation/selections/selected_images_v3_50k_v0.jsonl
```

The selection has 10,000 images.

Validation selection exists:

```text
data/tgvf_teacher/preparation/selections/selected_images_val_2k_v0.jsonl
```

The validation selection has 800 images, intended to produce about 2k accepted rows.

At the time of this record, there is no completed v3 validation run directory named `tgvf_v3_teacher_val_2k`. Existing validation run directories are old prompt/schema runs:

```text
data/tgvf_teacher/generated/runs/teacher_run_val_000001
data/tgvf_teacher/generated/runs/teacher_run_val_visual_cue_v1_000001
```

To generate the v3 2k validation set:

```bash
PYTHONPATH=src python -m tgvf_data.generate_teacher resume-sync \
  --selection data/tgvf_teacher/preparation/selections/selected_images_val_2k_v0.jsonl \
  --project-root /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm \
  --run-id tgvf_v3_teacher_val_2k \
  --target-accepted-samples 2000 \
  --model gpt-5.4 \
  --detail original \
  --prompt-version tgvf_v3_teacher_trajectory_visual_cue_v1 \
  --schema-version tgvf_teacher_schema_v3 \
  --max-output-tokens 3000 \
  --timeout-seconds 120 \
  --max-retries 5 \
  --confidence-threshold 0.75 \
  --seed 20260525 \
  --concurrency 8 \
  --wandb-mode disabled
```

## Multi-TGVF status

This 50k dataset does not contain multiple TGVF triggers within one trajectory.

Current forms:

```text
focus row:  trajectory_type = single_focus
direct row: trajectory_type = direct_answer
```

Multiple targets from the same image are flattened into separate rows. This is useful for same-image target specificity, but it is not the same as a single question that performs multiple sequential foveations.

This matches the current v3 default: `max_foveations=1` unless explicitly testing a multi-foveation ablation later.

## Bottom line

The 50k run reached the target with 50,022 accepted rows and preserved the core TGVF-v3 data identity:

- focus targets are explicit visual pointers
- evidence descriptions contain the answer-bearing visual fact
- direct/no-focus rows are included to prevent always-focusing behavior
- old data compatibility and resume/no-repeat behavior are preserved
- the generated data is audit-friendly through raw, parsed, accepted, rejected, report, and ledger outputs

The main follow-up is to generate the v3 2k validation set and decide whether Stage 2 should downsample direct rows or oversample visual-cue/mixed focus rows.
