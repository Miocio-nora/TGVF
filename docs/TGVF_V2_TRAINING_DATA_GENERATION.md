# TGVF V2 Training Data Generation

This document records the teacher-data generation pipeline used by the current
TGVF v2 training runs.

Important naming note: the model/module version is called TGVF v2, but the
teacher-data prompt/schema version used by the current v2 runs is
`visual_cue_v1`.

Code entry points:

- `src/tgvf_data/prepare.py`
- `src/tgvf_data/generate_teacher.py`

Current generated run:

```text
data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001
```

Current v2 training file:

```text
data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001/final/tgvf_teacher_items.accepted.jsonl
```

## Goal

The data generation pipeline creates teacher-guide items:

```text
image
question
target
evidence_description
```

Each item teaches TGVF what local visual evidence a target should retrieve.

The key separation is:

- `target`: a neutral foveation pointer, used later to produce target hidden states.
- `evidence_description`: the visible detail that the FVT readout should support.
- `short_answer`: used for quality filtering and metadata, not directly required by
  the TGVF training dataset class.

The current visual-cue data was introduced because purely semantic object names
were often too coarse. It asks the teacher to produce both semantic targets and
visual-cue targets such as small regions, marks, patches, text clusters, chart
marks, fields, and ambiguous local objects.

## Current Run Configuration

Persisted config:

```text
run id:
  teacher_run_visual_cue_v1_000001

selection:
  data/tgvf_teacher/preparation/selections/selected_images_20k_samples_v0.jsonl

project root:
  /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm

model:
  gpt-5.4

image detail:
  original

temperature:
  0.2

max output tokens:
  2000

timeout:
  120 seconds

max retries:
  5

target accepted samples:
  20000

confidence threshold:
  0.75

seed:
  20260525

prompt version:
  tgvf_teacher_guide_visual_cue_v1

schema version:
  tgvf_teacher_schema_visual_cue_v1

strict visual cue schema:
  false
```

A representative resume command is:

```bash
PYTHONPATH=src python -m tgvf_data.generate_teacher resume-sync \
  --selection data/tgvf_teacher/preparation/selections/selected_images_20k_samples_v0.jsonl \
  --project-root /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm \
  --run-id teacher_run_visual_cue_v1_000001 \
  --target-accepted-samples 20000 \
  --model gpt-5.4 \
  --detail original \
  --temperature 0.2 \
  --max-output-tokens 2000 \
  --timeout-seconds 120 \
  --max-retries 5 \
  --prompt-version tgvf_teacher_guide_visual_cue_v1 \
  --schema-version tgvf_teacher_schema_visual_cue_v1 \
  --confidence-threshold 0.75 \
  --seed 20260525
```

The persisted `config.yaml` does not record runtime concurrency. The generator
supports `--concurrency N` for sync requests.

## Image Selection

Image preparation is handled by `src/tgvf_data/prepare.py`.

The selected image file for the current run contains 7000 images:

```text
data/tgvf_teacher/preparation/selections/selected_images_20k_samples_v0.jsonl
```

Source mix:

```text
visual_genome      2800 images
textvqa + textocr  2100 images
docvqa             1400 images
chartqa             700 images
```

This matches the intended 20k accepted-item mix:

```text
visual_genome      8000 items
textvqa + textocr  6000 items
docvqa             4000 items
chartqa            2000 items
```

The selection records contain:

```text
stable_image_uid
source_dataset
source_profile
image_path
source_image_id
selection_run_id
planned_teacher_run_id
target_sample_budget_hint
status
metadata.source_manifest
metadata.split
```

## OpenAI Request Shape

For each selected image, `build_responses_payload` sends a Responses API request
with:

```python
{
    "model": "gpt-5.4",
    "input": [
        {
            "role": "developer",
            "content": [
                {"type": "input_text", "text": TEACHER_PROMPT_VISUAL_CUE_V1}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": build_user_message(image_record)},
                {
                    "type": "input_image",
                    "image_url": "data:<mime>;base64,...",
                    "detail": "original",
                },
            ],
        },
    ],
    "text": {
        "format": {
            "type": "json_schema",
            "name": "tgvf_teacher_schema_visual_cue_v1",
            "strict": True,
            "schema": ...
        }
    },
    "temperature": 0.2,
    "max_output_tokens": 2000,
}
```

So the teacher prompt is a `developer` instruction, and the per-image metadata
plus actual image are in the `user` message.

## User Message Template

The exact per-image user text is:

```text
Image metadata:
- stable_image_uid: {stable_image_uid}
- source_dataset: {source_dataset}
- source_profile: {source_profile}

Inspect the provided image and generate local visual evidence items according to the schema.

Source-profile hint:
{source_profile_hint}

If optional source context is available, include it below:

Optional source context:
{source_context_or_none}

Use source context only as a hint.
You must verify evidence from the image.
Do not copy source context if it is not visibly supported.
```

Current sync generation passes no optional source context, so this field is:

```text
(none)
```

Source-profile hints:

```text
natural_image:
  Prefer object_part, attribute, texture_material, spatial_relation, logo_symbol,
  and visible text if present.

scene_text:
  Prefer ocr_text, logo_symbol, local text regions, attribute, and spatial_relation.

document:
  Prefer document_field, table_cell, ocr_text, local layout regions.

chart:
  Prefer chart_value, axis labels, legend entries, plotted values, chart marks.

table:
  Prefer table_cell, row/column labels, document_field.

fallback:
  Prefer reliable local evidence; do not force unavailable types.
```

## Output Schema

Top-level required fields:

```text
image_id
source_dataset
source_profile
global_notes
items
```

Each item must contain:

```text
item_id
question
target
evidence_description
short_answer
evidence_type
locality
answer_type
visual_difficulty
visibility
target_leakage_risk
evidence_specificity
confidence
target_style
target_cues
```

Visual-cue-v1 added:

```text
target_style: semantic | visual_cue | mixed
target_cues: list of cue labels
```

Allowed `target_cues`:

```text
location
color
shape
size
texture
pattern
material
nearby_anchor
region_type
text_like
number_like
date_like
chart_like
table_like
symbol_like
object_part
spatial_relation
```

Allowed `evidence_type`:

```text
ocr_text
document_field
chart_value
table_cell
logo_symbol
object_part
attribute
texture_material
spatial_relation
counting
state_action
```

## Validation And Filtering

The teacher response is parsed, validated, deduplicated, then flattened into
training records.

Hard rejection reasons include:

```text
missing required fields
confidence < 0.75
invalid enum values
target_leakage_risk not in none/low
evidence_specificity != specific
target length outside 4 to 22 normalized words
evidence_description length outside 6 to 45 normalized words
global target such as "the image" or "the scene"
generic evidence description
short_answer leaked into target or question
numeric/date/code answer leaked into target or question
sensitive personal identifier in short_answer or evidence_description
duplicate target
duplicate short_answer/evidence_type pair
duplicate evidence_description
duplicate item_content_hash across generated data
```

Visual cue field issues are warnings by default unless
`--strict-visual-cue-schema` is set:

```text
visual_cue_target_has_fewer_than_two_cues
visually_generic_target
semantic_style_for_visual_cue_target
```

## Final Training Record

Accepted items are flattened into JSONL records. Important fields:

```text
uid
teacher_run_id
stable_image_uid
image
image_id
source_dataset
source_profile
question
target
evidence_description
short_answer
evidence_type
locality
answer_type
visual_difficulty
visibility
target_leakage_risk
evidence_specificity
confidence
target_style
target_cues
visual_cue_warnings
teacher_model
teacher_prompt_version
teacher_schema_version
raw_response_id
item_content_hash
created_at
```

The training dataset class only requires:

```text
image
question
target
evidence_description
```

but the extra metadata is important for auditing source mix, target style, and
quality.

Example accepted record shape:

```json
{
  "uid": "teacher_run_visual_cue_v1_000001:visual_genome:2389772:1",
  "stable_image_uid": "visual_genome:2389772",
  "source_dataset": "visual_genome",
  "source_profile": "natural_image",
  "question": "What is visible on the head of the lower sheep near its left ear area?",
  "target": "the dark patch on the lower sheep's head",
  "evidence_description": "There is a small dark marking on top of the lower sheep's head near the ear.",
  "short_answer": "A small dark marking near the ear.",
  "evidence_type": "attribute",
  "target_style": "visual_cue",
  "target_cues": ["location", "color", "size", "nearby_anchor", "object_part"],
  "teacher_model": "gpt-5.4",
  "teacher_prompt_version": "tgvf_teacher_guide_visual_cue_v1",
  "teacher_schema_version": "tgvf_teacher_schema_visual_cue_v1"
}
```

## Current Run Outputs

Main artifacts:

```text
config:
  data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001/config.yaml

prompt:
  data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001/prompt.txt

schema:
  data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001/schema.json

selected input:
  data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001/selected_images.input.jsonl

raw responses:
  data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001/raw_responses/responses.jsonl
  data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001/raw_responses/errors.jsonl

parsed:
  data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001/parsed/image_level_items.raw.jsonl
  data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001/parsed/image_level_items.validated.jsonl

final:
  data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001/final/tgvf_teacher_items.accepted.jsonl
  data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001/final/tgvf_teacher_items.rejected.jsonl

reports:
  data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001/reports/generation_summary.json
  data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001/reports/quality_filter_report.json
  data/tgvf_teacher/generated/runs/teacher_run_visual_cue_v1_000001/reports/cost_usage_report.json

global ledger:
  data/tgvf_teacher/generated/teacher_generation_ledger.jsonl

latest accepted copy:
  data/tgvf_teacher/generated/tgvf_teacher_items.latest.jsonl
```

Current summary:

```text
accepted_items:        20014
new_accepted_items:    19956
rejected_items:         1349
successful_calls:       4254
failed_calls:              3
image_calls:            4269
images with accepted:   4254

source mix:
  visual_genome       8001
  textvqa + textocr   6010
  docvqa              4000
  chartqa             2003

target styles:
  semantic            3566
  visual_cue          5045
  mixed              11403

usage:
  input tokens       15694353
  output tokens       3230274
  total tokens       18924627
```

## Exact Teacher Prompt

This is the full `tgvf_teacher_guide_visual_cue_v1` prompt stored in the run's
`prompt.txt`.

```text
You are a meticulous visual evidence annotator for training a multimodal model.

Your task is to inspect the image and produce local visual evidence items.

Each item should describe one local target that a model could intentionally focus on.

Important concept:

- The "target" is a neutral visual pointer.
- The target tells the model where or what to look at.
- The target must NOT reveal the answer.
- The "evidence_description" contains the actual visible detail.
- The evidence_description should be specific and grounded in the image.

We are training a visual foveation system. Prefer local, fine-grained, visually grounded
details over global captions.

The target does not always need to be a precise semantic object name.

Sometimes the target should be a visual-cue target: a local region described by visible
cues such as color, shape, size, texture, pattern, position, or nearby anchors.

This is especially useful when:
- the object category is uncertain;
- the target is a small region, mark, patch, field, symbol, or text cluster;
- naming the object precisely would reveal too much;
- the question depends on low-level visual evidence such as texture, text, shape, color,
  chart marks, document fields, or local attributes.

A good visual-cue target is semantically modest but visually locatable.

Examples of semantic targets:
- the small text printed below the barcode
- the date field near the bottom of the receipt
- the number inside the blue circle
- the label above the tallest chart bar
- the logo in the upper-right corner of the package

Examples of visual-cue targets:
- the small green object near the left edge
- the dark rectangular patch near the bottom
- the elongated striped region below the label
- the round blue mark in the upper-right corner
- the date-like small text cluster near the lower-right corner
- the barcode-like striped region on the package
- the shiny curved part beside the handle
- the patterned surface area on the metal part

Important:
- The teacher may see the detail clearly, but the target should still behave like a
  foveation pointer, not like the final answer.
- The target may be intentionally less specific than the evidence_description.
- The evidence_description should contain the clear, detailed visual fact.
- The target should help locate the region; the evidence_description should say what is
  actually visible there.

Generate between 3 and 6 items for this image.
Generate fewer items if fewer reliable local details are visible.

Prefer these evidence types when visible:

1. ocr_text: small text, labels, signs, package text, serial numbers
2. document_field: dates, totals, names, fields, form entries, receipt/document regions
3. chart_value: chart labels, axis values, bars, legends, plotted values
4. table_cell: table cells, row/column labels, local numeric entries
5. logo_symbol: logos, icons, badges, symbols
6. object_part: local parts of objects, buttons, ports, handles, edges
7. attribute: color, shape, condition, state, damage, status
8. texture_material: texture, material, surface pattern, roughness, fabric, grain
9. spatial_relation: local positional relationship between visible elements
10. counting: number of local visible elements in a region
11. state_action: local state or action visible in the image

Target rules:

- The target should be a short noun phrase.
- The target should usually be 5 to 18 words.
- The target should point to a local region, object part, symbol, text area, document
  field, chart region, table cell, visual mark, texture region, or ambiguous local object.
- The target may include spatial anchors such as "upper-right", "below the barcode",
  "near the left edge", "inside the blue circle", or "above the tallest bar".
- The target may include visual cues such as color, shape, size, texture, pattern, or
  material only when those cues help locate the target and are not the answer being asked.
- If the object category is uncertain, describe the region using visible cues instead of
  guessing a precise category.
- Prefer "the small green object near the left edge" over "something green".
- Prefer "the elongated striped region below the label" over "the thing below the label".
- Prefer "the date-like text cluster below the barcode" over "the expiration date EXP 08/2026".
- Prefer "the patterned area on the metal surface" over "the rough scratched texture" if
  the question asks what the texture is.

The target must not include:
- the answer value;
- the exact text, number, date, code, or label that should be read;
- the queried attribute value if that value is the answer;
- the final material/texture description if that description is the answer;
- the final count if the question asks for a count.

Do not write targets like:
- "the expiration date EXP 08/2026 below the barcode"
- "the red logo" if the question asks what color the logo is
- "the number 42" if the answer is 42
- "the rough scratched texture" if the evidence_description is supposed to describe the texture
- "the image"
- "the scene"
- "the object"
- "something"
- "something green" unless no better local description is possible

Evidence description rules:

- The evidence_description should be one concise sentence.
- It must describe only visible evidence.
- It should be specific, not generic.
- For text, numbers, dates, and labels, transcribe exactly if readable.
- For texture/material, describe the visible surface details.
- For spatial relations, describe the relation clearly.
- For visual-cue targets, resolve the visual cue into a clear evidence description.
- If uncertain, do not create the item.

Good target/evidence pairs:

Target:
the date-like small text printed below the barcode
Evidence:
The small text below the barcode reads EXP 08/2026.

Target:
the patterned area on the metal surface near the left edge
Evidence:
The metal surface has a brushed texture with fine horizontal lines.

Target:
the small green label near the bottom of the package
Evidence:
The green label contains the word ORGANIC in white letters.

Target:
the dark rectangular patch near the lower-right corner
Evidence:
The dark rectangular patch appears to be a small screen with pale text on it.

Target:
the round blue mark in the upper-right corner
Evidence:
The mark is a blue circular logo with white lettering inside.

Question rules:

- The question should naturally require looking at the target.
- The question must not reveal the answer.
- Prefer questions that require local visual evidence.
- Avoid questions answerable from common sense alone.
- The question may refer to the target using a semantic phrase or a visual-cue phrase.
- If the target is visually described, the question can ask what detail is visible there.

Target style rules:

- Set target_style to "semantic" when the target mainly names a known object, region,
  document field, table cell, chart label, symbol, or object part.
- Set target_style to "visual_cue" when the target mainly uses visible cues such as
  color, shape, size, texture, pattern, position, or nearby anchors instead of a precise
  object/category name.
- Set target_style to "mixed" when the target uses both a semantic anchor and visual cues.

Target cue rules:

- Fill target_cues with the visual cues used by the target.
- Allowed target_cues values are:
  location, color, shape, size, texture, pattern, material, nearby_anchor, region_type,
  text_like, number_like, date_like, chart_like, table_like, symbol_like, object_part,
  spatial_relation.
- Use an empty list only if no clear cue category applies.
- For visual_cue and mixed targets, target_cues should usually contain at least two cues.

Quality rules:

- Do not hallucinate.
- Skip details that are too blurry or uncertain.
- Avoid global scene captions.
- Avoid duplicate items.
- Prefer diversity across evidence_type.
- Prefer a mixture of semantic targets and visual-cue targets when the image supports it.
- Aim for at least 1 or 2 visual-cue targets per image when reliable local visual cues exist.
- Use confidence between 0.0 and 1.0.
- Only include items with confidence >= 0.75.
- Set target_leakage_risk to "none", "low", "medium", or "high".
- Avoid medium/high leakage items unless unavoidable.
- Set evidence_specificity to "specific" or "generic".
- Prefer only specific evidence.

Privacy and safety rules:

- Do not create items that reveal personal addresses, phone numbers, email addresses,
  government IDs, financial account numbers, or other sensitive personal identifiers.
- If such information is visible, skip it.
- Generic product labels, public signs, chart labels, package text, non-sensitive dates,
  and non-personal document fields are allowed.

Return strict JSON matching the provided schema.
Do not include markdown.
Do not include explanations outside JSON.
```
