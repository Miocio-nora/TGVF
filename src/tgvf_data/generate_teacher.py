from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - fallback exists for minimal environments.
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

from revisit_vlm.wandb_logging import WandbLogger, flatten_metrics
from tgvf_data.prepare import DEFAULT_SOURCE_MIX, read_jsonl, write_jsonl
from tgvf_data.tgvf_teacher_schema_v4 import (
    SCHEMA_VERSION_V4 as SCHEMA_VERSION_V4_CONST,
    TEACHER_VERSION_V4,
    teacher_output_schema_v4,
    validate_v4_image_level_output,
)
from tgvf_data.tgvf_v4_teacher import TEACHER_PROMPT_V4, build_v4_user_message

PROMPT_VERSION_V0 = "tgvf_teacher_guide_v0"
PROMPT_VERSION_VISUAL_CUE_V1 = "tgvf_teacher_guide_visual_cue_v1"
PROMPT_VERSION_V3 = "tgvf_v3_teacher_trajectory_visual_cue_v1"
PROMPT_VERSION_V4 = TEACHER_VERSION_V4
DEFAULT_TEACHER_PROMPT_VERSION = PROMPT_VERSION_V3
PROMPT_VERSION = DEFAULT_TEACHER_PROMPT_VERSION

SCHEMA_VERSION_V0 = "tgvf_teacher_schema_v0"
SCHEMA_VERSION_VISUAL_CUE_V1 = "tgvf_teacher_schema_visual_cue_v1"
SCHEMA_VERSION_V3 = "tgvf_teacher_schema_v3"
SCHEMA_VERSION_V4 = SCHEMA_VERSION_V4_CONST
DEFAULT_TEACHER_SCHEMA_VERSION = SCHEMA_VERSION_V3
SCHEMA_VERSION = DEFAULT_TEACHER_SCHEMA_VERSION
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_IMAGE_DETAIL = "original"
DEFAULT_FALLBACK_DETAIL = "high"
DEFAULT_RUN_ID = "teacher_run_000001"
DEFAULT_SELECTION = (
    "data/tgvf_teacher/preparation/selections/selected_images_20k_samples_v0.jsonl"
)
DEFAULT_TARGET_ACCEPTED_SAMPLES = 20_000
DEFAULT_SEED = 20260525
MANIFEST_VERSION = "tgvf_teacher_generation_v0"

EVIDENCE_TYPES = [
    "ocr_text",
    "document_field",
    "chart_value",
    "table_cell",
    "logo_symbol",
    "object_part",
    "attribute",
    "texture_material",
    "spatial_relation",
    "counting",
    "state_action",
]
DIRECT_EVIDENCE_TYPES = [
    "attribute",
    "object_part",
    "spatial_relation",
    "counting",
    "state_action",
    "other",
]
LOCALITIES = [
    "tiny_region",
    "small_region",
    "object_part",
    "single_object",
    "document_region",
    "chart_region",
    "table_region",
    "symbol_region",
]
DIRECT_LOCALITIES = ["single_object", "whole_image", "small_region"]
ANSWER_TYPES = [
    "text_string",
    "number",
    "date",
    "category",
    "attribute_value",
    "material_texture",
    "spatial_relation",
    "count",
    "boolean_state",
    "short_description",
]
DIRECT_ANSWER_TYPES = [
    "category",
    "attribute_value",
    "count",
    "boolean_state",
    "short_description",
]
ANSWER_FORMATS = ["short_text", "multiple_choice", "numeric", "date", "boolean", "free_text"]
VISUAL_DIFFICULTIES = ["clear", "medium", "hard"]
VISIBILITIES = ["clear", "partially_occluded", "low_contrast", "small", "blurry_but_readable"]
TARGET_LEAKAGE_RISKS = ["none", "low", "medium", "high"]
EVIDENCE_SPECIFICITIES = ["specific", "generic"]
TARGET_STYLES = ["semantic", "visual_cue", "mixed"]
NORMALIZED_TARGET_STYLES = TARGET_STYLES + ["none", "unknown"]
TARGET_CUES = [
    "location",
    "color",
    "shape",
    "size",
    "texture",
    "pattern",
    "material",
    "nearby_anchor",
    "region_type",
    "text_like",
    "number_like",
    "date_like",
    "chart_anchor",
    "table_anchor",
    "chart_like",
    "table_like",
    "symbol_like",
    "object_part",
    "relation",
    "spatial_relation",
]
SOURCE_PROFILES = [
    "natural_image",
    "scene_text",
    "document",
    "chart",
    "table",
    "mixed",
    "unknown",
]
ALLOWED_PROMPT_VERSIONS = [
    PROMPT_VERSION_V0,
    PROMPT_VERSION_VISUAL_CUE_V1,
    PROMPT_VERSION_V3,
    PROMPT_VERSION_V4,
]
ALLOWED_SCHEMA_VERSIONS = [
    SCHEMA_VERSION_V0,
    SCHEMA_VERSION_VISUAL_CUE_V1,
    SCHEMA_VERSION_V3,
    SCHEMA_VERSION_V4,
]
LEDGER_TERMINAL_SUCCESS = {"succeeded", "completed", "accepted"}
RETRYABLE_ERROR_MARKERS = (
    "rate limit",
    "timeout",
    "temporar",
    "connection",
    "server error",
    "5xx",
    "429",
    "500",
    "502",
    "503",
    "504",
)


@dataclass
class OpenAIConfig:
    model: str = DEFAULT_MODEL
    image_detail: str = DEFAULT_IMAGE_DETAIL
    fallback_detail: str = DEFAULT_FALLBACK_DETAIL
    temperature: float = 0.2
    max_output_tokens: int = 2000
    timeout_seconds: int = 120
    max_retries: int = 5
    allow_json_mode_fallback: bool = False


@dataclass
class GenerationConfig:
    target_accepted_samples: int = DEFAULT_TARGET_ACCEPTED_SAMPLES
    min_items_per_image: int = 1
    max_items_per_image: int = 6
    confidence_threshold: float = 0.75
    seed: int = DEFAULT_SEED
    prompt_version: str = PROMPT_VERSION
    schema_version: str = SCHEMA_VERSION
    strict_visual_cue_schema: bool = False
    allow_prompt_schema_mismatch: bool = False


@dataclass
class TeacherRunConfig:
    teacher_run_id: str
    selection_path: str
    project_root: str
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    image_backend: str = "sync_base64"
    pricing: dict[str, float] = field(default_factory=dict)


@dataclass
class TeacherResponse:
    response_id: str | None
    output_text: str
    raw_response: dict[str, Any]
    usage: dict[str, int]


@dataclass
class ParsedImageResult:
    image_record: dict[str, Any]
    image_output: dict[str, Any] | None
    accepted_items: list[dict[str, Any]]
    rejected_items: list[dict[str, Any]]
    status: str
    error: str | None
    raw_response_id: str | None
    usage: dict[str, int]


TEACHER_PROMPT_V0 = """You are a meticulous visual evidence annotator for training a multimodal model.

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
  field, chart region, or table cell.
- The target may include spatial anchors such as "upper-right", "below the barcode",
  "near the left edge".
- The target must not include the answer value.
- The target must not include the queried attribute value if that value is the answer.
- Do not write targets like "the red logo" if the question asks what color the logo is.
- Do not write targets like "the number 42" if the answer is 42.
- Do not write targets like "the rough texture" if the evidence_description is
  supposed to describe the texture.

Evidence description rules:

- The evidence_description should be one concise sentence.
- It must describe only visible evidence.
- It should be specific, not generic.
- For text, numbers, dates, and labels, transcribe exactly if readable.
- For texture/material, describe the visible surface details.
- For spatial relations, describe the relation clearly.
- If uncertain, do not create the item.

Question rules:

- The question should naturally require looking at the target.
- The question must not reveal the answer.
- Prefer questions that require local visual evidence.
- Avoid questions answerable from common sense alone.

Quality rules:

- Do not hallucinate.
- Skip details that are too blurry or uncertain.
- Avoid global scene captions.
- Avoid duplicate items.
- Prefer diversity across evidence_type.
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
"""


TEACHER_PROMPT_VISUAL_CUE_V1 = """You are a meticulous visual evidence annotator for training a multimodal model.

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
"""

TEACHER_PROMPT_V3 = """You are a meticulous visual evidence annotator for training a multimodal model.

We are building a Target-Guided Visual Foveation system.

Your task is to inspect the image and produce training data for two behaviors:

1. Focus behavior:
   The model should request a local visual focus target before answering.
   These examples will train the model to emit:
   <EVIDENCE_STATE>need_local_visual_evidence</EVIDENCE_STATE>
   <FOCUS>target</FOCUS>
   and later read foveated visual evidence.

2. Direct-answer behavior:
   The model should answer directly without foveation when the answer is already obvious from the whole image.
   These examples will train the model to emit:
   <EVIDENCE_STATE>sufficient_visual_evidence</EVIDENCE_STATE>
   <ANSWER>answer</ANSWER>

Important concept:

- The "target" is a neutral visual pointer.
- The target tells the vision system where or what to look at.
- The target must NOT reveal the answer.
- The "evidence_description" contains the actual visible detail.
- The evidence_description should be specific and grounded in the image.
- The target may be intentionally less specific than the evidence_description.
- The teacher may see the detail clearly, but the target should still behave like a foveation pointer, not like the final answer.

Generate:

- 3 to 6 focus_items if reliable local evidence exists.
- 0 to 2 direct_items if there are obvious whole-image questions that should not require foveation.

Generate fewer items if fewer reliable details are visible.
Do not hallucinate. If uncertain, skip the item.

Target styles:

A. Semantic target
Use this when the local region or object is clear.
Examples: the small text printed below the barcode; the date field near the bottom of the receipt; the number inside the blue circle; the label above the tallest chart bar; the logo in the upper-right corner of the package; the button near the right edge of the device; the table cell in the second row and third column.

B. Visual-cue target
Use this when the object category is uncertain, when naming the object precisely would reveal too much, or when visual cues are more appropriate than semantic labels. A visual-cue target describes a local region using visible cues such as color, shape, size, texture, pattern, position, or nearby anchors.
Examples: the small green object near the left edge; the dark rectangular patch near the bottom; the elongated striped region below the label; the round blue mark in the upper-right corner; the date-like small text cluster near the lower-right corner; the barcode-like striped region on the package; the shiny curved part beside the handle; the patterned surface area on the metal part; the number-like symbols inside the blue circle; the long horizontal bar near the bottom of the chart.

C. Mixed target
Use this when the target combines a semantic region and visual cues.
Examples: the small green label near the bottom of the package; the date-like text field on the lower part of the receipt; the blue circular logo near the upper-right corner; the textured metal strip along the lower edge.

Target rules:

- The target should be a short noun phrase.
- The target should usually be 5 to 18 words.
- The target must be local and visually locatable.
- The target may include spatial anchors such as upper-right, below the barcode, near the left edge, inside the blue circle, or above the tallest bar.
- The target may include visual cues only when those cues help locate the target and are not the answer being asked.
- If the object category is uncertain, describe the region using visible cues instead of guessing a precise category.
- Prefer "the small green object near the left edge" over "something green".
- Prefer "the elongated striped region below the label" over "the thing below the label".
- Prefer "the date-like text cluster below the barcode" over "the expiration date EXP 08/2026".
- Prefer "the patterned area on the metal surface" over "the rough scratched texture" if the question asks what the texture is.

The target must not include the answer value; exact text, number, date, code, label, or word that should be read; the queried attribute value if that value is the answer; the final material or texture description if that description is the answer; or the final count if the question asks for a count.

Bad targets: the image; the scene; the object; something; something green; the important part; the answer; the expiration date EXP 08/2026 below the barcode; the red logo if the question asks what color the logo is; the number 42 if the answer is 42; the rough scratched texture if the question asks what texture is visible.

Evidence description rules:

- The evidence_description should be one concise sentence.
- It must describe only visible evidence.
- It should be specific, not generic.
- For text, numbers, dates, and labels, transcribe exactly if readable.
- For texture/material, describe the visible surface details.
- For spatial relations, describe the relation clearly.
- For chart/table items, identify the local mark, axis label, legend, row, column, or cell clearly.
- For visual-cue targets, resolve the visual cue into a clear evidence description.
- If uncertain, do not create the item.

Question rules:

- The question should naturally require looking at the target.
- The question must not reveal the answer.
- Prefer questions that require local visual evidence.
- Avoid questions answerable from common sense alone.
- For focus_items, the question should make foveation useful.
- For direct_items, the question should be answerable without foveation.

Evidence types for focus_items: ocr_text, document_field, chart_value, table_cell, logo_symbol, object_part, attribute, texture_material, spatial_relation, counting, state_action.
Direct item evidence_type values: attribute, object_part, spatial_relation, counting, state_action, other.

Direct item rules:
Create direct_items only when the answer is clearly visible from the whole image and does not require local detail inspection. Good direct_items ask about obvious whole-image objects, animals, actions, state, count, or large foreground attributes. Bad direct_items ask for tiny text, receipt/footer dates, small chart labels, table cell values, small local numbers, or fine texture.

Target cues:
For each focus item, set target_cues as a list of cues used in the target. Allowed target_cues: location, color, shape, size, texture, pattern, nearby_anchor, region_type, text_like, number_like, chart_anchor, table_anchor, object_part, material, relation. For visual_cue targets, include at least two useful target_cues whenever possible.

Quality rules:

- Do not hallucinate.
- Skip details that are too blurry or uncertain.
- Avoid global scene captions in focus_items.
- Avoid duplicate items.
- Prefer diversity across evidence_type.
- Prefer a mixture of semantic, visual_cue, and mixed targets when the image supports it.
- Aim for roughly 60% semantic or mostly semantic targets, 30% visual_cue targets, and 10% mixed targets, but do not force this if the image does not support it.
- Use confidence between 0.0 and 1.0.
- Only include items with confidence >= 0.75.
- Set target_leakage_risk to none, low, medium, or high.
- Avoid medium/high leakage items unless unavoidable.
- Set evidence_specificity to specific or generic.
- Prefer only specific evidence.

Privacy and safety rules:

- Do not create items that reveal personal addresses, phone numbers, email addresses, government IDs, financial account numbers, or other sensitive personal identifiers.
- If such information is visible, skip it.
- Generic product labels, public signs, chart labels, package text, non-sensitive dates, and non-personal document fields are allowed.

Return strict JSON only. Do not include markdown. Do not include explanations outside JSON.

Use this schema exactly:

{
  "schema_version": "tgvf_teacher_schema_v3",
  "teacher_prompt_version": "tgvf_v3_teacher_trajectory_visual_cue_v1",
  "image_id": "<string or null>",
  "source_dataset": "<string or null>",
  "source_profile": "<natural_image | scene_text | document | chart | table | mixed | unknown>",
  "global_notes": "<short debug summary, not used for training>",
  "focus_items": [
    {
      "item_id": "<string>",
      "need_focus": true,
      "evidence_state": "need_local_visual_evidence",
      "trajectory_type": "single_focus",
      "question": "<question that naturally requires looking at the target>",
      "target": "<neutral local visual pointer without answer leakage>",
      "target_style": "<semantic | visual_cue | mixed>",
      "target_cues": ["<cue strings>"],
      "evidence_description": "<specific visible evidence sentence>",
      "short_answer": "<short answer if applicable>",
      "answer": "<final answer>",
      "answer_format": "<short_text | multiple_choice | numeric | date | boolean | free_text>",
      "value_span_text": "<answer-bearing span inside evidence_description, if any>",
      "evidence_type": "<ocr_text | document_field | chart_value | table_cell | logo_symbol | object_part | attribute | texture_material | spatial_relation | counting | state_action>",
      "locality": "<tiny_region | small_region | object_part | single_object | document_region | chart_region | table_region | symbol_region>",
      "answer_type": "<text_string | number | date | category | attribute_value | material_texture | spatial_relation | count | boolean_state | short_description>",
      "visual_difficulty": "<clear | medium | hard>",
      "visibility": "<clear | partially_occluded | low_contrast | small | blurry_but_readable>",
      "target_leakage_risk": "<none | low | medium | high>",
      "evidence_specificity": "<specific | generic>",
      "confidence": <float between 0 and 1>
    }
  ],
  "direct_items": [
    {
      "item_id": "<string>",
      "need_focus": false,
      "evidence_state": "sufficient_visual_evidence",
      "trajectory_type": "direct_answer",
      "question": "<question answerable from the whole image without local foveation>",
      "target": "",
      "target_style": "none",
      "target_cues": [],
      "evidence_description": "<brief visible support, optional but useful>",
      "short_answer": "<short answer>",
      "answer": "<final answer>",
      "answer_format": "<short_text | multiple_choice | numeric | date | boolean | free_text>",
      "value_span_text": "<answer-bearing span if any>",
      "evidence_type": "<attribute | object_part | spatial_relation | counting | state_action | other>",
      "locality": "<single_object | whole_image | small_region>",
      "answer_type": "<category | attribute_value | count | boolean_state | short_description>",
      "visual_difficulty": "clear",
      "visibility": "clear",
      "target_leakage_risk": "none",
      "evidence_specificity": "specific",
      "confidence": <float between 0 and 1>
    }
  ]
}
"""

TEACHER_PROMPT = TEACHER_PROMPT_V3


def validate_teacher_version_pair(
    prompt_version: str,
    schema_version: str,
    *,
    allow_mismatch: bool = False,
) -> None:
    if prompt_version not in ALLOWED_PROMPT_VERSIONS:
        raise ValueError(f"Unsupported teacher prompt version: {prompt_version}")
    if schema_version not in ALLOWED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported teacher schema version: {schema_version}")
    expected = {
        PROMPT_VERSION_V0: SCHEMA_VERSION_V0,
        PROMPT_VERSION_VISUAL_CUE_V1: SCHEMA_VERSION_VISUAL_CUE_V1,
        PROMPT_VERSION_V3: SCHEMA_VERSION_V3,
        PROMPT_VERSION_V4: SCHEMA_VERSION_V4,
    }[prompt_version]
    if schema_version != expected and not allow_mismatch:
        raise ValueError(
            "Teacher prompt/schema versions must match: "
            f"{prompt_version} expects {expected}, got {schema_version}. "
            "Use --allow-prompt-schema-mismatch to override."
        )


def get_teacher_prompt(prompt_version: str | None = None) -> str:
    version = prompt_version or DEFAULT_TEACHER_PROMPT_VERSION
    prompts = {
        PROMPT_VERSION_V0: TEACHER_PROMPT_V0,
        PROMPT_VERSION_VISUAL_CUE_V1: TEACHER_PROMPT_VISUAL_CUE_V1,
        PROMPT_VERSION_V3: TEACHER_PROMPT_V3,
        PROMPT_VERSION_V4: TEACHER_PROMPT_V4,
    }
    try:
        return prompts[version]
    except KeyError as exc:
        raise ValueError(f"Unsupported teacher prompt version: {version}") from exc


def teacher_output_schema(schema_version: str | None = None) -> dict[str, Any]:
    version = schema_version or DEFAULT_TEACHER_SCHEMA_VERSION
    if version not in ALLOWED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported teacher schema version: {version}")
    if version == SCHEMA_VERSION_V4:
        return teacher_output_schema_v4()
    if version == SCHEMA_VERSION_V3:
        return teacher_output_schema_v3()
    return teacher_output_schema_legacy(version)


def teacher_output_schema_legacy(version: str) -> dict[str, Any]:
    include_visual_cue_fields = version == SCHEMA_VERSION_VISUAL_CUE_V1
    item_required = [
        "item_id",
        "question",
        "target",
        "evidence_description",
        "short_answer",
        "evidence_type",
        "locality",
        "answer_type",
        "visual_difficulty",
        "visibility",
        "target_leakage_risk",
        "evidence_specificity",
        "confidence",
    ]
    item_properties = {
        "item_id": {"type": "string"},
        "question": {"type": "string"},
        "target": {"type": "string"},
        "evidence_description": {"type": "string"},
        "short_answer": {"type": "string"},
        "evidence_type": {"type": "string", "enum": EVIDENCE_TYPES},
        "locality": {"type": "string", "enum": LOCALITIES},
        "answer_type": {"type": "string", "enum": ANSWER_TYPES},
        "visual_difficulty": {"type": "string", "enum": VISUAL_DIFFICULTIES},
        "visibility": {"type": "string", "enum": VISIBILITIES},
        "target_leakage_risk": {"type": "string", "enum": TARGET_LEAKAGE_RISKS},
        "evidence_specificity": {"type": "string", "enum": EVIDENCE_SPECIFICITIES},
        "confidence": {"type": "number"},
    }
    if include_visual_cue_fields:
        item_required.extend(["target_style", "target_cues"])
        item_properties.update(
            {
                "target_style": {"type": "string", "enum": TARGET_STYLES},
                "target_cues": {
                    "type": "array",
                    "items": {"type": "string", "enum": TARGET_CUES},
                },
            }
        )
    item_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": item_required,
        "properties": item_properties,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "image_id",
            "source_dataset",
            "source_profile",
            "global_notes",
            "items",
        ],
        "properties": {
            "image_id": {"type": "string"},
            "source_dataset": {"type": "string"},
            "source_profile": {"type": "string"},
            "global_notes": {"type": "string"},
            "items": {
                "type": "array",
                "items": item_schema,
            },
        },
    }


def teacher_output_schema_v3() -> dict[str, Any]:
    focus_required = [
        "item_id",
        "need_focus",
        "evidence_state",
        "trajectory_type",
        "question",
        "target",
        "target_style",
        "target_cues",
        "evidence_description",
        "short_answer",
        "answer",
        "answer_format",
        "value_span_text",
        "evidence_type",
        "locality",
        "answer_type",
        "visual_difficulty",
        "visibility",
        "target_leakage_risk",
        "evidence_specificity",
        "confidence",
    ]
    direct_required = list(focus_required)
    focus_item_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": focus_required,
        "properties": {
            "item_id": {"type": "string"},
            "need_focus": {"type": "boolean", "enum": [True]},
            "evidence_state": {"type": "string", "enum": ["need_local_visual_evidence"]},
            "trajectory_type": {"type": "string", "enum": ["single_focus"]},
            "question": {"type": "string"},
            "target": {"type": "string"},
            "target_style": {"type": "string", "enum": TARGET_STYLES},
            "target_cues": {"type": "array", "items": {"type": "string", "enum": TARGET_CUES}},
            "evidence_description": {"type": "string"},
            "short_answer": {"type": "string"},
            "answer": {"type": "string"},
            "answer_format": {"type": "string", "enum": ANSWER_FORMATS},
            "value_span_text": {"type": "string"},
            "evidence_type": {"type": "string", "enum": EVIDENCE_TYPES},
            "locality": {"type": "string", "enum": LOCALITIES},
            "answer_type": {"type": "string", "enum": ANSWER_TYPES},
            "visual_difficulty": {"type": "string", "enum": VISUAL_DIFFICULTIES},
            "visibility": {"type": "string", "enum": VISIBILITIES},
            "target_leakage_risk": {"type": "string", "enum": TARGET_LEAKAGE_RISKS},
            "evidence_specificity": {"type": "string", "enum": EVIDENCE_SPECIFICITIES},
            "confidence": {"type": "number"},
        },
    }
    direct_item_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": direct_required,
        "properties": {
            "item_id": {"type": "string"},
            "need_focus": {"type": "boolean", "enum": [False]},
            "evidence_state": {"type": "string", "enum": ["sufficient_visual_evidence"]},
            "trajectory_type": {"type": "string", "enum": ["direct_answer"]},
            "question": {"type": "string"},
            "target": {"type": "string", "enum": [""]},
            "target_style": {"type": "string", "enum": ["none"]},
            "target_cues": {"type": "array", "items": {"type": "string"}, "maxItems": 0},
            "evidence_description": {"type": "string"},
            "short_answer": {"type": "string"},
            "answer": {"type": "string"},
            "answer_format": {"type": "string", "enum": ANSWER_FORMATS},
            "value_span_text": {"type": "string"},
            "evidence_type": {"type": "string", "enum": DIRECT_EVIDENCE_TYPES},
            "locality": {"type": "string", "enum": DIRECT_LOCALITIES},
            "answer_type": {"type": "string", "enum": DIRECT_ANSWER_TYPES},
            "visual_difficulty": {"type": "string", "enum": ["clear"]},
            "visibility": {"type": "string", "enum": ["clear"]},
            "target_leakage_risk": {"type": "string", "enum": ["none"]},
            "evidence_specificity": {"type": "string", "enum": ["specific"]},
            "confidence": {"type": "number"},
        },
    }
    string_or_null = {"type": ["string", "null"]}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "teacher_prompt_version",
            "image_id",
            "source_dataset",
            "source_profile",
            "global_notes",
            "focus_items",
            "direct_items",
        ],
        "properties": {
            "schema_version": {"type": "string", "enum": [SCHEMA_VERSION_V3]},
            "teacher_prompt_version": {"type": "string", "enum": [PROMPT_VERSION_V3]},
            "image_id": string_or_null,
            "source_dataset": string_or_null,
            "source_profile": {"type": "string", "enum": SOURCE_PROFILES},
            "global_notes": {"type": "string"},
            "focus_items": {"type": "array", "items": focus_item_schema},
            "direct_items": {"type": "array", "items": direct_item_schema},
        },
    }


def get_teacher_schema(schema_version: str | None = None) -> dict[str, Any]:
    return teacher_output_schema(schema_version)


def structured_text_format(
    schema: dict[str, Any] | None = None,
    *,
    schema_version: str | None = None,
) -> dict[str, Any]:
    version = schema_version or DEFAULT_TEACHER_SCHEMA_VERSION
    return {
        "format": {
            "type": "json_schema",
            "name": version,
            "strict": True,
            "schema": schema or get_teacher_schema(version),
        }
    }


def json_mode_text_format() -> dict[str, Any]:
    return {"format": {"type": "json_object"}}


def build_user_message(image_record: dict[str, Any], source_context: str = "") -> str:
    profile = image_record.get("source_profile", "unknown")
    hint = source_profile_hint(profile)
    context = source_context.strip()
    optional_block = (
        "Optional source context is provided below.\n\n"
        "Use it only as a hint.\n"
        "You must verify the answer from the image.\n"
        "Do not copy source context if it is not visibly supported.\n\n"
        f"{context}\n\n"
        "If the source question-answer pair is visually grounded and reliable, "
        "create one focus_item based on it. You may also create additional local "
        "visual evidence items from the image. If the source question is not visibly "
        "supported, ignore it."
        if context
        else "Optional source context:\n(none)"
    )
    return (
        "Image metadata:\n"
        f"- stable_image_uid: {image_record['stable_image_uid']}\n"
        f"- source_dataset: {image_record.get('source_dataset', '')}\n"
        f"- source_profile: {profile}\n\n"
        "Inspect the provided image and generate local visual evidence items according "
        "to the schema.\n\n"
        f"Source-profile hint:\n{hint}\n\n"
        f"{optional_block}"
    )


def source_context_from_record(image_record: dict[str, Any]) -> str:
    parts = []
    if image_record.get("source_question"):
        parts.append(f"source_question:\n{image_record['source_question']}")
    if image_record.get("source_answer"):
        parts.append(f"source_answer:\n{image_record['source_answer']}")
    metadata = image_record.get("source_metadata") or image_record.get("metadata")
    if metadata:
        parts.append("source_metadata:\n" + json.dumps(jsonable(metadata), sort_keys=True))
    return "\n\n".join(parts)


def source_profile_hint(source_profile: str) -> str:
    hints = {
        "natural_image": (
            "Prefer object_part, attribute, texture_material, spatial_relation, "
            "logo_symbol, and visible text if present."
        ),
        "scene_text": (
            "Prefer ocr_text, logo_symbol, local text regions, attribute, and "
            "spatial_relation."
        ),
        "document": "Prefer document_field, table_cell, ocr_text, local layout regions.",
        "chart": "Prefer chart_value, axis labels, legend entries, plotted values, chart marks.",
        "table": "Prefer table_cell, row/column labels, document_field.",
    }
    return hints.get(
        source_profile,
        "Prefer reliable local evidence; do not force unavailable types.",
    )


def build_responses_payload(
    *,
    image_record: dict[str, Any],
    image_reference: str,
    openai_config: OpenAIConfig,
    generation_config: GenerationConfig | None = None,
    source_context: str = "",
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generation_config = generation_config or GenerationConfig()
    user_message = (
        build_v4_user_message(image_record, source_context)
        if generation_config.prompt_version == PROMPT_VERSION_V4
        or generation_config.schema_version == SCHEMA_VERSION_V4
        else build_user_message(image_record, source_context)
    )
    return {
        "model": openai_config.model,
        "input": [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": get_teacher_prompt(generation_config.prompt_version),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_message,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_reference,
                        "detail": openai_config.image_detail,
                    },
                ],
            },
        ],
        "text": (
            json_mode_text_format()
            if openai_config.allow_json_mode_fallback
            else structured_text_format(schema, schema_version=generation_config.schema_version)
        ),
        "temperature": openai_config.temperature,
        "max_output_tokens": openai_config.max_output_tokens,
    }


class OpenAITeacherClient:
    def __init__(self, *, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError(
                "OpenAI Python SDK is not installed. Install it to run API commands."
            ) from exc
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def create_response(self, payload: dict[str, Any], *, timeout_seconds: int) -> TeacherResponse:
        response = self.client.with_options(timeout=timeout_seconds).responses.create(**payload)
        return response_to_teacher_response(response)

    def upload_batch_file(self, request_file: str | Path) -> Any:
        with Path(request_file).open("rb") as handle:
            return self.client.files.create(file=handle, purpose="batch")

    def create_batch(self, *, input_file_id: str, completion_window: str = "24h") -> Any:
        return self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/responses",
            completion_window=completion_window,
        )

    def retrieve_batch(self, batch_id: str) -> Any:
        return self.client.batches.retrieve(batch_id)

    def download_file_text(self, file_id: str) -> str:
        content = self.client.files.content(file_id)
        if hasattr(content, "text"):
            return content.text
        if hasattr(content, "read"):
            data = content.read()
            return data.decode("utf-8") if isinstance(data, bytes) else str(data)
        return str(content)


def call_teacher_with_retries(
    *,
    client: Any,
    payload: dict[str, Any],
    config: OpenAIConfig,
    sleep_fn: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> TeacherResponse:
    rng = rng or random.Random()
    last_error: Exception | None = None
    for attempt in range(config.max_retries + 1):
        try:
            return client.create_response(payload, timeout_seconds=config.timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - provider wrapper boundary.
            last_error = exc
            if not is_retryable_error(exc) or attempt >= config.max_retries:
                break
            delay = min(60.0, (2**attempt) + rng.random())
            sleep_fn(delay)
    raise RuntimeError(f"Teacher request failed after retries: {last_error}") from last_error


def response_to_teacher_response(response: Any) -> TeacherResponse:
    raw = _to_plain_dict(response)
    output_text = getattr(response, "output_text", None) or raw.get("output_text")
    if output_text is None:
        output_text = extract_output_text(raw)
    usage = normalize_usage(raw.get("usage") or getattr(response, "usage", None))
    return TeacherResponse(
        response_id=raw.get("id") or getattr(response, "id", None),
        output_text=output_text,
        raw_response=raw,
        usage=usage,
    )


def extract_output_text(raw_response: dict[str, Any]) -> str:
    texts = []
    for output in raw_response.get("output", []) or []:
        for content in output.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                texts.append(content["text"])
    return "\n".join(texts)


def normalize_usage(usage: Any) -> dict[str, int]:
    data = _to_plain_dict(usage) if usage is not None else {}
    input_tokens = int(data.get("input_tokens") or data.get("prompt_tokens") or 0)
    output_tokens = int(data.get("output_tokens") or data.get("completion_tokens") or 0)
    total_tokens = int(data.get("total_tokens") or input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def parse_teacher_output_text(text: str) -> dict[str, Any]:
    return json.loads(text)


def validate_image_level_output(
    image_output: dict[str, Any],
    image_record: dict[str, Any],
    *,
    config: GenerationConfig | None = None,
    teacher_run_id: str = DEFAULT_RUN_ID,
    model: str = DEFAULT_MODEL,
    raw_response_id: str | None = None,
    existing_item_hashes: set[str] | None = None,
    allow_duplicate_items: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = config or GenerationConfig()
    if config.schema_version == SCHEMA_VERSION_V4:
        return validate_v4_image_level_output(
            image_output,
            image_record,
            teacher_run_id=teacher_run_id,
            model=model,
            prompt_version=config.prompt_version,
            schema_version=config.schema_version,
            raw_response_id=raw_response_id,
            confidence_threshold=config.confidence_threshold,
            existing_item_hashes=existing_item_hashes,
            allow_duplicate_items=allow_duplicate_items,
        )
    accepted_candidates = []
    rejected = []
    for index, item_kind, item in iter_teacher_items(image_output):
        reasons, warnings = validate_item_with_warnings(
            item,
            confidence_threshold=config.confidence_threshold,
            schema_version=config.schema_version,
            strict_visual_cue_schema=config.strict_visual_cue_schema,
            item_kind=item_kind,
        )
        if reasons:
            rejected.append(
                rejected_record(
                    image_record=image_record,
                    item=item,
                    rejection_reasons=reasons,
                    visual_cue_warnings=warnings,
                    teacher_run_id=teacher_run_id,
                    model=model,
                    prompt_version=config.prompt_version,
                    schema_version=config.schema_version,
                    raw_response_id=raw_response_id,
                    item_index=index,
                    item_kind=item_kind,
                )
            )
        else:
            accepted_candidates.append((index, item, warnings, item_kind))

    accepted, duplicate_rejections = deduplicate_items(
        accepted_candidates,
        image_record=image_record,
        teacher_run_id=teacher_run_id,
        model=model,
        prompt_version=config.prompt_version,
        schema_version=config.schema_version,
        raw_response_id=raw_response_id,
        existing_item_hashes=existing_item_hashes or set(),
        allow_duplicate_items=allow_duplicate_items,
    )
    rejected.extend(duplicate_rejections)
    return accepted, rejected


def iter_teacher_items(image_output: dict[str, Any]) -> list[tuple[int, str, dict[str, Any]]]:
    items: list[tuple[int, str, dict[str, Any]]] = []
    if isinstance(image_output.get("focus_items"), list) or isinstance(
        image_output.get("direct_items"), list
    ):
        for index, item in enumerate(image_output.get("focus_items") or []):
            if isinstance(item, dict):
                items.append((index, "focus", item))
        offset = len(items)
        for index, item in enumerate(image_output.get("direct_items") or []):
            if isinstance(item, dict):
                items.append((offset + index, "direct", item))
        return items
    for index, item in enumerate(image_output.get("items") or []):
        if isinstance(item, dict):
            items.append((index, infer_item_kind(item), item))
    return items


def validate_item(
    item: dict[str, Any],
    *,
    confidence_threshold: float = 0.75,
    schema_version: str | None = None,
    strict_visual_cue_schema: bool = False,
    item_kind: str | None = None,
) -> list[str]:
    reasons, _warnings = validate_item_with_warnings(
        item,
        confidence_threshold=confidence_threshold,
        schema_version=schema_version,
        strict_visual_cue_schema=strict_visual_cue_schema,
        item_kind=item_kind,
    )
    return reasons


def validate_item_with_warnings(
    item: dict[str, Any],
    *,
    confidence_threshold: float = 0.75,
    schema_version: str | None = None,
    strict_visual_cue_schema: bool = False,
    item_kind: str | None = None,
) -> tuple[list[str], list[str]]:
    kind = infer_item_kind(item, item_kind=item_kind)
    normalized = normalize_teacher_item(item, item_kind=kind)
    if kind == "direct":
        return validate_direct_item_with_warnings(
            normalized,
            confidence_threshold=confidence_threshold,
        )
    return validate_focus_item_with_warnings(
        normalized,
        confidence_threshold=confidence_threshold,
        schema_version=schema_version,
        strict_visual_cue_schema=strict_visual_cue_schema,
    )


def validate_focus_item_with_warnings(
    item: dict[str, Any],
    *,
    confidence_threshold: float,
    schema_version: str | None,
    strict_visual_cue_schema: bool,
) -> tuple[list[str], list[str]]:
    reasons = []
    warnings = []
    for field_name in (
        "question",
        "target",
        "evidence_description",
        "short_answer",
        "evidence_type",
        "locality",
        "answer_type",
        "visual_difficulty",
        "visibility",
        "target_leakage_risk",
        "evidence_specificity",
    ):
        if not str(item.get(field_name, "")).strip():
            reasons.append(f"missing_{field_name}")

    if item.get("need_focus") is not True:
        reasons.append("need_focus_not_true")
    if item.get("evidence_state") != "need_local_visual_evidence":
        reasons.append("invalid_evidence_state")
    if item.get("trajectory_type") != "single_focus":
        reasons.append("invalid_trajectory_type")

    confidence = item.get("confidence")
    if not isinstance(confidence, int | float) or confidence < confidence_threshold:
        reasons.append("low_confidence")
    if item.get("evidence_type") not in EVIDENCE_TYPES:
        reasons.append("invalid_evidence_type")
    if item.get("locality") not in LOCALITIES:
        reasons.append("invalid_locality")
    if item.get("answer_type") not in ANSWER_TYPES:
        reasons.append("invalid_answer_type")
    if item.get("answer_format") not in ANSWER_FORMATS:
        reasons.append("invalid_answer_format")
    if item.get("visual_difficulty") not in VISUAL_DIFFICULTIES:
        reasons.append("invalid_visual_difficulty")
    if item.get("visibility") not in VISIBILITIES:
        reasons.append("invalid_visibility")
    if item.get("target_leakage_risk") not in {"none", "low"}:
        reasons.append("target_leakage_risk_too_high")
    if item.get("evidence_specificity") != "specific":
        reasons.append("generic_evidence_specificity")

    target = str(item.get("target", ""))
    question = str(item.get("question", ""))
    evidence = str(item.get("evidence_description", ""))
    answer_values = sorted(
        {str(item.get("short_answer", "")).strip(), str(item.get("answer", "")).strip()} - {""}
    )
    target_word_count = len(_words(target))
    evidence_word_count = len(_words(evidence))
    if target_word_count < 4 or target_word_count > 22:
        reasons.append("target_length_out_of_range")
    if evidence_word_count < 4 or evidence_word_count > 45:
        reasons.append("evidence_description_length_out_of_range")
    if target_is_global(target):
        reasons.append("global_target")
    if target_is_forbidden_generic(target):
        reasons.append("generic_target")
    if evidence_is_generic(evidence):
        reasons.append("generic_evidence_description")
    if evidence_has_uncertain_reading(evidence):
        reasons.append("uncertain_evidence_description")
    for answer_value in answer_values:
        reasons.extend(leakage_reasons(question, target, answer_value))
    if contains_sensitive_identifier(" ".join([target, evidence, *answer_values])):
        reasons.append("sensitive_personal_identifier")

    style_reasons, style_warnings = validate_visual_cue_fields(
        item,
        schema_version=schema_version,
        strict_visual_cue_schema=strict_visual_cue_schema,
    )
    reasons.extend(style_reasons)
    warnings.extend(style_warnings)
    return sorted(set(reasons)), sorted(set(warnings))


def validate_direct_item_with_warnings(
    item: dict[str, Any],
    *,
    confidence_threshold: float,
) -> tuple[list[str], list[str]]:
    reasons = []
    for field_name in ("question", "answer", "evidence_type", "locality", "answer_type"):
        if not str(item.get(field_name, "")).strip():
            reasons.append(f"missing_{field_name}")
    if item.get("need_focus") is not False:
        reasons.append("need_focus_not_false")
    if item.get("evidence_state") != "sufficient_visual_evidence":
        reasons.append("invalid_evidence_state")
    if item.get("trajectory_type") != "direct_answer":
        reasons.append("invalid_trajectory_type")
    if str(item.get("target", "")):
        reasons.append("direct_target_not_empty")
    if normalize_target_style(item) != "none":
        reasons.append("direct_target_style_not_none")
    if normalize_target_cues(item):
        reasons.append("direct_target_cues_not_empty")
    confidence = item.get("confidence")
    if not isinstance(confidence, int | float) or confidence < confidence_threshold:
        reasons.append("low_confidence")
    if item.get("evidence_type") not in DIRECT_EVIDENCE_TYPES:
        reasons.append("invalid_evidence_type")
    if item.get("locality") not in DIRECT_LOCALITIES:
        reasons.append("invalid_locality")
    if item.get("answer_type") not in DIRECT_ANSWER_TYPES:
        reasons.append("invalid_answer_type")
    if item.get("answer_format") not in ANSWER_FORMATS:
        reasons.append("invalid_answer_format")
    if item.get("visual_difficulty") != "clear":
        reasons.append("invalid_visual_difficulty")
    if item.get("visibility") != "clear":
        reasons.append("invalid_visibility")
    if item.get("target_leakage_risk") != "none":
        reasons.append("target_leakage_risk_too_high")
    if item.get("evidence_specificity") != "specific":
        reasons.append("generic_evidence_specificity")
    question = str(item.get("question", ""))
    evidence = str(item.get("evidence_description", ""))
    answer = str(item.get("answer", ""))
    if direct_question_requires_focus(question):
        reasons.append("direct_question_requires_focus")
    if evidence and evidence_is_generic(evidence):
        reasons.append("generic_evidence_description")
    if contains_sensitive_identifier(" ".join([question, evidence, answer])):
        reasons.append("sensitive_personal_identifier")
    return sorted(set(reasons)), []


def validate_visual_cue_fields(
    item: dict[str, Any],
    *,
    schema_version: str | None,
    strict_visual_cue_schema: bool,
) -> tuple[list[str], list[str]]:
    reasons = []
    warnings = []
    requires_style_fields = schema_version in {SCHEMA_VERSION_VISUAL_CUE_V1, SCHEMA_VERSION_V3}
    has_style = "target_style" in item
    has_cues = "target_cues" in item
    target = str(item.get("target", ""))

    if requires_style_fields and not has_style:
        reasons.append("missing_target_style")
    if requires_style_fields and not has_cues:
        reasons.append("missing_target_cues")

    style = normalize_target_style(item)
    cues = normalize_target_cues(item)
    legacy_normalized = bool(item.get("_legacy_normalized_target_fields"))
    if has_style and str(item.get("target_style")) not in TARGET_STYLES:
        if not (legacy_normalized and str(item.get("target_style")) in {"none", "unknown"}):
            reasons.append("invalid_target_style")
    if has_cues:
        raw_cues = item.get("target_cues")
        if not isinstance(raw_cues, list):
            reasons.append("invalid_target_cues")
        elif any(cue not in TARGET_CUES for cue in raw_cues):
            if not legacy_normalized:
                reasons.append("invalid_target_cue")

    if style in {"visual_cue", "mixed"} and len(cues) < 2:
        warnings.append("visual_cue_target_has_fewer_than_two_cues")
        if schema_version == SCHEMA_VERSION_V3 or strict_visual_cue_schema and not cues:
            reasons.append("visual_cue_target_missing_cues")
    if visually_generic_target(target):
        warnings.append("visually_generic_target")
        if schema_version == SCHEMA_VERSION_V3 or strict_visual_cue_schema:
            reasons.append("visually_generic_target")
    if style == "semantic" and appears_visual_cue_based(target, cues):
        warnings.append("semantic_style_for_visual_cue_target")
        if strict_visual_cue_schema:
            reasons.append("semantic_style_for_visual_cue_target")
    return reasons, warnings


def infer_item_kind(item: dict[str, Any], item_kind: str | None = None) -> str:
    if item_kind in {"focus", "direct"}:
        return item_kind
    if item.get("item_type") in {"no_refocus_continue", "no_refocus_answer"}:
        return "direct"
    if item.get("item_type") in {"single_refocus", "multi_refocus"}:
        return "focus"
    if item.get("need_focus") is False or item.get("trajectory_type") == "direct_answer":
        return "direct"
    return "focus"


def normalize_teacher_item(item: dict[str, Any], *, item_kind: str | None = None) -> dict[str, Any]:
    kind = infer_item_kind(item, item_kind=item_kind)
    normalized = dict(item)
    if kind == "direct":
        answer = str(normalized.get("answer") or normalized.get("short_answer") or "").strip()
        normalized.setdefault("need_focus", False)
        normalized.setdefault("evidence_state", "sufficient_visual_evidence")
        normalized.setdefault("trajectory_type", "direct_answer")
        normalized["target"] = str(normalized.get("target") or "")
        if "target_style" not in normalized or "target_cues" not in normalized:
            normalized["_legacy_normalized_target_fields"] = True
        normalized.setdefault("target_style", "none")
        normalized.setdefault("target_cues", [])
        normalized.setdefault("evidence_description", "")
        normalized.setdefault("short_answer", answer)
        normalized.setdefault("answer", answer)
        normalized.setdefault("answer_format", infer_answer_format(normalized))
        normalized.setdefault("value_span_text", answer)
        normalized.setdefault("visual_difficulty", "clear")
        normalized.setdefault("visibility", "clear")
        normalized.setdefault("target_leakage_risk", "none")
        normalized.setdefault("evidence_specificity", "specific")
        return normalized

    answer = str(normalized.get("answer") or normalized.get("short_answer") or "").strip()
    normalized.setdefault("need_focus", True)
    normalized.setdefault("evidence_state", "need_local_visual_evidence")
    normalized.setdefault("trajectory_type", "single_focus")
    if "target_style" not in normalized or "target_cues" not in normalized:
        normalized["_legacy_normalized_target_fields"] = True
    normalized.setdefault("target_style", "unknown")
    normalized.setdefault("target_cues", [])
    normalized.setdefault("answer", answer)
    normalized.setdefault("answer_format", infer_answer_format(normalized))
    normalized.setdefault("value_span_text", answer)
    return normalized


def infer_answer_format(item: dict[str, Any]) -> str:
    answer_type = str(item.get("answer_type") or "")
    answer = str(item.get("answer") or item.get("short_answer") or "")
    if answer_type == "date":
        return "date"
    if answer_type in {"number", "count"}:
        return "numeric"
    if answer_type == "boolean_state" or normalize_text(answer) in {"yes", "no", "true", "false"}:
        return "boolean"
    return "short_text"


def deduplicate_items(
    candidates: list[tuple[int, dict[str, Any], list[str], str]],
    *,
    image_record: dict[str, Any],
    teacher_run_id: str,
    model: str,
    prompt_version: str,
    schema_version: str,
    raw_response_id: str | None,
    existing_item_hashes: set[str],
    allow_duplicate_items: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(
        candidates,
        key=lambda pair: (float(pair[1].get("confidence", 0.0)), -pair[0]),
        reverse=True,
    )
    seen_targets: set[str] = set()
    seen_evidence: set[str] = set()
    seen_direct_questions: set[str] = set()
    accepted_ranked = []
    rejected = []
    for index, item, warnings, item_kind in ranked:
        normalized_item = normalize_teacher_item(item, item_kind=item_kind)
        normalized_kind = infer_item_kind(normalized_item, item_kind=item_kind)
        reasons = []
        target_key = normalize_text(normalized_item.get("target", ""))
        evidence_type = str(normalized_item.get("evidence_type", ""))
        evidence_key = normalize_text(normalized_item.get("evidence_description", ""))
        direct_question_key = normalize_text(normalized_item.get("question", ""))
        content_hash = item_content_hash(image_record["stable_image_uid"], normalized_item)
        if normalized_kind == "direct":
            if direct_question_key in seen_direct_questions:
                reasons.append("duplicate_direct_question")
        else:
            if target_key in seen_targets:
                reasons.append("duplicate_target")
        if evidence_key and evidence_key in seen_evidence:
            reasons.append("duplicate_evidence_description")
        if content_hash in existing_item_hashes and not allow_duplicate_items:
            reasons.append("duplicate_existing_item_content_hash")
        if reasons:
            rejected.append(
                rejected_record(
                    image_record=image_record,
                    item=normalized_item,
                    rejection_reasons=reasons,
                    visual_cue_warnings=warnings,
                    teacher_run_id=teacher_run_id,
                    model=model,
                    prompt_version=prompt_version,
                    schema_version=schema_version,
                    raw_response_id=raw_response_id,
                    item_index=index,
                    item_kind=normalized_kind,
                )
            )
            continue
        if normalized_kind == "direct":
            seen_direct_questions.add(direct_question_key)
        else:
            seen_targets.add(target_key)
        if evidence_key:
            seen_evidence.add(evidence_key)
        accepted_ranked.append((index, normalized_item, warnings, normalized_kind))
    accepted_ranked.sort(key=lambda pair: pair[0])
    return [
        flatten_item(
            image_record=image_record,
            item=item,
            teacher_run_id=teacher_run_id,
            model=model,
            prompt_version=prompt_version,
            schema_version=schema_version,
            visual_cue_warnings=warnings,
            raw_response_id=raw_response_id,
            item_index=out_index,
            item_kind=item_kind,
        )
        for out_index, (_original_index, item, warnings, item_kind) in enumerate(accepted_ranked)
    ], rejected


def flatten_item(
    *,
    image_record: dict[str, Any],
    item: dict[str, Any],
    teacher_run_id: str,
    model: str,
    prompt_version: str = PROMPT_VERSION,
    schema_version: str = SCHEMA_VERSION,
    visual_cue_warnings: list[str] | None = None,
    raw_response_id: str | None = None,
    item_index: int = 0,
    item_kind: str | None = None,
) -> dict[str, Any]:
    stable_uid = image_record["stable_image_uid"]
    kind = infer_item_kind(item, item_kind=item_kind)
    normalized = normalize_teacher_item(item, item_kind=kind)
    record = {
        "uid": f"{teacher_run_id}:{stable_uid}:{item_index}",
        "teacher_run_id": teacher_run_id,
        "stable_image_uid": stable_uid,
        "image": image_record["image_path"],
        "image_id": stable_uid,
        "source_dataset": image_record.get("source_dataset"),
        "source_profile": image_record.get("source_profile"),
        "schema_version": schema_version,
        "teacher_prompt_version": prompt_version,
        "teacher_schema_version": schema_version,
        "need_focus": bool(normalized.get("need_focus")),
        "evidence_state": normalized.get("evidence_state"),
        "trajectory_type": normalized.get("trajectory_type"),
        "question": str(normalized.get("question", "")).strip(),
        "target": str(normalized.get("target", "")).strip(),
        "target_style": normalize_target_style(normalized),
        "target_cues": normalize_target_cues(normalized),
        "evidence_description": str(normalized.get("evidence_description", "")).strip(),
        "short_answer": str(normalized.get("short_answer", "")).strip(),
        "answer": str(normalized.get("answer", "")).strip(),
        "answer_format": normalized.get("answer_format"),
        "value_span_text": str(normalized.get("value_span_text", "")).strip(),
        "evidence_type": normalized.get("evidence_type"),
        "locality": normalized.get("locality"),
        "answer_type": normalized.get("answer_type"),
        "visual_difficulty": normalized.get("visual_difficulty"),
        "visibility": normalized.get("visibility"),
        "target_leakage_risk": normalized.get("target_leakage_risk"),
        "evidence_specificity": normalized.get("evidence_specificity"),
        "confidence": float(normalized["confidence"]),
        "visual_cue_warnings": visual_cue_warnings or [],
        "teacher_model": model,
        "raw_response_id": raw_response_id,
        "item_content_hash": item_content_hash(stable_uid, normalized),
        "created_at": now_iso(),
    }
    return record


def rejected_record(
    *,
    image_record: dict[str, Any],
    item: dict[str, Any],
    rejection_reasons: list[str],
    visual_cue_warnings: list[str] | None = None,
    teacher_run_id: str,
    model: str,
    prompt_version: str = PROMPT_VERSION,
    schema_version: str = SCHEMA_VERSION,
    raw_response_id: str | None,
    item_index: int,
    item_kind: str | None = None,
) -> dict[str, Any]:
    kind = infer_item_kind(item, item_kind=item_kind)
    normalized = normalize_teacher_item(item, item_kind=kind)
    return {
        "teacher_run_id": teacher_run_id,
        "stable_image_uid": image_record["stable_image_uid"],
        "image": image_record["image_path"],
        "image_id": image_record["stable_image_uid"],
        "source_dataset": image_record.get("source_dataset"),
        "source_profile": image_record.get("source_profile"),
        "schema_version": schema_version,
        "teacher_prompt_version": prompt_version,
        "teacher_schema_version": schema_version,
        "need_focus": bool(normalized.get("need_focus")),
        "evidence_state": normalized.get("evidence_state"),
        "trajectory_type": normalized.get("trajectory_type"),
        "item_index": item_index,
        "item": normalized,
        "target_style": normalize_target_style(normalized),
        "target_cues": normalize_target_cues(normalized),
        "rejection_reasons": rejection_reasons,
        "visual_cue_warnings": visual_cue_warnings or [],
        "teacher_model": model,
        "raw_response_id": raw_response_id,
        "created_at": now_iso(),
    }


def parse_and_validate_response(
    *,
    image_record: dict[str, Any],
    response: TeacherResponse,
    teacher_run_id: str,
    config: GenerationConfig,
    model: str,
    existing_item_hashes: set[str] | None = None,
    allow_duplicate_items: bool = False,
) -> ParsedImageResult:
    try:
        image_output = parse_teacher_output_text(response.output_text)
    except Exception as exc:  # noqa: BLE001
        return ParsedImageResult(
            image_record=image_record,
            image_output=None,
            accepted_items=[],
            rejected_items=[],
            status="failed_parse",
            error=f"{type(exc).__name__}: {exc}",
            raw_response_id=response.response_id,
            usage=response.usage,
        )
    accepted, rejected = validate_image_level_output(
        image_output,
        image_record,
        config=config,
        teacher_run_id=teacher_run_id,
        model=model,
        raw_response_id=response.response_id,
        existing_item_hashes=existing_item_hashes,
        allow_duplicate_items=allow_duplicate_items,
    )
    status = "succeeded" if accepted else "failed_validation"
    return ParsedImageResult(
        image_record=image_record,
        image_output=image_output,
        accepted_items=accepted,
        rejected_items=rejected,
        status=status,
        error=None if accepted else "no_accepted_items",
        raw_response_id=response.response_id,
        usage=response.usage,
    )


def make_ledger_entry(
    *,
    teacher_run_id: str,
    image_record: dict[str, Any],
    status: str,
    accepted_item_count: int = 0,
    rejected_item_count: int = 0,
    request_custom_id: str | None = None,
    model: str = DEFAULT_MODEL,
    image_detail: str = DEFAULT_IMAGE_DETAIL,
    prompt_version: str = PROMPT_VERSION,
    schema_version: str = SCHEMA_VERSION,
    usage: dict[str, int] | None = None,
    error: str | None = None,
    num_focus_items_raw: int = 0,
    num_direct_items_raw: int = 0,
) -> dict[str, Any]:
    return {
        "teacher_run_id": teacher_run_id,
        "stable_image_uid": image_record["stable_image_uid"],
        "image": image_record.get("image_path"),
        "image_id": image_record.get("stable_image_uid"),
        "source_dataset": image_record.get("source_dataset"),
        "source_profile": image_record.get("source_profile"),
        "image_path": image_record.get("image_path"),
        "status": status,
        "accepted_item_count": accepted_item_count,
        "rejected_item_count": rejected_item_count,
        "num_focus_items_raw": int(num_focus_items_raw),
        "num_direct_items_raw": int(num_direct_items_raw),
        "num_rows_accepted": int(accepted_item_count),
        "num_rows_rejected": int(rejected_item_count),
        "request_custom_id": request_custom_id,
        "model": model,
        "teacher_model": model,
        "image_detail": image_detail,
        "prompt_version": prompt_version,
        "teacher_prompt_version": prompt_version,
        "schema_version": schema_version,
        "created_at": now_iso(),
        "usage": usage or zero_usage(),
        "error": error,
    }


def count_raw_item_kinds(image_output: dict[str, Any] | None) -> dict[str, int]:
    counts = {"focus": 0, "direct": 0}
    if image_output is None:
        return counts
    for _index, item_kind, _item in iter_teacher_items(image_output):
        counts[infer_item_kind({}, item_kind=item_kind)] += 1
    return counts


def read_successful_ledger_uids(
    ledger_path: str | Path,
    *,
    prompt_version: str | None = None,
    schema_version: str | None = None,
    model: str | None = None,
) -> set[str]:
    successful = set()
    for record in read_jsonl(ledger_path):
        if record.get("status") not in LEDGER_TERMINAL_SUCCESS:
            continue
        if prompt_version is not None and record.get("prompt_version") != prompt_version:
            continue
        if schema_version is not None and record.get("schema_version") != schema_version:
            continue
        if model is not None and record.get("model") != model:
            continue
        successful.add(record["stable_image_uid"])
    return successful


def load_existing_item_hashes(generated_root: str | Path) -> set[str]:
    root = Path(generated_root)
    hashes = set()
    for path in root.glob("runs/*/final/tgvf_teacher_items.accepted.jsonl"):
        for record in read_jsonl(path):
            if record.get("item_content_hash"):
                hashes.add(record["item_content_hash"])
    latest = root / "tgvf_teacher_items.latest.jsonl"
    for record in read_jsonl(latest):
        if record.get("item_content_hash"):
            hashes.add(record["item_content_hash"])
    return hashes


def resume_sync(
    *,
    selection: str | Path,
    project_root: str | Path,
    run_id: str,
    openai_config: OpenAIConfig,
    generation_config: GenerationConfig,
    limit_images: int | None = None,
    concurrency: int = 1,
    allow_regenerate: bool = False,
    allow_duplicate_items: bool = False,
    fail_fast: bool = False,
    client: Any | None = None,
    wandb_logger: WandbLogger | None = None,
) -> dict[str, Any]:
    validate_teacher_version_pair(
        generation_config.prompt_version,
        generation_config.schema_version,
        allow_mismatch=generation_config.allow_prompt_schema_mismatch,
    )
    run_paths = ensure_run_layout(project_root, run_id)
    run_config = TeacherRunConfig(
        teacher_run_id=run_id,
        selection_path=str(selection),
        project_root=str(project_root),
        openai=openai_config,
        generation=generation_config,
    )
    initialize_run_files(run_paths, run_config, selection)
    selected = read_jsonl(selection)
    if limit_images is not None:
        selected = selected[:limit_images]
    ledger_path = generated_root(project_root) / "teacher_generation_ledger.jsonl"
    successful = (
        set()
        if allow_regenerate
        else read_successful_ledger_uids(
            ledger_path,
            prompt_version=generation_config.prompt_version,
            schema_version=generation_config.schema_version,
            model=openai_config.model,
        )
    )
    existing_hashes = set() if allow_duplicate_items else load_existing_item_hashes(
        generated_root(project_root)
    )
    client = client or OpenAITeacherClient()

    candidates = []
    skipped_existing = 0
    for record in selected:
        if record["stable_image_uid"] in successful:
            skipped_existing += 1
            append_jsonl(
                ledger_path,
                make_ledger_entry(
                    teacher_run_id=run_id,
                    image_record=record,
                    status="skipped_existing",
                    model=openai_config.model,
                    image_detail=openai_config.image_detail,
                    prompt_version=generation_config.prompt_version,
                    schema_version=generation_config.schema_version,
                ),
            )
            continue
        if not Path(record["image_path"]).exists():
            append_jsonl(
                ledger_path,
                make_ledger_entry(
                    teacher_run_id=run_id,
                    image_record=record,
                    status="skipped_missing_image",
                    model=openai_config.model,
                    image_detail=openai_config.image_detail,
                    prompt_version=generation_config.prompt_version,
                    schema_version=generation_config.schema_version,
                    error="missing image file",
                ),
            )
            continue
        candidates.append(record)

    existing_accepted_records = read_jsonl(run_paths["accepted"])
    existing_accepted_items = len(existing_accepted_records)
    source_mix_targets = source_mix_quotas(generation_config.target_accepted_samples)
    summary = {
        "skipped_existing_images": skipped_existing,
        "newly_requested_images": 0,
        "successful_teacher_calls": 0,
        "failed_calls": 0,
        "existing_accepted_items": existing_accepted_items,
        "new_accepted_items": 0,
        "accepted_items": existing_accepted_items,
        "rejected_items": len(read_jsonl(run_paths["rejected"])),
        "source_mix_targets": source_mix_targets,
        "source_mix_counts": count_source_mix(existing_accepted_records),
    }
    progress = ProgressReporter(
        total_candidates=len(candidates),
        target_accepted_samples=generation_config.target_accepted_samples,
        existing_accepted_items=existing_accepted_items,
        skipped_existing_images=skipped_existing,
        wandb_logger=wandb_logger,
    )
    progress.print_start(summary)
    if concurrency <= 1:
        for record in candidates:
            if source_mix_targets_met(summary):
                break
            if source_mix_bucket(record) is not None and source_mix_bucket_filled(
                summary, source_mix_bucket(record)
            ):
                continue
            result = _process_one_sync(
                record,
                client=client,
                openai_config=openai_config,
                generation_config=generation_config,
                run_id=run_id,
                run_paths=run_paths,
                existing_item_hashes=existing_hashes,
                allow_duplicate_items=allow_duplicate_items,
            )
            _update_sync_summary(summary, result)
            progress.print_update(summary, result)
            existing_hashes.update(item["item_content_hash"] for item in result.accepted_items)
            if fail_fast and result.status.startswith("failed"):
                raise RuntimeError(result.error or result.status)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            for start in range(0, len(candidates), concurrency):
                if source_mix_targets_met(summary):
                    break
                chunk = [
                    record
                    for record in candidates[start : start + concurrency]
                    if source_mix_bucket(record) is None
                    or not source_mix_bucket_filled(summary, source_mix_bucket(record))
                ]
                if not chunk:
                    continue
                futures = [
                    executor.submit(
                        _process_one_sync,
                        record,
                        client=client,
                        openai_config=openai_config,
                        generation_config=generation_config,
                        run_id=run_id,
                        run_paths=run_paths,
                        existing_item_hashes=set(existing_hashes),
                        allow_duplicate_items=allow_duplicate_items,
                    )
                    for record in chunk
                ]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    _update_sync_summary(summary, result)
                    progress.print_update(summary, result)
                    existing_hashes.update(
                        item["item_content_hash"] for item in result.accepted_items
                    )
                    if fail_fast and result.status.startswith("failed"):
                        raise RuntimeError(result.error or result.status)
    update_latest_copy(run_paths["accepted"], generated_root(project_root))
    report = write_run_reports(run_paths=run_paths, run_id=run_id, summary=summary)
    if wandb_logger is not None and wandb_logger.enabled:
        wandb_logger.log(flatten_metrics(report, prefix="teacher/final"))
        wandb_logger.update_summary({"teacher": report})
    return report


class ProgressReporter:
    def __init__(
        self,
        *,
        total_candidates: int,
        target_accepted_samples: int,
        existing_accepted_items: int,
        skipped_existing_images: int,
        wandb_logger: WandbLogger | None = None,
    ) -> None:
        self.total_candidates = total_candidates
        self.target_accepted_samples = target_accepted_samples
        self.existing_accepted_items = existing_accepted_items
        self.skipped_existing_images = skipped_existing_images
        self.wandb_logger = wandb_logger
        self.start_time = time.monotonic()

    def print_start(self, summary: dict[str, Any]) -> None:
        self._print(
            "start "
            f"candidates={self.total_candidates} "
            f"skipped_existing={self.skipped_existing_images} "
            f"existing_accepted={self.existing_accepted_items} "
            f"target_accepted={self.target_accepted_samples}"
        )
        if self.wandb_logger is not None and self.wandb_logger.enabled:
            self.wandb_logger.log(
                {
                    "teacher/candidates": self.total_candidates,
                    "teacher/skipped_existing_images": self.skipped_existing_images,
                    "teacher/existing_accepted_items": self.existing_accepted_items,
                    "teacher/target_accepted_samples": self.target_accepted_samples,
                },
                step=0,
            )

    def print_update(self, summary: dict[str, Any], result: ParsedImageResult) -> None:
        done = int(summary.get("newly_requested_images", 0))
        accepted = int(summary.get("accepted_items", 0))
        remaining_candidates = max(self.total_candidates - done, 0)
        remaining_accepted = max(self.target_accepted_samples - accepted, 0)
        elapsed = max(time.monotonic() - self.start_time, 1e-6)
        images_per_minute = done / elapsed * 60.0
        eta = (remaining_candidates / images_per_minute) if images_per_minute > 0 else 0.0
        self._print(
            "progress "
            f"images={done}/{self.total_candidates} "
            f"ok={summary.get('successful_teacher_calls', 0)} "
            f"failed={summary.get('failed_calls', 0)} "
            f"accepted={accepted}/{self.target_accepted_samples} "
            f"new_accepted={summary.get('new_accepted_items', 0)} "
            f"rejected={summary.get('rejected_items', 0)} "
            f"remaining_accept={remaining_accepted} "
            f"last={result.image_record.get('source_dataset')}:{result.status} "
            f"last_items={len(result.accepted_items)} "
            f"rate={images_per_minute:.2f}img/min "
            f"eta_by_images={eta:.1f}min"
        )
        if self.wandb_logger is not None and self.wandb_logger.enabled:
            dataset = str(result.image_record.get("source_dataset") or "unknown")
            status = str(result.status or "unknown")
            usage = result.usage or zero_usage()
            self.wandb_logger.log(
                {
                    "teacher/images_processed": done,
                    "teacher/total_candidates": self.total_candidates,
                    "teacher/successful_calls": summary.get("successful_teacher_calls", 0),
                    "teacher/failed_calls": summary.get("failed_calls", 0),
                    "teacher/accepted_items": accepted,
                    "teacher/new_accepted_items": summary.get("new_accepted_items", 0),
                    "teacher/rejected_items": summary.get("rejected_items", 0),
                    "teacher/remaining_accepted_items": remaining_accepted,
                    "teacher/last_accepted_items": len(result.accepted_items),
                    "teacher/last_rejected_items": len(result.rejected_items),
                    "teacher/rate_images_per_minute": images_per_minute,
                    "teacher/eta_by_images_minutes": eta,
                    "teacher/usage_input_tokens": usage.get("input_tokens", 0),
                    "teacher/usage_output_tokens": usage.get("output_tokens", 0),
                    "teacher/usage_total_tokens": usage.get("total_tokens", 0),
                    f"teacher/source_dataset/{dataset}": 1,
                    f"teacher/status/{status}": 1,
                },
                step=done,
            )

    def _print(self, message: str) -> None:
        print(f"[tgvf-teacher] {now_iso()} {message}", file=sys.stderr, flush=True)


def _process_one_sync(
    image_record: dict[str, Any],
    *,
    client: Any,
    openai_config: OpenAIConfig,
    generation_config: GenerationConfig,
    run_id: str,
    run_paths: dict[str, Path],
    existing_item_hashes: set[str],
    allow_duplicate_items: bool,
) -> ParsedImageResult:
    custom_id = request_custom_id(run_id, image_record["stable_image_uid"])
    try:
        image_reference = image_to_data_url(image_record["image_path"])
        payload = build_responses_payload(
            image_record=image_record,
            image_reference=image_reference,
            openai_config=openai_config,
            generation_config=generation_config,
            source_context=source_context_from_record(image_record),
        )
        response = call_teacher_with_retries(
            client=client,
            payload=payload,
            config=openai_config,
        )
        append_jsonl(
            run_paths["responses"],
            {
                "custom_id": custom_id,
                "stable_image_uid": image_record["stable_image_uid"],
                "response": response.raw_response,
            },
        )
        parsed = parse_and_validate_response(
            image_record=image_record,
            response=response,
            teacher_run_id=run_id,
            config=generation_config,
            model=openai_config.model,
            existing_item_hashes=existing_item_hashes,
            allow_duplicate_items=allow_duplicate_items,
        )
    except Exception as exc:  # noqa: BLE001 - per-image failure boundary.
        error = f"{type(exc).__name__}: {exc}"
        append_jsonl(
            run_paths["errors"],
            {
                "custom_id": custom_id,
                "stable_image_uid": image_record["stable_image_uid"],
                "error": error,
            },
        )
        parsed = ParsedImageResult(
            image_record=image_record,
            image_output=None,
            accepted_items=[],
            rejected_items=[],
            status="failed_api" if is_retryable_error(exc) else "failed_validation",
            error=error,
            raw_response_id=None,
            usage=zero_usage(),
        )
    persist_parsed_result(
        parsed,
        run_paths=run_paths,
        run_id=run_id,
        custom_id=custom_id,
        model=openai_config.model,
        image_detail=openai_config.image_detail,
        prompt_version=generation_config.prompt_version,
        schema_version=generation_config.schema_version,
    )
    return parsed


def persist_parsed_result(
    parsed: ParsedImageResult,
    *,
    run_paths: dict[str, Path],
    run_id: str,
    custom_id: str,
    model: str,
    image_detail: str,
    prompt_version: str = PROMPT_VERSION,
    schema_version: str = SCHEMA_VERSION,
) -> None:
    if parsed.image_output is not None:
        append_jsonl(
            run_paths["raw_items"],
            {
                "stable_image_uid": parsed.image_record["stable_image_uid"],
                "raw_response_id": parsed.raw_response_id,
                "image_output": parsed.image_output,
            },
        )
        append_jsonl(
            run_paths["validated_items"],
            {
                "stable_image_uid": parsed.image_record["stable_image_uid"],
                "raw_response_id": parsed.raw_response_id,
                "status": parsed.status,
                "accepted_item_count": len(parsed.accepted_items),
                "rejected_item_count": len(parsed.rejected_items),
                "accepted_uids": [item["uid"] for item in parsed.accepted_items],
                "rejection_reasons": [
                    reason
                    for item in parsed.rejected_items
                    for reason in item.get("rejection_reasons", [])
                ],
                "visual_cue_warnings": [
                    warning
                    for item in [*parsed.accepted_items, *parsed.rejected_items]
                    for warning in item.get("visual_cue_warnings", [])
                ],
            },
        )
    for record in parsed.accepted_items:
        append_jsonl(run_paths["accepted"], record)
    for record in parsed.rejected_items:
        append_jsonl(run_paths["rejected"], record)
    raw_kind_counts = count_raw_item_kinds(parsed.image_output)
    append_jsonl(
        run_paths["ledger"],
        make_ledger_entry(
            teacher_run_id=run_id,
            image_record=parsed.image_record,
            status=parsed.status,
            accepted_item_count=len(parsed.accepted_items),
            rejected_item_count=len(parsed.rejected_items),
            request_custom_id=custom_id,
            model=model,
            image_detail=image_detail,
            prompt_version=prompt_version,
            schema_version=schema_version,
            usage=parsed.usage,
            error=parsed.error,
            num_focus_items_raw=raw_kind_counts["focus"],
            num_direct_items_raw=raw_kind_counts["direct"],
        ),
    )


def prepare_batch(
    *,
    selection: str | Path,
    project_root: str | Path,
    run_id: str,
    openai_config: OpenAIConfig,
    generation_config: GenerationConfig | None = None,
    target_accepted_samples: int = DEFAULT_TARGET_ACCEPTED_SAMPLES,
    limit_images: int | None = None,
    max_request_file_mb: int = 90,
    allow_regenerate: bool = False,
) -> dict[str, Any]:
    run_paths = ensure_run_layout(project_root, run_id)
    generation_config = generation_config or GenerationConfig(
        target_accepted_samples=target_accepted_samples
    )
    validate_teacher_version_pair(
        generation_config.prompt_version,
        generation_config.schema_version,
        allow_mismatch=generation_config.allow_prompt_schema_mismatch,
    )
    run_config = TeacherRunConfig(
        teacher_run_id=run_id,
        selection_path=str(selection),
        project_root=str(project_root),
        openai=openai_config,
        generation=generation_config,
        image_backend="batch_base64_chunked",
    )
    initialize_run_files(run_paths, run_config, selection)
    selected = read_jsonl(selection)
    if limit_images is not None:
        selected = selected[:limit_images]
    successful = (
        set()
        if allow_regenerate
        else read_successful_ledger_uids(
            generated_root(project_root) / "teacher_generation_ledger.jsonl",
            prompt_version=generation_config.prompt_version,
            schema_version=generation_config.schema_version,
            model=openai_config.model,
        )
    )

    request_index = []
    request_files = []
    max_bytes = max_request_file_mb * 1024 * 1024
    part_index = 0
    current_path = run_paths["requests"] / f"batch_requests_part_{part_index:03d}.jsonl"
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_handle = current_path.open("w")
    current_bytes = 0
    try:
        for image_record in selected:
            if image_record.get("stable_image_uid") in successful:
                continue
            if not Path(image_record["image_path"]).exists():
                continue
            custom_id = request_custom_id(run_id, image_record["stable_image_uid"])
            body = build_responses_payload(
                image_record=image_record,
                image_reference=image_to_data_url(image_record["image_path"]),
                openai_config=openai_config,
                generation_config=generation_config,
                source_context=source_context_from_record(image_record),
            )
            line = json.dumps(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                },
                sort_keys=True,
            )
            line_bytes = len(line.encode("utf-8")) + 1
            if current_bytes and current_bytes + line_bytes > max_bytes:
                current_handle.close()
                request_files.append(current_path)
                part_index += 1
                current_path = run_paths["requests"] / f"batch_requests_part_{part_index:03d}.jsonl"
                current_handle = current_path.open("w")
                current_bytes = 0
            current_handle.write(line + "\n")
            current_handle.flush()
            current_bytes += line_bytes
            request_index.append(
                {
                    "custom_id": custom_id,
                    "stable_image_uid": image_record["stable_image_uid"],
                    "image_path": image_record["image_path"],
                    "source_dataset": image_record.get("source_dataset"),
                    "source_profile": image_record.get("source_profile"),
                }
            )
    finally:
        current_handle.close()
    if current_bytes or not request_files:
        request_files.append(current_path)
    write_jsonl(request_index, run_paths["request_index"])
    preview = [json.loads(line) for line in request_files[0].read_text().splitlines()[:5]]
    write_jsonl(preview, run_paths["requests"] / "sync_requests_preview.jsonl")
    report = {
        "teacher_run_id": run_id,
        "request_file_count": len(request_files),
        "request_files": [str(path) for path in request_files if path.exists()],
        "request_count": len(request_index),
        "skipped_existing_images": len(successful),
        "target_accepted_samples": generation_config.target_accepted_samples,
        "prompt_version": generation_config.prompt_version,
        "schema_version": generation_config.schema_version,
        "model": openai_config.model,
        "image_detail": openai_config.image_detail,
        "created_at": now_iso(),
    }
    write_json(run_paths["reports"] / "run_state.json", report)
    return report


def submit_batch(
    *,
    run_dir: str | Path,
    client: OpenAITeacherClient | None = None,
    completion_window: str = "24h",
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    client = client or OpenAITeacherClient()
    request_files = sorted((run_dir / "requests").glob("batch_requests_part_*.jsonl"))
    batches = []
    for request_file in request_files:
        uploaded = client.upload_batch_file(request_file)
        file_id = getattr(uploaded, "id", None) or _to_plain_dict(uploaded).get("id")
        batch = client.create_batch(input_file_id=file_id, completion_window=completion_window)
        batch_data = _to_plain_dict(batch)
        batches.append(
            {
                "request_file": str(request_file),
                "input_file_id": file_id,
                "batch_id": batch_data.get("id"),
                "status": batch_data.get("status"),
                "created_at": now_iso(),
            }
        )
    state = load_json(run_dir / "reports" / "run_state.json", default={})
    state["batches"] = batches
    write_json(run_dir / "reports" / "run_state.json", state)
    return state


def poll_batch(*, run_dir: str | Path, client: OpenAITeacherClient | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    client = client or OpenAITeacherClient()
    state = load_json(run_dir / "reports" / "run_state.json", default={})
    batches = []
    for batch_info in state.get("batches", []):
        batch = client.retrieve_batch(batch_info["batch_id"])
        batch_data = _to_plain_dict(batch)
        batch_info = {**batch_info, **batch_data, "polled_at": now_iso()}
        if batch_data.get("output_file_id"):
            output_text = client.download_file_text(batch_data["output_file_id"])
            output_path = run_dir / "raw_responses" / f"{batch_info['batch_id']}.jsonl"
            output_path.write_text(output_text)
            batch_info["downloaded_output_path"] = str(output_path)
        if batch_data.get("error_file_id"):
            error_text = client.download_file_text(batch_data["error_file_id"])
            error_path = run_dir / "raw_responses" / f"{batch_info['batch_id']}.errors.jsonl"
            error_path.write_text(error_text)
            batch_info["downloaded_error_path"] = str(error_path)
        batches.append(batch_info)
    state["batches"] = batches
    write_json(run_dir / "reports" / "run_state.json", state)
    return state


def parse_batch_results(
    *,
    run_dir: str | Path,
    allow_duplicate_items: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config = load_run_config(run_dir)
    run_paths = run_layout_from_run_dir(run_dir)
    request_index = {
        record["custom_id"]: record
        for record in read_jsonl(run_paths["request_index"])
    }
    generated = run_dir.parents[1]
    existing_hashes = set() if allow_duplicate_items else load_existing_item_hashes(generated)
    response_files = sorted((run_dir / "raw_responses").glob("*.jsonl"))
    summary = {
        "skipped_existing_images": 0,
        "newly_requested_images": 0,
        "successful_teacher_calls": 0,
        "failed_calls": 0,
        "accepted_items": 0,
        "rejected_items": 0,
    }
    for response_file in response_files:
        if response_file.name.endswith(".errors.jsonl"):
            continue
        for line in response_file.read_text().splitlines():
            if not line.strip():
                continue
            batch_line = json.loads(line)
            custom_id = batch_line.get("custom_id")
            image_record = request_index.get(custom_id)
            if image_record is None:
                continue
            error = batch_line.get("error")
            if error:
                parsed = ParsedImageResult(
                    image_record=image_record,
                    image_output=None,
                    accepted_items=[],
                    rejected_items=[],
                    status="failed_api",
                    error=json.dumps(error, sort_keys=True),
                    raw_response_id=None,
                    usage=zero_usage(),
                )
            else:
                body = (batch_line.get("response") or {}).get("body") or {}
                response = TeacherResponse(
                    response_id=body.get("id"),
                    output_text=body.get("output_text") or extract_output_text(body),
                    raw_response=body,
                    usage=normalize_usage(body.get("usage")),
                )
                parsed = parse_and_validate_response(
                    image_record=image_record,
                    response=response,
                    teacher_run_id=config.teacher_run_id,
                    config=config.generation,
                    model=config.openai.model,
                    existing_item_hashes=existing_hashes,
                    allow_duplicate_items=allow_duplicate_items,
                )
            persist_parsed_result(
                parsed,
                run_paths=run_paths,
                run_id=config.teacher_run_id,
                custom_id=custom_id,
                model=config.openai.model,
                image_detail=config.openai.image_detail,
                prompt_version=config.generation.prompt_version,
                schema_version=config.generation.schema_version,
            )
            _update_sync_summary(summary, parsed)
            existing_hashes.update(item["item_content_hash"] for item in parsed.accepted_items)
    update_latest_copy(run_paths["accepted"], generated)
    return write_run_reports(run_paths=run_paths, run_id=config.teacher_run_id, summary=summary)


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    run_paths = run_layout_from_run_dir(run_dir)
    accepted = read_jsonl(run_paths["accepted"])
    rejected = read_jsonl(run_paths["rejected"])
    ledger = read_jsonl(run_paths["ledger"])
    summary = {
        "run_dir": str(run_dir),
        "accepted_items": len(accepted),
        "rejected_items": len(rejected),
        "ledger_entries": len(ledger),
        "ledger_status_counts": dict(Counter(record.get("status") for record in ledger)),
        "evidence_type_distribution": dict(
            Counter(
                evidence_type
                for record in accepted
                for evidence_type in (
                    record.get("evidence_types")
                    if isinstance(record.get("evidence_types"), list)
                    else [record.get("evidence_type")]
                )
                if evidence_type
            )
        ),
        "source_profile_distribution": dict(
            Counter(record.get("source_profile") for record in accepted)
        ),
        "source_dataset_distribution": dict(
            Counter(record.get("source_dataset") for record in accepted)
        ),
        "target_style_distribution": dict(
            Counter(normalize_target_style(record) for record in accepted)
        ),
        "target_cues_distribution": dict(
            Counter(cue for record in accepted for cue in normalize_target_cues(record))
        ),
        "answer_format_distribution": dict(
            Counter(record.get("answer_format") for record in accepted)
        ),
        "trajectory_type_distribution": dict(
            Counter(record.get("trajectory_type") for record in accepted)
        ),
        "focus_item_count": sum(1 for record in accepted if record.get("need_focus") is True),
        "direct_item_count": sum(1 for record in accepted if record.get("need_focus") is False),
        "image_group_size_histogram": image_group_size_histogram(accepted),
    }
    return summary


def write_run_reports(
    *,
    run_paths: dict[str, Path],
    run_id: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    accepted = read_jsonl(run_paths["accepted"])
    rejected = read_jsonl(run_paths["rejected"])
    ledger = [
        record
        for record in read_jsonl(run_paths["ledger"])
        if record.get("teacher_run_id") == run_id
    ]
    quality = quality_report(accepted, rejected, ledger)
    usage = usage_report(ledger)
    generation_summary = {
        "teacher_run_id": run_id,
        **summary,
        "quality_report": quality,
        "usage_report": usage,
        "created_at": now_iso(),
    }
    write_json(run_paths["reports"] / "quality_filter_report.json", quality)
    write_json(run_paths["reports"] / "cost_usage_report.json", usage)
    write_json(run_paths["reports"] / "generation_summary.json", generation_summary)
    write_json(run_paths["reports"] / "run_state.json", generation_summary)
    return generation_summary


def quality_report(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    rejection_reasons = Counter(
        reason for record in rejected for reason in record.get("rejection_reasons", [])
    )
    visual_cue_warnings = Counter(
        warning
        for record in [*accepted, *rejected]
        for warning in record.get("visual_cue_warnings", [])
    )
    confidence_bins = Counter(_confidence_bin(record.get("confidence", 0.0)) for record in accepted)
    image_calls = [
        record for record in ledger if not str(record.get("status", "")).startswith("skipped")
    ]
    zero_item_images = sum(1 for record in ledger if record.get("accepted_item_count", 0) == 0)
    target_style_counts = Counter(normalize_target_style(record) for record in accepted)
    target_cue_counts = Counter(
        cue for record in accepted for cue in normalize_target_cues(record)
    )
    v4_focus_descriptor_cue_counts = Counter(
        cue for record in accepted for cue in record.get("focus_descriptor_cues", [])
    )
    need_focus_counts = Counter(bool(record.get("need_focus")) for record in accepted)
    trajectory_counts = Counter(record.get("trajectory_type") or "unknown" for record in accepted)
    item_type_counts = Counter(record.get("item_type") or "unknown" for record in accepted)
    question_type_counts = Counter(record.get("question_type") or "unknown" for record in accepted)
    focus_category_counts = Counter(record.get("focus_category") or "unknown" for record in accepted)
    num_focus_step_counts = Counter(str(record.get("num_focus_steps", 0)) for record in accepted)
    group_histogram = image_group_size_histogram(accepted)
    return {
        "total_image_calls": len(image_calls),
        "total_parsed_image_responses": sum(1 for record in ledger if record.get("request_custom_id")),
        "images_with_zero_items": zero_item_images,
        "raw_item_count": len(accepted) + len(rejected),
        "accepted_item_count": len(accepted),
        "rejected_item_count": len(rejected),
        "focus_item_count": int(need_focus_counts.get(True, 0)),
        "direct_item_count": int(need_focus_counts.get(False, 0)),
        "hard_control_item_count": int(trajectory_counts.get("hard_control", 0)),
        "trajectory_type_distribution": dict(trajectory_counts),
        "item_type_distribution": dict(item_type_counts),
        "question_type_distribution": dict(question_type_counts),
        "focus_category_distribution": dict(focus_category_counts),
        "num_focus_steps_distribution": dict(num_focus_step_counts),
        "rejection_reason_histogram": dict(rejection_reasons),
        "rejection_reason_distribution": dict(rejection_reasons),
        "visual_cue_warning_histogram": dict(visual_cue_warnings),
        "target_style_distribution": dict(target_style_counts),
        "target_cues_distribution": dict(target_cue_counts),
        "focus_descriptor_cues_distribution": dict(v4_focus_descriptor_cue_counts),
        "visual_cue_item_count": int(target_style_counts.get("visual_cue", 0)),
        "semantic_item_count": int(target_style_counts.get("semantic", 0)),
        "mixed_item_count": int(target_style_counts.get("mixed", 0)),
        "unknown_item_count": int(target_style_counts.get("unknown", 0)),
        "average_accepted_visual_cue_items_per_image": _safe_div(
            int(target_style_counts.get("visual_cue", 0)), len(image_calls)
        ),
        "evidence_type_distribution": dict(
            Counter(
                evidence_type
                for record in accepted
                for evidence_type in record_evidence_types(record)
            )
        ),
        "source_profile_distribution": dict(
            Counter(record.get("source_profile") for record in accepted)
        ),
        "source_dataset_distribution": dict(
            Counter(record.get("source_dataset") for record in accepted)
        ),
        "answer_format_distribution": dict(
            Counter(record.get("answer_format") for record in accepted)
        ),
        "confidence_distribution": dict(confidence_bins),
        "target_leakage_risk_distribution": dict(
            Counter(record.get("target_leakage_risk") for record in accepted)
        ),
        "average_items_per_image": _safe_div(len(accepted) + len(rejected), len(image_calls)),
        "average_accepted_items_per_image": _safe_div(len(accepted), len(image_calls)),
        "visual_difficulty_distribution": dict(
            Counter(record.get("visual_difficulty") for record in accepted)
        ),
        "samples_by_visual_difficulty": dict(
            Counter(record.get("visual_difficulty") for record in accepted)
        ),
        "samples_by_answer_type": dict(Counter(record.get("answer_type") for record in accepted)),
        "image_group_size_histogram": group_histogram,
    }


def record_evidence_types(record: dict[str, Any]) -> list[str]:
    evidence_types = record.get("evidence_types")
    if isinstance(evidence_types, list):
        return [str(value) for value in evidence_types if value]
    evidence_type = record.get("evidence_type")
    return [str(evidence_type)] if evidence_type else []


def image_group_size_histogram(records: list[dict[str, Any]]) -> dict[str, int]:
    by_image = Counter(record.get("image_id") or record.get("stable_image_uid") for record in records)
    return dict(Counter(str(size) for size in by_image.values()))


def usage_report(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    usage = zero_usage()
    for record in ledger:
        record_usage = record.get("usage") or {}
        for key in usage:
            usage[key] += int(record_usage.get(key) or 0)
    accepted_items = sum(int(record.get("accepted_item_count") or 0) for record in ledger)
    image_calls = sum(1 for record in ledger if record.get("request_custom_id"))
    return {
        **usage,
        "image_calls": image_calls,
        "accepted_items": accepted_items,
        "tokens_per_image": _safe_div(usage["total_tokens"], image_calls),
        "tokens_per_accepted_item": _safe_div(usage["total_tokens"], accepted_items),
    }


def ensure_run_layout(project_root: str | Path, run_id: str) -> dict[str, Path]:
    root = generated_root(project_root)
    run_dir = root / "runs" / run_id
    paths = run_layout_from_run_dir(run_dir)
    for key, path in paths.items():
        if key in {"run_dir", "ledger"}:
            continue
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    paths["ledger"].parent.mkdir(parents=True, exist_ok=True)
    return paths


def run_layout_from_run_dir(run_dir: str | Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    generated = run_dir.parents[1]
    return {
        "run_dir": run_dir,
        "requests": run_dir / "requests",
        "raw_responses": run_dir / "raw_responses",
        "parsed": run_dir / "parsed",
        "final": run_dir / "final",
        "reports": run_dir / "reports",
        "logs": run_dir / "logs",
        "config": run_dir / "config.yaml",
        "prompt": run_dir / "prompt.txt",
        "schema": run_dir / "schema.json",
        "selected_input": run_dir / "selected_images.input.jsonl",
        "request_index": run_dir / "requests" / "request_index.jsonl",
        "responses": run_dir / "raw_responses" / "responses.jsonl",
        "errors": run_dir / "raw_responses" / "errors.jsonl",
        "raw_items": run_dir / "parsed" / "image_level_items.raw.jsonl",
        "validated_items": run_dir / "parsed" / "image_level_items.validated.jsonl",
        "accepted": run_dir / "final" / "tgvf_teacher_items.accepted.jsonl",
        "rejected": run_dir / "final" / "tgvf_teacher_items.rejected.jsonl",
        "ledger": generated / "teacher_generation_ledger.jsonl",
    }


def initialize_run_files(
    run_paths: dict[str, Path],
    config: TeacherRunConfig,
    selection_path: str | Path,
) -> None:
    for directory_key in ("requests", "raw_responses", "parsed", "final", "reports", "logs"):
        run_paths[directory_key].mkdir(parents=True, exist_ok=True)
    write_yaml(run_paths["config"], asdict(config))
    run_paths["prompt"].write_text(get_teacher_prompt(config.generation.prompt_version))
    write_json(run_paths["schema"], get_teacher_schema(config.generation.schema_version))
    if Path(selection_path).exists():
        shutil.copyfile(selection_path, run_paths["selected_input"])


def load_run_config(run_dir: str | Path) -> TeacherRunConfig:
    data = read_yaml(Path(run_dir) / "config.yaml")
    return TeacherRunConfig(
        teacher_run_id=data["teacher_run_id"],
        selection_path=data["selection_path"],
        project_root=data["project_root"],
        openai=OpenAIConfig(**data.get("openai", {})),
        generation=GenerationConfig(**data.get("generation", {})),
        image_backend=data.get("image_backend", "sync_base64"),
        pricing=data.get("pricing", {}),
    )


def generated_root(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / "tgvf_teacher" / "generated"


def request_custom_id(run_id: str, stable_image_uid: str) -> str:
    safe_uid = re.sub(r"[^A-Za-z0-9_-]+", "_", stable_image_uid).strip("_")
    return f"{run_id}__{safe_uid}"


def image_to_data_url(path: str | Path) -> str:
    path = Path(path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def item_content_hash(stable_image_uid: str, item: dict[str, Any]) -> str:
    text = "\n".join(
        [
            stable_image_uid,
            normalize_text(str(item.get("target", ""))),
            normalize_text(str(item.get("evidence_description", ""))),
        ]
    )
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def leakage_reasons(question: str, target: str, short_answer: str) -> list[str]:
    if not short_answer.strip():
        return []
    reasons = []
    answer = normalize_text(short_answer)
    target_norm = normalize_text(target)
    question_norm = normalize_text(question)
    if not answer:
        return []
    answer_tokens = _words(short_answer)
    target_tokens = _words(target)
    question_tokens = _words(question)
    single_alpha = len(answer) == 1 and answer.isalpha()
    if single_alpha:
        if answer in target_tokens:
            reasons.append("short_answer_in_target")
        if answer in question_tokens:
            reasons.append("short_answer_in_question")
    else:
        if answer in target_norm:
            reasons.append("short_answer_in_target")
        if answer in question_norm:
            reasons.append("short_answer_in_question")
    answer_token_set = set(answer_tokens)
    target_token_set = set(target_tokens)
    if len(answer_token_set) >= 2:
        overlap = answer_token_set & target_token_set
        if len(overlap) / len(answer_token_set) >= 0.75:
            reasons.append("most_answer_tokens_in_target")
    if looks_numeric_date_or_code(short_answer):
        if answer in target_norm:
            reasons.append("strict_answer_value_in_target")
        if answer in question_norm:
            reasons.append("strict_answer_value_in_question")
    return reasons


def normalize_target_style(item: dict[str, Any]) -> str:
    style = str(item.get("target_style", "")).strip()
    return style if style in NORMALIZED_TARGET_STYLES else "unknown"


def normalize_target_cues(item: dict[str, Any]) -> list[str]:
    cues = item.get("target_cues", [])
    if not isinstance(cues, list):
        return []
    return [str(cue) for cue in cues if str(cue) in TARGET_CUES]


def visually_generic_target(target: str) -> bool:
    norm = normalize_text(target)
    generic_targets = {
        "the object",
        "an object",
        "the thing",
        "a thing",
        "something",
        "something green",
        "the patch",
        "the mark",
        "the region",
        "the area",
        "the shape",
        "the cluster",
        "the interesting area",
    }
    if norm in generic_targets:
        return True
    vague_heads = {"object", "thing", "patch", "mark", "region", "area", "shape", "cluster"}
    words = _words(target)
    if len(words) <= 3 and any(word in vague_heads for word in words):
        return True
    if norm.endswith(" below") or norm.endswith(" above"):
        return True
    return False


def appears_visual_cue_based(target: str, cues: list[str]) -> bool:
    if not cues:
        return False
    semantic_cues = {
        "nearby_anchor",
        "region_type",
        "text_like",
        "number_like",
        "date_like",
        "chart_like",
        "table_like",
        "symbol_like",
        "object_part",
        "spatial_relation",
    }
    visual_only_cues = set(cues) - semantic_cues
    if len(visual_only_cues) >= 2:
        return True
    norm = normalize_text(target)
    visual_words = {
        "small",
        "large",
        "tiny",
        "green",
        "blue",
        "red",
        "dark",
        "light",
        "round",
        "rectangular",
        "striped",
        "patterned",
        "shiny",
        "curved",
        "elongated",
    }
    return bool(visual_only_cues and set(norm.split()) & visual_words)


def target_is_global(target: str) -> bool:
    norm = normalize_text(target)
    global_phrases = {
        "the image",
        "the scene",
        "the picture",
        "the photo",
        "the whole image",
        "the entire image",
        "the full image",
    }
    return norm in global_phrases or norm.startswith("the overall ")


def target_is_forbidden_generic(target: str) -> bool:
    norm = normalize_text(target)
    return norm in {
        "the image",
        "the scene",
        "the object",
        "the answer",
        "something",
        "something green",
        "the important part",
        "relevant visual evidence needed to answer the question",
    }


def evidence_has_uncertain_reading(evidence: str) -> bool:
    norm = normalize_text(evidence)
    uncertain_phrases = {
        "maybe",
        "possibly",
        "appears to say",
        "seems to say",
        "might say",
        "looks like it says",
        "not sure",
        "unclear",
    }
    return any(phrase in norm for phrase in uncertain_phrases)


def direct_question_requires_focus(question: str) -> bool:
    norm = normalize_text(question)
    focus_markers = {
        "tiny text",
        "small text",
        "fine print",
        "below the barcode",
        "barcode",
        "serial number",
        "small number",
        "tiny number",
        "small blue circle",
        "receipt footer",
        "date printed",
        "chart label",
        "axis label",
        "legend entry",
        "table cell",
        "row and column",
        "fine texture",
        "small label",
        "local value",
    }
    return any(marker in norm for marker in focus_markers)


def evidence_is_generic(evidence: str) -> bool:
    norm = normalize_text(evidence)
    generic = {
        "there is some text",
        "there is text",
        "it is an object",
        "an object is visible",
        "the object is visible",
        "some text is visible",
    }
    if norm in generic:
        return True
    if len(set(_words(evidence))) <= 3:
        return True
    return False


def contains_sensitive_identifier(text: str) -> bool:
    patterns = [
        r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b(?:\d[ -]*?){13,19}\b",
        r"\b\d{1,6}\s+[A-Za-z0-9 .'-]+(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|blvd)\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def looks_numeric_date_or_code(text: str) -> bool:
    stripped = text.strip()
    return bool(
        re.search(r"\d", stripped)
        and (
            re.fullmatch(r"[\w./:-]+", stripped)
            or re.search(r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b", stripped)
        )
    )


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _words(text: str) -> list[str]:
    return [word for word in normalize_text(text).split() if word]


def source_mix_bucket(record: dict[str, Any]) -> str | None:
    dataset = record.get("source_dataset")
    if dataset == "visual_genome":
        return "visual_genome"
    if dataset in {"textvqa", "textocr"}:
        return "textvqa_textocr"
    if dataset == "docvqa":
        return "docvqa"
    if dataset == "chartqa":
        return "chartqa"
    return None


def source_mix_quotas(total_samples: int) -> dict[str, int]:
    raw = {bucket: total_samples * fraction for bucket, fraction in DEFAULT_SOURCE_MIX.items()}
    quotas = {bucket: int(value) for bucket, value in raw.items()}
    remainder = total_samples - sum(quotas.values())
    for bucket, _value in sorted(
        raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True
    )[:remainder]:
        quotas[bucket] += 1
    return quotas


def count_source_mix(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        bucket = source_mix_bucket(record)
        if bucket is not None:
            counts[bucket] += 1
    return dict(counts)


def source_mix_bucket_filled(summary: dict[str, Any], bucket: str | None) -> bool:
    if bucket is None:
        return False
    targets = summary.get("source_mix_targets") or {}
    counts = summary.get("source_mix_counts") or {}
    return int(counts.get(bucket, 0)) >= int(targets.get(bucket, 0))


def source_mix_targets_met(summary: dict[str, Any]) -> bool:
    targets = summary.get("source_mix_targets") or {}
    counts = summary.get("source_mix_counts") or {}
    if not targets:
        return int(summary.get("accepted_items", 0)) >= int(
            summary.get("target_accepted_samples", 0)
        )
    return all(int(counts.get(bucket, 0)) >= int(target) for bucket, target in targets.items())


def export_balanced_accepted(
    *,
    accepted: str | Path,
    output: str | Path,
    target_samples: int,
    seed: int = DEFAULT_SEED,
    allow_shortfall: bool = False,
) -> dict[str, Any]:
    records = read_jsonl(accepted)
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in DEFAULT_SOURCE_MIX}
    unknown = []
    for record in records:
        bucket = source_mix_bucket(record)
        if bucket is None:
            unknown.append(record)
        else:
            buckets[bucket].append(record)
    for bucket_records in buckets.values():
        bucket_records.sort(key=lambda item: item.get("uid") or item.get("item_content_hash") or "")
        rng.shuffle(bucket_records)
    quotas = source_mix_quotas(target_samples)
    shortfalls = {
        bucket: quota - len(buckets.get(bucket, []))
        for bucket, quota in quotas.items()
        if len(buckets.get(bucket, [])) < quota
    }
    if shortfalls and not allow_shortfall:
        raise RuntimeError(f"Not enough accepted items for requested source mix: {shortfalls}")
    selected = []
    for bucket, quota in quotas.items():
        selected.extend(buckets.get(bucket, [])[:quota])
    if len(selected) < target_samples and allow_shortfall:
        selected_ids = {id(record) for record in selected}
        leftovers = [
            record
            for bucket in DEFAULT_SOURCE_MIX
            for record in buckets.get(bucket, [])
            if id(record) not in selected_ids
        ]
        leftovers.extend(unknown)
        selected.extend(leftovers[: target_samples - len(selected)])
    selected = selected[:target_samples]
    output = Path(output)
    write_jsonl(selected, output)
    report = {
        "accepted": str(accepted),
        "output": str(output),
        "target_samples": target_samples,
        "written_samples": len(selected),
        "input_source_mix_counts": count_source_mix(records),
        "output_source_mix_counts": count_source_mix(selected),
        "source_mix_targets": quotas,
        "source_mix_shortfalls": shortfalls,
        "seed": seed,
        "created_at": now_iso(),
    }
    write_json(output.with_suffix(output.suffix + ".report.json"), report)
    return report


def zero_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(jsonable(record), sort_keys=True) + "\n")
        handle.flush()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True))
    tmp.replace(path)


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        path.write_text(yaml.safe_dump(jsonable(data), sort_keys=False))
    else:
        path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True))


def read_yaml(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text()
    if yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)


def update_latest_copy(accepted_path: Path, root: Path) -> None:
    if accepted_path.exists():
        root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(accepted_path, root / "tgvf_teacher_items.latest.jsonl")


def is_retryable_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in RETRYABLE_ERROR_MARKERS)


def _update_sync_summary(summary: dict[str, Any], parsed: ParsedImageResult) -> None:
    summary["newly_requested_images"] += 1
    if parsed.status == "succeeded":
        summary["successful_teacher_calls"] += 1
    else:
        summary["failed_calls"] += 1
    summary["new_accepted_items"] += len(parsed.accepted_items)
    summary["accepted_items"] += len(parsed.accepted_items)
    summary["rejected_items"] += len(parsed.rejected_items)
    source_mix_counts = summary.setdefault("source_mix_counts", {})
    for item in parsed.accepted_items:
        bucket = source_mix_bucket(item)
        if bucket is not None:
            source_mix_counts[bucket] = int(source_mix_counts.get(bucket, 0)) + 1


def _confidence_bin(confidence: float) -> str:
    value = float(confidence)
    if value >= 0.95:
        return "0.95_1.00"
    if value >= 0.90:
        return "0.90_0.95"
    if value >= 0.80:
        return "0.80_0.90"
    if value >= 0.75:
        return "0.75_0.80"
    return "below_0.75"


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {
        key: item
        for key, item in vars(value).items()
        if not key.startswith("_")
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command in {"smoke-sync", "resume-sync"}:
        wandb_logger = wandb_logger_from_args(
            args,
            job_type=f"teacher-{args.command}",
            config={
                "command": args.command,
                "selection": args.selection,
                "project_root": args.project_root,
                "run_id": args.run_id,
                "openai": vars(openai_config_from_args(args)),
                "generation": vars(generation_config_from_args(args)),
                "limit_images": args.limit_images,
                "concurrency": args.concurrency,
                "allow_regenerate": args.allow_regenerate,
                "allow_duplicate_items": args.allow_duplicate_items,
            },
        )
        try:
            report = resume_sync(
                selection=args.selection,
                project_root=args.project_root,
                run_id=args.run_id,
                openai_config=openai_config_from_args(args),
                generation_config=generation_config_from_args(args),
                limit_images=args.limit_images,
                concurrency=args.concurrency,
                allow_regenerate=args.allow_regenerate,
                allow_duplicate_items=args.allow_duplicate_items,
                fail_fast=args.fail_fast,
                wandb_logger=wandb_logger,
            )
            if args.wandb_log_artifacts:
                run_paths = ensure_run_layout(args.project_root, args.run_id)
                wandb_logger.log_artifact(
                    name=f"{args.run_id}-teacher-final",
                    artifact_type="tgvf-teacher-data",
                    paths=[
                        run_paths["config"],
                        run_paths["prompt"],
                        run_paths["schema"],
                        run_paths["accepted"],
                        run_paths["rejected"],
                        run_paths["reports"] / "generation_summary.json",
                        run_paths["reports"] / "quality_filter_report.json",
                        run_paths["reports"] / "cost_usage_report.json",
                    ],
                    aliases=["latest", args.run_id],
                )
            print(json.dumps(report, indent=2))
        finally:
            wandb_logger.finish()
    elif args.command == "prepare-batch":
        report = prepare_batch(
            selection=args.selection,
            project_root=args.project_root,
            run_id=args.run_id,
            openai_config=openai_config_from_args(args),
            generation_config=generation_config_from_args(args),
            target_accepted_samples=args.target_accepted_samples,
            limit_images=args.limit_images,
            max_request_file_mb=args.max_request_file_mb,
            allow_regenerate=args.allow_regenerate,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "submit-batch":
        print(json.dumps(submit_batch(run_dir=args.run_dir), indent=2))
    elif args.command == "poll-batch":
        print(json.dumps(poll_batch(run_dir=args.run_dir), indent=2))
    elif args.command == "parse-results":
        print(
            json.dumps(
                parse_batch_results(
                    run_dir=args.run_dir,
                    allow_duplicate_items=args.allow_duplicate_items,
                ),
                indent=2,
            )
        )
    elif args.command == "export-balanced":
        print(
            json.dumps(
                export_balanced_accepted(
                    accepted=args.accepted,
                    output=args.output,
                    target_samples=args.target_samples,
                    seed=args.seed,
                    allow_shortfall=args.allow_shortfall,
                ),
                indent=2,
            )
        )
    elif args.command == "summarize":
        print(json.dumps(summarize_run(args.run_dir), indent=2))
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TGVF teacher-guide data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke-sync", "resume-sync"):
        command = subparsers.add_parser(name)
        add_generation_args(command)
        default_limit = 100 if name == "smoke-sync" else None
        command.add_argument("--limit-images", type=int, default=default_limit)
        command.add_argument("--concurrency", type=int, default=1)
        command.add_argument("--allow-regenerate", action="store_true")
        command.add_argument("--allow-duplicate-items", action="store_true")
        command.add_argument("--fail-fast", action="store_true")
        add_wandb_args(command)

    batch = subparsers.add_parser("prepare-batch")
    add_generation_args(batch)
    batch.add_argument("--limit-images", type=int, default=None)
    batch.add_argument("--max-request-file-mb", type=int, default=90)
    batch.add_argument("--allow-regenerate", action="store_true")
    add_wandb_args(batch)

    submit = subparsers.add_parser("submit-batch")
    submit.add_argument("--run-dir", required=True)

    poll = subparsers.add_parser("poll-batch")
    poll.add_argument("--run-dir", required=True)

    parse = subparsers.add_parser("parse-results")
    parse.add_argument("--run-dir", required=True)
    parse.add_argument("--allow-duplicate-items", action="store_true")

    export_balanced = subparsers.add_parser("export-balanced")
    export_balanced.add_argument("--accepted", required=True)
    export_balanced.add_argument("--output", required=True)
    export_balanced.add_argument("--target-samples", type=int, default=20_000)
    export_balanced.add_argument("--seed", type=int, default=DEFAULT_SEED)
    export_balanced.add_argument("--allow-shortfall", action="store_true")

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--run-dir", required=True)
    return parser


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", default=DEFAULT_SELECTION)
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--detail", default=DEFAULT_IMAGE_DETAIL)
    parser.add_argument("--fallback-detail", default=DEFAULT_FALLBACK_DETAIL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--allow-json-mode-fallback", action="store_true")
    parser.add_argument(
        "--prompt-version",
        default=DEFAULT_TEACHER_PROMPT_VERSION,
        choices=ALLOWED_PROMPT_VERSIONS,
    )
    parser.add_argument(
        "--schema-version",
        default=DEFAULT_TEACHER_SCHEMA_VERSION,
        choices=ALLOWED_SCHEMA_VERSIONS,
    )
    parser.add_argument("--allow-prompt-schema-mismatch", action="store_true")
    parser.add_argument("--strict-visual-cue-schema", action="store_true")
    parser.add_argument("--target-accepted-samples", type=int, default=20_000)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-dir", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-log-artifacts", action="store_true")


def wandb_logger_from_args(
    args: argparse.Namespace,
    *,
    job_type: str,
    config: dict[str, Any],
) -> WandbLogger:
    tags = [tag.strip() for tag in str(args.wandb_tags or "").split(",") if tag.strip()]
    return WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or getattr(args, "run_id", None),
        group=args.wandb_group,
        job_type=job_type,
        config=config,
        mode=args.wandb_mode,
        tags=tags or None,
        directory=args.wandb_dir,
    )


def openai_config_from_args(args: argparse.Namespace) -> OpenAIConfig:
    return OpenAIConfig(
        model=args.model,
        image_detail=args.detail,
        fallback_detail=args.fallback_detail,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        allow_json_mode_fallback=args.allow_json_mode_fallback,
    )


def generation_config_from_args(args: argparse.Namespace) -> GenerationConfig:
    validate_teacher_version_pair(
        args.prompt_version,
        args.schema_version,
        allow_mismatch=args.allow_prompt_schema_mismatch,
    )
    return GenerationConfig(
        target_accepted_samples=args.target_accepted_samples,
        confidence_threshold=args.confidence_threshold,
        seed=args.seed,
        prompt_version=args.prompt_version,
        schema_version=args.schema_version,
        strict_visual_cue_schema=args.strict_visual_cue_schema,
        allow_prompt_schema_mismatch=args.allow_prompt_schema_mismatch,
    )


if __name__ == "__main__":
    main()
