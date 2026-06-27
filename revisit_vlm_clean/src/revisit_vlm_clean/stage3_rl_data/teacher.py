"""GPT teacher request artifacts for Stage3 RL data generation."""

from __future__ import annotations

import json
from typing import Any

from revisit_vlm_clean.schema import _to_jsonable

from .schemas import now_iso, stable_hash

STAGE3_RL_TEACHER_REQUEST_SCHEMA_VERSION = "stage3_rl_teacher_request_v0"
STAGE3_RL_TEACHER_OUTPUT_SCHEMA_VERSION = "stage3_rl_teacher_bundle_v0"
STAGE3_RL_TEACHER_VERSION = "stage3_rl_gpt54_triage_v1"
DEFAULT_STAGE3_RL_TEACHER_MODEL = "gpt-5.4"

STAGE3_RL_TEACHER_DEVELOPER_PROMPT = """You are a meticulous TGVF Stage3 RL data annotator.

We are building data for a static-image Target-Guided Visual Foveation policy.
The policy sees an image and question, may emit one visual focus target, receives
focused visual evidence, and then answers.

Your task is to inspect the image and the provided source QA candidates. Decide
which source QAs are useful for RL, rewrite only when helpful, and generate
additional legacy-style questions only when the source QAs are insufficient.

The key distinction:
- Source QA is a candidate anchor, not a command.
- Source answers are useful gold anchors, but source questions may be too easy,
  too global, badly phrased, or unsuitable for a clean focus target.
- Generated questions are allowed, but they must follow the legacy TGVF V4
  visual-detail rules and must be visibly grounded.

Return strict JSON only. Do not include markdown. Do not hallucinate. Skip weak
or ambiguous evidence. Prefer fewer high-quality items over many weak items.
For each image, output one to four items only. Four is a hard maximum.
"""

STAGE3_RL_TEACHER_USER_PROMPT = """You are given one image and a bundle of source QA candidates.

Produce a Stage3 RL teacher bundle with this top-level schema:

{
  "schema_version": "stage3_rl_teacher_bundle_v0",
  "teacher_version": "stage3_rl_gpt54_triage_v0",
  "image_id": "source image id",
  "image_summary": "one concise whole-image description",
  "items": [...]
}

Each item must follow this schema:

{
  "item_id": "optional stable id, otherwise null",
  "provenance": "source_qa_kept | source_qa_rewritten | teacher_generated_legacy_style",
  "source_qa_id": "qa_id if derived from source QA, otherwise null",
  "source_decision": "keep | rewrite | generated",
  "reject_reason": null,
  "question": "natural open-answer question for RL",
  "original_question": "source question if any, otherwise null",
  "gold_answer": "plain answer text",
  "answer_aliases": ["optional aliases"],
  "answer_source": "source_gold | source_gold_rewritten_question | teacher_visible_evidence",
  "answer_type": "ocr_text | number | date | short_text | count | boolean_state | other",
  "eval_metric": "normalized_exact_match | numeric | date | token_f1",
  "evidence_type": "ocr_text | document_field | chart_value | table_cell | logo_symbol | object_part | color_attribute | texture_material | pattern | shape_boundary | spatial_relation | counting | state_action | diagram_visual_fact | math_reasoning | other",
  "difficulty_label": "direct_easy | local_easy | local_medium | local_hard | reasoning_hard",
  "tool_need_hint": "no_tool | optional_tool | useful_tool | likely_required",
  "question_family": "short bucket name",
  "target_spec": {
    "target_text": "6 to 24 word encoder-facing visual descriptor",
    "focus_type": "attribute | relative_position | ocr | document_field | chart_value | table_cell | object_part | texture_material | counting | other",
    "entities": [],
    "attribute_type": null,
    "relation_type": null,
    "source": "teacher",
    "confidence": 0.0
  },
  "quality": {
    "confidence": 0.0,
    "visually_verifiable": true,
    "answer_visible": true,
    "too_easy": false,
    "target_leakage_risk": "none | low | medium | high",
    "target_generic": false,
    "notes": []
  }
}

Source QA handling:
- For each useful source QA, output one item with provenance source_qa_kept or
  source_qa_rewritten.
- Keep the source gold answer when the answer is visibly supported.
- Rewrite only to make the question natural, open-answer, and RL-suitable.
- Do not pretend a simple object-identification answer becomes hard because the
  wording is complicated.
- If a source QA is too easy, global, unsupported, unsafe, or cannot have a clean
  non-leaking target, reject it by omitting an item and record the reason in
  bundle_rejections.
- Keep at most two direct_easy/no_tool source items. If the image supports richer
  local evidence, prefer rewriting or generating tool-useful local questions.

Generated item handling:
- If useful source QAs are too few, generate additional items from the image.
- Generated items must follow the legacy V4 TGVF rules: local visual detail,
  clean target, no answer leakage, plausible difficulty, and visible evidence.
- Generated questions should cover a mixture of difficulty levels. Stage3 RL
  needs direct/easy, optional-tool, useful-tool, and likely-required examples.
- Aim for 3 to 4 final items total. Do not output more than 4 items.
- For natural images, when evidence supports it, include at least two
  local_medium/local_hard items marked useful_tool or likely_required. For
  document/chart/scene-text images, include local field/value questions that
  genuinely benefit from focused evidence.
- Avoid filling the bundle with trivial color/object questions. One easy/no-tool
  calibration item is enough unless the image has no reliable harder evidence.
- Do not overproduce OCR/text/number items. Prefer color, texture/material,
  pattern, shape, object part, spatial relation, counting, and state/action when
  supported by the image.

Target rules inherited from legacy V4:
- target_text is not a tool-call object, not an instruction, and not a bare region
  name.
- It is an encoder-facing visual descriptor of the evidence to re-encode.
- It should usually be 6 to 24 words.
- It must not include the answer value, exact text/number/date/code being asked,
  final color/material/count/direction if that is the answer, or any option label.
- It must not contain a solved visual judgment that the question asks the model
  to make. Avoid target phrases such as "longest bar", "highest value",
  "light-colored container", "striped animal", "toasted texture", "matching
  item", or "left of the bottle" when those words reveal the answer.
- Prefer neutral localization over judgment: use row/axis/category labels,
  object identity from the question, nearby layout, or visible anchors that do
  not answer the question.
- Avoid answer-oriented verbs such as determine, judge, answer, verify, decide,
  infer, read, count, and whether.
- Good examples: "close-up surface texture, folds, and sheen of the glove";
  "wide shared view containing the dog, the river, and the space between them";
  "the second-row third-column cell with its row and column headers".
- Bad examples: "the longest horizontal bar"; "the light-colored drink
  container"; "the striped grazing animal"; "top bread slice with toasted
  texture".
- If the only concise target would reveal the answer, omit that item.

Open-answer policy:
- The RL prompt will not show choices.
- Source choices, if provided, are only hints for understanding the original
  annotation. Output plain answer text, not an option label.

Privacy and safety:
- Do not create items revealing personal addresses, phone numbers, emails,
  government IDs, financial account numbers, or other sensitive personal
  identifiers.
- Generic product labels, public signs, chart labels, package text, non-sensitive
  dates, and non-personal document fields are allowed.

Output strict JSON only.
"""


def stage3_rl_teacher_output_schema() -> dict[str, Any]:
    item_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "item_id",
            "provenance",
            "source_qa_id",
            "source_decision",
            "reject_reason",
            "question",
            "original_question",
            "gold_answer",
            "answer_aliases",
            "answer_source",
            "answer_type",
            "eval_metric",
            "evidence_type",
            "difficulty_label",
            "tool_need_hint",
            "question_family",
            "target_spec",
            "quality",
        ],
        "properties": {
            "item_id": {"type": ["string", "null"]},
            "provenance": {
                "type": "string",
                "enum": [
                    "source_qa_kept",
                    "source_qa_rewritten",
                    "teacher_generated_legacy_style",
                ],
            },
            "source_qa_id": {"type": ["string", "null"]},
            "source_decision": {
                "type": "string",
                "enum": ["keep", "rewrite", "generated"],
            },
            "reject_reason": {"type": ["string", "null"]},
            "question": {"type": "string"},
            "original_question": {"type": ["string", "null"]},
            "gold_answer": {"type": "string"},
            "answer_aliases": {"type": "array", "items": {"type": "string"}},
            "answer_source": {
                "type": "string",
                "enum": [
                    "source_gold",
                    "source_gold_rewritten_question",
                    "teacher_visible_evidence",
                ],
            },
            "answer_type": {
                "type": "string",
                "enum": [
                    "ocr_text",
                    "number",
                    "date",
                    "short_text",
                    "count",
                    "boolean_state",
                    "other",
                ],
            },
            "eval_metric": {
                "type": "string",
                "enum": ["normalized_exact_match", "numeric", "date", "token_f1"],
            },
            "evidence_type": {"type": "string"},
            "difficulty_label": {
                "type": "string",
                "enum": [
                    "direct_easy",
                    "local_easy",
                    "local_medium",
                    "local_hard",
                    "reasoning_hard",
                ],
            },
            "tool_need_hint": {
                "type": "string",
                "enum": ["no_tool", "optional_tool", "useful_tool", "likely_required"],
            },
            "question_family": {"type": "string"},
            "target_spec": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "target_text",
                    "focus_type",
                    "entities",
                    "attribute_type",
                    "relation_type",
                    "source",
                    "confidence",
                ],
                "properties": {
                    "target_text": {"type": "string"},
                    "focus_type": {"type": "string"},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "attribute_type": {"type": ["string", "null"]},
                    "relation_type": {"type": ["string", "null"]},
                    "source": {"type": "string", "enum": ["teacher"]},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
            "quality": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "confidence",
                    "visually_verifiable",
                    "answer_visible",
                    "too_easy",
                    "target_leakage_risk",
                    "target_generic",
                    "notes",
                ],
                "properties": {
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "visually_verifiable": {"type": "boolean"},
                    "answer_visible": {"type": "boolean"},
                    "too_easy": {"type": "boolean"},
                    "target_leakage_risk": {
                        "type": "string",
                        "enum": ["none", "low", "medium", "high"],
                    },
                    "target_generic": {"type": "boolean"},
                    "notes": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "teacher_version",
            "image_id",
            "image_summary",
            "items",
            "bundle_rejections",
        ],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": STAGE3_RL_TEACHER_OUTPUT_SCHEMA_VERSION,
            },
            "teacher_version": {"type": "string", "const": STAGE3_RL_TEACHER_VERSION},
            "image_id": {"type": "string"},
            "image_summary": {"type": "string"},
            "items": {"type": "array", "items": item_schema, "maxItems": 4},
            "bundle_rejections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["source_qa_id", "reason"],
                    "properties": {
                        "source_qa_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
    }


def build_stage3_rl_teacher_user_message(bundle: dict[str, Any]) -> str:
    source_qas = [_source_qa_for_prompt(item) for item in bundle.get("qa_items") or []]
    source_block = json.dumps(source_qas, ensure_ascii=False, sort_keys=True, indent=2)
    return (
        "Image metadata:\n"
        f"- stable_image_uid: {bundle.get('stable_image_uid', '')}\n"
        f"- image_id: {bundle.get('image_id', '')}\n"
        f"- source_dataset: {bundle.get('source_dataset', '')}\n"
        f"- source_split: {bundle.get('source_split', '')}\n"
        f"- source_profile: {bundle.get('source_profile', '')}\n\n"
        f"Source-profile hint:\n{_source_profile_hint(str(bundle.get('source_profile') or 'unknown'))}\n\n"
        f"{STAGE3_RL_TEACHER_USER_PROMPT}\n\n"
        "Source QA candidates:\n"
        f"{source_block}\n"
    )


def build_stage3_rl_teacher_request(
    *,
    bundle: dict[str, Any],
    run_id: str,
    model: str | None = None,
    image_detail: str = "original",
    temperature: float = 0.2,
    max_output_tokens: int = 3000,
) -> dict[str, Any]:
    model_name = model or DEFAULT_STAGE3_RL_TEACHER_MODEL
    request_id = stable_hash("stage3_rl_teacher_request", run_id, bundle.get("stable_image_uid"))
    user_message = build_stage3_rl_teacher_user_message(bundle)
    response_schema = stage3_rl_teacher_output_schema()
    custom_id = f"{run_id}__{str(bundle.get('stable_image_uid') or '').replace(':', '_')}"
    api_payload_template = {
        "model": model_name,
        "input": [
            {
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": STAGE3_RL_TEACHER_DEVELOPER_PROMPT}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_message},
                    {
                        "type": "input_image",
                        "image_url": "<FILLED_BY_TEACHER_RUNNER>",
                        "detail": image_detail,
                    },
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": STAGE3_RL_TEACHER_OUTPUT_SCHEMA_VERSION,
                "strict": True,
                "schema": response_schema,
            }
        },
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    return {
        "schema_version": STAGE3_RL_TEACHER_REQUEST_SCHEMA_VERSION,
        "request_id": request_id,
        "custom_id": custom_id,
        "created_at": now_iso(),
        "run_id": run_id,
        "teacher_version": STAGE3_RL_TEACHER_VERSION,
        "model": model_name,
        "image_path": bundle.get("image_path"),
        "stable_image_uid": bundle.get("stable_image_uid"),
        "image_id": bundle.get("image_id"),
        "source_dataset": bundle.get("source_dataset"),
        "source_split": bundle.get("source_split"),
        "source_profile": bundle.get("source_profile"),
        "source_qas": [_source_qa_for_prompt(item) for item in bundle.get("qa_items") or []],
        "prompt": {
            "developer": STAGE3_RL_TEACHER_DEVELOPER_PROMPT,
            "user": user_message,
        },
        "response_schema": response_schema,
        "api_payload_template": api_payload_template,
    }


def build_stage3_rl_teacher_requests(
    bundles: list[dict[str, Any]],
    *,
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    teacher_config = plan.get("teacher_config") or {}
    return [
        build_stage3_rl_teacher_request(
            bundle=bundle,
            run_id=str(plan.get("run_id") or "stage3_rl"),
            model=str(teacher_config.get("model") or plan.get("teacher_backend") or DEFAULT_STAGE3_RL_TEACHER_MODEL),
            image_detail=str(teacher_config.get("image_detail") or "original"),
            temperature=float(teacher_config.get("temperature", 0.2)),
            max_output_tokens=int(teacher_config.get("max_output_tokens", 5000)),
        )
        for bundle in bundles
        if bundle.get("qa_items")
    ]


def summarize_teacher_requests(requests: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, int] = {}
    source_qa_total = 0
    for request in requests:
        source = str(request.get("source_dataset") or "unknown")
        by_source[source] = by_source.get(source, 0) + 1
        source_qa_total += len(request.get("source_qas") or [])
    return _to_jsonable(
        {
            "schema_version": "stage3_rl_teacher_request_summary_v0",
            "teacher_version": STAGE3_RL_TEACHER_VERSION,
            "requests": len(requests),
            "source_qa_candidates": source_qa_total,
            "request_distribution_by_source": by_source,
            "created_at": now_iso(),
        }
    )


def _source_qa_for_prompt(qa_item: dict[str, Any]) -> dict[str, Any]:
    return {
        "qa_id": qa_item.get("qa_id"),
        "question": qa_item.get("question"),
        "gold_answer": qa_item.get("gold_answer"),
        "answer_aliases": qa_item.get("answer_aliases") or [],
        "original_choices": qa_item.get("original_choices") or [],
        "answer_format": qa_item.get("answer_format"),
        "answer_type": qa_item.get("answer_type"),
        "eval_metric": qa_item.get("eval_metric"),
        "evidence_type_hint": qa_item.get("evidence_type"),
        "difficulty_hint": qa_item.get("difficulty"),
        "question_family_hint": qa_item.get("question_family"),
        "metadata": qa_item.get("metadata") or {},
    }


def _source_profile_hint(source_profile: str) -> str:
    hints = {
        "natural_image": (
            "Prefer visual-detail questions about color, texture/material, pattern, "
            "shape boundaries, object parts, relations, counting, and state/action."
        ),
        "scene_text": (
            "Use OCR when useful, but also look for logo/symbol, color, object part, "
            "relation, and visible physical details. Do not let OCR dominate."
        ),
        "document": (
            "Prefer document fields, tables, local layout, and readable values. Include "
            "easy/direct cases only when they genuinely teach no-tool behavior."
        ),
        "chart": (
            "Prefer chart/table values, axes, legends, labels, and visible mark "
            "comparisons. Generate reasoning-hard only when the chart supports it."
        ),
        "table": "Prefer row/column/header focused questions and local cell values.",
    }
    return hints.get(source_profile, "Prefer reliable local visual evidence and balanced difficulty.")
