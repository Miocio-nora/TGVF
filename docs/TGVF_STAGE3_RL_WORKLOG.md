# TGVF Stage3 RL Worklog

This document records the working context for Stage3 RL dataset and reward
design. It is a planning log, not a finalized training recipe.

## Project Framing

The current project is a three-stage TGVF training pipeline.

TGVF means Target-Guided Visual Foveation. The model is trained to decide
whether current visual evidence is enough, select a local visual target when it
is not enough, invoke the TGVF tool to produce target-conditioned visual
evidence tokens, and then answer using that evidence.

## Three Stages

### Stage1: TGVF Module Cold Start

Purpose:

- Teach the TGVF module to read target descriptions.
- Given an image, question, and target, the module should produce visual
  evidence tokens aligned with the intended local target.

Important data property:

- Stage1 benefits from multiple QA/target rows per image.
- The matrix-CE / negative loss setup needs contrastive structure, especially
  same-image negatives where different targets on the same image should not be
  confused.

Interpretation:

- Stage1 is not mainly teaching final answer behavior.
- It is teaching target-conditioned visual readout.

### Stage2: LLM Tool-Use Cold Start

Purpose:

- Teach the LLM to use the TGVF tool.
- The LLM should learn when to request focus, how to phrase a target, how to
  continue after the TGVF visual evidence is appended, and how to produce an
  answer.

Important data property:

- Existing 50k SFT-style teacher data is used for cold start.
- The current SFT data teaches protocol shape and basic tool-use behavior.

Known current SFT source:

```text
data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k/final/tgvf_teacher_items.accepted.jsonl
```

This file has `sft_text` for each accepted item and is considered the current
50k SFT source.

### Stage3: RL

Purpose:

- Optimize the policy beyond SFT imitation.
- Improve task-level behavior: focus decisions, target quality, evidence use,
  answer correctness, and format stability.

Hard data constraint:

- Stage3 RL original images should not overlap with Stage1/Stage2 SFT original
  images.
- The exclusion should be image-level, not merely QA/item-level.
- Candidate keys for exclusion include image path, `image_id`,
  `stable_image_uid`, source dataset identifiers, and ideally image content hash
  if available.

Open design question:

- Stage1 needed multiple QA/target rows per image because of matrix CE and
  same-image negatives.
- Stage3 RL may or may not need multiple QA per image.
- This should be decided based on the reward design and rollout objective.

Preliminary view:

- Multiple QA per image can help if RL is meant to improve target selection and
  avoid generic image-level focus behavior.
- It is less obviously necessary if RL mostly optimizes answer formatting or
  post-D answer consumption.
- For GRPO-style training, the more essential grouping is multiple sampled
  completions per prompt, not necessarily multiple prompts per image.

## Dataset-Making Flow To Understand

Before producing the 20k RL dataset, we need to clearly understand and document
the full data pipeline:

```text
raw image pool
-> dataset registry
-> image manifest
-> image selection
-> teacher generation
-> schema validation and filtering
-> SFT rendering / sft_text
-> Stage1 split
-> Stage2 split
-> SFT image exclusion set
-> RL image selection from non-overlapping images
-> RL prompt/reference construction
-> policy rollout generation
-> reward scoring
-> RL train/dev/eval split
```

The 20k RL dataset should be built only after the SFT image exclusion set and
reward requirements are clear.

## Why Reward Design Comes First

Reward design affects data construction. For example:

- If reward needs answer correctness only, RL data needs reliable gold answers
  and answer normalizers.
- If reward evaluates focus trigger decisions, data may need labels or teacher
  references for whether local evidence is needed.
- If reward evaluates target quality, data may need teacher target references,
  local evidence descriptions, or a judge prompt.
- If reward evaluates evidence use, data may need reference evidence spans or
  post-tool evidence text.
- If reward penalizes answer leakage in targets, data must preserve answer,
  choices, target text, and leakage metadata.

Therefore, reward design should be discussed before committing to the 20k RL
data format and generation process.

## Reward Design Discussion Points

Candidate reward components:

```text
R_answer
  Final answer correctness, using exact match / multiple-choice parser /
  dataset-specific normalizer where possible.

R_parse_format
  Output is parseable and follows required answer/protocol format.

R_focus_trigger
  The model triggers focus when local visual evidence is needed and avoids
  unnecessary focus when evidence is already sufficient.

R_target_quality
  Focus target is local, visible, specific, non-generic, and does not leak the
  answer.

R_target_non_leakage
  Target should not contain too much premature judgment, inferred answer content,
  option text, or task-solving conclusion. It should describe where/what to look
  at, not decide the answer before the TGVF evidence is read.

R_tool_use
  Tool call is structurally valid and can be executed; appended TGVF evidence is
  consumed rather than ignored.

R_evidence
  Evidence text is grounded in the focused region and supports the final answer.

R_d_readout_grounding
  After TGVF returns D, the model should correctly read information from D. The
  answer/evidence should be attributable to the focused visual evidence rather
  than hallucinated from priors or copied from the question.

R_hallucination
  Penalize unsupported evidence, invented visual facts, unsupported OCR strings,
  and confident answer claims not backed by either the original image or D.

R_reasoning_quality
  Reasoning should be concise, task-relevant, and consistent with the evidence
  path. It should not rationalize a guessed answer after the fact.

R_efficiency
  Penalize unnecessary long reasoning, repeated focus, malformed actions, or
  excessive token use.
```

Open questions:

- Should RL optimize only final answer reward first, or include shaped rewards
  from the beginning?
- Should target quality be judged by rules, teacher references, a VLM judge, or
  outcome-based reward only?
- Should no-focus/direct cases be included in Stage3 RL, and at what ratio?
- Should the 20k RL data contain multiple QA per image?
- Should RL prompts preserve multiple-choice format or be converted to
  open-answer format?
- Which benchmarks or heldout teacher sets are allowed for reward calibration
  without contaminating final evaluation?

## Reward Design Notes

The reward should not be a single answer-only score. Stage3 is meant to improve
the whole TGVF policy, so reward design should cover at least:

```text
answer correctness
format correctness
whether tool use is needed
target quality
target non-leakage / no premature answer judgment
reasoning quality
D readout grounding
hallucination reduction
```

### Rule-Friendly Components

These can likely be scored with deterministic rules or benchmark parsers:

```text
answer correctness
answer parseability
protocol marker correctness
tool-call structural validity
target contains answer / option leakage
target too generic
excessive length
repeated malformed focus calls
```

### Reference-Dependent Components

These need teacher labels, gold answers, or structured references in the RL
dataset:

```text
whether tool use is needed
target overlap with teacher target or acceptable target family
evidence supports gold answer
D-readout answer correctness under forced/correct target
```

### Judge-Likely Components

These may require a VLM/LLM judge unless we design proxy tasks:

```text
target is visually local and sufficient
reasoning is faithful rather than post-hoc rationalization
evidence is grounded in D rather than hallucinated
unsupported visual/OCR claims
```

### Important Reward Tensions

Tool-use reward needs care. If the answer is already obvious, unnecessary focus
should be penalized. If local evidence is needed, failure to focus should be
penalized even if the model guesses correctly. This means RL data probably needs
both focus-needed and no-focus/direct cases.

## 2026-06-26 Clean Data Builder Implementation Notes

Implemented clean-native Stage3 RL data generation under:

```text
revisit_vlm_clean/src/revisit_vlm_clean/stage3_rl_data/
revisit_vlm_clean/README_STAGE3_RL_DATA.md
```

CLI shape:

```bash
tgvf_generate_data stage3-rl --write-plan ...
tgvf_generate_data stage3-rl --plan ... --preflight-only
tgvf_generate_data stage3-rl --plan ... --dry-run
tgvf_generate_data stage3-rl --plan ... --execute
```

Current implementation choices:

- Default source-QA adapters are Visual Genome QA, TextVQA, DocVQA, and
  ChartQA.
- Default output root is `revisit_vlm_clean/data/stage3_rl/v0_20k`.
- Default dataset root is `/home/dredvpn009/Flash_Storage/datasets`, matching
  the local clean benchmark defaults.
- Default SFT exclusions include the 50k teacher selection and accepted/Stage2
  manifests when those paths exist.
- Stage3 accepted prompts require TargetSpec by default.
- Source-provided targets and rule-built targets are validated for length,
  generic wording, task verbs, sensitive identifiers, and answer/choice leakage.
- Extension mode preserves prior accepted sample IDs and appends new prompts.

Important non-goals preserved:

- No GRPO training.
- No rollout runner.
- No reward judge.
- No FocusEvidence or ImageConsistency judge.
- No D generation or TGVF inference.
- No benchmark-derived training source.

Validation added:

```bash
PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests/test_stage3_rl_data.py
PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests/test_cli.py
```

Target quality also needs care. A target like "the orange beak" may help answer
the question but leaks answer content. A better target is "the bird's beak
surface" or "close-up of the bird's head and beak area". The reward should
prefer targets that locate evidence without deciding the answer.

For hallucination, answer correctness alone is insufficient. A model can guess
the correct answer with bad evidence, or produce plausible but unsupported OCR.
We need either reference evidence, judge checks, or special D-readout probes to
reward grounded use of D.

## Current Constraint Summary

- Existing 50k SFT data should be treated as Stage1/Stage2 cold-start data.
- Stage3 RL should use a separate image pool.
- The next step is reward design, because reward requirements determine what
  fields the 20k RL data must contain.

## 2026-06-27 Stage3 RL Source-QA + GPT-5.4 Triage Decision

Decision:

- Raw source QA should not be treated as final RL data.
- Source QA is useful as candidate context and often gives a better gold-answer
  anchor than free teacher generation.
- The source question itself may be too easy, too global, or not aligned with
  TGVF target/tool-policy needs.
- GPT-5.4 should inspect the image and source QA bundle, then decide whether to
  keep, rewrite, reject, or generate legacy-style local visual questions.
- Generated questions are allowed because the 50k SFT teacher data already used
  legacy V4 API generation and its distribution was acceptable.
- Stage3 differs from SFT because RL should include a broader difficulty mix,
  including direct/easy, optional-tool, useful-tool, and likely-required cases.

Implementation preparation:

- Added clean `teacher_triage` mode for Stage3 RL data preparation.
- This mode writes `teacher_requests.jsonl` and `teacher_request_summary.json`.
- It does not call the API and does not write final accepted RL prompts.
- Request provenance categories are:
  - `source_qa_kept`
  - `source_qa_rewritten`
  - `teacher_generated_legacy_style`
- The request prompt inherits legacy V4 target rules:
  - target is an encoder-facing visual descriptor;
  - target must avoid answer/choice leakage;
  - target must avoid task verbs such as determine, verify, read, and count;
  - weak, ambiguous, unsafe, or hallucinated evidence should be skipped.

Important correction:

- `source_qa-only` is now only a deterministic smoke/source-pool path.
- Formal Stage3 RL data should go through GPT-5.4 triage/generation before
  filter/balance/accepted manifests are considered final.

## 2026-06-27 Mixed 20 API Smoke Result

Status:

- The mixed 20-image API smoke is archived and excluded from formal samples.
- Archive path:
  `revisit_vlm_clean/data/stage3_rl/v0_teacher_triage_20k_20260627_041834/archives/smoke_mixed20_20260627_batch_6a3f3295/`.
- Root-level exclusion ledger:
  `revisit_vlm_clean/data/stage3_rl/v0_teacher_triage_20k_20260627_041834/api_archived_custom_ids.jsonl`.
- Archived custom ids are skipped by API prepare, parse, and finalize paths.

Smoke identity:

- Batch id: `batch_6a3f32958870819087b273c79e521920`.
- Images: 20.
- Source mix: Visual Genome 8, TextVQA 6, DocVQA 4, ChartQA 2.
- API status: 20 completed, 0 failed.
- Parsed JSON errors: 0.
- Runtime: about 9m42s OpenAI backend time, about 10m49s from submit to
  local parsed outputs.

Quality findings:

- Raw teacher items: 116, average 5.8 items/image.
- Existing deterministic filter accepted 114/116 before per-image cap.
- With max 4 QA/image, 79 would be accepted, but this smoke is not formal data.
- Problems observed:
  - too many `no_tool` and direct/easy items;
  - target occasionally included answer-adjacent judgments such as "longest",
    "light-colored", "striped", or "toasted";
  - API produced more items than needed, wasting tokens and later balance
    rejection.

Code corrections made after the smoke:

- Teacher prompt version moved to `stage3_rl_gpt54_triage_v1`.
- Teacher schema now enforces at most 4 items per image.
- Prompt now asks for 3-4 high-quality items, more useful-tool/local-medium+
  cases, and no more than one easy/no-tool calibration item unless necessary.
- Target rules now explicitly ban solved visual judgments inside target text.
- `max_output_tokens` default reduced from 5000 to 3000.
- Teacher items marked `quality.too_easy=true` are rejected by local filters.
- Added direct concurrent API runner for small smoke runs; Batch API should be
  reserved for larger chunks where queue overhead is amortized.

Operational decision:

- Do not use the current smoke output root as the official formal Stage3 RL
  sample set.
- Formal generation should use a new output root with the v1 prompt and archived
  smoke custom ids excluded.

## 2026-06-27 V1 Direct 20 API Smoke Result

Status:

- A second 20-image smoke was run with `stage3_rl_gpt54_triage_v1`.
- It used direct concurrent Responses API calls, not Batch API.
- It is archived and excluded from formal samples.
- Output root:
  `revisit_vlm_clean/data/stage3_rl/v1_direct_smoke20_20260627_030144/`.
- Archive:
  `revisit_vlm_clean/data/stage3_rl/v1_direct_smoke20_20260627_030144/archives/smoke_v1_direct20_20260627_030144/`.

Smoke identity:

- Source mix: Visual Genome 8, TextVQA 6, DocVQA 4, ChartQA 2.
- Teacher version: `stage3_rl_gpt54_triage_v1`.
- Max output tokens: 3000.
- Runner: direct concurrent API with `direct_workers=8`.
- Runtime: 25.9s measured by runner, 27s wall-clock.
- API errors: 0.
- Parsed outputs before archive: 20.
- After archive/reparse: `teacher_outputs.jsonl` has 0 active rows and
  `archived_outputs_skipped=20`.

Quality findings:

- Raw teacher items: 79, average 3.95 items/image.
- Per-image item cap worked: 19 images produced 4 items, 1 produced 3 items.
- Local deterministic filter:
  - valid: 77;
  - rejected: 2;
  - rejected reasons: one target leakage, one target task-verb false/edge case;
  - no balance rejection because all images were at or below max 4 QA/image.
- Tool distribution improved:
  - `useful_tool`: 44;
  - `likely_required`: 7;
  - `optional_tool`: 17;
  - `no_tool`: 11.
- Difficulty distribution improved:
  - `local_medium`: 43;
  - `local_easy`: 21;
  - `direct_easy`: 8;
  - `local_hard`: 5;
  - `reasoning_hard`: 2.

Remaining issues:

- One target still leaked an answer synonym:
  target "row of parked vehicles..." for answer "Automobiles".
- One target used "page count" and previously triggered
  `target_contains_task_verb` because the validator treated any occurrence of
  "count" as a forbidden task word. This was relaxed after discussion:
  noun-phrase locators like "page count field" are now allowed, while
  instruction-like targets such as "count the people" or "area to read the
  label" remain rejected.
- Chart target phrasing still has mild localization leakage in examples like
  "rightmost bar", although much less than the previous smoke.

Operational decision:

- Direct concurrent smoke is much better for small quality checks than Batch API.
- V1 prompt/schema is materially better than V0 and is the right baseline for
  the next smoke/formal candidate generation.

## 2026-06-29 Current Stage3 Training Recipe

Keep policy and judge model identities separate:

```text
policy model: /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking
policy checkpoint: clean Stage2 step1200 adapter checkpoint
judge model: qwen3_vl_32b_thinking / Qwen3-VL-32B-Thinking
```

The 32B model is for offline focus/grounding judge caches only. It must not be
used as the default policy model for the current 8B Stage2 adapter checkpoint.

The next prepared pilot is:

```text
output_root: outputs/stage3_grpo/all5_hint_no_block_frozenref_4gpu_g12_res768_20step_20260629
steps: 20
world_size: 4
group_size: 12
global_rollouts_per_step: 48
DeepStack: enabled, original_image_scope=no_block
reference_policy: frozen_stage2
tool labels: teacher_hint, hint_label_weight=1.0
reward: answer/tool/focus/ground/protocol all active
```

This is intentionally a pilot, not a 200-step formal lineage. The first thing to
inspect after launch is whether legal TGVF focus actions recover under the
current no-block DeepStack and frozen-reference setup.
