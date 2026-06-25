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
