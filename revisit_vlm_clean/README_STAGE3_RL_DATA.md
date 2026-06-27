# Stage3 RL Data Generation

This clean entrypoint builds the Stage3 RL source-QA pool. It produces
auditable image bundles and flattened QA prompts for later rollout, forced
probes, reward scoring, and GRPO training.

It does not run RL training, rollout generation, judge reward, TGVF inference,
D generation, or benchmark evaluation.

## Output

Default output root:

```bash
revisit_vlm_clean/data/stage3_rl/v0_20k
```

Each run writes:

```text
stage3_rl_data_plan.json
image_bundles.jsonl
qa_candidates.jsonl
teacher_requests.jsonl
teacher_request_summary.json
accepted_rl_prompts.jsonl
rejected_rl_prompts.jsonl
dedup_report.json
filter_report.json
balance_report.json
generation_ledger.jsonl
manifest_summary.json
```

`20k` means accepted QA prompts, not images. The builder organizes candidates by
image bundle, then flattens accepted rows to single QA prompts for RL rollout.

## Image Bundles vs QA Prompts

An image bundle is one source image with multiple QA candidates. This preserves
same-image structure for diagnostics and later target-utility analysis.

A QA prompt is one training/rollout unit:

```text
image + question + gold answer + optional choices + TargetSpec
```

GRPO groups should sample multiple rollouts per single QA prompt. They do not
need to treat a whole image bundle as one rollout unit.

## Raw Dataset Roots

The default local dataset root is:

```bash
/home/dredvpn009/Flash_Storage/datasets
```

Default v0 source-QA adapters:

```text
visual_genome/annotations/question_answers.json
textvqa/annotations/TextVQA_0.5.1_train.json
docvqa/annotations/train_v1.0_withQT.json
chartqa/annotations/train.jsonl
```

The builder does not copy raw images into the project.

## SFT Image Exclusion

Stage3 RL images must be disjoint from the current SFT image pool. By default,
the plan includes existing 50k SFT/Stage2 manifests when they are present, and
you can add more exclusion sets:

```bash
--exclude-manifest path/to/images.jsonl
```

Exclusion keys include `stable_image_uid`, `image_id`, image paths, source IDs,
and image hashes when available.

## Plan

Create the plan first:

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.generate_data stage3-rl \
  --write-plan \
  --run-id stage3_rl_v0_20k \
  --dataset-root /home/dredvpn009/Flash_Storage/datasets \
  --output-root revisit_vlm_clean/data/stage3_rl/v0_20k \
  --target-accepted-prompts 20000 \
  --seed 42
```

## Preflight

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.generate_data stage3-rl \
  --plan revisit_vlm_clean/data/stage3_rl/v0_20k/stage3_rl_data_plan.json \
  --preflight-only
```

Preflight checks dataset/source paths, exclusion manifests, output writability,
schema imports, benchmark-path blocking, and estimated source-QA count.

## Dry Run

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.generate_data stage3-rl \
  --plan revisit_vlm_clean/data/stage3_rl/v0_20k/stage3_rl_data_plan.json \
  --dry-run \
  --limit-images 50
```

Dry-run outputs go under `dry_run/` and do not update the final accepted
manifest.

## GPT-5.4 Teacher Triage Mode

The formal Stage3 path should not treat raw source QA as final RL data. Source
QA provides candidate anchors and gold answers, while GPT-5.4 decides whether a
candidate is suitable, rewrites it if useful, or generates a legacy-style local
visual question when source QA is weak.

Prepare GPT-5.4 request artifacts with:

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.generate_data stage3-rl \
  --write-plan \
  --run-id stage3_rl_v0_teacher_triage \
  --dataset-root /home/dredvpn009/Flash_Storage/datasets \
  --output-root revisit_vlm_clean/data/stage3_rl/v0_teacher_triage \
  --target-accepted-prompts 20000 \
  --qa-generation-mode teacher_triage \
  --teacher-backend gpt-5.4 \
  --seed 42
```

Then run a dry request build:

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.generate_data stage3-rl \
  --plan revisit_vlm_clean/data/stage3_rl/v0_teacher_triage/stage3_rl_data_plan.json \
  --dry-run \
  --limit-images 50
```

`teacher_triage` writes image bundles plus `teacher_requests.jsonl`. It does not
write final accepted RL prompts because API responses still need to be parsed,
filtered, and balanced. Each request asks GPT-5.4 to choose among:

```text
source_qa_kept
source_qa_rewritten
teacher_generated_legacy_style
```

The request prompt inherits the legacy V4 target rules: target text must be a
visual descriptor, must avoid answer leakage, and must not be an instruction such
as "read the number" or "determine the color". Stage3 adds `difficulty_label`
and `tool_need_hint` so later RL can include direct/easy, optional-tool,
useful-tool, and likely-required examples.

## Execute

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.generate_data stage3-rl \
  --plan revisit_vlm_clean/data/stage3_rl/v0_20k/stage3_rl_data_plan.json \
  --execute
```

Formal execute writes `generation_ledger.jsonl` and appends a data-generation
entry to `docs/EXPERIMENT_LEDGER.md` unless `--skip-experiment-ledger` is used.

## Extend

Create a new append-only version from an existing output:

```bash
PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.generate_data stage3-rl \
  --write-plan \
  --run-id stage3_rl_v1_50k \
  --extend-from revisit_vlm_clean/data/stage3_rl/v0_20k \
  --output-root revisit_vlm_clean/data/stage3_rl/v1_50k \
  --target-accepted-prompts 50000 \
  --seed 43
```

Extension keeps prior accepted rows at the front of the new accepted manifest
with unchanged sample IDs, then appends new prompts from non-prior images.

## TargetSpec

`TargetSpec` is a clean reference target for forced probes. It is not the only
correct target for free rollout.

The builder requires TargetSpec by default. Source-provided targets are
validated; missing targets are filled by deterministic source-QA rules when
possible. Accepted targets must be local visual cues, avoid answer/choice
leakage, avoid generic whole-image wording, and avoid task-solving verbs.

`forced_on_reference` should use `target_spec.target_text`. `forced_on_self_target`
should let the policy generate its own target after a forced focus prefix.
`tool_decision_label` is deliberately left null for later probes.

## Required Verifiable Fields

Accepted RL prompts include:

```text
sample_id
stable_image_uid
image_path
image_sha256
question
choices
original_choices
gold_answer
answer_aliases
answer_format
answer_type
eval_metric
evidence_type
difficulty
target_spec
forced_probe
generation_metadata
```

These are enough for later answer correctness, parse/format scoring, target
quality checks, and forced probe bookkeeping.

If source data has multiple-choice options, `choices` stays empty for the
default open-answer prompt. The source options are preserved in
`original_choices` and in `source_metadata.qa_metadata.choice_to_open_answer`.

## Non-Goals

This stage deliberately does not implement:

```text
GRPO training
rollout runner
online reward
judge reward
FocusEvidence / ImageConsistency judges
D generation
TGVF inference
benchmark evaluation
CoreDev/CoreFull eval
multi-focus data
```
