# TGVF VSTAR Force-Target Validation

Date: 2026-05-30

This note records the VSTAR validations around TGVF v2 conditioned-D inference,
force-mode target generation, deterministic rule targets, prompt variants, and
conditioned-image readout.

## Checkpoint And Default Evaluation Setup

Best checkpoint used in these validations:

```text
outputs/tgvf_fvt/20k_v2_matrixce_streaming_plus_neg_20260529_192910/tgvf_v2_bidirectional_matrixce/checkpoint_step_4254.pt
```

Common VSTAR command shape:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src PYTHONUNBUFFERED=1 python -m tgvf_eval.run \
  --benchmark vstar_bench --tier full \
  --method tgvf_conditioned_D_fresh \
  --model-path Qwen/Qwen2-VL-2B-Instruct \
  --tgvf-checkpoint outputs/tgvf_fvt/20k_v2_matrixce_streaming_plus_neg_20260529_192910/tgvf_v2_bidirectional_matrixce/checkpoint_step_4254.pt \
  --tgvf-variant tgvf_v2_bidirectional \
  --device cuda:0 --dtype bf16 \
  --answer-max-new-tokens 32 --capture-max-new-tokens 128 \
  --num-foveated-tokens none --spatial-merge-size 2 --image-budget mid \
  --force-target-source rule|llm \
  --run-id RUN_ID --no-resume
```

Important implementation details for the current v2 checkpoint:

- Conditioning happens before Qwen's frozen visual merger.
- The output token count is the original Qwen merged image-token count.
- Real source `image_grid_thw` and image-style mRoPE are used for the injected FVT chunk.
- Qwen visual merger is frozen and not trained.
- The fresh answer path clears the capture KV cache and uses a new first user turn:

```text
<|im_start|>user
<|vision_start|><FVT tokens><|vision_end|>
{full VSTAR multiple-choice question}
<|im_end|>
<|im_start|>assistant
```

## Prompt Variants

### Old Force Prompt

This prompt reproduced the previous best VSTAR fresh-D result.

```text
{question}

Before answering, select one local visual object or region to inspect. Name the thing to look at, not the answer value.
Good targets are short noun phrases like: the glove, the dustpan, the motorcycle helmet, the Apple logo, the two mentioned objects.
Bad targets include answer options, colors, JSON, pipes, or the words target/visual target description.
Output exactly one foveation request in this format:
<|foveate|><object or region phrase><|/foveate|>
Stop immediately after <|/foveate|>. Do not answer before foveating.
Do not output reasoning, JSON, separators, or tool metadata.
```

### New Short Force Prompt

This prompt was tested with both `question_without_choices` and full `question`.
The current finding is that full `question` is better than removing choices.

```text
You are selecting a visual focus target for a visual question.

Question:
{question}

Write the visual focus target that should be inspected before answering.

Rules:
- Output a short noun phrase only.
- Use the object, region, or relation mentioned in the question.
- If the question asks for an attribute, include the attribute type and the object, but not the attribute value.
- If the question asks for a left/right or above/below relation, include both mentioned entities, not the direction word.
- Do not include answer choices.
- Do not include answer values.
- Do not include colors, materials, numbers, dates, or directions if they are candidate answers.
- Do not invent an unrelated object.
- Do not output examples, JSON, separators, pipes, or special tokens.
- Length: 2 to 12 words.

Output exactly:
<|foveate|>TARGET<|/foveate|>
```

### Training Capture Prompt

This is the prompt used by training feature collection through
`build_tgvf_prompt(question)`.

```text
{question}

If fine-grained visual evidence is needed, output exactly one foveation request:
<|foveate|>visual target description<|/foveate|>
After <|/foveate|>, stop.
Do not answer the question yet.
Do not emit intent, mode, scope, JSON, tool metadata, explanations, or natural-language commentary.
Only emit the foveation request.
```

At the time this note was written, `src/tgvf_eval/prompts.py` is set to use this
training capture prompt for force mode.

## Deterministic Rule Target Builder

Added:

```text
src/tgvf_eval/target_rules.py
```

Added CLI:

```text
--force-target-source rule|llm|llm_with_rule_fallback
```

VSTAR force mode defaults to `rule` unless a source is explicitly passed.

Rule behavior:

- `direct_attributes`
  - `What is the {attribute} of {object}?`
  - target: `the {attribute} of {object}`
  - focus type: `attribute`
- `relative_position`
  - `Is {A} on the left or right side of {B}?`
  - target: `{A} and {B}`
  - focus type: `relative_position`
  - relation type: `left_or_right`

The builder covered all 191 VSTAR full samples in the tested split:

```text
direct_attributes: 115
relative_position: 76
parse coverage: 191/191
invalid rule targets: 0/191
```

Rule target examples:

```text
What is the material of the glove?
-> the material of the glove

What is the color of the dustpan?
-> the color of the dustpan

Is the telephone on the left or right side of the hand lamp?
-> the telephone and the hand lamp

Is the dog on the left or right side of the river?
-> the dog and the river
```

Validation rejects targets that contain answer option values, pipes, special
tokens, copied examples, forbidden atomic answers, or low overlap with question
entities.

## Main VSTAR Results

All rows below use VSTAR full, 191 samples, Qwen2-VL-2B, the same v2
bidirectional checkpoint, and fallback scoring.

| Setting | Run | Score | Correct | Direct | Relative | ABCD | Target source | Avg target words | Choice hit | Single word |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| direct Qwen | `vstar_full_bestpt_direct_qwen_20260530_020248` | 0.712042 | 136/191 | 0.747826 | 0.657895 | 53/86/27/25 | - | - | 0 | 0 |
| prompt-only force, no D | `vstar_full_bestpt_prompt_only_force_20260530_020248` | 0.623037 | 119/191 | 0.669565 | 0.552632 | 61/89/30/11 | - | - | 0 | 0 |
| original D, trimmed | `vstar_full_original_D_qwen_vmerge_trim_20260530_022430` | 0.586387 | 112/191 | 0.678261 | 0.447368 | 107/55/15/14 | - | - | 0 | 0 |
| old best fresh-D LLM | `vstar_full_bestpt_conditioned_D_fresh_20260530_032410` | 0.701571 | 134/191 | 0.739130 | 0.644737 | 54/80/31/26 | - | - | 0 | 0 |
| long prompt LLM | `vstar_full_bestpt_conditioned_D_fresh_longprompt_20260530_050545` | 0.685864 | 131/191 | 0.721739 | 0.631579 | 50/82/34/25 | - | - | 0 | 0 |
| rule + long prompt fixed | `vstar_full_bestpt_conditioned_D_fresh_rule_target_fixed_20260530_053812` | 0.691099 | 132/191 | 0.721739 | 0.644737 | 51/81/34/25 | rule:191 | 5.92 | 0 | 0 |
| rule + short prompt + question without choices | `vstar_full_bestpt_conditioned_D_fresh_rule_target_shortprompt_20260530_055321` | 0.685864 | 131/191 | 0.713043 | 0.644737 | 53/80/31/27 | rule:191 | 5.92 | 0 | 0 |
| LLM + short prompt + question without choices | `vstar_full_bestpt_conditioned_D_fresh_shortprompt_llm_repro_20260530_060105` | 0.685864 | 131/191 | 0.721739 | 0.631579 | 53/81/30/27 | llm:191 | 1.45 | 29 | 118 |
| LLM + short prompt + full question | `vstar_full_bestpt_conditioned_D_fresh_shortprompt_fullquestion_llm_20260530_061021` | 0.691099 | 132/191 | 0.721739 | 0.644737 | 55/78/30/28 | llm:191 | 1.19 | 164 | 163 |
| rule + short prompt + full question | `vstar_full_bestpt_conditioned_D_fresh_rule_target_fullquestion_shortprompt_20260530_061900` | 0.691099 | 132/191 | 0.713043 | 0.657895 | 54/79/31/27 | rule:191 | 5.92 | 0 | 0 |
| old prompt + LLM repro | `vstar_full_bestpt_conditioned_D_fresh_old_force_prompt_llm_repro_20260530_062900` | 0.701571 | 134/191 | 0.739130 | 0.644737 | 53/80/32/26 | llm:191 | 2.49 | 51 | 59 |
| old prompt + rule | `vstar_full_bestpt_conditioned_D_fresh_old_force_prompt_rule_target_20260530_063600` | 0.696335 | 133/191 | 0.721739 | 0.657895 | 54/80/32/25 | rule:191 | 5.92 | 0 | 0 |
| training capture prompt + rule | `vstar_full_bestpt_conditioned_D_fresh_training_capture_prompt_rule_target_20260530_064600` | 0.685864 | 131/191 | 0.713043 | 0.644737 | 53/82/32/24 | rule:191 | 5.92 | 0 | 0 |
| training capture prompt + LLM | `vstar_full_bestpt_conditioned_D_fresh_training_capture_prompt_llm_target_20260530_065000` | 0.659686 | 126/191 | 0.695652 | 0.605263 | 53/81/29/28 | llm:191 | 10.45 | 85 | 2 |

Additional original-D checks:

| Setting | Run | Score | Correct | Direct | Relative | ABCD |
|---|---|---:|---:|---:|---:|---:|
| original D, qwen vmerge | `vstar_full_original_D_qwen_vmerge_20260530_021745` | 0.575916 | 110/191 | 0.608696 | 0.526316 | 92/70/16/13 |
| original D + repeat options | `vstar_full_original_D_repeatopts_20260530_031308` | 0.554974 | 106/191 | 0.591304 | 0.500000 | 132/33/13/13 |

## Important Comparisons

### Old Prompt Reproduces Old Best

The old prompt with LLM-generated targets reproduced the previous best fresh-D
score:

```text
old best:          134/191 = 0.701571
old prompt repro:  134/191 = 0.701571
```

Differences between the old best run and the repro:

```text
parsed_answer differences: 1
raw_output differences:   1
target differences:       3
score flips:              0
```

The changed targets were all pathological long targets, so the score stayed the
same.

### Full Question Helps Compared With Question Without Choices

For rule target with the short prompt:

```text
question_without_choices: 131/191
full question:            132/191
```

For LLM target with the short prompt:

```text
question_without_choices: 131/191
full question:            132/191
```

Conclusion: the full multiple-choice question should be kept in force capture.

### Old Prompt + Rule Is Close But Not Best

Old prompt + rule target:

```text
133/191 = 0.696335
direct:   0.721739
relative: 0.657895
```

Old prompt + LLM target:

```text
134/191 = 0.701571
direct:   0.739130
relative: 0.644737
```

Rule targets are cleaner and improve relative-position accuracy, but the old
LLM targets still win by one sample overall because direct attributes improve.

Representative flips:

| Sample | Gold | Old LLM pred | Old rule pred | LLM target | Rule target |
|---:|---|---|---|---|---|
| 190 | A | B | A | `chair on the right side of the street` | `the white chair and the street` |
| 75 | A | A | B | `backpack` | `the color of the backpack` |
| 81 | A | A | D | `green umbrella` | `the color of the flag` |

### Training Capture Prompt Did Not Help

The hypothesis was that using the exact training capture prompt might better
match the target hidden-state distribution.

Results did not support that under the current VSTAR fresh-D answer pipeline:

```text
old prompt + rule:              133/191
training capture prompt + rule: 131/191

old prompt + LLM:               134/191
training capture prompt + LLM:  126/191
```

For rule targets, the target text is identical across prompt variants, so this
is a direct test of capture-prompt context. The training prompt still performed
worse.

The training prompt LLM case failed mostly because the model often generated
placeholder or malformed targets:

```text
visual target description
<|endoftext|>What is the color of the car? ...
```

### Rule Targets Are Cleaner But Not Always Better For VSTAR

Rule targets:

```text
choice_value_hit: 0
single_word_target: 0
avg target words: 5.92
```

Old prompt LLM targets:

```text
choice_value_hit: 51
single_word_target: 59
avg target words: 2.49
```

Training prompt LLM targets:

```text
choice_value_hit: 85
single_word_target: 2
avg target words: 10.45
many special-token / copied-question failures
```

Interpretation:

- Rule target improves cleanliness and interpretability.
- VSTAR multiple-choice direct-attribute answering can still benefit from short
  object-only targets or even leaked target strings.
- Clean target generation alone is not sufficient for higher VSTAR accuracy.

## Conditioned-D Description Readout

Description sanity-check output:

```text
eval_outputs/tgvf_conditioned_image_descriptions/vstar_best_v2_three_way_descriptions_20260530_023644.jsonl
```

The script compared:

- native Qwen image description
- target-conditioned D description
- original Qwen D description
- target-conditioned D with target text in the prompt
- original D with target text in the prompt

Example observations:

```text
sample 0, target: glove
conditional D + target -> "A person is wearing blue gloves."
original D + target    -> "glove"

sample 1, target: the dustpan
conditional D + target -> "The dustpan is on the sidewalk."
original D + target    -> restates the prompt rather than describing evidence

sample 10, target: man's helmet
conditional D + target -> "The man is wearing a yellow helmet."
original D + target    -> restates focused-evidence prompt
```

This supports that target-conditioned D is carrying readable local visual
evidence when the LLM is explicitly asked to describe the focused target.

However, without target text, conditioned D often produces a broad scene
description similar to native image description. The target text remains
important for readout.

The readout debug confirmed real-grid injection:

```text
position_ids_source: source_image_grid_mrope
fake_image_grid_thw: null
image_pad_mm_type_is_image: true
image_position_ids_are_3d: true
visual_tower_called_for_fvt: false
```

## Current Conclusions

1. `tgvf_conditioned_D_fresh` is the correct VSTAR evaluation path for the
   current v2 checkpoint because it clears capture KV cache and lets the LLM see
   the conditioned D plus the full question as a fresh first turn.

2. The best current VSTAR conditioned-D result remains:

```text
old prompt + LLM target: 134/191 = 0.701571
```

3. Direct Qwen remains higher on VSTAR:

```text
direct Qwen: 136/191 = 0.712042
```

4. Deterministic rule targets are technically correct and much cleaner, but the
   best rule run is one sample below old-prompt LLM:

```text
old prompt + rule target: 133/191 = 0.696335
```

5. Matching the training capture prompt did not improve VSTAR. The exact
   training prompt causes severe LLM target-generation failures and slightly
   hurts even with forced rule targets.

6. Full multiple-choice question should be used during capture. Removing answer
   choices changed target hidden states and generally reduced score by about one
   sample in these tests.

7. Original Qwen D as injected D performs poorly and produces a strong A bias in
   some variants. The target-conditioned D path is meaningful, but the final
   VSTAR answer layer remains sensitive to target text and prompt context.

## Recommended Default For Further VSTAR Work

For reporting best current conditioned-D:

```text
force prompt: old prompt
target source: llm
method: tgvf_conditioned_D_fresh
question: full VSTAR question with answer choices
```

For interpretable / leak-free analysis:

```text
force prompt: old prompt
target source: rule
method: tgvf_conditioned_D_fresh
question: full VSTAR question with answer choices
```

For future model work, the likely improvement is not another inference prompt
tweak. The more structural direction is to train/evaluate with target
conditioning that is robust to:

- short object-only targets
- attribute-relation targets
- malformed or partially leaked LLM targets
- prompt-context shifts during target hidden-state capture
