# Clean Project Eval And Mask Notes

This note explains two option surfaces that must remain explicit in the clean
project: post-TGVF continuation and Stage2 original-image-key masking.

## Why This Exists

The historical project mixed several ideas under short labels such as "free",
"softforce", "old mask", and "natural continue". Those labels are not enough to
define an experiment. Clean-project runs must record the exact continuation
mode, forward mode, parser/scorer, and Stage2 mask semantics.

## Post-TGVF Continuation

After focus is captured and D is appended, the evaluator still needs to decide
what text prefix, if any, is appended before answer generation.

Historical continuation modes:

- `natural_continue`
- `answer_only`
- `evidence_then_answer`
- `think_then_answer`

Clean-project benchmark rule:

- Use `natural_continue` only for the clean benchmark runner.
- Keep the other continuation modes only as archive/diagnostic references unless
  the user explicitly reopens them.
- The clean benchmark runner mode surface is limited to `original`,
  `tgvf_free`, `tgvf_force`, and `tgvf_softforce`.
- `original` and `tgvf_free` must not include any extra prompt suffix.
- `tgvf_softforce` may use one explicitly configured soft prompt, but the exact
  text must be recorded in `run_config.txt`.

### `natural_continue`

Protocol C behavior:

```text
<|tgvf_start|>[D visual tokens]<|tgvf_end|>
```

Then the model continues naturally with no extra evidence/answer prefix.

Why it matters:

- The strongest historical 512 TGVF VStar result used this style:
  `20260616 TGVF 512 natural_continue = 52.88`.
- It can produce correct answers even when the output format contains repeated
  or loose fragments.
- It is a real eval option and must not be silently replaced by structured
  continuation.

### Archived Structured Continuations

`evidence_then_answer` and related modes append an explicit trajectory prefix
after D, such as a think/evidence segment before the answer.

Why they matter:

- They are better diagnostic tools for checking whether the model consumes
  correct D differently from random/no/wrong D.
- They are closer to Stage2 supervised trajectory structure.
- They are not automatically comparable to `natural_continue`.

Archive/diagnostic rule:

- Existing results using structured continuation must be labeled and must not be
  compared as the clean benchmark setting.
- Do not keep structured continuation as a clean runner option.

## Protocol / im_end Behavior

The clean Qwen3 path is the im_end-trained tool-observation protocol:

```text
tgvf_protocol=protocol_c_tool_observation
focus_action_im_end=true
toolobs_action_stop=im_end
```

This means the focus action is expected to end with `<|im_end|>` after
`<|focus_end|>`. The TGVF append path must not add a duplicate leading
`<|im_end|>` when the captured action already ended with one.

Old focus-end-only behavior is historical checkpoint compatibility, not clean
mainline behavior.

## Stage2 Original-Image-Key Mask

Stage2 training can block post-TGVF queries from attending to original image
visual keys.

Do not conflate two related but distinct settings:

- training-time original-image-key mask behavior;
- benchmark/inference-time DeepStack `original_image_scope`.

The relevant deterministic dimensions are:

- `mask_original_image_after_tgvf`: enabled/disabled
- scope:
  - `evidence_only`
  - `through_answer`

The stochastic mask probability option must be recorded explicitly. The current
strong Qwen3 Stage2 checkpoint used `mask_original_image_after_tgvf_prob=0.75`
with DeepStack enabled during Stage2 training.

### `through_answer`

The post-TGVF path cannot attend to original image visual keys through the answer
tokens.

Historical note:

- Important early baselines, including `20260619 open-answer row-only`, used old
  behavior equivalent to `through_answer` with effective probability `1.0`.

### `evidence_only`

The post-TGVF path cannot attend to original image visual keys during
evidence/readout, but answer tokens can attend to original image visual keys.

Historical note:

- Existing `evidence_only` experiments are not clean comparisons against
  `through_answer` because other variables changed, including stochastic mask
  probability and/or Stage1 lineage.

Clean-project rule:

- Use `DeepStack enabled + original_image_scope=no_block +
  post_tgvf_forward_mode=kv_cache` as the Qwen3 benchmark/inference mainline.
- Treat older `through_answer + no_kv_full_sequence` benchmark results as
  side/reference unless the task is explicitly reproducing those runs.
- Keep `evidence_only` only as an explicitly named ablation/reference setting.
- Do not use "old mask behavior" as a config label. Write exact tuples, for
  example:

```text
training_mask_original_image_after_tgvf=true
training_mask_scope=through_answer
training_mask_probability=0.75
eval_deepstack_original_image_scope=no_block
eval_post_tgvf_forward_mode=kv_cache
```

## Forward Mode Is Separate

Post-D generation also has a forward-mode question:

- KV continuation
- no-KV full-sequence continuation

This is separate from continuation text. The clean project must label both:

```text
post_tgvf_continuation=natural_continue
post_tgvf_forward_mode=kv_cache
```

or:

```text
post_tgvf_continuation=natural_continue
post_tgvf_forward_mode=no_kv_full_sequence
```

Forward mode must be recorded, but result differences must not be explained as
KV/no-KV effects unless DeepStack state and other inputs are controlled. For
Qwen3, DeepStack on/off is a first-class eval variable.

## Clean Eval Families

Clean project evaluation has three separate families:

- `internal_diagnostic`: Stage1 readout, Stage2 protocol, D/no-D/random-D,
  query sensitivity, and similar probes. These are not benchmark-table runners.
- `project_native_external`: the main clean benchmark runner for experimental
  tables, with TGVF-specific fields such as trigger, focus, D condition,
  parser identity, and DeepStack state.
- `valkit`: first-class external validation family. It remains separate from
  project-native external results unless sample identity, parser/scorer, and
  output semantics are proven equivalent.

## Parser / Scorer Split

Clean eval separates model-output parsing from benchmark scoring.

`model_output_parser` records:

- focus action;
- answer text;
- malformed output;
- trigger state;
- focus-valid state;
- D condition.

`benchmark_scorer` computes accuracy. Official benchmark scorers are preferred
when available. If fallback scoring is used, the output must include:

```text
scorer_name=fallback
official_tool_used=false
```

Fallback scores must not be reported as official benchmark scores.

## Minimum Eval Identity

Every clean eval result should record:

- checkpoint and processor;
- eval family:
  - Stage1/Stage2 diagnostic;
  - project-native external benchmark;
  - ValKit/VLMEvalKit;
- benchmark source and sample identity;
- benchmark population label, for example `blink_all_subtasks_full` or
  `mmmu_pro_standard10_test_full`;
- post-TGVF continuation mode;
- post-TGVF forward mode;
- DeepStack state;
- prompt suffix / softforce prompt text, if any;
- parser/scorer path;
- official scorer status;
- trigger/force policy;
- max image resolution;
- answer parse rate and malformed rate;
- D condition if applicable.
