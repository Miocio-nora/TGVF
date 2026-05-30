# TGVF Inference Pipeline And Prompts

This document records the current benchmark inference paths and every prompt /
instruction template that participates in TGVF inference.

Code entry points:

- Benchmark runner: `src/tgvf_eval/run.py`
- Benchmark model runner: `src/tgvf_eval/model_runner.py`
- Benchmark prompts: `src/tgvf_eval/prompts.py`
- Capture fallback prompt: `src/revisit_vlm/tgvf_capture.py`
- FVT/text answer-turn append prompts: `src/revisit_vlm/tgvf_foveal.py`
- Module inference wrapper: `src/revisit_vlm/tgvf_inference.py`

## Current Benchmark Paths

### 1. `direct_qwen`

Purpose: Qwen2-VL baseline with no foveation and no TGVF module.

Flow:

```text
image/video + direct benchmark prompt
  -> Qwen2-VL generate answer
```

Implementation:

- `QwenTGVFModelRunner._run_direct`
- First user message is built by `_single_user_messages`.
- The text is `build_prompt(sample.question, config).prompt`.
- `build_qwen2vl_tgvf_inputs(..., messages=messages)` is called with explicit
  messages, so capture fallback wrapping is not used.

Prompt content:

```text
{question}

{cot_line_if_enabled}
```

where `cot_line_if_enabled` is either empty or:

```text
Think step by step internally, then provide the final answer in the required format.
```

### 2. `tgvf_prompt_only_force`

Purpose: two-stage foveation protocol without injecting FVT tokens. This isolates
the effect of forced foveation + answer-turn continuation from the effect of FVT
injection.

Flow:

```text
image/video + force foveation prompt
  -> force output <|foveate|>
  -> model generates target text
  -> force output <|/foveate|>
  -> append a new text-only user answer turn into KV cache
  -> continue generation for the answer
```

Implementation:

- `QwenTGVFModelRunner._run_prompt_only`
- Capture is done by `capture_tgvf_single_pass`.
- Force mode uses `ForcedFoveationBracketWrapper`.
- Answer turn is appended by `append_answer_turn_and_open`.
- No FVT tokens are appended.
- No second full forward is used.

First-stage prompt:

```text
{question}

Before answering, select one local visual object or region to inspect. Name the thing to look at, not the answer value.
Good targets are short noun phrases like: the glove, the dustpan, the motorcycle helmet, the Apple logo, the two mentioned objects.
Bad targets include answer options, colors, JSON, pipes, or the words target/visual target description.
Output exactly one foveation request in this format:
<|foveate|><object or region phrase><|/foveate|>
Stop immediately after <|/foveate|>. Do not answer before foveating.
Do not output reasoning, JSON, separators, or tool metadata.
{cot_line_if_enabled}
```

Forced generated sequence:

```text
<|foveate|>{generated_target_text}<|/foveate|>
```

Default multiple-choice answer-turn instruction:

```text
Now answer the original multiple-choice question.
Output exactly one letter: A, B, C, D.
Do not explain.
```

Full appended text-only answer turn:

```text
<|im_end|>
<|im_start|>user
Now answer the original multiple-choice question.
Output exactly one letter: A, B, C, D.
Do not explain.
<|im_end|>
<|im_start|>assistant
```

If `repeat_options_in_continuation=True`, the text-only instruction becomes:

```text
Answer the original multiple-choice question.

Question:
{original_question}

Options:
A. {option_text_0}
B. {option_text_1}
C. {option_text_2}
D. {option_text_3}

Use the original image.
Output exactly one letter: A, B, C, D.
Do not explain.
```

For non-multiple-choice text-only continuation:

```text
Now answer the user's original question. Do not mention this continuation step.
```

### 3. `tgvf_module_force`

Purpose: two-stage foveation protocol with TGVF visual-token injection.

Flow:

```text
image/video + force foveation prompt
  -> force output <|foveate|>
  -> model generates target text
  -> force output <|/foveate|>
  -> capture target hidden states and pre-merge visual tokens
  -> D = TGVF(target_hidden_states, pre_merge_visual_tokens)
  -> append a new user turn containing D as pseudo-image tokens
  -> append FVT answer instruction
  -> continue generation for the answer
```

Implementation:

- `QwenTGVFModelRunner._run_correct_module`
- `run_tgvf_inference`
- `Qwen2VLPreMergeVisualHook` captures pre-merge visual tokens.
- `append_fvt_result_and_open_answer_turn` appends FVT and opens the assistant turn.
- No second full forward is used.

First-stage prompt and forced foveation sequence are identical to
`tgvf_prompt_only_force`.

Default multiple-choice FVT answer instruction:

```text
The visual tokens above are focused evidence for the target:
{target_text}

Use this evidence to answer the original multiple-choice question.
Output exactly one letter: A, B, C, D.
Do not explain.
```

Full appended FVT answer turn:

```text
<|im_end|>
<|im_start|>user
<|vision_start|><image_pad repeated for each FVT token><|vision_end|>
The visual tokens above are focused evidence for the target:
{target_text}

Use this evidence to answer the original multiple-choice question.
Output exactly one letter: A, B, C, D.
Do not explain.
<|im_end|>
<|im_start|>assistant
```

The `<image_pad>` token embeddings are replaced with the TGVF output tensor `D`.

If `repeat_options_in_continuation=True`, the FVT instruction becomes:

```text
The visual tokens above are focused evidence for the target:
{target_text}

Answer the original multiple-choice question.

Question:
{original_question}

Options:
A. {option_text_0}
B. {option_text_1}
C. {option_text_2}
D. {option_text_3}

Use the original image and the focused visual evidence.
Output exactly one letter: A, B, C, D.
Do not explain.
```

If only `benchmark_answer_format == "multiple_choice"` is set and option letters
are not available:

```text
The visual tokens above are focused evidence for the target:
{target_text}

Use this evidence to answer the original multiple-choice question.
Output exactly one option letter.
Do not explain.
```

For non-multiple-choice FVT continuation:

```text
The visual tokens above are focused evidence for the target:
{target_text}

Use this evidence to answer the user's original question.
Do not mention the foveation process.
```

## Message Construction

Current benchmark paths pass explicit messages into `build_qwen2vl_tgvf_inputs`.
This is important because it prevents the lower-level capture helper from wrapping
an already-built benchmark prompt a second time.

Current benchmark first-turn message shape:

```python
[
    {
        "role": "user",
        "content": [
            {"type": "image", "image": sample.primary_media, ...image_budget_kwargs},
            {"type": "text", "text": build_prompt(sample.question, config).prompt},
        ],
    }
]
```

For videos, `_vision_content_items` expands the video into sampled image frames
when frame sampling succeeds.

The processor then applies:

```python
processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

## Benchmark Prompt Templates

These are defined in `src/tgvf_eval/prompts.py`.

### Force Prompt

Used by:

- `tgvf_prompt_only_force`
- `tgvf_module_force`
- `tgvf_no_D`
- `tgvf_random_D`
- `tgvf_wrong_D`

Template:

```text
{question}

Before answering, select one local visual object or region to inspect. Name the thing to look at, not the answer value.
Good targets are short noun phrases like: the glove, the dustpan, the motorcycle helmet, the Apple logo, the two mentioned objects.
Bad targets include answer options, colors, JSON, pipes, or the words target/visual target description.
Output exactly one foveation request in this format:
<|foveate|><object or region phrase><|/foveate|>
Stop immediately after <|/foveate|>. Do not answer before foveating.
Do not output reasoning, JSON, separators, or tool metadata.
{cot_line_if_enabled}
```

### Free Prompt

Used by:

- `tgvf_prompt_only_free`
- `tgvf_module_free`

Template:

```text
{question}

If local visual evidence is needed, output exactly one foveation request in this format:
<|foveate|>target<|/foveate|>
Then stop immediately after <|/foveate|>. If no local visual evidence is needed, answer directly.
{cot_line_if_enabled}
```

In free mode, there is no forced wrapper. If the model does not emit a complete
foveation span, the runner returns the generated text directly.

### Direct Prompt

Used by:

- `direct_qwen`

Template:

```text
{question}

{cot_line_if_enabled}
```

### COT Line

When `cot_enabled=False`, this is empty.

When `cot_enabled=True`:

```text
Think step by step internally, then provide the final answer in the required format.
```

## Capture Fallback Prompt

Defined in `src/revisit_vlm/tgvf_capture.py` as `build_tgvf_prompt`.

Current benchmark eval paths do not use this fallback because they pass explicit
messages into `capture_tgvf_single_pass`. It is still used by lower-level scripts
or callers that call `build_qwen2vl_tgvf_inputs(..., messages=None)`.

Template:

```text
{question}

If fine-grained visual evidence is needed, output exactly one foveation request:
<|foveate|>visual target description<|/foveate|>
After <|/foveate|>, stop.
Do not answer the question yet.
Do not emit intent, mode, scope, JSON, tool metadata, explanations, or natural-language commentary.
Only emit the foveation request.
```

Important: benchmark eval should not pass an already-built force/free prompt as
`question` while leaving `messages=None`, because that would double-wrap the
prompt.

## Text-Only Answer Instructions

Defined in `build_text_answer_instruction`.

### With Repeated Options

Condition:

- `repeat_options_in_continuation=True`
- `option_texts` is present

Template:

```text
Answer the original multiple-choice question.

Question:
{original_question}

Options:
{letter_0}. {option_text_0}
{letter_1}. {option_text_1}
...

Use the original image.
Output exactly one letter: {letter_0}, {letter_1}, ...
Do not explain.
```

### With Option Letters

Condition:

- `option_letters` is present
- `repeat_options_in_continuation=False`

Template:

```text
Now answer the original multiple-choice question.
Output exactly one letter: {letters}.
Do not explain.
```

For V*Bench with four choices, this becomes:

```text
Now answer the original multiple-choice question.
Output exactly one letter: A, B, C, D.
Do not explain.
```

### Multiple Choice Without Explicit Letters

Condition:

- `benchmark_answer_format == "multiple_choice"`
- `option_letters` is not present

Template:

```text
Now answer the original multiple-choice question.
Output exactly one option letter.
Do not explain.
```

### Open Answer

Template:

```text
Now answer the user's original question. Do not mention this continuation step.
```

## FVT Answer Instructions

Defined in `build_fvt_answer_instruction`.

### With Repeated Options

Condition:

- `repeat_options_in_continuation=True`
- `option_texts` is present

Template:

```text
The visual tokens above are focused evidence for the target:
{target_text}

Answer the original multiple-choice question.

Question:
{original_question}

Options:
{letter_0}. {option_text_0}
{letter_1}. {option_text_1}
...

Use the original image and the focused visual evidence.
Output exactly one letter: {letter_0}, {letter_1}, ...
Do not explain.
```

### With Option Letters

Condition:

- `option_letters` is present
- `repeat_options_in_continuation=False`

Template:

```text
The visual tokens above are focused evidence for the target:
{target_text}

Use this evidence to answer the original multiple-choice question.
Output exactly one letter: {letters}.
Do not explain.
```

For V*Bench with four choices, this becomes:

```text
The visual tokens above are focused evidence for the target:
{target_text}

Use this evidence to answer the original multiple-choice question.
Output exactly one letter: A, B, C, D.
Do not explain.
```

### Multiple Choice Without Explicit Letters

Condition:

- `benchmark_answer_format == "multiple_choice"`
- `option_letters` is not present

Template:

```text
The visual tokens above are focused evidence for the target:
{target_text}

Use this evidence to answer the original multiple-choice question.
Output exactly one option letter.
Do not explain.
```

### Open Answer

Template:

```text
The visual tokens above are focused evidence for the target:
{target_text}

Use this evidence to answer the user's original question.
Do not mention the foveation process.
```

## Legacy / Low-Level Continuation Instruction

Defined in both `src/tgvf_eval/prompts.py` and `src/revisit_vlm/tgvf_inference.py`.

Current normal benchmark module inference does not use this as the appended
answer instruction; it uses `append_fvt_result_and_open_answer_turn` instead.
The string remains for lower-level or legacy paths.

Template:

```text
The newly provided visual tokens are focused evidence for the target you requested. Use them to answer the user's original question. Do not mention the foveation process.
```

## FVT Position / Grid Behavior

For `fvt_append_mode="qwen_native_pseudo_image"`, FVT is appended as a native
Qwen2-VL pseudo image:

```text
<|vision_start|><image_pad repeated M times><|vision_end|>
```

Then the `<image_pad>` embeddings are replaced by `D`.

Current behavior:

- v1 fixed-M modules use a fake near-square grid, for example M=128 gives
  `[[1, 16, 32]]`.
- v2 modules condition Qwen pre-merge visual tokens before merge. The common
  inference path then calls the frozen Qwen visual merger to produce the final
  LLM-side `D`, sets `debug_metadata["tgvf_version"] == "v2"`, and passes
  `capture.image_grid_thw` to the append path, so the FVT append can preserve
  real source grid metadata.

## Control Conditions

Current control module modes use the same answer-turn protocol as the correct
module path:

- `tgvf_no_D`: force foveation, then append text-only answer turn.
- `tgvf_random_D`: force foveation, compute D shape, replace D with random
  tensor, append FVT answer turn.
- `tgvf_wrong_D`: force foveation, compute D shape, locally shuffle/flip D,
  append FVT answer turn.

The intended control variable is now D itself, not the surrounding chat protocol.
