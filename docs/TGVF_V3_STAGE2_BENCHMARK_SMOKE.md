# TGVF-v3 Stage2 Benchmark Smoke Notes

Date: 2026-06-01

This note records the current TGVF-v3 Stage2 benchmark smoke setup and the main results observed so far.

## Version Label

This set of experiments should be referred to as:

`TGVF-v3 Stage2 Qwen3-VL-Thinking evidence-trajectory smoke`

Backbone:

`Qwen3-VL-8B-Thinking`

Stage2 checkpoint:

`outputs/tgvf_v3_stage2_8b/stage2_8b_fast_8gpu_1200step_full/train/checkpoint_step_1200.pt`

Important scope:

- These are smoke / diagnostic benchmark runs.
- They are not final official benchmark submissions.
- The main purpose is to compare protocol stability and `correct_D` versus controls.
- The runs use local Transformers inference, not hosted serving APIs.

## Visible Protocol

The v3 protocol uses plain-text markers, not tokenizer special tokens:

```text
<EVIDENCE_STATE>...</EVIDENCE_STATE>
<FOCUS>...</FOCUS>
<TGVF>...</TGVF>
<EVIDENCE>...</EVIDENCE>
<ANSWER>...</ANSWER>
```

The markers are ordinary tokenized text. No tokenizer resize was used.

For focus samples, the foveation query is only the text span inside:

```text
<FOCUS>target text</FOCUS>
```

The marker tokens are excluded from `H_q`.

## Inference Pattern

The current useful force-TGVF pattern is:

```text
User:
[image]
Question with choices

Assistant forced prefix:
<EVIDENCE_STATE>need_local_visual_evidence</EVIDENCE_STATE>
<FOCUS>

Model generates:
target</FOCUS>

Runtime:
capture H_q from target text
compute D = TGVF(H_q, V_pre)
append:
<TGVF>[visual D tokens]</TGVF>

Continuation:
<EVIDENCE>

Model generates:
evidence text
</EVIDENCE>
<ANSWER>...</ANSWER>
```

This is the current preferred benchmark diagnostic continuation:

`evidence_then_answer`

The earlier `answer_only` continuation:

```text
<TGVF>[D]</TGVF>
<ANSWER>
```

was less aligned with Stage2 training. It often caused the model to output answer-like fragments or option text without a stable evidence trajectory, especially on MMMU-Pro.

## Compared Methods

The current standard comparison set is:

| Method | Meaning |
|---|---|
| `base_direct_qwen3_think` | Original Qwen3-VL-8B-Thinking, no Stage2 checkpoint, normal thinking mode |
| `base_direct_qwen3_no_think` | Original Qwen3-VL-8B-Thinking, no Stage2 checkpoint, thinking closed with `</think>` prefill |
| `stage2_direct_qwen3` | Stage2 checkpoint loaded, direct answer path, no TGVF trigger |
| `stage2_force_correct_D` | Stage2 checkpoint, forced focus, target-conditioned TGVF D, `evidence_then_answer` |
| `stage2_force_random_D` | Stage2 checkpoint, forced focus, random calibrated D control, `evidence_then_answer` |

Important naming rule:

- `base_direct_qwen3_*` means no LoRA / no Stage2 checkpoint.
- `stage2_direct_qwen3` means the Stage2 checkpoint is loaded but TGVF is not triggered.
- `stage2_force_*` means the Stage2 checkpoint is loaded and the force focus path is used.

## Parser and Scoring

For multiple-choice benchmarks, the strict parser checks:

- explicit `<ANSWER>A</ANSWER>` style output;
- answer phrases such as `answer is A`;
- single-line option letter output;
- fallback option-text matching.

The fallback option-text matching matters because the model often outputs content such as:

```text
red plastic stool
```

instead of:

```text
D
```

This is acceptable for smoke diagnostics, but official benchmark reporting should use each benchmark's official parser/scorer where available.

## Dataset Mismatch Found

The current v3 Stage2 train data has almost no benchmark-style multiple-choice format.

Train 50k:

| Field | Count |
|---|---:|
| total rows | 50,022 |
| focus rows | 35,542 |
| no-focus rows | 14,480 |
| `answer_format=multiple_choice` | 57 |
| non-empty `choices` field | 0 |

Val 2k:

| Field | Count |
|---|---:|
| total rows | 2,023 |
| focus rows | 1,382 |
| no-focus rows | 641 |
| `answer_format=multiple_choice` | 2 |
| non-empty `choices` field | 0 |

This explains why benchmark outputs often contain answer content or evidence text instead of a clean option letter.

## VSTAR Full Results

Dataset:

`/nvmesv/dredvpn009/datasets/benchmarks/vstar_bench/snapshot/test_questions.jsonl`

Size:

`191`

### Evidence-then-answer continuation

Output:

`outputs/tgvf_v3_stage2_8b/vstar_full_191_20260601_force_evidence_then_answer/merged_summary_strict.json`

| Method | Accuracy | Parse rate |
|---|---:|---:|
| `base_direct_qwen3_think` | 49.74% | 100.00% |
| `base_direct_qwen3_no_think` | 45.03% | 85.34% |
| `stage2_direct_qwen3` | 49.74% | 100.00% |
| `stage2_force_correct_D` | 53.40% | 95.29% |
| `stage2_force_random_D` | 48.69% | 91.10% |

Category breakdown:

| Method | direct_attributes | relative_position |
|---|---:|---:|
| `stage2_direct_qwen3` | 44.35% | 57.89% |
| `stage2_force_correct_D` | 48.70% | 60.53% |
| `stage2_force_random_D` | 43.48% | 56.58% |

Protocol metrics:

| Method | focus valid | append success | evidence close | answer tag |
|---|---:|---:|---:|---:|
| `stage2_force_correct_D` | 100.00% | 100.00% | 47.12% | 55.50% |
| `stage2_force_random_D` | 100.00% | 100.00% | 25.13% | 35.08% |

Main VSTAR conclusion:

- `correct_D` beats `random_D` by `+4.71 pts`.
- `correct_D` beats `stage2_direct_qwen3` by `+3.66 pts`.
- Both VSTAR subcategories improve under `correct_D`.
- The evidence/answer formatting is not fully stable yet, but the correct-D control is meaningfully better than random-D.

### Answer-only continuation

Earlier answer-only runs were less diagnostic:

| Method | Accuracy |
|---|---:|
| `stage2_force_correct_D answer_only` | 51.31% |
| `stage2_force_random_D answer_only` | 51.83% |

In answer-only mode, `random_D` was not worse than `correct_D`, so that setting did not support the D-content claim.

## MMMU-Pro 300 Results

Dataset adapter:

`mmmu_pro`

Current local preferred file:

`standard (10 options)/test-00000-of-00002.parquet`

Sample count:

`300`

Output:

`outputs/tgvf_v3_stage2_8b/mmmu_pro_300_20260601_stage2_force_evidence/merged_summary.json`

### Evidence-then-answer continuation

| Method | Accuracy | Parse rate |
|---|---:|---:|
| `base_direct_qwen3_think` | 17.33% | 51.33% |
| `base_direct_qwen3_no_think` | 26.33% | 63.00% |
| `stage2_direct_qwen3` | 33.33% | 92.33% |
| `stage2_force_correct_D` | 33.67% | 88.00% |
| `stage2_force_random_D` | 25.00% | 76.00% |

Protocol metrics:

| Method | focus valid | append success | evidence close | answer tag |
|---|---:|---:|---:|---:|
| `stage2_force_correct_D` | 100.00% | 100.00% | 81.67% | 82.33% |
| `stage2_force_random_D` | 100.00% | 100.00% | 59.00% | 66.67% |

Main MMMU-Pro conclusion:

- `correct_D` beats `random_D` by `+8.67 pts`.
- `correct_D` is roughly tied with `stage2_direct_qwen3`.
- Stage2 improves output format substantially versus base Qwen3 under the current parser.
- The correct-D versus random-D gap is the most important positive signal here.

### Long-thinking baseline rerun, 2026-06-02

The first `base_direct_qwen3_think` MMMU-Pro run used only `max_answer_tokens=256`.
That setting truncated many thinking outputs before the model reached a final option.

Rerun:

`outputs/qwen3_base_direct/mmmu_pro_300_20260602_think_maxtok1024/merged_summary.json`

| Method | Max answer tokens | Accuracy | Parse rate | `<|im_end|>` rate | Avg output chars |
|---|---:|---:|---:|---:|---:|
| `base_direct_qwen3_think` old | 256 | 17.33% | 51.33% | 9.67% | 991.3 |
| `base_direct_qwen3_think` rerun | 1024 | 34.00% | 99.67% | 38.67% | 2984.7 |

Interpretation:

- The original `256`-token thinking baseline was severely underestimated.
- With `1024` tokens, base Qwen3-Thinking reaches `34.00%`, roughly matching `stage2_direct_qwen3` at `33.33%` and `stage2_force_correct_D` at `33.67%` under the current fallback parser.
- Future comparisons on reasoning-heavy benchmarks must specify the thinking token budget.
- The TGVF control claim remains: `stage2_force_correct_D` still beats `stage2_force_random_D` by `+8.67 pts` in the Stage2 force setting.

Additional long-thinking reruns:

| Benchmark | Max answer tokens | Accuracy | Parse rate | `<|im_end|>` rate | Avg output chars |
|---|---:|---:|---:|---:|---:|
| `mathvista` | 1024 | 34.67% | 100.00% | 65.67% | 2185.2 |
| `mathverse` | 2048 | 25.33% | 100.00% | 51.67% | 5128.6 |

Conclusion:

- The original short-token `base_direct_qwen3_think` rows for MMMU-Pro, MathVista, and MathVerse were token-budget limited.
- Future base-thinking comparisons should use the long-token rows, not the earlier 256-token rows.

## Additional 5-Variant Benchmark Diagnostics, 2026-06-01 to 2026-06-02

This section records the complete five-method diagnostic results for the newer external benchmark runs.

Important caveat:

- These are still smoke / diagnostic results.
- They use the current local fallback parsers and scorers unless noted otherwise.
- They are not official benchmark submissions.
- The most reliable TGVF-specific diagnostic remains `force_correct_D` versus `force_random_D`.

The five compared variants are:

| Variant | Meaning |
|---|---|
| `base_direct_qwen3_think` | Base Qwen3-VL-8B-Thinking, no Stage2 checkpoint, thinking enabled |
| `base_direct_qwen3_no_think` | Base Qwen3-VL-8B-Thinking, no Stage2 checkpoint, thinking suppressed |
| `stage2_direct_qwen3` | Stage2 checkpoint loaded, direct answer path, no TGVF trigger |
| `stage2_force_correct_D` | Stage2 checkpoint, forced focus, target-conditioned TGVF D |
| `stage2_force_random_D` | Stage2 checkpoint, forced focus, random calibrated D control |

### OCRBench v2 300

Scope:

`ocrbench_v2`, light sample, 300 examples.

Scorer:

`fallback_normalized_exact`

Outputs:

- `outputs/tgvf_v3_stage2_8b/ocrbench_v2_300_20260601_stage2_force_evidence`
- `outputs/qwen3_base_direct/ocrbench_v2_300_20260601_think`
- `outputs/qwen3_base_direct/ocrbench_v2_300_20260601_nothink`

| Variant | Accuracy | Parse rate |
|---|---:|---:|
| `base_direct_qwen3_think` | 0.00% | 100.00% |
| `base_direct_qwen3_no_think` | 0.00% | 100.00% |
| `stage2_direct_qwen3` | 32.33% | 100.00% |
| `stage2_force_correct_D` | 20.00% | 100.00% |
| `stage2_force_random_D` | 12.33% | 100.00% |

Interpretation:

- `stage2_direct_qwen3` is strongest under the current fallback exact scorer.
- `stage2_force_correct_D` beats `stage2_force_random_D` by `+7.67 pts`.
- Base direct scoring is not reliable here because OCRBench v2 is open OCR / structured extraction and the base outputs are not normalized into the benchmark's expected answer format.
- This should not be reported as official OCRBench v2 performance without the official scorer / answer normalizer.

### BLINK Counting Validation 120

Scope:

`blink`, local `Counting/val`, 120 examples.

Reason for validation split:

The local BLINK test split has hidden labels. The adapter was switched to `Counting/val` for local scored diagnostics.

Scorer:

`fallback_mcq`

Outputs:

- `outputs/tgvf_v3_stage2_8b/blink_counting_val_120_20260601_stage2_force_evidence`
- `outputs/qwen3_base_direct/blink_counting_val_120_20260601_think`
- `outputs/qwen3_base_direct/blink_counting_val_120_20260601_nothink`

| Variant | Accuracy | Parse rate |
|---|---:|---:|
| `base_direct_qwen3_think` | 60.83% | 96.67% |
| `base_direct_qwen3_no_think` | 64.17% | 99.17% |
| `stage2_direct_qwen3` | 62.50% | 100.00% |
| `stage2_force_correct_D` | 55.00% | 94.17% |
| `stage2_force_random_D` | 51.67% | 90.83% |

Interpretation:

- `base_direct_qwen3_no_think` is strongest on this BLINK Counting subset.
- `stage2_direct_qwen3` is close to base no-think and above base think.
- `stage2_force_correct_D` beats `stage2_force_random_D` by `+3.33 pts`.
- This is not a full BLINK result; it is only the Counting validation subset.

### MathVista Testmini 300

Scope:

`mathvista`, testmini sample, 300 examples.

Scorer:

`fallback_mixed_mcq_open_exact`

Outputs:

- `outputs/tgvf_v3_stage2_8b/mathvista_300_20260601_stage2_force_evidence`
- `outputs/qwen3_base_direct/mathvista_300_20260601_think`
- `outputs/qwen3_base_direct/mathvista_300_20260601_nothink`

| Variant | Accuracy | Parse rate |
|---|---:|---:|
| `base_direct_qwen3_think` | 24.33% | 99.67% |
| `base_direct_qwen3_no_think` | 28.00% | 97.33% |
| `stage2_direct_qwen3` | 54.00% | 99.67% |
| `stage2_force_correct_D` | 41.67% | 98.33% |
| `stage2_force_random_D` | 35.33% | 97.00% |

Interpretation:

- `stage2_direct_qwen3` is strongest under the fallback mixed scorer.
- `stage2_force_correct_D` beats `stage2_force_random_D` by `+6.33 pts`.
- MathVista contains a mix of multiple-choice and free-form answers. The fallback open-answer exact scorer is rough and can under-credit long-form correct reasoning.
- The original base-thinking row used a short token budget and should be treated as an under-estimate.

Long-thinking baseline rerun:

`outputs/qwen3_base_direct/mathvista_300_20260602_think_maxtok1024/merged_summary.json`

| Method | Max answer tokens | Accuracy | Parse rate | `<|im_end|>` rate | Avg output chars |
|---|---:|---:|---:|---:|---:|
| `base_direct_qwen3_think` old | 256 | 24.33% | 99.67% | not recorded here | not recorded here |
| `base_direct_qwen3_think` rerun | 1024 | 34.67% | 100.00% | 65.67% | 2185.2 |

Updated MathVista interpretation:

- Long-token base Qwen3-Thinking improves from `24.33%` to `34.67%`.
- `stage2_direct_qwen3` remains strongest at `54.00%` under the current fallback scorer.
- `stage2_force_correct_D` remains above `stage2_force_random_D` by `+6.33 pts`.

### MathVerse 300

Scope:

`mathverse`, testmini sample, 300 examples.

Scorer:

Current fallback parser/scorer. MathVerse gold answers are mixed and include option letters, formulas, numbers, and free-form strings.

Outputs:

- `outputs/tgvf_v3_stage2_8b/mathverse_300_20260601_stage2_force_evidence`
- `outputs/qwen3_base_direct/mathverse_300_20260601_think`
- `outputs/qwen3_base_direct/mathverse_300_20260601_nothink`

| Variant | Accuracy | Parse rate |
|---|---:|---:|
| `base_direct_qwen3_think` | 4.67% | 19.67% |
| `base_direct_qwen3_no_think` | 2.33% | 9.67% |
| `stage2_direct_qwen3` | 24.00% | 43.00% |
| `stage2_force_correct_D` | 20.00% | 40.67% |
| `stage2_force_random_D` | 12.67% | 32.00% |

Interpretation:

- `stage2_direct_qwen3` is strongest under the current fallback parser.
- `stage2_force_correct_D` beats `stage2_force_random_D` by `+7.33 pts`.
- The original base thinking rows are heavily token-limited / parse-limited.
- The original base-thinking row should be treated as an under-estimate.

Long-thinking baseline rerun:

`outputs/qwen3_base_direct/mathverse_300_20260602_think_maxtok2048/merged_summary.json`

| Method | Max answer tokens | Accuracy | Parse rate | `<|im_end|>` rate | Avg output chars |
|---|---:|---:|---:|---:|---:|
| `base_direct_qwen3_think` old | 256 | 4.67% | 19.67% | not recorded here | not recorded here |
| `base_direct_qwen3_think` rerun | 2048 | 25.33% | 100.00% | 51.67% | 5128.6 |

Updated MathVerse interpretation:

- Long-token base Qwen3-Thinking improves from `4.67%` to `25.33%`.
- `stage2_direct_qwen3` is close at `24.00%` under the current fallback scorer.
- `stage2_force_correct_D` remains above `stage2_force_random_D` by `+7.33 pts`.

## HR-Bench 4K 300 Results

Dataset adapter:

`hr_bench_4k`

Sample count:

`300`

Output:

`outputs/tgvf_v3_stage2_8b/hr_bench_4k_300_20260601_stage2_force_evidence/merged_summary.json`

### Evidence-then-answer continuation

| Method | Accuracy | Parse rate |
|---|---:|---:|
| `base_direct_qwen3_think` | 48.00% | 89.33% |
| `base_direct_qwen3_no_think` | 47.00% | 87.00% |
| `stage2_direct_qwen3` | 48.00% | 100.00% |
| `stage2_force_correct_D` | 49.67% | 92.00% |
| `stage2_force_random_D` | 44.33% | 83.67% |

Protocol metrics:

| Method | focus valid | append success | evidence close | answer tag |
|---|---:|---:|---:|---:|
| `stage2_force_correct_D` | 100.00% | 100.00% | 68.00% | 76.00% |
| `stage2_force_random_D` | 100.00% | 100.00% | 46.00% | 60.33% |

Main HR-Bench conclusion:

- `correct_D` beats `random_D` by `+5.33 pts`.
- `correct_D` beats `stage2_direct_qwen3` by `+1.67 pts`.
- `correct_D` beats `base_direct_qwen3_think` by `+1.67 pts`.
- Stage2 direct matches base direct think in accuracy but has better parse stability.
- The correct-D path also has better evidence/answer format stability than random-D.

## Why Evidence-then-answer Works Better

Stage2 was trained on focus trajectories shaped like:

```text
<EVIDENCE_STATE>need_local_visual_evidence</EVIDENCE_STATE>
<FOCUS>target</FOCUS>
<TGVF>[D]</TGVF>
<EVIDENCE>evidence_description</EVIDENCE>
<ANSWER>answer</ANSWER>
```

Therefore, opening:

```text
<EVIDENCE>
```

after the TGVF block matches the training distribution better than opening:

```text
<ANSWER>
```

This was especially clear on MMMU-Pro:

| Method | answer-only acc | evidence-then-answer acc |
|---|---:|---:|
| `stage2_force_correct_D` | 6.25% on 16-sample smoke | 50.00% on 16-sample smoke |
| `stage2_force_random_D` | 6.25% on 16-sample smoke | 43.75% on 16-sample smoke |

## Current Interpretation

Positive findings:

- The v3 force focus protocol is mechanically stable.
- Single-pass capture is active; benchmark summaries report `second_full_forward_used_any=false`.
- `correct_D > random_D` on VSTAR full, MMMU-Pro 300, HR-Bench 4K 300, OCRBench v2 300, BLINK Counting val 120, MathVista 300, and MathVerse 300 when using `evidence_then_answer`.
- The Stage2 checkpoint strongly improves protocol formatting versus base Qwen3 on MMMU-Pro.

Limitations:

- The current Stage2 dataset has almost no real benchmark-style multiple-choice training.
- Option-letter output is not fully stable.
- Evidence close and answer tag rates are still imperfect.
- Some benchmark gains may come from Stage2 protocol/prompt distribution rather than D content alone, so random-D controls are required.
- Base Qwen3-Thinking must be run with an adequate token budget on reasoning-heavy benchmarks; the earlier 256-token rows underestimated the base model.
- Full official benchmark reporting still needs benchmark-specific official scorers/parsers.

## Immediate Next Steps

Recommended next work:

1. Add a Stage2 MCQ adaptation dataset with explicit `choices` and letter-only answer targets.
2. Keep `evidence_then_answer` as the default force-TGVF benchmark continuation.
3. Report every benchmark with `correct_D` and `random_D` controls.
4. Add `--skip-direct` and shared capture/control execution to speed up force benchmark runs.
5. Re-run formal benchmark reports with official parsers/scorers where available.
