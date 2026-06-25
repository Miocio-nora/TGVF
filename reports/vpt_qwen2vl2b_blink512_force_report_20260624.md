# VPT/Qwen2-VL-2B BLINK-512 Diagnostic Report

Date: 2026-06-24  
Workspace: `/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm`  
Benchmark: BLINK local validation full, 1901 samples  
Image setting: `max_image_resolution=512`, `max_pixels=262144`  
Scorer: project fallback scorer; official BLINK scorer path exists but is not wired in this evaluator

## Purpose

The original VPT prompt-trigger tests looked suspicious because both soft-force prompts
returned 0 action-token triggers. We ran explicit forced-action diagnostics to separate:

- self-trigger failure,
- second-round input construction failure,
- model/checkpoint behavior after a clean second-round execution.

## Evaluated Systems

| System | Model | Prompt/action setting |
| --- | --- | --- |
| Qwen2-VL baseline | `Qwen/Qwen2-VL-2B-Instruct` | direct answer, no tool |
| VPT direct old compat | `rp-yu/Qwen2-VL-2b-VPT-CLIP` | direct answer, old compatibility shim |
| VPT soft-force reencode | `rp-yu/Qwen2-VL-2b-VPT-CLIP` | prompt phrase: `Require additional perception features...` |
| VPT soft-force region | `rp-yu/Qwen2-VL-2b-VPT-CLIP` | prompt phrase: `Identify the region...` |
| VPT direct fixed | `rp-yu/Qwen2-VL-2b-VPT-CLIP` | direct answer, fixed visual unwrap |
| VPT force CLIP | `rp-yu/Qwen2-VL-2b-VPT-CLIP` | forced `<|clip_action_start|><|clip_action|><|clip_action_end|>` |
| VPT force full-region | `rp-yu/Qwen2-VL-2b-VPT-CLIP` | forced full 8x8 region token span |

## Main Results

| Run | Accuracy | Trigger | Second round | Errors | Valid baseline? |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen2-VL-2B direct | 41.35 | n/a | n/a | 0 | Yes |
| VPT direct, old compat unwrap | 37.30 | 0/1901 | 0/1901 | 0 | No |
| VPT soft-force reencode prompt | 36.61 | 0/1901 | 0/1901 | 0 | No, old compat |
| VPT soft-force region prompt | 36.30 | 0/1901 | 0/1901 | 0 | No, old compat |
| VPT direct, fixed `pooler_output` unwrap | 43.82 | 0/1901 | 0/1901 | 0 | Yes |
| VPT force CLIP, fixed unwrap | 43.66 | 1901/1901 | 1901/1901 | 0 | Yes |
| VPT force full-region, fixed unwrap | 44.08 | 1901/1901 | 1901/1901 | 0 | Sanity only |

## Bugs Found

### 1. CLIP second-round processor incompatibility

The initial force-CLIP smoke failed on every row:

```text
TypeError: Qwen2VLImageProcessorKwargs.__init__() got an unexpected keyword argument 'videos'
```

Cause: the compatibility evaluator passed `videos=None` into the current
`Qwen2VLImageProcessor`, which no longer accepts that argument.

Fix: call the image processor without `videos` when the current implementation rejects it.

### 2. Wrong visual output selected for current HF Qwen2-VL

After fixing the processor call, force-CLIP hit a token split mismatch:

```text
split_with_sizes expects split_sizes to sum exactly to 3808, but got split_sizes=[315, 315, 322]
```

Cause: current HF Qwen2-VL visual tower returns:

- `last_hidden_state`: pre-merge visual patch tokens,
- `pooler_output`: merged visual tokens.

VPT's projector computes expected per-image token counts as:

```text
grid_thw.prod // spatial_merge_size^2
```

That is the merged-token scale. The old compatibility shim used `last_hidden_state`,
which is the wrong scale for the current HF output object.

Fix: prefer `pooler_output` when available; fall back to `last_hidden_state` only when
there is no pooled/merged output.

## Interpretation

The user's suspicion was correct: the earlier VPT direct/soft-force numbers should not
be treated as exact baselines. They were affected by a compatibility shim bug, and the
soft-force prompts did not exercise the second-round path at all because trigger rate was
0/1901.

After the compatibility fix, VPT direct reaches 43.82, above the Qwen2-VL direct baseline
of 41.35. Forced CLIP runs cleanly but does not improve over corrected VPT direct
under this BLINK-512 setup: 43.66 vs 43.82.

The full-region force score is 44.08, but this variant simply sends a full 8x8 crop of
the first image through the region pathway. BLINK often uses multiple images, so this is
a pipeline sanity check rather than a fair region-selection result.

## Output Artifacts

- Qwen2 direct baseline:
  `eval_outputs/qwen2vl2b_blink_full_512_20260623_215737/merged/blink__direct_qwen.merged_summary.json`
- VPT old direct:
  `eval_outputs/vpt_qwen2vl2b_clip_blink_full_512_20260624_000122/merged/blink__vpt_clip_direct.merged_summary.json`
- VPT soft-force reencode:
  `eval_outputs/vpt_qwen2vl2b_clip_blink_softforce_reencode_512_20260624_005920/merged/blink__vpt_clip_softforce_reencode.merged_summary.json`
- VPT soft-force region:
  `eval_outputs/vpt_qwen2vl2b_clip_blink_softforce_region_512_20260624_005920/merged/blink__vpt_clip_softforce_region.merged_summary.json`
- VPT force CLIP:
  `eval_outputs/vpt_qwen2vl2b_clip_blink_force_clip_512_20260624_012010/merged/blink__vpt_clip_force_clip.merged_summary.json`
- VPT force full-region:
  `eval_outputs/vpt_qwen2vl2b_clip_blink_force_region_full_512_20260624_012010/merged/blink__vpt_clip_force_region_full.merged_summary.json`
- VPT fixed direct:
  `eval_outputs/vpt_qwen2vl2b_clip_blink_direct_pooler_512_20260624_013020/merged/blink__vpt_clip_direct_pooler.merged_summary.json`

## Practical Consequences

1. Any future VPT/Qwen2-VL evaluation on the current HF stack must use merged visual
   tokens from `pooler_output`.
2. The old 37.30 VPT direct result is invalid as a current baseline.
3. Self-triggering remains unsolved for this checkpoint: direct/soft prompt runs emitted
   0 action tokens.
4. Forced CLIP is runnable and clean, but it is not beneficial when forced globally on
   every BLINK sample.
