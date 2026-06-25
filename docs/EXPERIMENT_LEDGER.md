# TGVF Experiment Ledger

This file is the source of truth for experiment intent, configuration, results, and analysis.
Do not rely on chat context or script defaults for experiment identity.

## Required Process

Before starting any training or evaluation run:

1. Add a `PLANNED` entry in this file.
2. Fill in the exact baseline anchor and the exact intended diff.
3. List all critical paths: Stage1 checkpoint, Stage1 processor, Stage2 checkpoint or output, train data, eval data, benchmark output.
4. State which variables are allowed to change.
5. If the run is an ablation, verify that no other variables changed.

After the run starts:

1. Update status to `RUNNING`.
2. Add tmux session name, command/script, GPU set, and timestamp.

After the run finishes:

1. Update status to `DONE`, `FAILED`, or `STOPPED`.
2. Add key metrics, output paths, and elapsed times.
3. Add conclusion and whether the result is comparable to the intended baseline.

If a mistake is found:

1. Mark the entry as `INVALID_FOR_<baseline>` or `SIDE_RESULT`.
2. Explain why it is invalid.
3. Do not use the result as evidence for that baseline.

## Entry Template

```text
### EXP-YYYYMMDD-HHMM-short-name

- Status:
- Question:
- Baseline anchor:
- Intended diff:
- Allowed changed variables:
- Not allowed to change:
- Code commit / worktree:
- Stage1 checkpoint:
- Stage1 processor:
- Stage2 checkpoint/output:
- Train data:
- Validation data:
- Benchmark output:
- Script / command:
- GPUs:
- tmux:
- Started:
- Finished:
- Metrics:
- Analysis:
- Conclusion:
- Comparable to baseline:
- Follow-up:
```

## Baseline Anchors

### BASE-20260619-open-answer-rowonly

- Purpose: Best known row-only/open-answer VStar baseline before current mask-prob ablations.
- Stage1 checkpoint: `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_imend_bidirectional_2gpu_bs4_accum4_gbs32_2000step_20260619_014148/train/checkpoint_step_2000.pt`
- Stage1 processor: `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_imend_bidirectional_2gpu_bs4_accum4_gbs32_2000step_20260619_014148/train/processor_step_2000`
- Stage1 type: row-only/default token-row setup, not `full_mask`.
- Stage2 checkpoint: `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt`
- Stage2 processor: `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200`
- Stage2 train data: `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl`
- Stage2 val data: `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`
- Old mask behavior: old code default, `mask_original_image_after_tgvf=True`, no probability/scope config.
- VStar free result: 51.31 overall, 45.22 direct, 60.53 relative.
- VStar softforce result: 47.64 overall, 42.61 direct, 55.26 relative.
- Source files:
  - `outputs/vlmevalkit/vstar_512_open_answer_2gpu_20260619_014148/TGVF-Qwen3VL-8B-ToolObs-Env-Free-512/T20260619-122819/status.json`
  - `outputs/vlmevalkit/vstar_512_open_answer_softforce_20260619_143025/TGVF-Qwen3VL-8B-ToolObs-Env-Free-512/T20260619-142751/status.json`

### BASE-20260617-rowonly-original-stage2

- Purpose: Older row-only Stage1 + original clean_imend Stage2 baseline.
- Stage1 checkpoint: `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617/train/checkpoint_step_2000.pt`
- Stage1 processor: `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617/train/processor_step_2000`
- Stage2 checkpoint: `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/checkpoint_step_1200.pt`
- Stage2 train data: `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl`
- VStar free results:
  - 45.55 overall, 39.13 direct, 55.26 relative.
  - 47.12 overall, 40.87 direct, 56.58 relative on later rerun.
- Protocol eval:
  - no_focus_direct: 78.13
  - force correct_D: 89.84
  - teacher-forced correct_D: 92.97
- Important note: strong protocol eval did not transfer to VStar free.

### BASE-20260621-fullmask

- Purpose: Full-mask Stage1/Stage2 reference.
- Stage1 checkpoint: `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_imend_fullmask_bidirectional_2gpu_bs4_accum4_gbs32_2000step_20260620_230349/train/checkpoint_step_2000.pt`
- Stage2 checkpoint: `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_fullmask_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260620_230349/checkpoint_step_1200.pt`
- Full benchmark output: `outputs/full_benchmarks_fullmask_4gpu_20260621_101055`
- VStar free: 50.79 overall, 40.87 direct, 65.79 relative, trigger 44.50.
- VStar softforce: 50.79 overall, 45.22 direct, 59.21 relative, trigger 81.15.
- HR free: 54.00, trigger 17.13.
- HR softforce: 53.75, trigger 74.63.
- OCR free: 20.61, trigger 3.32.
- OCR softforce: 19.95, trigger 48.72.
- BLINK free: 58.33, trigger 5.00.
- BLINK softforce: 55.00, trigger 59.17.

## Experiment Entries

### EXP-20260622-021650-maskprob50-evidence-only

- Status: DONE.
- Question: Does probabilistic Stage2 original-image masking with answer-stage original access improve train/inference alignment?
- Baseline anchor: Not a clean ablation of `BASE-20260619-open-answer-rowonly`; uses different Stage1.
- Intended diff:
  - Add `mask_original_image_after_tgvf_prob=0.5`.
  - Change Stage2 mask scope to evidence/readout only; answer stage can attend original image keys.
- Allowed changed variables:
  - Stage2 mask probability.
  - Stage2 mask scope.
- Not allowed to change:
  - Should have been baseline Stage1 if comparing to 20260619, but it was not.
- Code commit / worktree:
  - Started after commit `25d4451` plus current working-tree behavior at the time.
- Stage1 checkpoint:
  - `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617/train/checkpoint_step_2000.pt`
- Stage1 processor:
  - `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617/train/processor_step_2000`
- Stage2 output:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_maskprob50_evidenceonly_from_clean_imend_stage1_4gpu_bs16_accum2_focus80_value1_1200step_20260622_021650`
- Train data:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl`
- Validation data:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`
- Benchmark output:
  - `outputs/full_benchmarks_maskprob50_evidenceonly_4gpu_20260622_105128`
- Script / command:
  - Stage2 via `scripts/run_tgvf_v3_protocol_c_stage2_from_stage1c_4gpu.sh`.
  - Full benchmark via `scripts/run_tgvf_v3_fullmask_stage1_stage2_fullbench_2gpu.sh` with `RUN_CHAIN=0`.
- GPUs:
  - Stage2: 0,1,2,3.
  - Full benchmark: 0,1,2,3.
- Metrics:
  - Stage2 train wall time: about 5:34.
  - Protocol eval wall time: about 50:47.
  - Protocol eval:
    - no_focus_direct: 78.13.
    - free_router_end2end: 66.78.
    - free trigger: 15.41.
    - force correct_D: 59.38.
    - teacher-forced correct_D: 61.72.
  - VStar free: 47.64 overall, 40.87 direct, 57.89 relative, trigger 57.07.
  - VStar softforce: 46.60 overall, 41.74 direct, 53.95 relative, trigger 89.01.
  - HR free: 53.38, trigger 19.63.
- Analysis:
  - Not better than fullmask on VStar.
  - Direct stayed similar; relative_position dropped versus fullmask.
- Conclusion:
  - Evidence-only answer-unmasked with prob 0.5 did not recover VStar.
- Comparable to baseline:
  - Comparable to later 20260622 experiments using the same 20260617 Stage1.
  - Not a valid ablation of `BASE-20260619-open-answer-rowonly`.
- Follow-up:
  - Any ablation against 20260619 must use the 20260619 Stage1 checkpoint.

### EXP-20260622-133909-oldmask075-through-answer

- Status: DONE.
- Question:
  - Test old attention-mask semantics with `mask_original_image_after_tgvf_prob=0.75`.
- Baseline anchor:
  - Intended comparison drifted during discussion.
  - Actual run uses `BASE-20260617-rowonly-original-stage2` Stage1, not `BASE-20260619-open-answer-rowonly`.
- Intended diff:
  - Use old through-answer original image key mask behavior.
  - Set mask probability to 0.75.
- Allowed changed variables:
  - Stage2 mask probability.
  - Stage2 mask scope.
- Not allowed to change:
  - If comparing to `BASE-20260619-open-answer-rowonly`, Stage1 should not have changed. This run violates that.
- Code commit / worktree:
  - Base commit: `25d4451`.
  - Additional uncommitted code changes added:
    - `--mask-original-image-after-tgvf-scope`.
    - `evidence_only|through_answer` scope switch.
    - `scripts/run_tgvf_v3_protocol_c_stage2_oldmask075_fullbench_0_3.sh`.
- Stage1 checkpoint:
  - `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617/train/checkpoint_step_2000.pt`
- Stage1 processor:
  - `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617/train/processor_step_2000`
- Stage2 output:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_oldmaskprob075_throughanswer_from_clean_rowonly_stage1_4gpu_bs16_accum2_focus80_value1_1200step_20260622_133909`
- Train data:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl`
- Validation data:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`
- Benchmark output:
  - `outputs/full_benchmarks_oldmaskprob075_throughanswer_4gpu_20260622_133909`
- Script / command:
  - `scripts/run_tgvf_v3_protocol_c_stage2_oldmask075_fullbench_0_3.sh`
- GPUs:
  - 0,1,2,3.
- tmux:
  - `oldmask075_g0_3_20260622_133909`
- Started:
  - 2026-06-22T13:39:09+09:00.
- Finished:
  - 2026-06-22T22:12:39+09:00.
- Progress:
  - Stage2 training DONE. Wall time: 5:32:20.
  - Protocol eval DONE. Wall time: 51:48.
  - Full benchmark DONE.
- Metrics:
  - Training val:
    - val loss: 0.9045.
    - val focus mask active rate: 0.625.
    - val boundary acc mean: 0.95045.
  - Protocol eval:
    - no_focus_direct: 78.13.
    - free_router_end2end: 65.97.
    - free trigger: 13.89.
    - force correct_D: 59.38.
    - teacher-forced correct_D: 60.94.
  - VStar free: 47.12 overall, 41.74 direct, 55.26 relative, trigger 52.36.
  - VStar softforce: 48.17 overall, 42.61 direct, 56.58 relative, trigger 88.48.
  - HR free: 55.00, answer parse 96.25, trigger 20.50.
  - HR softforce: 54.13, answer parse 95.25, trigger 66.63.
  - OCR free: 20.84, answer parse 100.00, trigger 5.84.
  - OCR softforce: 20.65, answer parse 100.00, trigger 35.55.
  - BLINK free: 58.33, answer parse 96.67, trigger 10.83.
  - BLINK softforce: 58.33, answer parse 95.00, trigger 41.67.
- Analysis:
  - This run does not explain the gap to `BASE-20260619-open-answer-rowonly`, because it uses the 20260617 Stage1.
  - It is only valid as a comparison against experiments that also use the 20260617 Stage1.
  - VStar remains weak relative to 20260619 and fullmask references.
  - HR free is improved versus recent references.
  - OCR stays around the previous open-answer/free level but remains below original Qwen.
  - BLINK is below the 20260620 open-answer/free reference.
- Conclusion:
  - Side result, not valid evidence for 20260619 ablation.
- Comparable to baseline:
  - Not comparable to `BASE-20260619-open-answer-rowonly`.
  - Comparable to `EXP-20260622-021650-maskprob50-evidence-only` on the same 20260617 Stage1.
- Follow-up:
  - To answer the intended question, rerun using the 20260619 Stage1 checkpoint and processor.

### EXP-20260622-code-mask-scope-option

- Status: IMPLEMENTED, uncommitted as of 2026-06-22T21:26:39+09:00.
- Question:
  - Support both new evidence-only masking and old through-answer masking without reverting code.
- Motivation:
  - Need clean ablations of Stage2 original-image-key masking semantics.
- Code commit / worktree:
  - Base commit: `25d4451`.
  - Modified files:
    - `scripts/train_tgvf_v3_stage2.py`
    - `src/revisit_vlm/tgvf_v3_stage2.py`
    - `src/revisit_vlm/tgvf_v3_stage2_fast.py`
    - `scripts/run_tgvf_v3_protocol_c_stage2_from_stage1c_4gpu.sh`
    - `tests/test_tgvf_v3_stage2_mask.py`
  - Added file:
    - `scripts/run_tgvf_v3_protocol_c_stage2_oldmask075_fullbench_0_3.sh`
- Change:
  - Add `--mask-original-image-after-tgvf-scope evidence_only|through_answer`.
  - `evidence_only`: mask original image keys until answer starts.
  - `through_answer`: old behavior, mask original image keys through answer.
  - Add mask probability logging and tests.
- Verification:
  - `python -m py_compile scripts/train_tgvf_v3_stage2.py src/revisit_vlm/tgvf_v3_stage2.py src/revisit_vlm/tgvf_v3_stage2_fast.py`
  - `pytest -q tests/test_tgvf_v3_stage2_mask.py`
  - `bash -n scripts/run_tgvf_v3_protocol_c_stage2_from_stage1c_4gpu.sh`
  - `bash -n scripts/run_tgvf_v3_protocol_c_stage2_oldmask075_fullbench_0_3.sh`
  - `git diff --check`
- Analysis:
  - Code change is useful, but future experiment entries must bind the correct Stage1 explicitly.
- Conclusion:
  - Keep the option, but do not use script defaults as experiment identity.

### EXP-20260623-005426-oldmask075-correct-stage1

- Status: DONE.
- Question:
  - Rerun the old-mask `mask_original_image_after_tgvf_prob=0.75` / `through_answer`
    Stage2 experiment with the correct 20260619 open-answer row-only Stage1.
- Baseline anchor:
  - `BASE-20260619-open-answer-rowonly`.
- Intended diff:
  - Use correct 20260619 Stage1 instead of the incorrect 20260617 Stage1 used by
    `EXP-20260622-133909-oldmask075-through-answer`.
  - Stage2 mask probability: 0.75.
  - Stage2 mask scope: `through_answer`, matching old attention-mask semantics.
  - Run Stage2 training on GPUs 4,5,6,7.
  - Run protocol eval and full benchmark on GPUs 0,1,2,3.
- Allowed changed variables:
  - Stage2 mask probability and scope.
  - Physical GPU placement.
  - Training world size is 4 GPU with `bs16 accum2`, same effective global batch
    as the 20260619 2GPU `bs16 accum4` Stage2 baseline.
- Not allowed to change:
  - Stage1 checkpoint/processor from `BASE-20260619-open-answer-rowonly`.
  - Open-answer Stage2 train/test data.
  - Max image resolution 512.
  - Full benchmark mode: free and softforce only, no extra original rerun.
- Code commit / worktree:
  - Base commit: `25d4451`.
  - Uses uncommitted mask-scope code and new wrapper script.
- Stage1 checkpoint:
  - `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_imend_bidirectional_2gpu_bs4_accum4_gbs32_2000step_20260619_014148/train/checkpoint_step_2000.pt`
- Stage1 processor:
  - `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_imend_bidirectional_2gpu_bs4_accum4_gbs32_2000step_20260619_014148/train/processor_step_2000`
- Stage2 output:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_oldmaskprob075_throughanswer_from_20260619_stage1_train47_eval03_4gpu_bs16_accum2_focus80_value1_1200step_20260623_005426`
- Train data:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl`
- Validation data:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`
- Benchmark output:
  - `outputs/full_benchmarks_oldmaskprob075_throughanswer_correct_stage1_4gpu_20260623_005426`
- Script / command:
  - `STAMP=20260623_005426 TRAIN_GPUS=4,5,6,7 EVAL_GPUS=0,1,2,3 BENCH_GPUS=0,1,2,3 bash scripts/run_tgvf_v3_protocol_c_stage2_oldmask075_correct_stage1_train47_eval03.sh`
- GPUs:
  - Train: 4,5,6,7.
  - Protocol eval: 0,1,2,3.
  - Full benchmark: 0,1,2,3.
- tmux:
  - `oldmask075_correct_s1_t47_e03_20260623_005426`
- Started:
  - 2026-06-23T00:56:56+09:00.
- Finished:
  - 2026-06-23T09:31:51+09:00.
- Metrics:
  - Stage2 training wall time: 5:34:10.
  - Protocol eval wall time: 53:33.
  - Full benchmark wall time: about 2:07:11.
  - Training val:
    - val loss: 0.9009.
    - val focus mask active rate: 0.59375.
    - val boundary acc mean: 0.94595.
    - val value span match rate: 0.54688.
  - Protocol eval:
    - no_focus_direct: 78.91.
    - free_router_end2end: 63.67.
    - free trigger: 18.33.
    - free correct_D: 36.36.
    - force overall: 24.53.
    - force correct_D: 44.53.
    - teacher-forced overall: 26.41.
    - teacher-forced correct_D: 42.97.
  - Full benchmark:
    - VStar free: 47.12 overall, 39.13 direct, 59.21 relative.
    - VStar softforce: 50.26 overall, 44.35 direct, 59.21 relative.
    - HR free: 54.38, answer parse 97.13, trigger 22.50.
    - HR softforce: 54.25, answer parse 97.25, trigger 71.38.
    - OCR free: 21.02, answer parse 100.00, trigger 5.35.
    - OCR softforce: 20.92, answer parse 100.00, trigger 44.10.
    - BLINK free: 59.17, answer parse 98.33, trigger 7.50.
    - BLINK softforce: 65.00, answer parse 96.67, trigger 51.67.
- Analysis:
  - Using the correct 20260619 Stage1 did not recover free-mode VStar.
  - Against `BASE-20260619-open-answer-rowonly`, free changed:
    - VStar: 51.31 -> 47.12, -4.19.
    - HR: 53.38 -> 54.38, +1.00.
    - OCR: 20.82 -> 21.02, +0.21.
    - BLINK: 65.83 -> 59.17, -6.67.
  - Against `BASE-20260619-open-answer-rowonly`, softforce changed:
    - VStar: 47.64 -> 50.26, +2.62.
    - HR: 54.38 -> 54.25, -0.13.
    - OCR: 20.67 -> 20.92, +0.24.
    - BLINK: 62.50 -> 65.00, +2.50.
  - Protocol correct_D degraded badly:
    - force correct_D only 44.53.
    - teacher-forced correct_D only 42.97.
  - Direct Stage1-style readout regression was added after this protocol result:
    - Command output root:
      `eval_outputs/readout_stage2_oldmask075_correct_stage1_20260623_005426_step1200`
    - Compared against the corresponding 20260619 Stage1 readout report using the
      same Stage1 focus test JSONL, processor, Protocol C tool-observation prompt,
      `focus_action_im_end=true`, max image resolution 512, and 200 samples.
    - Stage1 readout:
      - mean_nll_correct_D: 1.5912109375.
      - mean_delta_correct_vs_target_only: 0.660390625.
      - mean_delta_correct_vs_random: 0.97888671875.
      - pct_correct_D_beats_target_only: 0.99.
      - pct_correct_D_beats_random: 1.0.
      - pct_correct_D_beats_wrong_same: 0.84.
      - pct_correct_D_beats_wrong_diff: 0.84375.
    - Stage2 checkpoint readout:
      - mean_nll_correct_D: 1.589931640625.
      - mean_delta_correct_vs_target_only: 0.661669921875.
      - mean_delta_correct_vs_random: 0.980166015625.
      - pct_correct_D_beats_target_only: 0.99.
      - pct_correct_D_beats_random: 1.0.
      - pct_correct_D_beats_wrong_same: 0.83.
      - pct_correct_D_beats_wrong_diff: 0.8125.
    - Interpretation:
      - This does not support the hypothesis that Stage2 destroyed the Stage1
        target-description / D-readout ability of the TGVF module.
      - `force correct_D` is an end-to-end Stage2 protocol answer metric, not a
        direct Stage1 readout metric. It mixes D construction, post-D attention,
        LoRA/token-row answer behavior, generation format, and answer parsing.
      - Therefore the collapse in protocol `force correct_D` should be analyzed
        as a post-D Stage2 interaction / answer-policy failure unless a full
        Stage2-model readout evaluator shows otherwise.
- Conclusion:
  - Not a good free-mode direction.
  - Softforce is promising on VStar/BLINK, but the protocol D metrics are too weak
    to treat this as a clean mechanism improvement.
  - Corrected mechanism conclusion: the Stage2 TGVF module preserved Stage1-style
    description readout on the 200-sample diagnostic; current failures are not
    explained by loss of the Stage1 target-description alignment alone.
- Comparable to baseline:
  - Intended to be comparable to `BASE-20260619-open-answer-rowonly` except for
    the intended Stage2 mask probability/scope ablation and 4GPU physical training
    layout with matched effective global batch.
- Follow-up:
  - Do not pick this as default for free mode.
  - If continuing this branch, inspect why e2e `force correct_D` collapsed despite
    preserved Stage1-style readout. Priority checks:
    - full Stage2-model readout with LoRA/token rows loaded.
    - post-D attention mask/original-key access.
    - generated evidence text quality before final answer.
    - answer parser/format failures under force and teacher-forced modes.

### DIAG-20260623-stage2-protocol-teacher-forced-retest

- Status: PARTIAL DONE.
- Question:
  - Re-test 20260619 and 20260623 Stage2 under the same Stage2 protocol evaluator
    to check whether the low `force correct_D` style metrics are a protocol/log
    artifact or a real post-D degradation.
- Output root:
  - `eval_outputs/protocol_force_correctD_retest_20260623`
- Compared checkpoints:
  - 20260619 open-answer Stage2:
    `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt`
  - 20260623 oldmask075 through-answer Stage2:
    `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_oldmaskprob075_throughanswer_from_20260619_stage1_train47_eval03_4gpu_bs16_accum2_focus80_value1_1200step_20260623_005426/checkpoint_step_1200.pt`
- Fixed eval settings:
  - Eval JSONL:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`
  - Protocol: `protocol_c_tool_observation`.
  - Max image resolution: 512.
  - Max focus samples: 128.
  - D controls: `correct_D,no_D,random_D,wrong_same_image_D,wrong_diff_image_D`.
  - Blocks started: `force_focus_targets,teacher_forced_post_tgvf,force_end2end`.
  - `force_end2end` was stopped at user request after `teacher_forced_post_tgvf`
    completed.
- Results:
  - `force_focus_targets`:
    - 20260619: focus_valid_rate 1.0, focus_miss_rate 0.0.
    - 20260623: focus_valid_rate 1.0, focus_miss_rate 0.0.
  - `teacher_forced_post_tgvf / correct_D`:
    - 20260619: accuracy 52.34, answer_parse_rate 75.00, append_success_rate 100.00.
    - 20260623: accuracy 42.97, answer_parse_rate 62.50, append_success_rate 100.00.
  - `teacher_forced_post_tgvf / random_D`:
    - 20260619: accuracy 28.13, answer_parse_rate 42.19.
    - 20260623: accuracy 28.91, answer_parse_rate 42.97.
  - `teacher_forced_post_tgvf / wrong_same_image_D`:
    - 20260619: accuracy 28.13, answer_parse_rate 42.19.
    - 20260623: accuracy 29.69, answer_parse_rate 42.97.
  - `teacher_forced_post_tgvf / wrong_diff_image_D`:
    - 20260619: accuracy 27.34, answer_parse_rate 40.63.
    - 20260623: accuracy 29.69, answer_parse_rate 41.41.
  - `teacher_forced_post_tgvf / no_D`:
    - 20260619: accuracy 0.00, answer_parse_rate 3.13.
    - 20260623: accuracy 0.78, answer_parse_rate 3.13.
- Interpretation:
  - The evaluator is not obviously broken:
    - forced focus targets are valid for both runs.
    - `correct_D` is much better than `no_D/random_D/wrong_D` in both runs.
    - D append succeeds at 100% for `correct_D`.
  - The problem is visible before free/force target-generation noise:
    - Under teacher-forced post-TGVF conditions, 20260623 `correct_D` drops by
      9.38 accuracy points and 12.50 parse-rate points versus 20260619.
    - The controls do not drop; wrong/random D are roughly unchanged or slightly
      higher, so the useful-D margin narrows sharply.
  - Combined with the direct Stage1 readout regression, this suggests the D
    representation itself is intact, but the 20260623 Stage2 post-D answer path
    became worse at consuming correct D and formatting/terminating the answer.

### DIAG-20260623-qwen2vl2b-blink-full-512

- Status: DONE.
- Request:
  - Test raw Qwen2-VL-2B on BLINK full at max image resolution 512 using GPUs
    4-7.
- Purpose:
  - Provide a non-TGVF Qwen2-VL-2B BLINK-full reference under the same local
    BLINK validation snapshot.
- Model:
  - `Qwen/Qwen2-VL-2B-Instruct` from local Hugging Face cache.
- Evaluation settings:
  - Benchmark: BLINK.
  - Split: local validation labels, all 14 BLINK subtasks.
  - Samples: 1901 total validation examples.
  - Input media: all non-empty BLINK `image_*` fields per example are passed to
    the Qwen2-VL message; examples contain 1-4 images.
  - Max image resolution: 512, implemented as `max_pixels=512*512`.
  - Method: `direct_qwen`, no TGVF, no extra TGVF prompt.
  - Sharding: 4 shards, one process each on GPUs 4,5,6,7.
- Code notes:
  - `BlinkAdapter` now loads all `snapshot/*/val-*.parquet` files instead of
    only `Counting/val`.
  - `QwenTGVFModelRunner` now passes every item in `sample.media` into the
    multimodal message instead of only `primary_media`.
  - `src/tgvf_eval/run.py` now supports `--max-image-resolution`,
    `--num-shards`, and `--shard-index`.
  - Repro script:
    `scripts/run_qwen2vl2b_blink_full_512_sharded.sh`.
- Output root:
  - Active run: `eval_outputs/qwen2vl2b_blink_full_512_20260623_215737`
  - Abandoned empty launch:
    `eval_outputs/qwen2vl2b_blink_full_512_20260623_215306` due to a shell
    pids/subshell bug in the first version of the sharded launcher; no model
    rows were produced there.
- tmux:
  - `qwen2vl2b_blink_full_512_20260623_215737`.
- Result:
  - Completed all 4 shards, 1901/1901 scored, 0 row errors.
  - Merged accuracy: 41.35%.
  - Per-subtask accuracy:
    - Art Style: 49.57.
    - Counting: 52.50.
    - Forensic Detection: 25.76.
    - Functional Correspondence: 26.92.
    - IQ Test: 25.33.
    - Jigsaw: 54.00.
    - Multi-view Reasoning: 44.36.
    - Object Localization: 50.00.
    - Relative Depth: 51.61.
    - Relative Reflectance: 25.37.
    - Semantic Correspondence: 31.65.
    - Spatial Relation: 76.92.
    - Visual Correspondence: 25.58.
    - Visual Similarity: 45.19.
  - Output distribution:
    - Parsed A: 1020.
    - Parsed B: 557.
    - Parsed C: 271.
    - Parsed D: 53.
  - Scoring caveat:
    - This uses the project fallback scorer. BLINK official code path exists
      locally, but the adapter-specific official scorer is not wired in
      `tgvf_eval` v0.

### DIAG-20260624-vpt-qwen2vl2b-clip-blink-full-512

- Status: DONE.
- Request:
  - Pull and evaluate the VPT checkpoint from the example project on BLINK using
    settings comparable to the Qwen2-VL-2B baseline above.
- Baseline to compare:
  - `DIAG-20260623-qwen2vl2b-blink-full-512`
  - Qwen2-VL-2B-Instruct direct no-CoT, BLINK full val, max image resolution 512:
    41.35%.
- Model:
  - `rp-yu/Qwen2-VL-2b-VPT-CLIP`.
  - Base model: `Qwen/Qwen2-VL-2B-Instruct`.
  - Local HF snapshot:
    `models--rp-yu--Qwen2-VL-2b-VPT-CLIP/snapshots/93b0efe8dfee12c2bdaa9faca21378750771c930`.
- Evaluation settings:
  - Benchmark: BLINK.
  - Split: local validation labels, all 14 BLINK subtasks.
  - Samples: 1901 total validation examples.
  - Prompt: direct no-CoT, same question/options surface as the Qwen2 baseline.
  - Input media: all non-empty BLINK `image_*` fields per example.
  - Max image resolution: 512, implemented as `max_pixels=512*512`.
  - Scorer: project fallback scorer, same caveat as Qwen2 baseline.
  - VPT second round: `auto`; only used if the direct first-round output contains
    a CLIP action token. Smoke tests on 8 samples had trigger rate 0.
- Legacy safety:
  - Existing `tgvf_eval.run` and example project files are not used as mutable
    integration points for VPT.
  - VPT compatibility is isolated in:
    `scripts/eval_vpt_qwen2_blink_full_512.py`.
  - Sharded launcher:
    `scripts/run_vpt_qwen2vl2b_clip_blink_full_512_sharded.sh`.
- Compatibility notes:
  - Local example-project VPT code is 4.45-era and not directly importable under
    current Transformers 5.8.1.
  - The evaluator loads the local VPT modeling file through an in-process
    compatibility layer only.
  - Source-level runtime shims:
    - remove obsolete `AutoModelForVision2Seq` import.
    - fall back from old `Qwen2TokenizerFast` import to current Qwen2 tokenizer.
    - handle `cache_position is None` from Transformers 5 generation.
    - map `self.model.embed_tokens` to
      `self.model.language_model.embed_tokens`.
    - map `self.visual` to `self.model.visual`.
    - unwrap current visual tower `BaseModelOutputWithPooling` to tensor output.
  - Checkpoint key remaps:
    - `visual.* -> model.visual.*`
    - `model.* -> model.language_model.*`
    - `depth_projector.* -> clip_projector.*`
  - Strict load smoke: `missing=0`, `unexpected=0`.
- Smoke:
  - `limit=1`: passed, raw output `B`, no error.
  - `limit=8`: passed, 8/8 no errors, media counts include 1/2/3 images,
    score 50.00, trigger rate 0.
- Output root:
  - `eval_outputs/vpt_qwen2vl2b_clip_blink_full_512_20260624_000122`
- tmux:
  - `vpt_qwen2vl2b_clip_blink_full_512_20260624_000122`.
- Result:
  - Completed all 4 shards, 1901/1901 scored, 0 row errors.
  - Merged accuracy: 37.30%.
  - Compared to Qwen2-VL-2B-Instruct direct no-CoT baseline 41.35:
    - Delta: -4.05 points.
  - Trigger / VPT use:
    - CLIP action trigger rate: 0.00.
    - Second-round VPT rate: 0.00.
    - Therefore this run measures the VPT checkpoint under the same direct
      no-extra-prompt surface, but does not exercise the VPT second perception
      mechanism.
  - Output distribution:
    - Parsed A: 854.
    - Parsed B: 443.
    - Parsed C: 96.
    - Parsed D: 508.
    - Max raw output length: 1.
  - Per-subtask accuracy and delta vs Qwen2 baseline:
    - Art Style: 47.01 vs 49.57, delta -2.56.
    - Counting: 40.83 vs 52.50, delta -11.67.
    - Forensic Detection: 26.52 vs 25.76, delta +0.76.
    - Functional Correspondence: 28.46 vs 26.92, delta +1.54.
    - IQ Test: 20.67 vs 25.33, delta -4.67.
    - Jigsaw: 52.67 vs 54.00, delta -1.33.
    - Multi-view Reasoning: 44.36 vs 44.36, delta +0.00.
    - Object Localization: 45.90 vs 50.00, delta -4.10.
    - Relative Depth: 48.39 vs 51.61, delta -3.23.
    - Relative Reflectance: 26.87 vs 25.37, delta +1.49.
    - Semantic Correspondence: 25.18 vs 31.65, delta -6.47.
    - Spatial Relation: 46.15 vs 76.92, delta -30.77.
    - Visual Correspondence: 27.33 vs 25.58, delta +1.74.
    - Visual Similarity: 47.41 vs 45.19, delta +2.22.
  - Interpretation:
    - As a like-for-like direct no-CoT comparison, this VPT-CLIP checkpoint is
      worse than raw Qwen2-VL-2B-Instruct on BLINK full by 4.05 points.
    - The run is not evidence against the VPT mechanism itself, because the model
      never emitted CLIP action tokens under the no-extra-prompt BLINK setting.
    - If we want to evaluate the actual VPT mechanism, the next run should use a
      VPT trigger/instruction or the official VPT two-round evaluation style.

## Reconstructed Historical Entries

These entries were reconstructed from chat context, output files, configs, and tests on
2026-06-22. Treat them as the current historical record; if a raw log contradicts an
entry, update this file immediately.

### TECH-20260618-chat-inference-parser-loader

- Status: DONE.
- Question:
  - Are chat inference parsing, weight loading, and Stage2 loss weights aligned with
    training/evaluation?
- Relevant files:
  - `scripts/chat_tgvf_v3_stage2.py`
  - `scripts/chat_tgvf_v3.py`
  - `src/revisit_vlm/tgvf_v3_stage2.py`
  - `tests/test_tgvf_chat_loading.py`
- Findings:
  - Stage2 default weighted span loss stayed aligned with legacy `value1`:
    - `evidence_state=0.2`
    - `focus_target=1.5`
    - `evidence=1.0`
    - `value_span=1.0`
    - `answer=1.0`
    - `no_focus_evidence_state=0.2`
    - `no_focus_answer=1.0`
  - Chat loading was checked against PEFT/base-model missing-key handling.
  - Free mode was checked as no output correction and no extra prompt by default.
  - The old free-router prompt option was removed from the chat path.
  - Chat/eval max image resolution for current Stage2 tests was 512.
  - Raw Qwen chat was also launched for comparison under `outputs/chat_qwen3_vl`.
- Output roots:
  - `outputs/chat_tgvf_v3_stage2/tgvf_clean_chat_512_20260618_140431`
  - `outputs/chat_tgvf_v3_stage2/tgvf_clean_chat_noprompt_20260618_141310`
  - `outputs/chat_qwen3_vl`
- Conclusion:
  - Chat parser/loading was made consistent enough for manual testing.
  - Future changes to parser or load validation must include tests equivalent to
    `tests/test_tgvf_chat_loading.py`.

### DATA-20260618-stage2-choice-bias-open-answer

- Status: DONE.
- Question:
  - How much of the Stage2 dialog data is multiple-choice, and should it be converted
    to direct answer/open-answer format?
- Old data:
  - Train: `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl`
  - Test: `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`
- Old data stats:
  - Train: 46,883 total, 33,159 multiple-choice, 70.73%.
  - Test: 1,002 total, 711 multiple-choice, 70.96%.
  - Train multiple-choice labels:
    - A: 12,207 / 33,159 = 36.81%.
    - B: 13,640 / 33,159 = 41.14%.
    - C: 6,865 / 33,159 = 20.70%.
    - D: 446 / 33,159 = 1.35%.
  - Test multiple-choice labels:
    - A: 235 / 711 = 33.05%.
    - B: 298 / 711 = 41.91%.
    - C: 166 / 711 = 23.35%.
    - D: 12 / 711 = 1.69%.
- Open-answer data:
  - Train: `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl`
  - Test: `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`
- Open-answer conversion:
  - All examples have `answer_format=short_text`.
  - `choices=[]`.
  - Converted old MC examples keep provenance under
    `metadata.choice_to_open_answer`.
  - Example: old `answer="B. orange"` became `answer="orange"`,
    `short_answer="orange"`, `value_span_text="orange"`.
- Code:
  - `scripts/convert_stage2_choices_to_open_answer.py`
  - `tests/test_stage2_choice_conversion.py`
- Analysis:
  - Old data has severe option-label imbalance, especially very low D.
  - Training answer letters can create shortcut bias unrelated to visual grounding.
  - Open-answer/direct-answer format was necessary to reduce option priors.
- Conclusion:
  - Do not use old MC answer format as the primary Stage2 target unless explicitly
    studying option-label bias.

### HIST-20260614-to-20260616-vstar-eval-debug

- Status: DONE.
- Question:
  - Why did early VStar results vary strongly with prompt, resolution, and capture
    fixes?
- Context:
  - These were evaluation/debug runs, not clean training ablations.
  - They included NoRes, HF default pixels, Max16K pixels, exact MCQ, postfix,
    natural_continue, and TGVF capture fixes.
- Key VStar results:

| Date | Model / setting | Overall | Direct | Relative | Source |
| --- | --- | ---: | ---: | ---: | --- |
| 2026-06-14 | Qwen NoRes initial | 47.64 | 45.22 | 51.32 | `outputs/vlmevalkit/vstar_nores_latest_and_qwen_20260614/.../T20260614-221055/status.json` |
| 2026-06-14 | Qwen NoRes official/exact | 52.36 | 53.91 | 50.00 | `outputs/vlmevalkit/vstar_nores_exact_mcq_1024_latest_and_qwen_20260614/.../T20260614-234709/status.json` |
| 2026-06-15 | Qwen HFDefault | 67.54 | 63.48 | 73.68 | `outputs/vlmevalkit/vstar_hfdefault_exact_mcq_latest_and_qwen_20260614/.../T20260615-001139/status.json` |
| 2026-06-15 | Qwen Max16K exact | 74.35 | 79.13 | 67.11 | `outputs/vlmevalkit/vstar_max16k_exact_mcq_latest_and_qwen_20260614/.../T20260615-001940/status.json` |
| 2026-06-15 | Qwen Max16K postfix | 77.49 | 79.13 | 75.00 | `outputs/vlmevalkit/vstar_max16k_qwen_postfix_20260615/.../T20260615-014609/status.json` |
| 2026-06-15 | TGVF Max16K early fixed | 8.90 | 6.09 | 13.16 | `outputs/vlmevalkit/vstar_max16k_tgvf_fixed_20260614/.../T20260615-002751/status.json` |
| 2026-06-15 | TGVF Max16K finalfix5 | 69.11 | 74.78 | 60.53 | `outputs/vlmevalkit/vstar_max16k_tgvf_finalfix5_20260614/.../T20260615-013243/status.json` |
| 2026-06-15 | TGVF HFDefault | 63.35 | 65.22 | 60.53 | `outputs/vlmevalkit/vstar_hfdefault_tgvf_20260615/.../T20260615-015638/status.json` |
| 2026-06-15 | Qwen 512 | 51.31 | 46.09 | 59.21 | `outputs/vlmevalkit/vstar_512_qwen_tgvf_20260615/.../T20260615-020815/status.json` |
| 2026-06-15 | TGVF 512 | 46.07 | 40.00 | 55.26 | `outputs/vlmevalkit/vstar_512_qwen_tgvf_20260615/.../T20260615-021449/status.json` |
| 2026-06-16 | TGVF 512 natural_continue | 52.88 | 44.35 | 65.79 | `outputs/vlmevalkit/vstar_512_natural_continue_tgvf_20260616/.../T20260616-141338/status.json` |
| 2026-06-16 | TGVF Max16K natural_continue | 30.37 | 14.78 | 53.95 | `outputs/vlmevalkit/vstar_max16k_natural_continue_tgvf_20260616/.../T20260616-142810/status.json` |

- Analysis:
  - Early VStar numbers were heavily affected by evaluation formatting and capture
    correctness.
  - Max16K can produce much higher Qwen/TGVF VStar than 512, so 512-only comparisons
    must not be compared directly with Max16K runs.
  - The remembered "52+" result is the 2026-06-16 TGVF 512 natural_continue run
    and later should not be mixed with different post-TGVF continuation semantics.
- Conclusion:
  - Use these runs as debugging history, not as a clean baseline for current 512
    open-answer Stage2 ablations.

### EXP-20260611-toolobs-focus-imend-stage1-stage2

- Status: DONE.
- Purpose:
  - Earlier Protocol C tool-observation Stage1/Stage2 run used for chat testing and
    historical comparison.
- Stage1:
  - `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260611_021244/train/checkpoint_step_2000.pt`
  - Config highlights:
    - `freeze_qwen=True`
    - `attention_mask_mode=weak_strict_original_image_keys_4d`
    - `mask_original_image_after_tgvf=True`
    - `protocol_c_token_rows_trainable.enabled=True`
    - `row_gradient_mask_active=True`
- Stage2:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_multifocus_focus_imend_from_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260611_021244/checkpoint_step_1200.pt`
- Protocol eval:
  - no_focus_direct: 78.91.
  - free_router_end2end: 67.78.
  - free trigger: 42.53.
  - force overall: 47.03.
  - force correct_D: 87.50.
  - teacher-forced overall: 49.84.
  - teacher-forced correct_D: 89.84.
- Analysis:
  - Protocol eval showed strong correct_D under force/teacher-forced conditions.
  - That did not by itself guarantee strong free-mode benchmark transfer.
- Conclusion:
  - Keep as historical toolobs checkpoint, but do not compare directly with clean
    open-answer baselines.

### EXP-20260617-rowonly-clean-imend

- Status: DONE.
- Purpose:
  - Clean row-only Stage1 + old clean_imend Stage2 baseline.
- Stage1:
  - `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617/train/checkpoint_step_2000.pt`
  - Config highlights:
    - `row_only_parameters=True`
    - `optimizer_param_count=32768`
    - `row_gradient_mask_active=False`
    - `freeze_qwen=True`
    - `mask_original_image_after_tgvf=True`
- Stage2:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/checkpoint_step_1200.pt`
- Stage2 config:
  - `max_image_resolution=512`
  - `world_size=4`
  - `gradient_accumulation_steps=2`
  - `max_steps=1200`
  - old default `mask_original_image_after_tgvf=True`
  - legacy Stage2 loss weights as in `TECH-20260618-chat-inference-parser-loader`
- Protocol eval:
  - no_focus_direct: 78.13.
  - force overall: 50.63.
  - force correct_D: 89.84.
  - teacher-forced overall: 51.09.
  - teacher-forced correct_D: 92.97.
- VStar:
  - 2026-06-18 first run: 45.55 overall, 39.13 direct, 55.26 relative.
  - 2026-06-18 rerun: 47.12 overall, 40.87 direct, 56.58 relative.
- Analysis:
  - Correct_D protocol metrics were high, but VStar free remained weak.
  - This shows protocol eval alone was not sufficient to explain benchmark transfer.
- Conclusion:
  - Valid baseline for experiments using the same 20260617 Stage1.
  - Not valid as a baseline for 20260619 open-answer 2GPU Stage1 experiments.

### EXP-20260619-open-answer-2gpu

- Status: DONE.
- Purpose:
  - Main 512 open-answer row-only baseline after removing MC answer-letter targets.
- Stage1:
  - `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_imend_bidirectional_2gpu_bs4_accum4_gbs32_2000step_20260619_014148/train/checkpoint_step_2000.pt`
  - Config highlights:
    - `row_only_parameters=True`
    - `optimizer_param_count=32768`
    - `row_gradient_mask_active=False`
    - `freeze_qwen=True`
    - `gradient_accumulation_steps=4`
- Stage2:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt`
- Stage2 config:
  - `max_image_resolution=512`
  - `world_size=2`
  - `gradient_accumulation_steps=4`
  - `max_steps=1200`
  - old default `mask_original_image_after_tgvf=True`
  - legacy Stage2 loss weights.
- Protocol eval:
  - Available local protocol files include `no_focus_direct` and `force_focus_targets`
    only, not a complete free/force/teacher-forced summary.
  - no_focus_direct: 78.13.
- VStar results:
  - Free, no extra prompt: 51.31 overall, 45.22 direct, 60.53 relative.
  - Direct mode: 31.41 overall, 15.65 direct, 55.26 relative.
  - Force mode: 49.21 overall, 46.09 direct, 53.95 relative.
  - Soft force, prompt phrase `use focus tool`: 47.64 overall, 42.61 direct,
    55.26 relative.
- Comparison:
  - Original Qwen 512 VStar: 51.31 overall, 46.09 direct, 59.21 relative.
  - Open-answer free matched original overall on VStar but changed which samples
    were correct.
- Analysis:
  - This is the main baseline for later 512 open-answer work.
  - Any current ablation claiming comparison to this baseline must use this exact
    Stage1 checkpoint/processor unless the Stage1 change is explicitly the ablation.
- Conclusion:
  - Treat as `BASE-20260619-open-answer-rowonly`.

### ANALYSIS-20260619-vstar-disagreement-and-D-ablation

- Status: DONE.
- Purpose:
  - Understand where TGVF free succeeds/fails relative to original Qwen and whether
    post-TGVF D tokens causally help.
- Mechanism note:
  - TGVF is not crop-based in this code path.
  - The model emits a focus target, then the Stage1/Stage2 machinery generates and
    appends foveated visual tokens D from the selected target and image features.
  - The relevant questions are:
    - Was the pre-focus decision already wrong?
    - Was the focus target useful?
    - Did D contain usable evidence?
    - Did original visual keys dominate or interfere after D was appended?
- Disagreement visualization:
  - Output root: `outputs/analysis/vstar_complement_visuals_20260619`
  - Manifest: `outputs/analysis/vstar_complement_visuals_20260619/manifest.csv`
  - Original correct / free wrong: 24 images.
  - Original wrong / free correct: 24 images.
  - Each visual contains the image plus question/model information for inspection.
- Diagnostic outputs:
  - `outputs/diagnostics/vstar_free_d_conditions_complement_20260619_154445/vstar_force_summary.json`
  - `outputs/diagnostics/vstar_free_d_conditions_vlmeval_png_triggered_20260619_160126/vstar_force_summary.json`
  - `outputs/diagnostics/vstar_free_d_conditions_vlmeval_png_triggered_clean_unmasked_20260619_172756/vstar_force_summary.json`
  - `outputs/diagnostics/vstar_free_d_conditions_vlmeval_png_triggered_clean_maskorig_20260619_172756/vstar_force_summary.json`
- Key diagnostics:
  - Complement direct/free_direct_or_miss, n=48 each:
    - direct_qwen3 accuracy: 39.58.
    - free_direct_or_miss accuracy: 39.58.
    - free_direct_or_miss focus_valid_rate: 0.00.
  - Triggered VLMEval PNG diagnostic, n=24 per D condition:
    - free_correct_D accuracy: 54.17.
    - free_no_D accuracy: 0.00, answer parse 0.00.
    - free_random_D accuracy: 50.00.
  - Clean unmasked triggered diagnostic, n=24 per D condition:
    - free_correct_D accuracy: 54.17.
    - free_no_D accuracy: 37.50.
    - free_random_D accuracy: 54.17.
  - Clean mask-original-visual-keys diagnostic, n=24 per D condition:
    - free_correct_D accuracy: 50.00.
    - free_no_D accuracy: 41.67.
    - free_random_D accuracy: 54.17.
- Analysis:
  - Correct D was not consistently stronger than random D on these small triggered
    subsets.
  - Masking original visual keys after TGVF did not clearly improve the small
    diagnostic subset.
  - This supports the hypothesis that failures are not only about target location;
    D quality/readout and original-key interaction both need controlled tests.
- Conclusion:
  - Use disagreement folders and D ablations for qualitative failure analysis.
  - Do not explain failures as "crop" failures; that is the wrong mechanism.

### EXP-20260620-original-free-softforce-full-summary

- Status: DONE.
- Purpose:
  - Summarize original, free, and softforce under current 512 setup across VStar,
    HRBench4K, OCRBenchV2, and BLINK.
- Summary source:
  - `outputs/analysis/benchmark_original_free_softforce_summary_20260620.csv`
- Results:

| Benchmark | n | Original | Free | Softforce | Free trigger | Softforce trigger |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VStar | 191 | 51.31 | 51.31 | 47.64 | 47.12 | 73.82 |
| HRBench4K | 800 | 52.63 | 53.38 | 54.38 | 17.00 | 64.50 |
| OCRBenchV2 | 2467 | 23.23 | 20.82 | 20.67 | 4.09 | 41.59 |
| BLINK | 120 | 61.67 | 65.83 | 62.50 | 5.83 | 45.83 |

- Analysis:
  - Free improved HR and BLINK over original, matched VStar, and hurt OCR.
  - Softforce helped HR relative to free but hurt VStar/BLINK/OCR.
  - Future repeated full benchmark does not always need original; free and softforce
    are enough when original anchor is unchanged.
- Conclusion:
  - This table is the cross-benchmark reference for the 20260619 open-answer era.

### EXP-20260620-fullmask-stage1-stage2

- Status: DONE.
- Purpose:
  - Test `--protocol-token-row-mode full_mask`.
- Stage1:
  - `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_imend_fullmask_bidirectional_2gpu_bs4_accum4_gbs32_2000step_20260620_230349/train/checkpoint_step_2000.pt`
  - Config highlights:
    - `protocol_token_row_mode=full_mask`
    - `row_only_parameters=False`
    - `optimizer_param_count=1242505216`
    - `row_gradient_mask_active=True`
    - `freeze_qwen=True`
- Stage2:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_fullmask_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260620_230349/checkpoint_step_1200.pt`
- Full benchmark output:
  - `outputs/full_benchmarks_fullmask_4gpu_20260621_101055`
- Results:

| Benchmark | Free | Softforce | Free trigger | Softforce trigger |
| --- | ---: | ---: | ---: | ---: |
| VStar | 50.79 | 50.79 | 44.50 | 81.15 |
| HRBench4K | 54.00 | 53.75 | 17.13 | 74.63 |
| OCRBenchV2 | 20.61 | 19.95 | 3.32 | 48.72 |
| BLINK | 58.33 | 55.00 | 5.00 | 59.17 |

- Protocol eval:
  - no_focus_direct: 79.69.
  - teacher-forced correct_D: 40.63.
  - Full protocol force/free summary was not complete locally.
- Analysis:
  - Full-mask was not an obvious win.
  - It also changes the Stage1 trainable parameter regime substantially, so it is
    not comparable to row-only runs as a small ablation.
- Decision:
  - User decided to default back to no `full_mask`.
- Conclusion:
  - Keep as a side baseline, not the default setting.

### TECH-20260621-vpt-mask-comparison

- Status: DONE.
- Question:
  - In the example VPT project, is the training mask removed at inference, and should
    TGVF Stage2 similarly allow original image access during answer generation?
- Findings:
  - VPT training `para_mask_ratio=0.5` is stochastic parameter/mask application
    inside the training mask path, not "mask exactly half the image tokens" in the
    same way as our attention-key mask.
  - VPT evaluation/inference config sets mask ratio to 0, so the mask is effectively
    removed at inference.
  - VPT masking is implemented by zeroing selected input embeddings after the first
    forward, not by attention-key blocking.
  - TGVF Stage2 old behavior masked original image keys after TGVF during training,
    while free inference did not impose the same mask by default.
- Analysis:
  - There is a real train/inference mismatch risk.
  - Because VPT and TGVF mask mechanics differ, copying `para_mask_ratio=0.5` is not
    equivalent. TGVF needs explicit ablations:
    - mask probability.
    - mask scope through evidence only vs through answer.
    - whether post-TGVF inference blocks original visual keys.
- Code outcome:
  - `--mask-original-image-after-tgvf-prob` was added.
  - `--mask-original-image-after-tgvf-scope evidence_only|through_answer` was added.
- Conclusion:
  - The current correct comparison is not "VPT mask vs TGVF mask"; it is old
    through-answer original-key blocking vs answer-unmasked/evidence-only blocking
    under identical Stage1 and data.

### OPS-20260621-to-20260622-git-snapshots

- Status: DONE.
- Commits:
  - `eccffc9 snapshot tgvf v3 protocol c experiments`
  - `25d4451 add probabilistic evidence-only stage2 image masking`
- Purpose:
  - Preserve the current working state before additional mask-probability/scope
    ablations.
- Current uncommitted scope after those commits:
  - `scripts/run_tgvf_v3_protocol_c_stage2_from_stage1c_4gpu.sh`
  - `scripts/train_tgvf_v3_stage2.py`
  - `src/revisit_vlm/tgvf_v3_stage2.py`
  - `src/revisit_vlm/tgvf_v3_stage2_fast.py`
  - `tests/test_tgvf_v3_stage2_mask.py`
  - `docs/EXPERIMENT_LEDGER.md`
  - `scripts/run_tgvf_v3_protocol_c_stage2_oldmask075_fullbench_0_3.sh`
- Verification for current uncommitted code:
  - `python -m py_compile scripts/train_tgvf_v3_stage2.py src/revisit_vlm/tgvf_v3_stage2.py src/revisit_vlm/tgvf_v3_stage2_fast.py`
  - `pytest -q tests/test_tgvf_v3_stage2_mask.py`
  - `bash -n scripts/run_tgvf_v3_protocol_c_stage2_from_stage1c_4gpu.sh`
  - `bash -n scripts/run_tgvf_v3_protocol_c_stage2_oldmask075_fullbench_0_3.sh`
  - `git diff --check`

## Backfill Queue

- Update `EXP-20260624-qwen2vl2b-tgvf-stage1-stage2-512-0_3` after the active
  Qwen2-VL-2B Stage1/Stage2 training run finishes or blocks.
- Update `EXP-20260622-133909-oldmask075-through-answer` when the active tmux run
  finishes.
- Backfill exact metrics for minor 2026-06-14 VStar capturefix intermediate runs
  if they become relevant. Main outcome is already recorded under
  `HIST-20260614-to-20260616-vstar-eval-debug`.
- For every future run, add `PLANNED` before launch and bind it to one baseline
  anchor. Do not rely on script names or defaults.

## Corrective Action Log

### 2026-06-22 Baseline Mismatch Error

- Problem:
  - `EXP-20260622-133909-oldmask075-through-answer` was launched with the 20260617 Stage1.
  - During discussion, it was compared against `BASE-20260619-open-answer-rowonly`.
- Why this is wrong:
  - The Stage1 checkpoint differs.
  - Therefore the run cannot answer whether changing mask probability/scope on the 20260619 baseline helps or hurts.
- Root cause:
  - Script default was used instead of explicitly anchoring the run to the intended baseline.
  - No planned experiment entry or baseline diff was written before launch.
- Required prevention:
  - No future run should start without a `PLANNED` ledger entry.
  - Ablations must list an exact baseline anchor and an exact allowed diff.
  - If Stage1 differs, the result must be marked as not comparable to that baseline.

### DIAG-20260624-vpt-qwen2vl2b-clip-blink-softforce-512

- Status: DONE.
- Baseline anchors:
  - `DIAG-20260623-qwen2vl2b-blink-full-512`: Qwen2-VL-2B-Instruct direct BLINK full
    max-res 512, 41.35.
  - `DIAG-20260624-vpt-qwen2vl2b-clip-blink-full-512`: VPT-CLIP direct BLINK full
    max-res 512, 37.30, trigger/second-round 0/1901.
- Purpose:
  - Test the two original VPT prompt-trigger soft-force styles under the same BLINK
    full max-res 512 setting.
- Model:
  - `rp-yu/Qwen2-VL-2b-VPT-CLIP`, loaded through the isolated compatibility script.
  - Base processor/template from `Qwen/Qwen2-VL-2B-Instruct`.
- Shared settings:
  - Benchmark: BLINK local validation full, 1901 samples.
  - Scoring: project fallback scorer, same as Qwen2 and VPT direct diagnostics.
  - Image limit: `max_image_resolution=512`, `max_pixels=262144`.
  - Generation: deterministic, `max_new_tokens=128`, `use_cache=False`.
  - Legacy safety: no example-project source files modified.
- Variant A:
  - Method: `vpt_clip_softforce_reencode`.
  - Prompt trigger: `Require additional perception features, and then answer the question`.
  - Expected second round: `<clip_image>` re-encode when the model emits CLIP action tokens.
- Variant B:
  - Method: `vpt_clip_softforce_region`.
  - Prompt trigger: `Identify the region that can help you answer the question, and then answer the question`.
  - Expected second round: original VPT 8x8 region-token crop path.
  - BLINK multi-image caveat: region crop follows original VPT code and crops only the
    first input image; this is logged in per-row debug.
- Planned outputs:
  - Reencode tmux: `vpt_softforce_reencode_blink_20260624_005920`
  - Reencode output: `eval_outputs/vpt_qwen2vl2b_clip_blink_softforce_reencode_512_20260624_005920`
  - Region tmux: `vpt_softforce_region_blink_20260624_005920`
  - Region output: `eval_outputs/vpt_qwen2vl2b_clip_blink_softforce_region_512_20260624_005920`
- Smoke:
  - `limit=8` for both variants completed without errors.
  - Both variants had trigger rate 0/8, so the full run tests whether the original
    prompt phrases induce self-triggering at all; no action token is forced.
- Results:

| Run | Accuracy | Trigger | Second round | Errors |
| --- | ---: | ---: | ---: | ---: |
| Qwen2-VL-2B direct | 41.35 | n/a | n/a | 0 |
| VPT-CLIP direct | 37.30 | 0/1901 | 0/1901 | 0 |
| VPT-CLIP softforce reencode prompt | 36.61 | 0/1901 | 0/1901 | 0 |
| VPT-CLIP softforce region prompt | 36.30 | 0/1901 | 0/1901 | 0 |

- Output summaries:
  - Reencode:
    `eval_outputs/vpt_qwen2vl2b_clip_blink_softforce_reencode_512_20260624_005920/merged/blink__vpt_clip_softforce_reencode.merged_summary.json`
  - Region:
    `eval_outputs/vpt_qwen2vl2b_clip_blink_softforce_region_512_20260624_005920/merged/blink__vpt_clip_softforce_region.merged_summary.json`
- Analysis:
  - Neither original VPT prompt phrase induced tool/action-token emission on this
    `rp-yu/Qwen2-VL-2b-VPT-CLIP` checkpoint under BLINK full max-res 512.
  - Sanity check: full merged raw outputs have max length 1 for direct/reencode/region,
    and first-round raw text contains 0 CLIP action tokens and 0 region action tokens.
    The zero trigger rate is not a parser miss.
  - Parsed option distributions:
    - Direct: A=854, B=443, C=96, D=508.
    - Reencode prompt: A=1093, B=300, C=86, D=422.
    - Region prompt: A=1045, B=296, C=46, D=514.
  - Therefore these two runs did not exercise the actual second-round CLIP or region
    crop mechanisms; they measure prompt perturbation of the direct answer path.
  - Both prompt perturbations hurt slightly relative to VPT direct, so this checkpoint
    likely needs an explicit forced-action path if we want to evaluate the VPT
    mechanism itself rather than self-trigger behavior.

### DIAG-20260624-vpt-qwen2vl2b-clip-blink-force-512

- Status: DONE.
- Baseline anchor:
  - `DIAG-20260624-vpt-qwen2vl2b-clip-blink-softforce-512`: both original prompt
    triggers produced 0/1901 action tokens, so they did not exercise VPT second-round
    mechanics.
- Purpose:
  - Separate self-trigger failure from second-round mechanism failure by forcing the
    assistant action token before second-round generation.
- Model/settings:
  - Same `rp-yu/Qwen2-VL-2b-VPT-CLIP` compatibility evaluator as the VPT direct and
    soft-force BLINK runs.
  - BLINK full val, max-res 512, project fallback scorer, deterministic generation.
- Variant A:
  - Method: `vpt_clip_force_clip`.
  - Forced first-round assistant text:
    `<|clip_action_start|><|clip_action|><|clip_action_end|>`.
  - Second round uses the evaluator's all-input-media CLIP re-encode path.
- Variant B:
  - Method: `vpt_clip_force_region_full`.
  - Forced first-round assistant text: full 8x8 region tokens.
  - Region crop follows original VPT's first-image-only crop path; this is a pipeline
    sanity check, not a fair multi-image BLINK region selection test.
- Planned sequence:
  - `limit=8` smoke for both variants.
  - Full 1901-sample runs only if smoke confirms nonzero second-round use and no row
    errors.
- Smoke findings:
  - Initial CLIP force exposed a compatibility bug: current HF image processor rejects
    the old `videos=None` argument in the CLIP image-processor call.
  - After fixing that, CLIP force exposed the deeper visual-output unwrap bug:
    current HF Qwen2-VL visual tower returns raw pre-merge tokens in `last_hidden_state`
    and merged tokens in `pooler_output`; VPT's projector expects merged tokens because
    its split sizes are `grid_thw.prod // spatial_merge_size^2`.
  - The compatibility shim now prefers `pooler_output` when available.
  - Post-fix smoke:
    - Direct `limit=8`: 62.5, trigger 0/8, second round 0/8, errors 0.
    - CLIP force `limit=8`: 62.5, trigger 8/8, second round 8/8, errors 0.
    - Region-full force `limit=8`: 62.5, trigger 8/8, second round 8/8, errors 0.
- Active full outputs:
  - `eval_outputs/vpt_qwen2vl2b_clip_blink_force_clip_512_20260624_012010`
  - `eval_outputs/vpt_qwen2vl2b_clip_blink_force_region_full_512_20260624_012010`
- Full results:

| Run | Accuracy | Trigger | Second round | Errors |
| --- | ---: | ---: | ---: | ---: |
| VPT-CLIP force CLIP | 43.66 | 1901/1901 | 1901/1901 | 0 |
| VPT-CLIP force full-region | 44.08 | 1901/1901 | 1901/1901 | 0 |

- Output summaries:
  - `eval_outputs/vpt_qwen2vl2b_clip_blink_force_clip_512_20260624_012010/merged/blink__vpt_clip_force_clip.merged_summary.json`
  - `eval_outputs/vpt_qwen2vl2b_clip_blink_force_region_full_512_20260624_012010/merged/blink__vpt_clip_force_region_full.merged_summary.json`
- Interpretation:
  - Forced CLIP and forced region-full both run cleanly after the visual unwrap fix.
  - The earlier VPT direct/soft prompt numbers should not be used as exact baselines
    anymore, because they were produced before the compatibility shim selected
    merged visual tokens via `pooler_output`.
  - A fixed direct rerun is required before claiming that forced second-round VPT
    improves over direct VPT.

### DIAG-20260624-vpt-qwen2vl2b-clip-blink-direct-pooler-512

- Status: DONE.
- Purpose:
  - Re-run VPT direct BLINK full after the compatibility shim fix
    (`pooler_output` over `last_hidden_state`) so force results have a valid baseline.
- Model/settings:
  - `rp-yu/Qwen2-VL-2b-VPT-CLIP`.
  - BLINK full val, max-res 512, project fallback scorer.
  - No trigger prompt, no forced action, no CoT.
- Active output:
  - `eval_outputs/vpt_qwen2vl2b_clip_blink_direct_pooler_512_20260624_013020`
- Result:
  - Accuracy: 43.82.
  - Trigger: 0/1901.
  - Second round: 0/1901.
  - Errors: 0.
  - Summary:
    `eval_outputs/vpt_qwen2vl2b_clip_blink_direct_pooler_512_20260624_013020/merged/blink__vpt_clip_direct_pooler.merged_summary.json`
- Corrected comparison table:

| Run | Accuracy | Trigger | Second round | Notes |
| --- | ---: | ---: | ---: | --- |
| Qwen2-VL-2B direct | 41.35 | n/a | n/a | unaffected baseline |
| VPT direct, old compat unwrap | 37.30 | 0/1901 | 0/1901 | invalid as exact baseline |
| VPT direct, fixed `pooler_output` unwrap | 43.82 | 0/1901 | 0/1901 | corrected baseline |
| VPT force CLIP, fixed unwrap | 43.66 | 1901/1901 | 1901/1901 | forced second round |
| VPT force full-region, fixed unwrap | 44.08 | 1901/1901 | 1901/1901 | first-image-only sanity |

- Analysis:
  - The user's suspicion was correct: the old compatibility evaluator had a real
    visual-output unwrap bug, hidden by the fact that soft prompts never exercised
    the CLIP second-round path.
  - After fixing the visual token source, VPT direct improves from 37.30 to 43.82 on
    BLINK full 512.
  - Forced CLIP does not beat the corrected direct baseline: 43.66 vs 43.82.
    This suggests the VPT CLIP second-round mechanism is cleanly runnable but not
    beneficial on this BLINK setting when forced for every sample.
  - Region-full is 44.08, but because it simply re-feeds a full crop of the first
    image and BLINK often has multiple images, it should be treated only as a
    pipeline sanity check.

### EXP-20260624-qwen2vl2b-tgvf-stage1-stage2-512-0_3

- Status: DONE.
- User request:
  - Train our TGVF Stage1 and Stage2 based on `Qwen/Qwen2-VL-2B-Instruct`.
  - Use GPUs 0-3.
  - Use `max_image_resolution=512`.
  - Preserve current assets.
  - Write a report for the just-finished VPT/Qwen2 diagnostic results.
- Asset safety:
  - New launcher only: `scripts/run_tgvf_v3_qwen2vl2b_stage1_stage2_512_0_3.sh`.
  - New output roots:
    - `outputs/tgvf_v3_protocol_c_stage1_qwen2vl2b/...`
    - `outputs/tgvf_v3_protocol_c_qwen2vl2b/...`
  - The launcher refuses to write into non-empty output dirs unless
    `ALLOW_EXISTING_OUTPUT=1` is explicitly set.
  - Existing Qwen3 launchers and checkpoints are not overwritten.
- Qwen2 compatibility:
  - `load_qwen3_vl` now branches by `AutoConfig.model_type`.
  - `qwen3_vl` still uses `Qwen3VLForConditionalGeneration`.
  - `qwen2_vl` uses `Qwen2VLForConditionalGeneration`.
  - This is intentionally a narrow loader branch so the Qwen3 path is unchanged.
  - `build_qwen3_inputs` now reads `processor.image_processor.patch_size`.
    Qwen3 remains patch size 16; Qwen2 uses patch size 14, avoiding invalid
    512-resized shapes that are not divisible by Qwen2's 28-token merge factor.
- Smoke:
  - Loader smoke loaded `Qwen/Qwen2-VL-2B-Instruct` as
    `Qwen2VLForConditionalGeneration` with `Qwen2VLProcessor`.
  - Mock dispatch smoke confirmed Qwen3 still dispatches to
    `Qwen3VLForConditionalGeneration`.
  - First Stage1 smoke failed before this patch with Qwen2 image processor reshape:
    invalid 512x320-style shape from hard-coded `image_patch_size=16`.
  - Processor smoke after the fix produced valid Qwen2 grid `[[1, 36, 22]]`.
  - Full chain smoke passed:
    - Stage1 1 step:
      `outputs/tgvf_v3_protocol_c_stage1_qwen2vl2b/protocol_c_toolobs_stage1_qwen2vl2b_v4data_clean_imend_bidirectional_4gpu_bs4_accum2_gbs32_1step_maxres512_20260624_qwen2_smoke2_015659/train/checkpoint_step_1.pt`
    - Stage2 1 step:
      `outputs/tgvf_v3_protocol_c_qwen2vl2b/protocol_c_toolobs_stage2_qwen2vl2b_v4data_clean_imend_open_answer_multifocus_focus_imend_from_stage1_4gpu_bs16_accum2_focus80_value1_1step_maxres512_20260624_qwen2_smoke2_015659/checkpoint_step_1.pt`
- Planned training settings:
  - Stage1:
    - data: `tgvf_v4_teacher_50k_clean_imend` Protocol C focus split.
    - global batch: `4 GPUs * bs4 * accum2 = 32`.
    - steps: 2000.
    - max res: 512.
    - row mode: `row_only`.
  - Stage2:
    - data: `tgvf_v4_teacher_50k_clean_imend_open_answer` Protocol C split.
    - global batch: `4 GPUs * bs16 * accum2 = 128`.
    - steps: 1200.
    - max res: 512.
    - loss weights: legacy defaults including `evidence_state=0.2`,
      `focus_target=1.5`, `evidence=1.0`, `value_span=1.0`, `answer=1.0`,
      `no_focus_evidence_state=0.2`, `no_focus_answer=1.0`.
    - current mask defaults: `mask_original_image_after_tgvf=true`,
      `prob=1.0`, `scope=evidence_only`.
- Report:
  - `reports/vpt_qwen2vl2b_blink512_force_report_20260624.md`
- Full run:
  - tmux: `qwen2vl2b_tgvf_stage1_stage2_512_0_3_20260624_020233`
  - Stage1 output:
    `outputs/tgvf_v3_protocol_c_stage1_qwen2vl2b/protocol_c_toolobs_stage1_qwen2vl2b_v4data_clean_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_maxres512_20260624_020233`
  - Stage2 output:
    `outputs/tgvf_v3_protocol_c_qwen2vl2b/protocol_c_toolobs_stage2_qwen2vl2b_v4data_clean_imend_open_answer_multifocus_focus_imend_from_stage1_4gpu_bs16_accum2_focus80_value1_1200step_maxres512_20260624_020233`
- Timing:
  - Started: 2026-06-24 02:03 JST.
  - Stage1 train finished: 2026-06-24 03:34 JST, wall `5485.76s`.
  - Stage1 eval finished: 2026-06-24 03:46 JST.
  - Stage2 train finished: 2026-06-24 08:21 JST, wall `4:35:16`.
  - Stage2 protocol eval finished: 2026-06-24 09:32 JST, wall `1:11:09`.
- Metrics:
  - Stage1 readout eval:
    - samples: 200.
    - `pct_correct_D_beats_target_only=0.995`.
    - `pct_correct_D_beats_wrong_same=0.925`.
    - `mean_delta_correct_vs_target_only=0.6179`.
    - `mean_delta_correct_vs_wrong_same=0.2503`.
  - Stage1 query sensitivity:
    - groups: 46, items: 200.
    - `retrieval_top1=0.775`.
    - `mrr=0.8739`.
    - `mean_diagonal_gap=0.1394`.
  - Stage1 distribution:
    - `avg_manifold_loss=4.9296`.
    - `norm_ratio_D_to_Vmerge=1.8531`.
    - `finite_rate=1.0`.
    - `collapse_warning=false`.
  - Stage2 validation at step 1200:
    - samples: 128.
    - loss: `1.0561`.
    - focus/no-focus: `111/17`.
    - `value_span_match_rate=0.5469`.
    - `protocol_c_boundary_acc_focus_start=1.0`.
    - `protocol_c_boundary_acc_focus_end=0.8829`.
    - `protocol_c_boundary_acc_mean=0.9414`.
  - Stage2 protocol eval at step 1200:
    - `no_focus_direct`: n=128, parse=1.0, acc=0.1641.
    - `free_router_end2end`: n=256, parse=0.8359, acc=0.1367,
      trigger=0.0, focus miss=1.0.
    - `force_focus_targets`: n=128, focus valid=0.7734, parse=0.0.
    - `teacher_forced_post_tgvf`: n=640, append success=0.9875,
      focus valid=1.0, parse=0.0, acc=0.0.
    - `force_end2end`: n=524, append success=0.9294,
      focus valid=0.9447, parse=0.0, acc=0.0.
- Analysis:
  - Stage1 looks healthy: correct-D beats target-only and wrong-same at high rates,
    query sensitivity is non-trivial, and D distribution does not collapse.
  - Stage2 validation loss/boundary numbers alone are not enough; the end-to-end
    protocol eval fails after TGVF insertion. Actual samples continue with
    malformed/repetitive `<think>` text or fragments such as `The<|im_end|>`
    instead of emitting a parseable answer.
  - Free mode never triggers focus in this eval (`trigger=0.0`), while forced and
    teacher-forced modes do append D but still do not recover answer generation.
  - This points to a Qwen2-specific Stage2 protocol-generation problem, not a
    Stage1 D/readout failure.
  - Full benchmark was not run by this chain; only Stage1 eval and Stage2 protocol
    eval were executed.
- Conclusion:
  - The Qwen2 Stage1 adaptation is technically working, but this Qwen2 Stage2 run is
    not usable as a TGVF end-to-end checkpoint yet.
  - Next debugging should focus on Qwen2 Stage2 answer-format supervision/template
    and post-TGVF continuation behavior before spending time on full benchmarks.

### EXP-20260624-qwen2vl2b-no-think-stage1-stage2-512-0_3

- Status: INTERRUPTED_PARTIAL.
- Question:
  - Does a Qwen2-specific no-think tool-observation protocol fix the Stage2
    continuation/answer parsing failure seen in `EXP-20260624-qwen2vl2b-tgvf-stage1-stage2-512-0_3`?
- Baseline anchor:
  - `EXP-20260624-qwen2vl2b-tgvf-stage1-stage2-512-0_3`.
- Intended diff:
  - New protocol: `protocol_c_tool_observation_qwen2_no_think`.
  - Focus action: `<|focus_start|>{target}<|focus_end|><|im_end|>`.
  - TGVF tool turn unchanged: `<|im_start|>tool`, `<|tgvf_start|>`, D, `<|tgvf_end|><|im_end|>`.
  - Post-TGVF answer turn: `<|evidence_start|>{evidence}<|evidence_end|>\n{answer}<|im_end|>`.
  - No-focus samples: direct `{answer}<|im_end|>`, no `<think>` span.
  - Stage1 is retrained because both focus-action `H_q` context and readout context changed.
- Allowed changed variables:
  - Protocol text format and protocol token set.
  - Stage1 checkpoint/processor due to protocol retraining.
- Not allowed to change:
  - Base model: `Qwen/Qwen2-VL-2B-Instruct`.
  - Data files.
  - Max image resolution: 512.
  - Stage1/Stage2 steps, batch, LR, mask defaults, and legacy Stage2 loss weights.
- Code changes:
  - Added `protocol_c_tool_observation_qwen2_no_think` to protocol render/parser/token handling.
  - Generalized Stage1/Stage2 protocol token-row save/restore from fixed four Protocol C tokens to the active protocol token set.
  - Added launcher: `scripts/run_tgvf_v3_qwen2vl2b_no_think_stage1_stage2_512_0_3.sh`.
- Smoke:
  - Python compile passed for protocol, Stage1/Stage2 training, fast Stage2, and protocol eval modules.
  - Shell syntax passed for the no-think launcher and the Stage1/Stage2 wrapper scripts.
  - Render/parser smoke confirmed:
    - tokens: focus start/end, tgvf start/end, evidence start/end.
    - focus action has no `<think>` and parses a valid focus target.
    - post-TGVF evidence-tag output parses final answer.
    - direct no-focus answer strips trailing `<|im_end|>`.
  - 1-step chain smoke passed with `WANDB_MODE=disabled`, `RUN_STAGE1_EVAL=0`, `RUN_PROTOCOL_EVAL=0`:
    - Stage1 smoke checkpoint:
      `outputs/tgvf_v3_protocol_c_stage1_qwen2vl2b/protocol_c_toolobs_qwen2_no_think_stage1_qwen2vl2b_v4data_clean_imend_bidirectional_4gpu_bs4_accum2_gbs32_1step_maxres512_20260624_qwen2_no_think_smoke_120200/train/checkpoint_step_1.pt`
    - Stage2 smoke checkpoint:
      `outputs/tgvf_v3_protocol_c_qwen2vl2b/protocol_c_toolobs_qwen2_no_think_stage2_qwen2vl2b_v4data_clean_imend_open_answer_multifocus_focus_imend_from_stage1_4gpu_bs16_accum2_focus80_value1_1step_maxres512_20260624_qwen2_no_think_smoke_120200/checkpoint_step_1.pt`
    - Stage1 token-row check: six rows saved, shape `(6, 1536)` for input and output embeddings.
    - Stage2 step-1 debug used `protocol_c_tool_observation_qwen2_no_think`, focus ratio `0.875`, value span match `0.5625`, mask scope `evidence_only`.
- Stage1 output:
  - `outputs/tgvf_v3_protocol_c_stage1_qwen2vl2b/protocol_c_toolobs_qwen2_no_think_stage1_qwen2vl2b_v4data_clean_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_maxres512_20260624_120501`
- Stage2 output:
  - `outputs/tgvf_v3_protocol_c_qwen2vl2b/protocol_c_toolobs_qwen2_no_think_stage2_qwen2vl2b_v4data_clean_imend_open_answer_multifocus_focus_imend_from_stage1_4gpu_bs16_accum2_focus80_value1_1200step_maxres512_20260624_120501`
- Train data:
  - Stage1: `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl`.
  - Stage2: `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl`.
- Validation data:
  - Stage1: `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl`.
  - Stage2: `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`.
- Script / command:
  - `STAMP=20260624_120501 WANDB_MODE=online CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_tgvf_v3_qwen2vl2b_no_think_stage1_stage2_512_0_3.sh`.
- GPUs:
  - 0,1,2,3.
- tmux:
  - `qwen2vl2b_no_think_stage1_stage2_512_0_3_20260624_120501`.
- Started:
  - 2026-06-24 12:05 JST.
- W&B:
  - Online sync enabled.
  - Stage1 run id: `vt5gyw1q`.
  - Stage1 URL: `https://wandb.ai/mio_nora/tgvf-v3/runs/vt5gyw1q`.
  - Stage2 run id: `3nayd69w`.
  - Stage2 URL: `https://wandb.ai/mio_nora/tgvf-v3/runs/3nayd69w`.
- Active progress:
  - Stage1 step 1 logged successfully.
  - `tgvf_protocol=protocol_c_tool_observation_qwen2_no_think`.
  - `protocol_c_token_row_param_count=9216`, matching six protocol tokens at hidden size 1536.
  - step-1 `loss_total=5.2152`, `loss_gen=3.6016`, `loss_same_image_negative=1.4688`, `finite_rate=1.0`.
  - Stage1 train finished at 2026-06-24 13:36 JST, wall `5483.90s`.
  - Stage1 eval finished at 2026-06-24 13:49 JST.
  - Stage2 started at 2026-06-24 13:49 JST and is running.
- Stage1 metrics:
  - Readout:
    - samples: 200.
    - `pct_correct_D_beats_target_only=1.0`.
    - `pct_correct_D_beats_wrong_same=0.915`.
    - `mean_delta_correct_vs_target_only=0.9514`.
    - `mean_delta_correct_vs_wrong_same=0.3256`.
  - Query sensitivity:
    - groups: 46, items: 200.
    - `retrieval_top1=0.710`.
    - `mrr=0.8367`.
    - `mean_diagonal_gap=0.1317`.
  - Distribution:
    - `avg_manifold_loss=3.9833`.
    - `norm_ratio_D_to_Vmerge=1.6143`.
    - `finite_rate=1.0`.
    - `collapse_warning=false`.
- Stage1 comparison to previous Qwen2 thinking Stage1:
  - Readout correct-D vs target-only improved: `0.995 -> 1.000`.
  - Readout delta vs target-only improved: `0.6179 -> 0.9514`.
  - Readout delta vs wrong-same improved: `0.2503 -> 0.3256`.
  - Correct-D beats wrong-same slightly decreased: `0.925 -> 0.915`.
  - Query retrieval worsened: top1 `0.775 -> 0.710`, MRR `0.8739 -> 0.8367`.
  - D norm/manifold is less inflated: norm ratio `1.8531 -> 1.6143`, manifold loss `4.9296 -> 3.9833`.
  - Interpretation: no-think Stage1 is not collapsed and has stronger readout separation, but target-specific retrieval is weaker than the thinking-protocol Stage1.
- Finished:
  - 2026-06-24 18:45 JST.
  - End-to-end wall from launch to protocol eval summary: about 6h40m.
  - Stage1 train wall: `5483.90s` / about 1h31m.
  - Stage2 train wall: `3:54:59`.
  - Protocol eval wall: `3645.27s` / about 1h01m.
- Metrics:
  - Stage2 checkpoint:
    - `outputs/tgvf_v3_protocol_c_qwen2vl2b/protocol_c_toolobs_qwen2_no_think_stage2_qwen2vl2b_v4data_clean_imend_open_answer_multifocus_focus_imend_from_stage1_4gpu_bs16_accum2_focus80_value1_1200step_maxres512_20260624_120501/checkpoint_step_1200.pt`
  - Stage2 validation at step 1200:
    - `loss=1.2223`.
    - `focus_ratio=0.8672`, `no_focus_ratio=0.1328`.
    - `value_span_match_rate=0.5469`.
    - boundary acc: focus_start `1.0000`, focus_end `0.8829`, evidence_start `1.0000`, evidence_end `1.0000`, mean `0.9707`.
  - Protocol eval output:
    - `outputs/tgvf_v3_protocol_c_qwen2vl2b/protocol_c_toolobs_qwen2_no_think_stage2_qwen2vl2b_v4data_clean_imend_open_answer_multifocus_focus_imend_from_stage1_4gpu_bs16_accum2_focus80_value1_1200step_maxres512_20260624_120501/protocol_eval_step_1200/summary.json`
  - Protocol eval:
    - `no_focus_direct`: parse `1.0000`, accuracy `0.5469`, malformed `0.2031`.
    - `free_router_end2end`: parse `0.9844`, accuracy `0.4531`, trigger `0.0000`, focus_miss `1.0000`, malformed `0.2773`.
    - `force_focus_targets`: focus_valid `0.8594`, append_success `0.0000`, answer_parse `0.0000`.
    - `teacher_forced_post_tgvf`: append_success `0.9875`, focus_valid `1.0000`, answer_parse `0.1469`, accuracy `0.0188`.
    - `teacher_forced_post_tgvf/correct_D`: answer_parse `0.0078`, accuracy `0.0000`.
    - `teacher_forced_post_tgvf/no_D`: answer_parse `0.6563`, accuracy `0.0859`.
    - `force_end2end`: append_success `0.9560`, focus_valid `0.9683`, answer_parse `0.0951`, accuracy `0.0055`.
    - `force_end2end/correct_D`: append_success `0.8594`, answer_parse `0.0000`, accuracy `0.0000`.
- Analysis:
  - The no-think protocol fixed the most obvious Qwen2 incompatibility for direct/free answer parsing:
    - Previous Qwen2 thinking protocol had `no_focus_direct` accuracy `0.1641` and `free_router_end2end` parse `0.8359`, accuracy `0.1367`.
    - No-think improves these to `0.5469` direct and `0.4531` free with parse `0.9844`.
  - It did not make the TGVF route functional:
    - Free routing still never chooses focus: trigger remains `0.0000`.
    - Forced/generated focus targets are often syntactically valid, but answer continuation after TGVF is still broken: `force_end2end/correct_D` parse `0.0000`.
    - Teacher-forced correct D is also essentially unusable: `correct_D` parse `0.0078`, accuracy `0.0000`.
  - The surprising diagnostic is that `no_D` after teacher forcing parses and answers better than `correct_D`.
    This points away from a pure parser bug and toward the model treating the inserted D/evidence context as a bad continuation context for Qwen2 under this protocol/training setup.
  - Stage1 did learn a usable D manifold/readout signal, but Stage2 did not learn to consume that D as an answer-enabling context.
    The likely bottleneck is Stage2 post-tool continuation behavior, not Stage1 D generation alone.
  - Root-cause localization after inspecting eval JSONL:
    - Direct/free answer parsing is mostly healthy, so the final-answer parser and Qwen2 no-think direct format are not the main failure.
    - Force target generation is often syntactically valid, so target-span parsing is not the main failure.
    - With visual D inserted, continuations usually start as degenerate text such as `The...`, `The<|im_end|>`, or repeated `the`, and rarely emit `<|evidence_end|>`.
    - Without D, the same post-tool scaffold emits `<|evidence_end|>` far more often, even though answers remain weak.
    - The concrete failure is therefore Qwen2 post-D continuation under the current FVT append/training setup.
      Candidate mechanisms are: autoregressive KV append of a second visual-token block is not equivalent to the Stage2 full-sequence teacher-forced training path for Qwen2, or the trained D distribution is a strong LM-state perturbation for Qwen2.
    - A double `<|im_end|>` prefix mismatch and an off-by-one FVT scatter error were checked against recorded protocol text/helper layout and are not supported by the evidence.
  - Follow-up eval-only ablation `outputs/diagnostics/qwen2_no_think_protocol_eval_append_aligned_32_20260624`:
    - Code changed eval append to call `render_tgvf_prefix_suffix(..., include_leading_im_end=not protocol_uses_tool_observation(protocol))`, matching Stage2 training for tool-observation protocols.
    - No retraining; same checkpoint, 32 focus samples, blocks `teacher_forced_post_tgvf,force_end2end`, D conditions `correct_D,no_D`.
    - `teacher_forced_post_tgvf/correct_D`: parse `0.03125`, accuracy `0.0`, evidence_end rate `0.03125`.
    - `force_end2end/correct_D`: parse `0.0`, accuracy `0.0`, evidence_end rate `0.0`.
    - `no_D` remained much more parseable: teacher-forced parse `0.625`, force parse `0.5862`.
    - Conclusion: repeated leading `<|im_end|>` in eval was a real train/eval mismatch, but not the main cause of Qwen2 post-D collapse.
  - Follow-up no-KV full-sequence prefill ablation `outputs/diagnostics/qwen2_no_think_protocol_eval_fullseq_32_20260624`:
    - Code added `--append-prefill-mode full_sequence` to `eval/eval_v3_stage2_protocol.py`.
    - Instead of appending the tool/D segment to existing `past_key_values`, this path builds the full sequence `prompt + action + tool/D + evidence_start`, scatters original image embeddings and D embeddings into `inputs_embeds`, computes full position ids/mm token types, runs one full forward, then continues autoregressively.
    - Same checkpoint, 32 focus samples, blocks `teacher_forced_post_tgvf,force_end2end`, D conditions `correct_D,no_D`.
    - `teacher_forced_post_tgvf/correct_D`: parse `0.03125`, accuracy `0.0`, evidence_end rate `0.09375`.
    - `force_end2end/correct_D`: parse `0.0`, accuracy `0.0`, evidence_end rate `0.0625`.
    - Compared to KV-append-aligned, correct-D parse did not improve: teacher-forced `0.03125 -> 0.03125`, force `0.0 -> 0.0`.
    - Conclusion: the main failure is not explained by preserving/reusing the action KV cache during D append.
      The evidence now points more strongly to Stage2 not learning a usable post-D continuation for Qwen2 under this protocol/training setup.
  - Follow-up post-D likelihood/logit probe:
    - Script: `scripts/probe_qwen2_post_d.py`.
    - Output: `outputs/diagnostics/qwen2_no_think_post_d_probe_16_scaled_20260624`.
    - Setup: same checkpoint, first 16 focus validation samples, teacher-forced gold continuation after `<|evidence_start|>`, conditions `correct_D`, `correct_D_scaled_to_vmerge_norm`, `zero_D`, `random_calibrated_D`, `no_D`.
    - `correct_D`: mean NLL `4.4481`, first-token prob `0.7252`, evidence_end NLL `11.1509`, median evidence_end rank `2028.5`, D norm `93.69`, v_merge norm `58.11`, D-v_merge cosine `0.168`.
    - `zero_D`: mean NLL `4.0377`, first-token prob `0.7275`, evidence_end NLL `7.1131`, median evidence_end rank `98.5`.
    - `random_calibrated_D`: mean NLL `4.0358`, evidence_end NLL `7.1692`, median evidence_end rank `104.5`.
    - `correct_D_scaled_to_vmerge_norm`: mean NLL `4.0804`, evidence_end NLL `7.1234`, median evidence_end rank `120.5`, D norm `58.12`.
    - Pairwise: unscaled `correct_D` is worse than scaled `correct_D` on mean NLL for `12/16` samples and worse on evidence_end NLL for `14/16`; scaling reduces evidence_end NLL by `4.03` on average.
    - Scaled `correct_D` is statistically similar to `zero_D/random_calibrated_D`, not better.
    - Conclusion: the main immediate post-D collapse is strongly linked to D scale/norm mismatch. However, once scale is corrected, D still does not show clear semantic information gain over zero/random D on gold continuation likelihood.
- Conclusion:
  - DONE, but not successful as a TGVF model.
  - Qwen2 no-think protocol is a necessary format fix for Qwen2 direct/free parsing, yet Stage2 still fails specifically on post-TGVF answer generation.
  - Do not benchmark this checkpoint as a meaningful TGVF route result unless the purpose is to diagnose Qwen2 post-tool failure.
- Comparable to baseline:
  - Comparable as a Qwen2 protocol-format ablation, with Stage1 retrained intentionally because protocol context changed.

### DIAG-20260624-qwen3-stage1-stage2-scale

- Question:
  - Check whether current Qwen3 Stage1/Stage2 training has a D scale problem.
- Stage1 source:
  - Existing foveated visual token distribution reports under
    `outputs/tgvf_v3_protocol_c_stage1_8b/*/eval/fvt_distribution/fvt_distribution_report.json`.
- Stage1 metrics:
  - `protocol_c_stage1_encoder_reencode_two_branch_4gpu_2000step`:
    D norm `20.705`, V_merge norm `20.647`, ratio `1.003`.
  - `protocol_c_stage1_encoder_reencode_two_branch_film_aggressive_4gpu_2000step_online`:
    D norm `21.009`, V_merge norm `20.647`, ratio `1.018`.
  - `protocol_c_stage1_v4data_bidirectional_4gpu_2000step`:
    D norm `120.902`, V_merge norm `20.580`, ratio `6.060`, manifold loss `3.305`.
  - `protocol_c_toolobs_stage1_v4data_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260611_021244`:
    D norm `106.242`, V_merge norm `20.580`, ratio `5.350`, manifold loss `2.608`.
  - `protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617`:
    D norm `103.450`, V_merge norm `20.684`, ratio `5.174`, manifold loss `2.380`.
  - `protocol_c_toolobs_stage1_v4data_clean_imend_bidirectional_2gpu_bs4_accum4_gbs32_2000step_20260619_014148`:
    D norm `104.320`, V_merge norm `20.684`, ratio `5.217`, manifold loss `2.458`.
  - `protocol_c_toolobs_stage1_v4data_clean_imend_fullmask_bidirectional_2gpu_bs4_accum4_gbs32_2000step_20260620_230349`:
    D norm `99.637`, V_merge norm `20.684`, ratio `4.983`, manifold loss `2.236`.
- Stage2 source:
  - Output: `outputs/diagnostics/qwen3_stage2_scale_probe_4_20260624/summary.json`.
  - Method: load each Stage2 checkpoint on Qwen3-VL-8B-Thinking, teacher-force focus action on the first 4 focus validation samples, compute D via the loaded foveal module, and compare mean token norm against the original image `V_merge` norm.
- Stage2 metrics:
  - `20260619_open_answer`:
    D norm `105.75`, V_merge norm `20.94`, ratio `5.05`, D/V cosine `0.127`.
  - `20260623_oldmask075_correct_s1`:
    D norm `105.86`, V_merge norm `20.94`, ratio `5.06`, D/V cosine `0.127`.
  - `20260617_rowonly`:
    D norm `98.67`, V_merge norm `20.94`, ratio `4.71`, D/V cosine `0.121`.
  - `20260620_fullmask`:
    D norm `98.40`, V_merge norm `20.94`, ratio `4.70`, D/V cosine `0.117`.
- Config check:
  - Qwen3 tool-observation Stage1 uses `visual_token_manifold=0.01`.
  - Qwen3 tool-observation Stage2 uses `visual_token_manifold=0.0`.
  - Stage2 therefore does not continue enforcing the D manifold/norm; it consumes the Stage1 D distribution through answer/evidence losses.
- Analysis:
  - Yes, current Qwen3 tool-observation Stage1/Stage2 has a large D scale mismatch relative to original merged visual tokens: roughly `4.7x-5.2x`.
  - This is not introduced by the 20260623 mask ablation; 20260619 and 20260623 have almost identical Stage2 scale.
  - The mismatch already exists in Stage1 for the bidirectional/tool-observation family, then persists into Stage2.
  - Earlier reencode/two-branch Stage1 variants did not have this scale mismatch; their D/V ratios were about `1.0`.
  - Therefore the scale issue is tied to the current bidirectional/tool-observation Stage1/FVT formulation, not to Qwen3 as a base model in general.

### DIAG-20260624-qwen3-merger-input-scale

- Question:
  - Check whether the scale problem already exists in the tensor sent into the frozen Qwen visual merger, or whether it appears only after the merger.
- Method:
  - Measure `conditioned_pre_merge_visual_tokens` from `TGVFv2Bidirectional` before `finalize_tgvf_output_with_frozen_qwen_merger`.
  - Compare against original tapped `V_pre`.
  - Then measure post-merger `D` against original `V_merge`.
  - Use first 4 valid focus validation samples, max image resolution 512.
- Outputs:
  - Stage2 probe: `outputs/diagnostics/qwen3_merger_input_scale_probe_4_20260624/summary.json`.
  - Stage1 probe: `outputs/diagnostics/qwen3_stage1_merger_input_scale_probe_4_20260624/summary.json`.
- Stage2 metrics:
  - `20260619_stage2_open_answer`:
    - original `V_pre` norm `13.68`, conditioned pre-merge norm `28.49`, pre ratio `2.083`, delta/pre ratio `1.825`, conditioned/original cosine `0.466`, post-merger D/V_merge ratio `5.050`.
  - `20260623_stage2_oldmask075_correct_s1`:
    - original `V_pre` norm `13.68`, conditioned pre-merge norm `28.53`, pre ratio `2.086`, delta/pre ratio `1.828`, conditioned/original cosine `0.466`, post-merger D/V_merge ratio `5.055`.
  - `20260617_stage2_rowonly`:
    - original `V_pre` norm `13.68`, conditioned pre-merge norm `28.47`, pre ratio `2.081`, delta/pre ratio `1.770`, conditioned/original cosine `0.514`, post-merger D/V_merge ratio `4.712`.
  - `20260620_stage2_fullmask`:
    - original `V_pre` norm `13.68`, conditioned pre-merge norm `28.33`, pre ratio `2.071`, delta/pre ratio `1.803`, conditioned/original cosine `0.477`, post-merger D/V_merge ratio `4.699`.
- Stage1 metrics:
  - `20260619_stage1`:
    - original `V_pre` norm `13.68`, conditioned pre-merge norm `32.70`, pre ratio `2.391`, delta/pre ratio `2.168`, conditioned/original cosine `0.408`, post-merger D/V_merge ratio `4.999`.
  - `20260617_stage1_rowonly`:
    - original `V_pre` norm `13.68`, conditioned pre-merge norm `34.99`, pre ratio `2.558`, delta/pre ratio `2.302`, conditioned/original cosine `0.426`, post-merger D/V_merge ratio `4.959`.
  - `20260620_stage1_fullmask`:
    - original `V_pre` norm `13.68`, conditioned pre-merge norm `32.89`, pre ratio `2.404`, delta/pre ratio `2.173`, conditioned/original cosine `0.415`, post-merger D/V_merge ratio `4.781`.
- Analysis:
  - The tensor sent into the frozen Qwen visual merger is already off-scale and off-manifold.
  - The bidirectional toolobs Stage1 produces `conditioned_pre_merge_visual_tokens` about `2.4x-2.6x` the original `V_pre` norm, with a delta around `2.2x-2.3x` the original pre-merge norm.
  - Stage2 reduces this pre-merger ratio to about `2.07x-2.09x`, but does not restore it to the original pre-merge manifold.
  - The frozen Qwen merger then maps these off-manifold pre-merge tokens to LM-side D tokens about `4.7x-5.1x` the original `V_merge` norm.
  - Therefore the scale problem is not merely a frozen-merger amplification artifact; it starts before the merger in the learned TGVF residual update.

### DIAG-20260624-qwen3-stage1-manifold-loss-weight

- Question:
  - Check Qwen3 Stage1 manifold loss magnitude, its share of total loss, and why it did not prevent D scale drift.
- Code:
  - `_safe_visual_token_manifold_loss` computes:
    - `MSE(mean(D), mean(V_merge)) + MSE(std(D), std(V_merge))`.
  - Stage1 total loss is:
    - `gen * loss_gen + visual_token_manifold * loss_man + same_image_negative * loss_same + contrastive_alignment * loss_contrastive`.
  - Current Qwen3 toolobs Stage1 configs use `visual_token_manifold=0.01`.
- Parsed train logs:
  - `20260619`:
    - all logged steps: total `1.8602`, raw manifold `2.5396`, weighted manifold `0.0254`, share `1.43%`, D/V ratio `5.21`.
    - last 10 logs: total `1.5867`, raw manifold `2.4726`, weighted manifold `0.0247`, share `1.58%`, D/V ratio `5.08`.
    - final step: total `1.2546`, raw manifold `2.4599`, weighted manifold `0.0246`, share `1.96%`, D/V ratio `4.79`.
  - `20260617_rowonly`:
    - all logged steps: total `1.9006`, raw manifold `2.4561`, weighted manifold `0.0246`, share `1.37%`, D/V ratio `5.05`.
    - last 10 logs: total `1.6106`, raw manifold `2.3092`, weighted manifold `0.0231`, share `1.47%`, D/V ratio `4.97`.
    - final step: total `1.5569`, raw manifold `2.3967`, weighted manifold `0.0240`, share `1.54%`, D/V ratio `4.75`.
  - `20260620_fullmask`:
    - all logged steps: total `1.8628`, raw manifold `2.3582`, weighted manifold `0.0236`, share `1.32%`, D/V ratio `5.02`.
    - last 10 logs: total `1.5851`, raw manifold `2.2149`, weighted manifold `0.0221`, share `1.42%`, D/V ratio `4.80`.
    - final step: total `1.2566`, raw manifold `2.2127`, weighted manifold `0.0221`, share `1.76%`, D/V ratio `4.57`.
  - `20260611`:
    - all logged steps: total `1.8245`, raw manifold `2.6775`, weighted manifold `0.0268`, share `1.58%`, D/V ratio `5.03`.
    - last 10 logs: total `1.6234`, raw manifold `2.5853`, weighted manifold `0.0259`, share `1.62%`, D/V ratio `4.81`.
    - final step: total `1.8285`, raw manifold `2.5666`, weighted manifold `0.0257`, share `1.40%`, D/V ratio `4.87`.
- Analysis:
  - The manifold loss is active but weak. Raw manifold is around `2.2-2.7`, but after multiplying by `0.01` it contributes only `0.022-0.027` to total loss.
  - This is about `1.3%-2.0%` of total logged loss, while generation and same-image matrix CE dominate optimization.
  - The objective matches only feature-wise mean and std after merger. It does not directly constrain per-token norm, pre-merger residual norm, or cosine/proximity to the original pre-merge token manifold.
  - The learned bidirectional residual can therefore become large enough to encode the target/evidence signal while paying only a small weighted manifold penalty.
  - Stage2 sets `visual_token_manifold=0.0`, so any remaining scale mismatch is not corrected there.
- Decision:
  - Raise the default Stage1 manifold weight from `0.01` to `0.1`.
  - Updated code defaults:
    - `src/revisit_vlm/tgvf_training.py`: `LossWeights.visual_token_manifold=0.1`.
    - `scripts/train_tgvf_v3_stage1.py`: `--loss-visual-token-manifold` default `0.1`.
    - `scripts/train_tgvf_fvt.py`: `--loss-visual-token-manifold` default `0.1`.
    - Stage1 wrapper scripts default `LOSS_VISUAL_TOKEN_MANIFOLD=0.1`.
  - Stage2 remains default `0.0` for now. If we want Stage2 manifold regularization, run it as an explicit ablation rather than silently changing the Stage2 answer-continuation objective.

### BUGFIX-20260624-qwen2-eval-and-stage2-format-audit

- Trigger:
  - Qwen2-VL-2B no-think Stage1+Stage2 did not merely underperform; force/correct-D and teacher-forced post-TGVF collapsed at the protocol/parse level.
  - This should be treated as a code-path failure until proven otherwise, not as a small alignment or scale effect.
- Affected checkpoint:
  - `outputs/tgvf_v3_protocol_c_qwen2vl2b/protocol_c_toolobs_qwen2_no_think_stage2_qwen2vl2b_v4data_clean_imend_open_answer_multifocus_focus_imend_from_stage1_4gpu_bs16_accum2_focus80_value1_1200step_maxres512_20260624_120501/checkpoint_step_1200.pt`
- Findings:
  - Evaluation continuation bug:
    - `continue_generation_qwen3()` manually decoded with `input_ids=next_token`, full `past_key_values`, full `attention_mask`, and a hand-built 1D `position_ids`.
    - For Qwen2-VL, this bypasses `prepare_inputs_for_generation()` and overrides the model's multimodal RoPE position path.
    - Qwen2-VL forward explicitly computes 3D position ids when `position_ids is None`; passing the hand-built position ids prevents that code path.
    - Probe: direct no-position bare forward fails with attention/cache shape mismatch; `prepare_inputs_for_generation()` is the correct incremental API.
  - Stage2 fast single-focus format bug:
    - The actual Qwen2 Stage2 run used `fast_batched_stage2=True`.
    - In `tgvf_v3_stage2_fast._prepare_focus_final()`, single-focus readout used `render_focus_readout_answer_text(...)` without `append_im_end=protocol_uses_tool_observation(protocol)`.
    - In `_prepare_no_focus_final()`, no-focus output had the same missing `append_im_end`.
    - Slow Stage2 and multi-focus fast Stage2 already used `append_im_end=True` for tool-observation protocols.
    - The failed dataset is mostly single-focus: `39,312 / 39,655` focus samples, so this mismatch covers the dominant Stage2 training path.
- Evidence from one real Qwen2 sample:
  - Training action text has `<|focus_end|><|im_end|>`.
  - Teacher-forced eval capture also generated `<|focus_end|><|im_end|>`, so the focus/tool boundary itself was not missing.
  - Append chunk was `tool -> <|tgvf_start|> visual tokens <|tgvf_end|><|im_end|> assistant -> <|evidence_start|>`, matching the intended prefix.
  - Before the fast-path fix, single-focus final target ended with `...<|evidence_end|>\nanswer` instead of `...answer<|im_end|>`.
- Code changes:
  - `src/revisit_vlm/qwen3_vl_tgvf.py`
    - `continue_generation_qwen3()` now uses `model.prepare_inputs_for_generation(...)` when available, instead of always forcing hand-built 1D position ids.
  - `src/revisit_vlm/tgvf_v3_stage2_fast.py`
    - Single-focus final readout now passes `append_im_end=protocol_uses_tool_observation(protocol)`.
    - No-focus fast output now passes `append_im_end=protocol_uses_tool_observation(protocol)`.
  - `tests/test_qwen3_vl_tgvf.py`
    - Added assertion that continuation uses the model prepare hook for every decoded token.
- Verification:
  - `python -m py_compile src/revisit_vlm/qwen3_vl_tgvf.py src/revisit_vlm/tgvf_v3_stage2_fast.py`
  - `pytest -q tests/test_qwen3_vl_tgvf.py::test_continue_generation_stops_on_repetitive_tail`
- Current interpretation:
  - Existing Qwen2 no-think benchmark results are not clean because eval continuation used a wrong incremental generation path.
  - The existing Qwen2 no-think Stage2 checkpoint is also not clean because dominant single-focus fast-path targets were trained without final `<|im_end|>`.
  - A clean Qwen2 conclusion requires rerunning Stage2 after the fast-path target fix, then reevaluating with the continuation fix.

### AUDIT-20260624-d-position-and-embedding-invariants

- Question:
  - Confirm whether Qwen2 and Qwen3 training feed D as real visual tokens with correct embeddings, `mm_token_type_ids`, `image_grid_thw`, and M-RoPE `position_ids`.
  - Also check whether the surrounding original-image embeddings are clean.
- Probe:
  - Real Qwen2-VL-2B-Instruct and local Qwen3-VL-2B-Thinking.
  - Sample image: `/home/dredvpn009/Flash_Storage/datasets/visual_genome/VG_100K_2/2410492.jpg`.
  - Construct a minimal prompt plus appended TGVF visual block at max resolution 512.
  - Compare:
    - D placeholder count vs source merged visual token count.
    - D `mm_token_type_ids`.
    - D M-RoPE position shape and whether the D span is a 3D visual span.
    - Whether D embeddings are replaced by D.
    - Whether original image embeddings equal tapped `V_merge`.
    - Whether eval append chunk positions equal full-sequence positions for the same sequence.
- Qwen2 probe result:
  - source/D count: `216`.
  - `mm_original_all_image=true`, `mm_fvt_all_image=true`.
  - `position_shape=[3,1,492]`, `fvt_position_3d=true`.
  - `stage1_fvt_embed_equals_d=true`.
  - `stage2fast_original_embed_equals_vmerge=true`.
  - `stage2fast_fvt_embed_equals_d=true`.
  - `stage1_original_embed_equals_vmerge=false`, mean abs diff vs `V_merge` = `1.0108`.
  - eval append chunk position equals training full-sequence chunk position: `true`.
- Qwen3 probe result:
  - source/D count: `160`.
  - `mm_original_all_image=true`, `mm_fvt_all_image=true`.
  - `position_shape=[3,1,370]`, `fvt_position_3d=true`.
  - `stage1_fvt_embed_equals_d=true`.
  - `stage2fast_original_embed_equals_vmerge=true`.
  - `stage2fast_fvt_embed_equals_d=true`.
  - `stage1_original_embed_equals_vmerge=false`, mean abs diff vs `V_merge` = `0.4481`.
  - eval append chunk position equals training full-sequence chunk position: `true`.
- Interpretation:
  - The D span itself is correctly represented as visual tokens in both Qwen2 and Qwen3:
    - the placeholder count matches the source merged visual token count;
    - the D span has image token type ids;
    - the D span has 3D visual M-RoPE positions from the real source grid;
    - the D span embeddings are replaced by the generated D tensor.
  - Stage2 fast training is also clean for original image embeddings:
    - it scatters original `V_merge` into original image token positions;
    - it scatters D into the appended TGVF image token positions.
  - Stage1 readout and the legacy slow Stage2 path are not clean in the same way:
    - they use full `inputs_embeds` and replace only the D span;
    - original image placeholder positions remain ordinary `<|image_pad|>` token embeddings, not true `V_merge`;
    - post-TGVF masks block original image keys, so this may have been partially hidden, but it is not an original-model-equivalent context.
  - Eval append chunk positions are aligned with training full-sequence D chunk positions for both Qwen2 and Qwen3.
  - Eval continuation after append still needs a separate stricter fix:
    - using the model prepare hook is not enough unless the continuation uses the append-after-D M-RoPE delta;
    - the correct delta is derived from the append/full sequence positions as `max(position_ids)+1-full_sequence_len`.
- Status:
  - Do not describe the current pipeline as fully clean.
  - Clean Stage2-fast D position/embedding: yes.
  - Clean Stage1 original-image embedding context: no.
  - Clean eval continuation position after D: not yet guaranteed.

### BUGFIX-20260624-stage1-readout-original-image-embedding

- Trigger:
  - `AUDIT-20260624-d-position-and-embedding-invariants` found that Stage1 readout full forward replaced the appended D span but did not replace original image placeholder embeddings with the true Qwen merged visual tokens `V_merge`.
  - This meant Stage1 readout loss saw original image positions as `<|image_pad|>` token embeddings, not original-model visual embeddings.
- Fix:
  - `prepare_v3_stage1_readout_inputs(...)` now accepts optional `merged_visual_tokens`.
  - When provided, it scatters `merged_visual_tokens` into `source_visual_token_indices` before scattering D into the appended TGVF image span.
  - Stage1 training now passes `feature.merged_visual_tokens` for:
    - positive readouts;
    - cyclic-margin same-image negatives;
    - matrix-CE same-image negatives.
  - For negative D readouts, the original image embeddings are from the row/capture sample, not from the D source sample.
  - Stage1-style eval/readout helpers now also pass the cached row item's `merged_visual_tokens`.
- Metadata:
  - Readout inputs now expose `original_image_embeds_replaced`.
  - Eval cache/readout NLL metadata also records `original_image_embeds_replaced`.
- Verification:
  - `python -m py_compile src/revisit_vlm/tgvf_v3_stage1.py eval/v3_common.py eval/eval_v3_readout.py eval/eval_v3_query_sensitivity.py tests/test_tgvf_v3_stage1.py`
  - `pytest -q tests/test_tgvf_v3_stage1.py::test_v3_stage1_readout_loss_backprops_to_d_not_frozen_qwen`
  - `pytest -q tests/test_tgvf_v3_eval.py::test_compute_v3_eval_item_forwards_protocol_and_focus_im_end tests/test_tgvf_v3_eval.py::test_compute_v3_readout_nll_forwards_protocol_and_focus_im_end`
- Important implication:
  - Existing Stage1 checkpoints before this fix were trained with the old Stage1 readout context.
  - Any new "clean" Stage1 run should be treated as a new setting, even if all CLI hyperparameters are identical.

### AUDIT-20260624-qwen2-stage2-collapse-source

- Trigger:
  - Qwen2-VL-2B no-think Stage2 looked like a hard collapse after TGVF:
    - no-focus/direct eval was reasonable;
    - post-D correct_D generation almost never produced a parseable evidence/answer sequence.
  - User asked whether the issue is in Qwen2 Stage2 training, or a Qwen3-specific path that is not compatible with Qwen2.
- Checkpoints compared:
  - Qwen2 no-think Stage2:
    - `outputs/tgvf_v3_protocol_c_qwen2vl2b/protocol_c_toolobs_qwen2_no_think_stage2_qwen2vl2b_v4data_clean_imend_open_answer_multifocus_focus_imend_from_stage1_4gpu_bs16_accum2_focus80_value1_1200step_maxres512_20260624_120501/checkpoint_step_1200.pt`
  - Qwen2 think Stage2:
    - `outputs/tgvf_v3_protocol_c_qwen2vl2b/protocol_c_toolobs_stage2_qwen2vl2b_v4data_clean_imend_open_answer_multifocus_focus_imend_from_stage1_4gpu_bs16_accum2_focus80_value1_1200step_maxres512_20260624_020233/checkpoint_step_1200.pt`
  - Qwen3 row-only reference:
    - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/checkpoint_step_1200.pt`
- Config finding:
  - Qwen2 and Qwen3 Stage2 macro settings are mostly aligned:
    - `fast_batched_stage2=true`;
    - max image resolution 512;
    - global batch size 128;
    - LoRA target modules q/k/v/o/gate/up/down;
    - `modules_to_save=["embed_tokens","lm_head"]`;
    - Stage2 uses source visual token count for D and frozen Qwen visual merger.
  - Qwen2-specific difference:
    - `Qwen/Qwen2-VL-2B-Instruct` has `tie_word_embeddings=True`;
    - PEFT warns because tied embeddings are in `modules_to_save` while `ensure_weight_tying=False`.
    - This is a risk, but current probes do not make it the primary explanation.
- Existing protocol-eval symptom:
  - Qwen2 no-think Stage2 protocol eval at step 1200:
    - `no_focus_direct`: answer parse 1.0, accuracy 0.546875.
    - `free_router_end2end`: no TGVF trigger, accuracy 0.453125.
    - `teacher_forced_post_tgvf correct_D`: answer parse 0.0078125, accuracy 0.0.
    - `force_end2end correct_D`: answer parse 0.0, accuracy 0.0.
  - Typical D-path generation:
    - after `<|evidence_start|>`, Qwen2 emits junk such as `Thebook.com` or repeated words and often never closes `<|evidence_end|>`.
- Training-style forward probe:
  - Reloaded the Qwen2 checkpoint and called the actual Stage2 validation path:
    - `validate_stage2 -> v3_stage2_batched_training_step`.
  - On 8 focus samples:
    - loss about `1.2119`;
    - `protocol_c_boundary_acc_evidence_start=1.0`;
    - `protocol_c_boundary_acc_evidence_end=1.0`.
  - Mask ablation inside the training-style path did not change this:
    - old evidence-only mask, no mask, and through-answer mask all kept evidence_end boundary acc at 1.0.
- D-path alignment probe:
  - Compared training fast-path H_q/D with protocol-eval capture H_q/D on the same 8 samples.
  - Results:
    - mean H_q cosine overlap about `0.9546`;
    - mean D cosine overlap about `0.9835`;
    - D norms are similar between paths.
  - Interpretation:
    - protocol eval is not producing a wildly different D tensor;
    - the collapse is downstream of D append, during continuation.
- KV-continuation probe:
  - `scripts/probe_qwen2_post_d.py` measures teacher-forced gold continuation NLL after append-D using the eval KV continuation path.
  - Before fixes, correct_D had:
    - mean NLL about `4.3989`;
    - mean evidence_end NLL about `11.5540`;
    - median evidence_end rank about `6595.5`.
  - Full-sequence prefill with KV continuation did not fix it:
    - evidence_end rank still thousands.
- Partial fix applied:
  - `Qwen3AppendResult.model_kwargs` now carries `tgvf_next_position_ids`, derived from the append prefill `position_ids[:, :, -1:] + 1`.
  - `continue_generation_qwen3` uses this stored 3D text position for post-D decoding instead of asking the model to infer positions from plain 1D length.
  - The force/chat shared append path in `eval/eval_v3_vstar_force.py` and protocol-eval append path in `eval/eval_v3_stage2_protocol.py` now populate the same field.
  - Direct post-D decode also passes explicit `cache_position`.
  - `scripts/probe_qwen2_post_d.py` was updated to use the same next-position/cache-position logic for teacher-forced probing.
- Partial-fix result:
  - Qwen2 KV probe improved early-token/mean NLL but did not restore evidence_end:
    - correct_D mean NLL improved to about `2.82` on 8 samples;
    - evidence_end median rank was still thousands.
  - With explicit cache_position on a 4-sample quick probe:
    - correct_D mean NLL about `2.38`;
    - evidence_end median rank still about `1884`.
- No-KV full-sequence confirmation:
  - Added `--continuation-forward-mode no_kv_full_sequence` to `scripts/probe_qwen2_post_d.py`.
  - This probe does not reuse post-D `past_key_values`.
  - For each gold continuation token it rebuilds the full sequence, scatters original `V_merge` and D, recomputes full Qwen VL M-RoPE position ids, and forwards with `use_cache=False`.
  - Qwen2 no-think 4-sample result:
    - correct_D mean NLL about `0.9728`;
    - correct_D mean evidence_end NLL about `0.0021`;
    - correct_D median evidence_end rank `1.0`.
  - This confirms the Qwen2 collapse is caused by the post-D KV continuation path, not by Stage2 full-sequence training being unable to model the evidence/answer sequence.
- Current interpretation:
  - Qwen2 Stage2 training itself is not the main collapse source.
  - The core mismatch is that training learns/validates final readout in a full-sequence teacher-forced forward, while evaluation/chat continue from an appended-D KV cache.
  - Qwen2 is much more sensitive to this KV-continuation mismatch than Qwen3.
  - We should not trust Qwen2 post-D KV generation numbers as a method verdict until a no-KV full-sequence continuation path is implemented and benchmarked.
- Next required fix:
  - Add a no-KV full-sequence continuation mode for post-D generation:
    - rebuild full `inputs_embeds` with original `V_merge` and appended D;
    - recompute full Qwen VL position ids for the entire sequence after every generated token or in small chunks;
    - use this as the correctness path for Qwen2 validation, even if slower.
  - Only after that should Qwen2 Stage2 be judged.
- Verification commands run:
  - `python -m py_compile src/revisit_vlm/qwen3_vl_tgvf.py eval/eval_v3_stage2_protocol.py eval/eval_v3_vstar_force.py scripts/probe_qwen2_post_d.py scripts/chat_tgvf_v3.py`
  - `CUDA_VISIBLE_DEVICES=0 python scripts/probe_qwen2_post_d.py --output-dir outputs/debug_qwen2_post_d_probe_20260624 --max-focus 8 --device cuda:0 --device-map cuda:0 --dtype bfloat16 --attn-implementation sdpa`
  - `CUDA_VISIBLE_DEVICES=0 python scripts/probe_qwen2_post_d.py --output-dir outputs/debug_qwen2_post_d_probe_fullseq_20260624 --max-focus 8 --device cuda:0 --device-map cuda:0 --dtype bfloat16 --attn-implementation sdpa --append-prefill-mode full_sequence`
  - `CUDA_VISIBLE_DEVICES=0 python scripts/probe_qwen2_post_d.py --output-dir outputs/debug_qwen2_post_d_probe_after_posfix_20260624 --max-focus 8 --device cuda:0 --device-map cuda:0 --dtype bfloat16 --attn-implementation sdpa`
  - `CUDA_VISIBLE_DEVICES=0 python scripts/probe_qwen2_post_d.py --output-dir outputs/debug_qwen2_post_d_probe_after_cachepos_20260624 --max-focus 4 --device cuda:0 --device-map cuda:0 --dtype bfloat16 --attn-implementation sdpa`
  - `CUDA_VISIBLE_DEVICES=0 python scripts/probe_qwen2_post_d.py --output-dir outputs/debug_qwen2_post_d_probe_nokv_fullseq_4_20260625 --max-focus 4 --device cuda:0 --device-map cuda:0 --dtype bfloat16 --attn-implementation sdpa --append-prefill-mode full_sequence --continuation-forward-mode no_kv_full_sequence`

### AUDIT-20260625-qwen3-kv-vs-nokv-continuation

- Question:
  - Check whether previous Qwen3 benchmark validation also used the problematic post-D KV continuation path.
  - Check whether Qwen3 shows the same KV/full-sequence mismatch as Qwen2.
- Code/history finding:
  - The repository only has two committed snapshots, so fine-grained date history is unavailable.
  - In committed snapshot `eccffc9`, `eval/eval_v3_stage2_protocol.py` already appends post-D tokens using `past_key_values=capture.past_key_values` and then calls `continue_generation_qwen3`.
  - `append_prefill_mode` is not present in committed history and is a later working-tree option.
  - Old protocol eval configs have `append_prefill_mode=None`, which maps to the default KV append behavior.
  - `eval/eval_v3_vstar_force.py` and the benchmark/chat force path also use `append_answer_only(...)`, which appends D with `past_key_values=capture.past_key_values`; previous external benchmark results therefore used KV continuation unless a separate second/full-forward path was explicitly enabled.
- Qwen3 reference checkpoint:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/checkpoint_step_1200.pt`
- Qwen3 previous protocol eval result:
  - `teacher_forced_post_tgvf correct_D` in the saved protocol eval had answer parse about `0.9844` and accuracy about `0.9297`.
  - This already suggested Qwen3 did not suffer the same severe post-D KV collapse as Qwen2.
- Probe update:
  - `scripts/probe_qwen2_post_d.py` now strips the already-prefilled `<think>\n` prefix for think protocols and uses `</think>` as the close-marker token for Qwen3, while still using `<|evidence_end|>` for Qwen2 evidence-tag protocols.
- Qwen3 2-sample KV probe:
  - Command:
    - `CUDA_VISIBLE_DEVICES=0 python scripts/probe_qwen2_post_d.py --stage2-checkpoint outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --tgvf-protocol protocol_c_tool_observation --output-dir outputs/debug_qwen3_post_d_probe_kv_2_20260625 --max-focus 2 --device cuda:0 --device-map cuda:0 --dtype bfloat16 --attn-implementation sdpa`
  - correct_D:
    - mean NLL about `1.0578`;
    - close-marker (`</think>`) NLL about `0.0000017`;
    - close-marker rank `1.0`.
- Qwen3 2-sample no-KV full-sequence probe:
  - Command:
    - `CUDA_VISIBLE_DEVICES=0 python scripts/probe_qwen2_post_d.py --stage2-checkpoint outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --tgvf-protocol protocol_c_tool_observation --output-dir outputs/debug_qwen3_post_d_probe_nokv_2_20260625 --max-focus 2 --device cuda:0 --device-map cuda:0 --dtype bfloat16 --attn-implementation sdpa --append-prefill-mode full_sequence --continuation-forward-mode no_kv_full_sequence`
  - correct_D:
    - mean NLL about `1.0233`;
    - close-marker (`</think>`) NLL about `0.0000013`;
    - close-marker rank `1.0`.
- Interpretation:
  - Previous Qwen3 protocol/external benchmark validation did use KV continuation.
  - Qwen3 does not show the severe Qwen2 KV collapse in this small close-marker/gold-continuation probe.
  - The Qwen3 benchmark numbers are still from a path that is not strictly identical to training full-sequence forward, so they should be labeled as KV-continuation evals.
  - For rigorous comparison across Qwen2/Qwen3 and future reports, add a no-KV full-sequence benchmark mode and report it separately from KV mode.

### AUDIT-20260625-formal-benchmark-default-nokv

- Status: DONE.
- Question:
  - Move the Qwen2/Qwen3 post-D benchmark path from the old KV continuation to a training-equivalent no-KV full-sequence continuation by default.
- Motivation:
  - Qwen2 Stage2 validation/probe showed that full-sequence forward predicts the post-D evidence close marker correctly, while KV continuation collapses.
  - Treating the old KV path as the benchmark default can understate the method, especially for Qwen2.
- Code change:
  - `eval/eval_v3_vstar_force.py` now has `--post-tgvf-forward-mode {no_kv_full_sequence,kv_cache}`, default `no_kv_full_sequence`.
  - `continue_after_append(...)` now dispatches to `continue_generation_qwen3_no_kv_full_sequence(...)` by default.
  - The no-KV path rebuilds the complete sequence every generation step, scatters original image `V_merge` tokens and appended D into `inputs_embeds`, recomputes full Qwen VL position ids, and forwards with `use_cache=False`.
  - `eval/eval_v3_mmmu_force.py` now uses `continue_after_append(...)` instead of directly calling `continue_generation_qwen3(...)`.
  - Official benchmark wrapper scripts now explicitly pass/log `POST_TGVF_FORWARD_MODE`, defaulting to `no_kv_full_sequence`.
- Verification:
  - `python -m py_compile eval/eval_v3_mmmu_force.py eval/eval_v3_vstar_force.py scripts/probe_qwen2_post_d.py src/revisit_vlm/qwen3_vl_tgvf.py`
  - `git diff --check`
  - Qwen2 formal BLINK smoke, limit 1:
    - checkpoint: `outputs/tgvf_v3_protocol_c_qwen2vl2b/protocol_c_toolobs_qwen2_no_think_stage2_qwen2vl2b_v4data_clean_imend_open_answer_multifocus_focus_imend_from_stage1_4gpu_bs16_accum2_focus80_value1_1200step_maxres512_20260624_120501/checkpoint_step_1200.pt`
    - `post_tgvf_forward_mode=no_kv_full_sequence`;
    - `second_full_forward_used_any=true`;
    - append success 1.0, answer parse 1.0 on 1 sample.
- Policy:
  - Future post-D benchmark and chat correctness checks should default to no-KV full-sequence unless explicitly studying KV-cache behavior.
  - Any result using `kv_cache` must be labeled as a KV-continuation result, not silently compared against no-KV/full-sequence results.
- Current benchmark:
  - Model: `Qwen/Qwen2-VL-2B-Instruct`.
  - TGVF protocol: `protocol_c_tool_observation_qwen2_no_think`.
  - Checkpoint: `outputs/tgvf_v3_protocol_c_qwen2vl2b/protocol_c_toolobs_qwen2_no_think_stage2_qwen2vl2b_v4data_clean_imend_open_answer_multifocus_focus_imend_from_stage1_4gpu_bs16_accum2_focus80_value1_1200step_maxres512_20260624_120501/checkpoint_step_1200.pt`.
  - Benchmark: BLINK full, max image resolution 512.
  - Modes: force correct_D, free correct_D.
  - GPUs: 0,1,2,3.
  - tmux: `qwen2_blink_nokv_0_3_20260625`.
  - Output root: `outputs/no_kv_benchmarks/qwen2_blink_full_512_20260625_004823`.
  - Started: 2026-06-25T00:48:23+09:00.
  - Finished: 2026-06-25T01:23:00+09:00.
  - Metrics:
    - force correct_D:
      - `n_total_rows=1901`;
      - `force_correct_D n=1528`, accuracy `19.11`, answer parse `46.73`, focus valid `100.00`, append success `100.00`;
      - `force_focus_failed n=373`, accuracy `0.00`.
      - `second_full_forward_used_any=true`.
    - free correct_D:
      - `free_direct_or_miss n=1901`, accuracy `21.67`, answer parse `55.08`, trigger rate `0.00`;
      - no post-D continuation was used because no free sample triggered focus;
      - `second_full_forward_used_any=false`.
  - Table output: `outputs/no_kv_benchmarks/qwen2_blink_full_512_20260625_004823/table.json`.
  - Analysis:
    - Formal no-KV post-D continuation works mechanically on the benchmark path.
    - Qwen2 BLINK force does not recover: the model often fails focus generation (`373/1901`) and still has low parse/accuracy even when forced focus is captured.
    - Qwen2 free never triggers TGVF on BLINK full under this no-think protocol, so its result is effectively direct answering, not a TGVF result.
    - Therefore the Qwen2 BLINK failure is not explained solely by the post-D KV mismatch found in probes; focus/action quality and final answer formatting remain major failures.
  - Follow-up:
    - Run matching Qwen3 BLINK full no-KV after Qwen2 completes.
    - Produce a table covering Qwen2 and Qwen3 with mode, accuracy, parse rate, trigger/focus fields, and output paths.

### EXP-20260625-012725-qwen3-blink-full-nokv

- Status: STOPPED.
- Question:
  - Measure Qwen3 BLINK full with the new default no-KV full-sequence post-D continuation, matching the Qwen2 formal benchmark path where possible.
- Baseline anchor:
  - `BASE-20260617-rowonly-original-stage2`.
- Intended diff:
  - Evaluation only: use `post_tgvf_forward_mode=no_kv_full_sequence` instead of old post-D KV continuation.
- Allowed changed variables:
  - Post-D evaluation forward mode.
  - Qwen3 protocol keeps its thinking/tool-observation format.
- Not allowed to change:
  - Stage2 checkpoint, BLINK full split, max image resolution 512.
- Stage2 checkpoint:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/checkpoint_step_1200.pt`
- Stage2 processor:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/processor_step_1200`
- Benchmark output:
  - `outputs/no_kv_benchmarks/qwen3_blink_full_512_20260625_012725`
- Script / command:
  - tmux inline runner using `eval/eval_v3_mmmu_force.py`.
  - `--benchmark blink --tier full --max-image-resolution 512 --eval-mode force/free --d-conditions correct_D --post-tgvf-continuation evidence_then_answer --post-tgvf-forward-mode no_kv_full_sequence`.
- GPUs:
  - 0,1,2,3.
- tmux:
  - `qwen3_blink_nokv_0_3_20260625`.
- Started:
  - 2026-06-25T01:27:25+09:00.
- Metrics:
  - No merged benchmark metrics.
  - Stopped before completing force mode; four shards had reached about `25/475` samples.
- Notes:
  - Model loaded successfully on 4 shards, about 23GB per GPU.
  - Qwen3 uses `max_action_tokens=128`, `max_answer_tokens=256`, unlike the Qwen2 no-think run which used shorter answer continuation.
  - User requested stop before completion.

### EXP-20260625-132715-qwen3-blink120-nokv-oldmask075

- Status: INVALID_SIDE_RESULT.
- Question:
  - Re-test Qwen3 BLINK under the historical full-benchmark table setting (`n=120`, not current 1901 full BLINK) with no-KV full-sequence post-D continuation.
- Baseline anchor:
  - Compare primarily against `EXP-20260623-005426-oldmask075-correct-stage1` BLINK rows:
    - free: n=120, accuracy 59.17, parse 98.33, trigger 7.50.
    - softforce: n=120, accuracy 65.00, parse 96.67, trigger 51.67.
- Intended diff:
  - Eval only: `post_tgvf_forward_mode=no_kv_full_sequence` instead of the old KV continuation path.
- Allowed changed variables:
  - Post-D evaluation forward mode.
- Not allowed to change:
  - Stage2 checkpoint/processor.
  - BLINK sample count: explicit `--limit 120`.
  - Max image resolution 512.
  - Continuation surface: `natural_continue`.
  - free and softforce modes.
- Stage2 checkpoint:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_oldmaskprob075_throughanswer_from_20260619_stage1_train47_eval03_4gpu_bs16_accum2_focus80_value1_1200step_20260623_005426/checkpoint_step_1200.pt`
- Stage2 processor:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_oldmaskprob075_throughanswer_from_20260619_stage1_train47_eval03_4gpu_bs16_accum2_focus80_value1_1200step_20260623_005426/processor_step_1200`
- Benchmark output:
  - `outputs/no_kv_benchmarks/qwen3_blink120_oldmask075_correct_stage1_nokv_20260625_132715`
- Script / command:
  - `STAMP=20260625_132715 bash scripts/run_qwen3_blink120_nokv_4_7.sh`
  - Fixed args include `--benchmark blink --tier full --limit 120 --max-image-resolution 512 --eval-mode free --post-tgvf-continuation natural_continue --post-tgvf-forward-mode no_kv_full_sequence`.
  - Modes: `free`, `free_softforce_focus` via `--question-suffix "use focus tool"`.
- GPUs:
  - 4,5,6,7.
- tmux:
  - `qwen3_blink120_nokv_4_7_20260625_132715`.
- Started:
  - 2026-06-25T13:27:15+09:00.
- Metrics:
  - Free mode partial merged result: n=120, accuracy 43.33, parse 97.50, trigger 3.33.
- Invalid reason:
  - This did not evaluate the historical BLINK Counting-120 population.
  - Row audit showed old IDs like `val-00000-of-00001-0`, but this run produced
    IDs like `Art_Style/val-00000-of-00001-11`.
  - Therefore it changed sample identity, not only post-D forward mode.
  - Do not use this run for the KV vs no-KV judgment.

### EXP-20260625-173956-qwen3-blink-counting120-nokv-standard

- Status: DONE.
- Question:
  - Judge Qwen3 BLINK Counting-120 under the exact historical standard, changing
    only post-D forward mode from KV continuation to no-KV full-sequence.
- Baseline anchor:
  - `EXP-20260623-005426-oldmask075-correct-stage1` BLINK rows:
    - free: n=120, accuracy 59.17, parse 98.33, trigger 7.50.
    - softforce: n=120, accuracy 65.00, parse 96.67, trigger 51.67.
- Intended diff:
  - Eval only: `post_tgvf_forward_mode=no_kv_full_sequence` instead of the old
    KV continuation path.
- Allowed changed variables:
  - Post-D evaluation forward mode only.
- Not allowed to change:
  - Stage2 checkpoint/processor.
  - Benchmark population: BLINK Counting validation parquet, all 120 rows.
  - Max image resolution 512.
  - Continuation surface: `natural_continue`.
  - Eval modes: `free` and `free_softforce_focus`.
  - Scoring: BLINK official exact-match scorer, no LLM judge.
- Stage2 checkpoint:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_oldmaskprob075_throughanswer_from_20260619_stage1_train47_eval03_4gpu_bs16_accum2_focus80_value1_1200step_20260623_005426/checkpoint_step_1200.pt`
- Stage2 processor:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_oldmaskprob075_throughanswer_from_20260619_stage1_train47_eval03_4gpu_bs16_accum2_focus80_value1_1200step_20260623_005426/processor_step_1200`
- Benchmark source:
  - `/nvmesv/dredvpn009/datasets/benchmarks/blink/snapshot/Counting/val-00000-of-00001.parquet`
  - official scorer symlinked from `/nvmesv/dredvpn009/datasets/benchmarks/blink/official_code`
- Benchmark output:
  - `outputs/no_kv_benchmarks/qwen3_blink_counting120_standard_nokv_20260625_173956`
- Script / command:
  - `STAMP=20260625_173956 RUN_ROOT=outputs/no_kv_benchmarks/qwen3_blink_counting120_standard_nokv_20260625_173956 bash scripts/run_qwen3_blink120_nokv_4_7.sh`
  - Fixed args include `--benchmark blink --tier full --max-image-resolution 512 --eval-mode free --post-tgvf-continuation natural_continue --post-tgvf-forward-mode no_kv_full_sequence`.
  - No `--limit`; sample count is 120 because the temporary benchmark root exposes only the Counting validation parquet.
  - Modes: `free`, `free_softforce_focus` via `--question-suffix "use focus tool"`.
- GPUs:
  - 4,5,6,7.
- tmux:
  - `qwen3_blink_counting120_nokv_4_7_20260625_173956`
- Started:
  - 2026-06-25T17:41:12+09:00.
- Finished:
  - 2026-06-25T17:49:42+09:00.
- Elapsed:
  - 8m30s.
- Metrics:
  - free:
    - n: 120.
    - accuracy: 60.00.
    - answer parse rate: 100.00.
    - trigger rate: 7.50.
  - free_softforce_focus:
    - n: 120.
    - accuracy: 70.00.
    - answer parse rate: 100.00.
    - trigger rate: 51.67.
- Validation:
  - Old KV baseline and this no-KV run have identical sample order after stripping
    the new adapter prefix `Counting/`.
  - All four shards in both modes used `official_blink_exact_match`.
  - All shard summaries report `limit=None` and the temporary
    `benchmark_root_counting_only`.
  - Merged summaries report `second_full_forward_used_any=True`, confirming the
    no-KV full-sequence path was used.
- Result files:
  - `outputs/no_kv_benchmarks/qwen3_blink_counting120_standard_nokv_20260625_173956/summary_blink120_nokv.json`
  - `outputs/no_kv_benchmarks/qwen3_blink_counting120_standard_nokv_20260625_173956/blink/free/merged_summary.json`
  - `outputs/no_kv_benchmarks/qwen3_blink_counting120_standard_nokv_20260625_173956/blink/free_softforce_focus/merged_summary.json`
- Conclusion:
  - Under the controlled BLINK Counting-120 setting for the 20260623 oldmask075
    correct-stage1 checkpoint, changing only post-D continuation from KV to
    no-KV full-sequence does not hurt Qwen3.
  - no-KV is slightly higher than the 20260623 KV baseline in both modes:
    free 59.17 -> 60.00, softforce 65.00 -> 70.00.
  - This is not the no-KV counterpart of the 20260620 cross-benchmark reference
    table where BLINK was free 65.83 and softforce 62.50. That table used the
    20260619 open-answer checkpoint:
    `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt`.

### EXP-20260625-175818-qwen3-blink-counting120-nokv-20260619-open-answer

- Status: DONE.
- Question:
  - Re-run the 20260620 BLINK Counting-120 open-answer reference checkpoint with
    no-KV full-sequence post-D continuation.
- Baseline anchor:
  - `EXP-20260620-original-free-softforce-full-summary` BLINK row:
    - free: n=120, accuracy 65.83, trigger 5.83.
    - softforce: n=120, accuracy 62.50, trigger 45.83.
- Intended diff:
  - Eval only: `post_tgvf_forward_mode=no_kv_full_sequence` instead of the old
    KV continuation path.
- Allowed changed variables:
  - Post-D evaluation forward mode only.
- Not allowed to change:
  - Stage2 checkpoint/processor.
  - Benchmark population: BLINK Counting validation parquet, all 120 rows.
  - Max image resolution 512.
  - Continuation surface: `natural_continue`.
  - Eval modes: `free` and `free_softforce_focus`.
  - Scoring: BLINK official exact-match scorer, no LLM judge.
- Stage2 checkpoint:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt`
- Stage2 processor:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200`
- Benchmark source:
  - `/nvmesv/dredvpn009/datasets/benchmarks/blink/snapshot/Counting/val-00000-of-00001.parquet`
  - official scorer symlinked from `/nvmesv/dredvpn009/datasets/benchmarks/blink/official_code`
- Benchmark output:
  - `outputs/no_kv_benchmarks/qwen3_blink_counting120_20260619_open_answer_nokv_20260625_175818`
- Script / command:
  - `STAMP=20260625_175818 RUN_ROOT=outputs/no_kv_benchmarks/qwen3_blink_counting120_20260619_open_answer_nokv_20260625_175818 STAGE2_CKPT=outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt STAGE2_PROCESSOR=outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200 bash scripts/run_qwen3_blink120_nokv_4_7.sh`
  - No `--limit`; sample count is 120 because the temporary benchmark root exposes only the Counting validation parquet.
  - Modes: `free`, `free_softforce_focus` via `--question-suffix "use focus tool"`.
- GPUs:
  - 4,5,6,7.
- tmux:
  - `qwen3_blink_counting120_20260619_nokv_4_7_20260625_175818`
- Started:
  - 2026-06-25T17:59:30+09:00.
- Finished:
  - 2026-06-25T18:07:41+09:00.
- Elapsed:
  - 8m11s.
- Metrics:
  - free:
    - n: 120.
    - accuracy: 65.83.
    - answer parse rate: 100.00.
    - trigger rate: 5.83.
  - free_softforce_focus:
    - n: 120.
    - accuracy: 64.17.
    - answer parse rate: 100.00.
    - trigger rate: 45.83.
- Validation:
  - Old KV baseline and this no-KV run have identical sample order after stripping
    the new adapter prefix `Counting/`.
  - All four shards in both modes used `official_blink_exact_match`.
  - All shard summaries report `limit=None`.
  - Merged summaries report `second_full_forward_used_any=True`, confirming the
    no-KV full-sequence path was used for triggered post-D continuation.
- Result files:
  - `outputs/no_kv_benchmarks/qwen3_blink_counting120_20260619_open_answer_nokv_20260625_175818/summary_blink120_nokv.json`
  - `outputs/no_kv_benchmarks/qwen3_blink_counting120_20260619_open_answer_nokv_20260625_175818/blink/free/merged_summary.json`
  - `outputs/no_kv_benchmarks/qwen3_blink_counting120_20260619_open_answer_nokv_20260625_175818/blink/free_softforce_focus/merged_summary.json`
- Conclusion:
  - On the true 20260620 BLINK Counting-120 reference checkpoint, no-KV leaves
    low-trigger free unchanged: 65.83 -> 65.83.
  - no-KV improves high-trigger softforce modestly: 62.50 -> 64.17.
  - Together with the 20260623 oldmask075 result, the pattern is consistent:
    higher trigger rate exposes more post-D continuation sensitivity, but the
    gain size is checkpoint-dependent.

### EXP-20260625-184230-qwen3-blink-counting120-nokv-maskprob50-evidenceonly

- Status: DONE.
- Question:
  - Re-test the `mask_original_image_after_tgvf_prob=0.5` evidence-only Stage2
    checkpoint under BLINK Counting-120 no-KV full-sequence post-D continuation.
- Baseline anchor:
  - Same checkpoint under old KV BLINK Counting-120 was not found locally as a
    merged BLINK output. Treat this first as a standalone no-KV measurement.
  - This checkpoint is not a clean ablation of the 20260619 open-answer baseline
    because it uses the 20260617 Stage1 checkpoint.
- Intended diff:
  - Evaluation setting: `post_tgvf_forward_mode=no_kv_full_sequence`.
- Stage2 checkpoint:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_maskprob50_evidenceonly_from_clean_imend_stage1_4gpu_bs16_accum2_focus80_value1_1200step_20260622_021650/checkpoint_step_1200.pt`
- Stage2 processor:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_maskprob50_evidenceonly_from_clean_imend_stage1_4gpu_bs16_accum2_focus80_value1_1200step_20260622_021650/processor_step_1200`
- Stage1 lineage:
  - `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617/train/checkpoint_step_2000.pt`
- Stage2 training highlights:
  - `mask_original_image_after_tgvf=true`
  - `mask_original_image_after_tgvf_prob=0.5`
  - evidence-only scope per experiment ledger
  - open-answer Stage2 data
  - 4GPU, batch size 16, grad accum 2, global batch 128, 1200 steps
- Benchmark source:
  - BLINK Counting validation parquet, all 120 rows.
  - official BLINK exact-match scorer, no LLM judge.
- Benchmark output:
  - `outputs/no_kv_benchmarks/qwen3_blink_counting120_maskprob50_evidenceonly_nokv_20260625_184230`
- Script / command:
  - `STAMP=20260625_184230 RUN_ROOT=outputs/no_kv_benchmarks/qwen3_blink_counting120_maskprob50_evidenceonly_nokv_20260625_184230 STAGE2_CKPT=outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_maskprob50_evidenceonly_from_clean_imend_stage1_4gpu_bs16_accum2_focus80_value1_1200step_20260622_021650/checkpoint_step_1200.pt STAGE2_PROCESSOR=outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_maskprob50_evidenceonly_from_clean_imend_stage1_4gpu_bs16_accum2_focus80_value1_1200step_20260622_021650/processor_step_1200 bash scripts/run_qwen3_blink120_nokv_4_7.sh`
  - Modes: `free`, `free_softforce_focus`.
- GPUs:
  - 4,5,6,7.
- tmux:
  - `qwen3_blink_counting120_maskprob50_nokv_4_7_20260625_184230`
- Started:
  - 2026-06-25T18:44:47+09:00.
- Finished:
  - 2026-06-25T18:53:13+09:00.
- Elapsed:
  - 8m26s.
- Metrics:
  - free:
    - n: 120.
    - accuracy: 63.33.
    - answer parse rate: 100.00.
    - trigger rate: 10.00.
  - free_softforce_focus:
    - n: 120.
    - accuracy: 62.50.
    - answer parse rate: 100.00.
    - trigger rate: 42.50.
- Validation:
  - Evaluated 120 unique BLINK Counting validation samples.
  - All four shards in both modes used `official_blink_exact_match`.
  - All shard summaries report `limit=None`.
  - Merged summaries report `second_full_forward_used_any=True`.
- Result files:
  - `outputs/no_kv_benchmarks/qwen3_blink_counting120_maskprob50_evidenceonly_nokv_20260625_184230/summary_blink120_nokv.json`
  - `outputs/no_kv_benchmarks/qwen3_blink_counting120_maskprob50_evidenceonly_nokv_20260625_184230/blink/free/merged_summary.json`
  - `outputs/no_kv_benchmarks/qwen3_blink_counting120_maskprob50_evidenceonly_nokv_20260625_184230/blink/free_softforce_focus/merged_summary.json`
- Conclusion:
  - The prob=0.5 evidence-only checkpoint does not show the same high-trigger
    BLINK lift as the 20260623 prob=0.75 through-answer checkpoint.
  - Under no-KV BLINK Counting-120 it reaches free 63.33 and softforce 62.50.
    Compared with the prob=0.75 through-answer no-KV result, free is higher
    (63.33 vs 60.00) but the high-trigger softforce mode is much lower
    (62.50 vs 70.00).
  - This supports separating two effects: no-KV fixes post-D continuation, but
    the Stage2 mask/training setting still determines whether triggered TGVF is
    actually beneficial.

### EXP-20260625-185857-qwen3-blink-counting120-kv-maskprob50-evidenceonly

- Status: DONE.
- Question:
  - Run the same prob=0.5 evidence-only Stage2 checkpoint on BLINK Counting-120
    with the old KV continuation path, to compare directly against
    `EXP-20260625-184230-qwen3-blink-counting120-nokv-maskprob50-evidenceonly`.
- Intended diff from the no-KV run:
  - `post_tgvf_forward_mode=kv_cache` instead of `no_kv_full_sequence`.
- Held fixed:
  - Stage2 checkpoint/processor.
  - BLINK Counting validation population, all 120 rows.
  - `natural_continue`.
  - free and free_softforce_focus modes.
  - official BLINK exact-match scorer.
  - max image resolution 512.
- Stage2 checkpoint:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_maskprob50_evidenceonly_from_clean_imend_stage1_4gpu_bs16_accum2_focus80_value1_1200step_20260622_021650/checkpoint_step_1200.pt`
- Stage2 processor:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_maskprob50_evidenceonly_from_clean_imend_stage1_4gpu_bs16_accum2_focus80_value1_1200step_20260622_021650/processor_step_1200`
- Benchmark output:
  - `outputs/kv_benchmarks/qwen3_blink_counting120_maskprob50_evidenceonly_kv_20260625_185857`
- Script / command:
  - `STAMP=20260625_185857 RUN_ROOT=outputs/kv_benchmarks/qwen3_blink_counting120_maskprob50_evidenceonly_kv_20260625_185857 POST_TGVF_FORWARD_MODE=kv_cache STAGE2_CKPT=outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_maskprob50_evidenceonly_from_clean_imend_stage1_4gpu_bs16_accum2_focus80_value1_1200step_20260622_021650/checkpoint_step_1200.pt STAGE2_PROCESSOR=outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_maskprob50_evidenceonly_from_clean_imend_stage1_4gpu_bs16_accum2_focus80_value1_1200step_20260622_021650/processor_step_1200 bash scripts/run_qwen3_blink120_nokv_4_7.sh`
- GPUs:
  - 4,5,6,7.
- tmux:
  - `qwen3_blink_counting120_maskprob50_kv_4_7_20260625_185857`
- Started:
  - 2026-06-25T18:59:50+09:00.
- Finished:
  - 2026-06-25T19:06:03+09:00.
- Elapsed:
  - 6m13s.
- Metrics:
  - free:
    - n: 120.
    - accuracy: 62.50.
    - answer parse rate: 100.00.
    - trigger rate: 10.00.
  - free_softforce_focus:
    - n: 120.
    - accuracy: 63.33.
    - answer parse rate: 100.00.
    - trigger rate: 42.50.
- Validation:
  - Same sample order as the no-KV prob=0.5 run.
  - All shards used `official_blink_exact_match`.
  - Merged summaries report `second_full_forward_used_any=False`, confirming the
    KV continuation path.
- Result files:
  - `outputs/kv_benchmarks/qwen3_blink_counting120_maskprob50_evidenceonly_kv_20260625_185857/summary_blink120_kv_cache.json`
  - `outputs/kv_benchmarks/qwen3_blink_counting120_maskprob50_evidenceonly_kv_20260625_185857/blink/free/merged_summary.json`
  - `outputs/kv_benchmarks/qwen3_blink_counting120_maskprob50_evidenceonly_kv_20260625_185857/blink/free_softforce_focus/merged_summary.json`
- Comparison to no-KV:
  - free: KV 62.50 -> no-KV 63.33, +0.83.
  - softforce: KV 63.33 -> no-KV 62.50, -0.83.
- Conclusion:
  - For this prob=0.5 evidence-only checkpoint, no-KV does not create the
    high-trigger improvement observed in the prob=0.75 through-answer checkpoint.
  - The post-D continuation fix is not sufficient by itself; the Stage2 mask /
    training setting still determines whether triggered TGVF helps.

### EXP-20260625-191315-qwen3-other-benchmarks-nokv-19-vs-075

- Status: RUNNING.
- Question:
  - Compare the 20260619 open-answer checkpoint and the 20260623 prob=0.75
    through-answer checkpoint on non-BLINK benchmarks under the no-KV
    full-sequence post-D continuation standard.
- Motivation:
  - BLINK Counting-120 showed that no-KV especially matters when TGVF trigger
    rate is high. Need check whether the same pattern appears on VStar, HR, and
    OCR.
- Checkpoints:
  - ckpt19:
    `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt`
  - ckpt075:
    `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_oldmaskprob075_throughanswer_from_20260619_stage1_train47_eval03_4gpu_bs16_accum2_focus80_value1_1200step_20260623_005426/checkpoint_step_1200.pt`
- Benchmark population:
  - `vstar_bench`, tier full.
  - `hr_bench_4k`, tier full.
  - `ocrbench_v2`, tier full under current adapter.
- Modes:
  - `free`.
  - `free_softforce_focus` with question suffix `use focus tool`.
- Held fixed:
  - `post_tgvf_continuation=natural_continue`.
  - `post_tgvf_forward_mode=no_kv_full_sequence`.
  - `tgvf_protocol=protocol_c_tool_observation`.
  - max image resolution 512.
  - official/project scoring backend `auto`, no LLM judge.
- Script / command:
  - `STAMP=20260625_191315 RUN_ROOT=outputs/no_kv_benchmarks/qwen3_other_benchmarks_19_vs_075_nokv_20260625_191315 BENCH_GPUS=4,5,6,7 bash scripts/run_qwen3_two_ckpts_other_benchmarks_nokv_4gpu.sh`
- Output:
  - `outputs/no_kv_benchmarks/qwen3_other_benchmarks_19_vs_075_nokv_20260625_191315`
- GPUs:
  - 4,5,6,7.
- tmux:
  - `qwen3_other_benchmarks_19_vs_075_nokv_4_7_20260625_191315`
- Started:
  - 2026-06-25T19:13:15+09:00.
- Stopped:
  - 2026-06-25T19:48+09:00.
- Stop reason:
  - `ckpt19/hr_bench_4k/free_softforce_focus` entered an abnormal long-running
    state: all four shards stayed at logged `{"index": 1, "total": 200}` for
    about 10 minutes, while GPU processes remained active and no merged summary
    was produced. The tmux session was killed to avoid wasting GPUs and to avoid
    treating the incomplete run as a full benchmark result.
- Metrics:
  - VStar, ckpt19, free:
    - n: 191.
    - accuracy: 49.21.
    - trigger rate: 0.00.
  - VStar, ckpt19, free_softforce_focus:
    - n: 191.
    - accuracy: 46.07.
    - trigger rate: 40.84.
  - VStar, ckpt075, free:
    - n: 191.
    - accuracy: 51.83.
    - trigger rate: 1.57.
  - VStar, ckpt075, free_softforce_focus:
    - n: 191.
    - accuracy: 49.74.
    - trigger rate: 63.87.
  - HR-Bench-4K, ckpt19, free:
    - n: 800.
    - accuracy: 53.25.
    - trigger rate: 17.00.
- Missing metrics:
  - HR-Bench-4K, ckpt19, free_softforce_focus.
  - HR-Bench-4K, ckpt075, free/free_softforce_focus.
  - OCRBench-v2, both checkpoints and both modes.
- Result files:
  - `outputs/no_kv_benchmarks/qwen3_other_benchmarks_19_vs_075_nokv_20260625_191315/ckpt19/vstar_bench/free/merged_summary.json`
  - `outputs/no_kv_benchmarks/qwen3_other_benchmarks_19_vs_075_nokv_20260625_191315/ckpt19/vstar_bench/free_softforce_focus/merged_summary.json`
  - `outputs/no_kv_benchmarks/qwen3_other_benchmarks_19_vs_075_nokv_20260625_191315/ckpt075/vstar_bench/free/merged_summary.json`
  - `outputs/no_kv_benchmarks/qwen3_other_benchmarks_19_vs_075_nokv_20260625_191315/ckpt075/vstar_bench/free_softforce_focus/merged_summary.json`
  - `outputs/no_kv_benchmarks/qwen3_other_benchmarks_19_vs_075_nokv_20260625_191315/ckpt19/hr_bench_4k/free/merged_summary.json`
- Interim conclusion:
  - On VStar, the 20260623 prob=0.75 through-answer checkpoint is stronger than
    the 20260619 open-answer checkpoint under no-KV in both free and softforce,
    but softforce hurts both checkpoints despite higher trigger rates.
  - HR-Bench free for ckpt19 is available, but the paired ckpt075/free and
    softforce rows are not yet available, so HR cannot be used for ckpt
    comparison from this interrupted run.
  - The HR softforce/no-KV path needs inspection before running OCR full; OCR
    would otherwise inherit the same long-generation risk.

### DIAG-20260625-qwen3-native-vs-manual-prefix-cache

- Status: DONE.
- Question:
  - Check whether differences besides DeepStack have measurable effects between native image prefix and manual `inputs_embeds` prefix for Qwen3 Stage2 20260623 prob=0.75.
- Checkpoint:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_oldmaskprob075_throughanswer_from_20260619_stage1_train47_eval03_4gpu_bs16_accum2_focus80_value1_1200step_20260623_005426/checkpoint_step_1200.pt`
- Method:
  - One teacher-forced Stage2 sample, max image resolution 512.
  - Compare native pixel/image prefix cache against manual `inputs_embeds` prefix with original visual positions replaced by tapped `v_merge`.
  - Then append the same `correct_D` with `natural_continue` and compare logits again.
- Result:
  - Sample: `visual_genome:2410492`.
  - Prefix logits native vs manual:
    - max abs diff: 1.5703.
    - mean abs diff: 0.2883.
    - cosine: 0.9978.
    - top token: both `<|im_end|>`.
  - After appending the same D, native-cache+D vs manual-cache+D:
    - max abs diff: 3.0547.
    - mean abs diff: 0.4216.
    - cosine: 0.9939.
    - top token: both `<think>`.
  - Position / rope:
    - manual position ids vs recomputed position ids max abs diff: 0.
    - native rope deltas and manual rope deltas: both `[[-144]]`.
    - native and manual cache length: both 380.
- Conclusion:
  - There is a measurable effect from native prefix vs manual prefix even before D append, and it grows after D append.
  - In this probe, position ids / rope deltas / cache length do not explain the difference.
  - The remaining concrete implementation difference is the native image path side effects, especially DeepStack / native visual forward behavior, versus manual `v_merge` inputs_embeds.

### DIAG-20260625-qwen3-deepstack-ablation-native-vs-manual

- Status: DONE.
- Question:
  - Check whether the measured native-prefix vs manual-prefix difference is actually caused by DeepStack rather than generic forward-entry, position ids, rope deltas, or cache length.
- Checkpoint:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_oldmaskprob075_throughanswer_from_20260619_stage1_train47_eval03_4gpu_bs16_accum2_focus80_value1_1200step_20260623_005426/checkpoint_step_1200.pt`
- Output:
  - `outputs/diagnostics/qwen3_deepstack_ablation_20260625_205846/result.json`
- Method:
  - One teacher-forced Stage2 sample `visual_genome:2410492`, max image resolution 512.
  - Compare three prefix caches: native with DeepStack, native with `_deepstack_process` monkeypatched to no-op, and manual `inputs_embeds` with original image positions replaced by tapped `v_merge`.
  - Append the same `correct_D` with `natural_continue`, then compare post-D logits.
- Results:
  - Prefix native+DeepStack vs manual:
    - max abs diff 1.5703, mean abs diff 0.2883, cosine 0.9978.
  - Prefix native no-DeepStack vs manual:
    - max abs diff 0.0, mean abs diff 0.0, cosine 1.0.
  - After D, native+DeepStack cache vs manual cache:
    - max abs diff 3.0547, mean abs diff 0.4216, cosine 0.9939.
  - After D, native no-DeepStack cache vs manual cache:
    - max abs diff 0.0, mean abs diff 0.0, cosine 1.0.
  - Position / rope / cache length:
    - manual vs computed position ids max diff 0.
    - rope deltas all `[[-144]]`.
    - cache length all 380.
- Conclusion:
  - For this controlled sample, the native/manual difference is fully explained by DeepStack injection.
  - Forward-entry, position ids, rope deltas, and cache length do not produce residual difference once DeepStack is disabled.
  - This supports the interpretation that old KV eval mixed a native DeepStack prefix cache with manual TGVF D append, while training/no-KV answer loss uses the manual no-DeepStack full-sequence path.

### EXP-20260626-002759-clean-stage2-vstar1-smoke

- Status: RUNNING.
- Question:
  - Can the new `revisit_vlm_clean` executable runner load a real historical Qwen3 Stage2 checkpoint through the narrow legacy adapter and run one deterministic path-backed VStar sample end to end?
- Baseline anchor:
  - `BASE-20260617-rowonly-original-stage2`.
- Intended diff:
  - This is not an accuracy comparison.
  - It validates the clean runner contract, deterministic diagnostic manifest, `run_config.txt` launch identity output, and Stage2 legacy bridge on one VStar row.
- Allowed changed variables:
  - Clean wrapper/adapter path.
  - Single diagnostic sample manifest.
  - Smaller `max_answer_tokens=32` for smoke runtime.
- Not allowed to change:
  - Stage2 checkpoint.
  - Stage1 processor recorded in checkpoint.
  - TGVF protocol.
  - Sample id.
  - Max image resolution 512.
  - DeepStack state: clean default off.
  - Post-TGVF continuation: `natural_continue`.
- Code commit / worktree:
  - Planned from branch `clean/tgvf-clean-project-20260625`.
  - Pre-ledger code commit: `eed5a59534e23351fc5a8914ff54e5d6fa51a7f0`.
  - Launch must occur from a tracked-clean worktree after this PLANNED entry is committed; exact runtime commit will be recorded in `run_config.txt`.
  - Untracked paths ignored for code identity: `logs/`, `third_party/`.
- Stage1 checkpoint:
  - `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617/train/checkpoint_step_2000.pt`
- Stage1 processor:
  - `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617/train/processor_step_2000`
- Stage2 checkpoint/output:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/checkpoint_step_1200.pt`
- Train data:
  - Recorded in checkpoint config:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl`
- Validation data:
  - Stage2 runtime validation jsonl:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`
  - Runtime identity check returned `n_rows=1002`, `need_focus=857`, `no_focus=145`.
- Benchmark sample:
  - Manifest:
    `revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_first_1_20260626.json`
  - Manifest hash:
    `851e301ea0730ee90086c83565135fa5c5fcc5d33b962fd074444d85241d6995`
  - Sample id:
    `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/0_000000`
  - Source:
    `vstar_bench/snapshot/test_questions.jsonl`, row index `0`.
  - Image:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks/vstar_bench/snapshot/direct_attributes/sa_4690.jpg`
  - Gold:
    `A`.
- Benchmark output:
  - `outputs/clean_smokes/stage2_tgvf_force_vstar1_20260626_002759`
- Script / command:
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.benchmark --run-id clean_stage2_tgvf_force_vstar1_20260626_002759 --checkpoint-path outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617/train/processor_step_2000 --mode tgvf_force --post-tgvf-forward-mode kv_cache --tgvf-protocol protocol_c_tool_observation --population-id vstar_test_questions_191 --manifest-path revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_first_1_20260626.json --manifest-hash 851e301ea0730ee90086c83565135fa5c5fcc5d33b962fd074444d85241d6995 --benchmark-root /home/dredvpn009/Flash_Storage/datasets/benchmarks --output-dir outputs/clean_smokes/stage2_tgvf_force_vstar1_20260626_002759 --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 32 --execute --runner-backend tgvf_stage2_qwen3 --dtype bfloat16 --device cuda:0 --device-map cuda:0 --attn-implementation sdpa --stage2-checkpoint outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/checkpoint_step_1200.pt --stage2-eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl --stage2-d-condition correct_D --force-prefix-mode target_hint`
- GPUs:
  - Planned: GPU 0 only.
- tmux:
  - `clean_stage2_vstar1_20260626_002759`.
- Started:
  - 2026-06-26T00:30:36+09:00.
- Finished:
  - 2026-06-26T00:31:59+09:00.
- Metrics:
  - `n_rows=1`.
  - `accuracy=1.0`.
  - `answer_parse_rate=1.0`.
  - `trigger_rate=1.0`.
  - `focus_valid_rate=1.0`.
  - `append_success_rate=1.0`.
  - `malformed_rate=0.0`.
  - Row wall time: 16.11 seconds.
  - Captured D shape: `[234, 4096]`.
  - Captured H_q shape: `[17, 4096]`.
  - Focus target:
    `close-up of the blue glove on the vendor's hand with its smooth surface and fit`.
  - Raw final answer:
    `The glove looks like a standard disposable rubber glove used in food service.\n</think>\n(A) rubber<|im_end|>`.
  - Parsed answer:
    `A`.
  - Score:
    `1.0`.
- Analysis:
  - The clean executable runner successfully loaded the historical Qwen3 Stage2 checkpoint and LoRA adapter through `tgvf_stage2_qwen3`.
  - `run_config.txt`, `run_config.json`, `sample_manifest.json`, `rows.jsonl`, and `summary.json` were written.
  - Runtime code identity was recorded in `run_config.txt` as commit `f3984ee37bb5212a3ff1b32b160cec46ddefbe0b` with `dirty_worktree=False`.
  - The bridge used the checkpoint-consistent Stage2 validation jsonl under `tgvf_v4_teacher_50k_clean_imend`, not the open-answer split.
  - The earlier bridge issue where clean choices could be passed into legacy `prompt_question` and duplicate answer choices was fixed before launch.
  - Legacy debug emitted `target_answer_leakage_flag=true`; this one-row smoke is therefore not evidence about focus-target quality statistics.
- Conclusion:
  - Clean Stage2 one-row force smoke passed.
  - The clean wrapper, diagnostic manifest, launch identity output, Stage2 runtime validation, and legacy bridge are now executable on a real checkpoint.
- Comparable to baseline:
  - No; diagnostic launch smoke only.
- Follow-up:
  - If this passes, run the same clean backend on a small multi-row manifest before any benchmark claims.
