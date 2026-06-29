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

- Status: STOPPED.
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
- Ledger correction:
  - Status was left as `RUNNING` after the 2026-06-25 stop. Updated to
    `STOPPED` on 2026-06-26 before launching clean-native smoke work; no new
    metrics were added.

### EXP-20260626-030329-clean-native-stage2-vstar1-smoke

- Status: DONE.
- Question:
  - Does the clean-native `tgvf_stage2_qwen3_native` backend load and execute a
    one-sample Qwen3 Stage2 TGVF force smoke without using the historical
    `Stage2ProtocolEvaluator` class?
- Baseline anchor:
  - Diagnostic only; not a benchmark comparison.
  - Uses `BASE-20260619-open-answer-rowonly` checkpoint because it is a known
    working open-answer Qwen3 Stage2 artifact.
- Intended diff:
  - Use the clean project backend `tgvf_stage2_qwen3_native` instead of the
    historical Stage2 evaluator bridge.
  - Use committed diagnostic manifest
    `revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_first_1_20260626.json`.
- Allowed changed variables:
  - Backend implementation path: clean-native only.
  - Output directory for smoke artifacts.
- Not allowed to change:
  - Checkpoint and processor identity.
  - Manifest sample id/hash.
  - Protocol: `protocol_c_tool_observation`.
  - Mode: `tgvf_force`.
  - Forward mode: `kv_cache`.
  - Continuation: `natural_continue`.
  - Max image resolution: 512.
  - Scoring backend: `auto`.
- Code commit / worktree:
  - Commit: `09247b2ff6834b93e63e03e7fa3b53489c4cd029`.
  - Worktree dirty only from this RUNNING ledger update at launch.
- Stage1 checkpoint:
  - Not used directly by benchmark runner.
- Stage1 processor:
  - Not used directly by benchmark runner.
- Stage2 checkpoint/output:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt`
- Stage2 processor:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200`
- Train data:
  - Not used.
- Validation data:
  - Stage2 runtime identity file:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`
  - Diagnostic manifest:
    `revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_first_1_20260626.json`
  - Manifest hash:
    `851e301ea0730ee90086c83565135fa5c5fcc5d33b962fd074444d85241d6995`
  - Benchmark source:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks/vstar_bench/snapshot/test_questions.jsonl`
  - Sample count: 1.
- Benchmark output:
  - `outputs/clean_native_smoke/qwen3_stage2_native_vstar1_force_20260626_030329`
- Script / command:
  - Planned command:
    `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.benchmark --run-id clean_native_stage2_vstar1_force_20260626_030329 --checkpoint-path outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200 --mode tgvf_force --post-tgvf-forward-mode kv_cache --subset-id diagnostic_vstar_first_1_20260626 --manifest-path revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_first_1_20260626.json --manifest-hash 851e301ea0730ee90086c83565135fa5c5fcc5d33b962fd074444d85241d6995 --benchmark-root /home/dredvpn009/Flash_Storage/datasets/benchmarks --output-dir outputs/clean_native_smoke/qwen3_stage2_native_vstar1_force_20260626_030329 --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 64 --tgvf-protocol protocol_c_tool_observation --scoring-backend auto --runner-backend tgvf_stage2_qwen3_native --dtype bfloat16 --device cuda:0 --device-map cuda:0 --attn-implementation sdpa --stage2-checkpoint outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --stage2-eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl --stage2-d-condition correct_D --force-prefix-mode target_hint --execute`
- GPUs:
  - Planned: physical GPU 2 via `CUDA_VISIBLE_DEVICES=2`, runtime device
    `cuda:0`.
- tmux:
  - None planned unless foreground command exceeds interactive runtime.
- Started:
  - 2026-06-26T03:09:27+09:00.
- Finished:
  - 2026-06-26T03:11:38+09:00.
- Metrics:
  - n: 1.
  - accuracy: 1.0.
  - answer parse rate: 1.0.
  - trigger rate: 1.0.
  - focus valid rate: 1.0.
  - append success rate: 1.0.
  - row error: null.
  - wall time for row: 37.41 seconds.
  - parsed answer: `A`.
  - score: 1.0.
  - generated focus target:
    `close-up of the glove surface, sheen, and fit on the hand`.
  - FVT shape: `[234, 4096]`.
  - append path:
    `clean_native_qwen3_visual_special_tokens_embedding_replace`.
- Analysis:
  - The clean-native backend loaded the Qwen3 model, Stage2 LoRA, and foveal
    module, captured a forced focus target, produced D, appended visual D, and
    continued generation without row error.
  - `summary.json` and `run_config.txt` record
    `runner_backend=tgvf_stage2_qwen3_native`,
    `resolved_backend=tgvf_stage2_qwen3_native`, `kv_cache`, DeepStack off, and
    the diagnostic manifest hash.
  - Checkpoint validation showed the checkpoint config processor points to the
    20260619 Stage1 processor; `diff -qr` confirmed that processor directory is
    identical to the Stage2 `processor_step_1200` directory used in the command.
  - This is a one-sample forced-path runtime smoke only. It does not validate
    free/softforce trigger behavior, no-KV full-sequence behavior, or benchmark
    accuracy.
- Conclusion:
  - PASS for clean-native Stage2 Qwen3 forced-path GPU smoke.
  - The native backend is no longer merely import/test clean; it can execute a
    real checkpoint on a real image through post-D continuation.
- Comparable to baseline:
  - No. Diagnostic smoke only.
- Follow-up:
  - Run the same diagnostic manifest for `tgvf_free` and `tgvf_softforce`.
  - Add a fixed diagnostic comparison against `tgvf_stage2_qwen3_legacy`.
  - Then run a small fixed manifest before any benchmark-scale claim.

### EXP-20260626-031442-clean-native-stage2-vstar1-free-softforce-smoke

- Status: DONE.
- Question:
  - Do the clean-native `tgvf_stage2_qwen3_native` free and softforce paths run
    on the same fixed one-sample VStar diagnostic manifest?
- Baseline anchor:
  - Diagnostic only; follows the successful force-path smoke
    `EXP-20260626-030329-clean-native-stage2-vstar1-smoke`.
- Intended diff:
  - Change mode from `tgvf_force` to `tgvf_free` and `tgvf_softforce`.
  - Keep checkpoint, processor, manifest, protocol, forward mode, max
    resolution, and scoring fixed.
- Allowed changed variables:
  - Mode.
  - Softforce prompt text for the softforce run: `Use focus tool.`
  - Output directories.
- Not allowed to change:
  - Checkpoint and processor identity.
  - Manifest sample id/hash.
  - Protocol: `protocol_c_tool_observation`.
  - Forward mode: `kv_cache`.
  - Continuation: `natural_continue`.
  - Max image resolution: 512.
  - Scoring backend: `auto`.
- Code commit / worktree:
  - Commit: `de04b2dda7f4de004cb331497013ef8885a7c979`.
  - Worktree dirty only from this RUNNING ledger update at launch.
- Stage1 checkpoint:
  - Not used directly by benchmark runner.
- Stage1 processor:
  - Not used directly by benchmark runner.
- Stage2 checkpoint/output:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt`
- Stage2 processor:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200`
- Train data:
  - Not used.
- Validation data:
  - Stage2 runtime identity file:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`
  - Diagnostic manifest:
    `revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_first_1_20260626.json`
  - Manifest hash:
    `851e301ea0730ee90086c83565135fa5c5fcc5d33b962fd074444d85241d6995`
  - Benchmark source:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks/vstar_bench/snapshot/test_questions.jsonl`
  - Sample count: 1.
- Benchmark output:
  - Free:
    `outputs/clean_native_smoke/qwen3_stage2_native_vstar1_free_20260626_031442`
  - Softforce:
    `outputs/clean_native_smoke/qwen3_stage2_native_vstar1_softforce_20260626_031442`
- Script / command:
  - Free:
    `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.benchmark --run-id clean_native_stage2_vstar1_free_20260626_031442 --checkpoint-path outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200 --mode tgvf_free --post-tgvf-forward-mode kv_cache --subset-id diagnostic_vstar_first_1_20260626 --manifest-path revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_first_1_20260626.json --manifest-hash 851e301ea0730ee90086c83565135fa5c5fcc5d33b962fd074444d85241d6995 --benchmark-root /home/dredvpn009/Flash_Storage/datasets/benchmarks --output-dir outputs/clean_native_smoke/qwen3_stage2_native_vstar1_free_20260626_031442 --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 64 --tgvf-protocol protocol_c_tool_observation --scoring-backend auto --runner-backend tgvf_stage2_qwen3_native --dtype bfloat16 --device cuda:0 --device-map cuda:0 --attn-implementation sdpa --stage2-checkpoint outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --stage2-eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl --stage2-d-condition correct_D --force-prefix-mode target_hint --execute`
  - Softforce:
    `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.benchmark --run-id clean_native_stage2_vstar1_softforce_20260626_031442 --checkpoint-path outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200 --mode tgvf_softforce --softforce-prompt-text "Use focus tool." --post-tgvf-forward-mode kv_cache --subset-id diagnostic_vstar_first_1_20260626 --manifest-path revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_first_1_20260626.json --manifest-hash 851e301ea0730ee90086c83565135fa5c5fcc5d33b962fd074444d85241d6995 --benchmark-root /home/dredvpn009/Flash_Storage/datasets/benchmarks --output-dir outputs/clean_native_smoke/qwen3_stage2_native_vstar1_softforce_20260626_031442 --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 64 --tgvf-protocol protocol_c_tool_observation --scoring-backend auto --runner-backend tgvf_stage2_qwen3_native --dtype bfloat16 --device cuda:0 --device-map cuda:0 --attn-implementation sdpa --stage2-checkpoint outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --stage2-eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl --stage2-d-condition correct_D --force-prefix-mode target_hint --execute`
- GPUs:
  - Planned: physical GPU 3 via `CUDA_VISIBLE_DEVICES=3`, runtime device
    `cuda:0`.
- tmux:
  - None planned unless foreground command exceeds interactive runtime.
- Started:
  - 2026-06-26T03:14:42+09:00.
- Finished:
  - 2026-06-26T03:19:33+09:00.
- Metrics:
  - Free:
    - n: 1.
    - accuracy: 1.0.
    - answer parse rate: 1.0.
    - trigger rate: 0.0.
    - focus valid rate: 0.0.
    - append success rate: null.
    - row error: null.
    - parsed answer: `A`.
    - score: 1.0.
    - wall time for row: 31.69 seconds.
  - Softforce:
    - n: 1.
    - accuracy: 1.0.
    - answer parse rate: 1.0.
    - trigger rate: 0.0.
    - focus valid rate: 0.0.
    - append success rate: null.
    - row error: null.
    - parsed answer: `A`.
    - score: 1.0.
    - wall time for row: 31.72 seconds.
- Analysis:
  - Both free and softforce clean-native runs loaded the real Qwen3 Stage2
    checkpoint and produced valid scored outputs with no row errors.
  - Neither run triggered focus on this single VStar row; both took the
    direct/no-trigger branch with `append_success=null`.
  - Therefore this entry validates free/softforce runtime and parsing for the
    no-trigger branch only. Triggered post-D execution is covered by the force
    smoke, but a trigger-positive free/softforce diagnostic remains needed.
- Conclusion:
  - PASS for clean-native Stage2 Qwen3 free/softforce no-trigger GPU smoke.
  - Not sufficient as evidence for free/softforce triggered-TGVF behavior.
- Comparable to baseline:
  - No. Diagnostic smoke only.
- Follow-up:
  - Find or create a fixed diagnostic sample/prompt where softforce triggers,
    then run clean-native and legacy bridge on that same sample.
  - After trigger-positive diagnostic passes, run a small fixed manifest before
    any benchmark-scale claim.

### EXP-20260626-032556-clean-native-stage2-vstar-trigger-softforce-smoke

- Status: DONE.
- Question:
  - Does clean-native `tgvf_stage2_qwen3_native` softforce trigger and complete
    post-D continuation on a fixed VStar sample selected from historical
    trigger-positive ckpt19 softforce rows?
- Baseline anchor:
  - Diagnostic only.
  - Candidate sample evidence:
    `outputs/no_kv_benchmarks/qwen3_other_benchmarks_19_vs_075_nokv_20260625_191315/ckpt19/vstar_bench/free_softforce_focus/merged_rows.jsonl`
    had row id `10` with `trigger_focus_decision=true`,
    `focus_valid=true`, and `append_success=true`.
- Intended diff:
  - Use the new committed clean diagnostic manifest
    `revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_softforce_trigger_row10_20260626.json`.
  - Run only the clean-native softforce path first.
- Allowed changed variables:
  - Diagnostic manifest sample: VStar row_index 10.
  - Mode: `tgvf_softforce`.
  - Softforce prompt text: `Use focus tool.`
  - Output directory.
- Not allowed to change:
  - Checkpoint and processor identity.
  - Manifest sample id/hash.
  - Protocol: `protocol_c_tool_observation`.
  - Forward mode: `kv_cache`.
  - Continuation: `natural_continue`.
  - Max image resolution: 512.
  - Scoring backend: `auto`.
- Code commit / worktree:
  - Commit: `f357d5df7727a2f7e67ca642961fe0bf1e35aee4`.
  - Worktree dirty only from this RUNNING ledger update at launch.
- Stage1 checkpoint:
  - Not used directly by benchmark runner.
- Stage1 processor:
  - Not used directly by benchmark runner.
- Stage2 checkpoint/output:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt`
- Stage2 processor:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200`
- Train data:
  - Not used.
- Validation data:
  - Stage2 runtime identity file:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`
  - Diagnostic manifest:
    `revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_softforce_trigger_row10_20260626.json`
  - Manifest hash:
    `6421ea792f5db73555c895a7b28a4bacfb8eefa6046b1d35080b17b5081cfb44`
  - Benchmark source:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks/vstar_bench/snapshot/test_questions.jsonl`
  - Sample count: 1.
- Benchmark output:
  - `outputs/clean_native_smoke/qwen3_stage2_native_vstar1_trigger_softforce_20260626_032556`
- Script / command:
  - `CUDA_VISIBLE_DEVICES=4 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.benchmark --run-id clean_native_stage2_vstar1_trigger_softforce_20260626_032556 --checkpoint-path outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200 --mode tgvf_softforce --softforce-prompt-text "Use focus tool." --post-tgvf-forward-mode kv_cache --subset-id diagnostic_vstar_softforce_trigger_row10_20260626 --manifest-path revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_softforce_trigger_row10_20260626.json --manifest-hash 6421ea792f5db73555c895a7b28a4bacfb8eefa6046b1d35080b17b5081cfb44 --benchmark-root /home/dredvpn009/Flash_Storage/datasets/benchmarks --output-dir outputs/clean_native_smoke/qwen3_stage2_native_vstar1_trigger_softforce_20260626_032556 --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 64 --tgvf-protocol protocol_c_tool_observation --scoring-backend auto --runner-backend tgvf_stage2_qwen3_native --dtype bfloat16 --device cuda:0 --device-map cuda:0 --attn-implementation sdpa --stage2-checkpoint outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --stage2-eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl --stage2-d-condition correct_D --force-prefix-mode target_hint --execute`
- GPUs:
  - Planned: physical GPU 4 via `CUDA_VISIBLE_DEVICES=4`, runtime device
    `cuda:0`.
- tmux:
  - None planned unless foreground command exceeds interactive runtime.
- Started:
  - 2026-06-26T03:25:56+09:00.
- Finished:
  - 2026-06-26T03:29:25+09:00.
- Metrics:
  - n: 1.
  - accuracy: 1.0.
  - answer parse rate: 1.0.
  - trigger rate: 1.0.
  - focus valid rate: 1.0.
  - append success rate: 1.0.
  - row error: null.
  - parsed answer: `C`.
  - score: 1.0.
  - wall time for row: 43.55 seconds.
  - generated focus target:
    `small worker figure near the right side with helmet and nearby construction materials`.
  - FVT shape: `[234, 4096]`.
  - append path:
    `clean_native_qwen3_visual_special_tokens_embedding_replace`.
- Analysis:
  - The selected trigger-positive diagnostic worked under the clean-native
    backend: softforce triggered focus, produced a valid target, appended D, and
    answered correctly with no row error.
  - This validates the clean-native free/softforce triggered post-D branch on a
    real checkpoint and real image.
- Conclusion:
  - PASS for clean-native Stage2 Qwen3 softforce trigger-positive GPU smoke.
- Comparable to baseline:
  - No. Diagnostic smoke only.
- Follow-up:
  - If this triggers and appends successfully, run the same manifest with
    `tgvf_stage2_qwen3_legacy`.

### EXP-20260626-032925-legacy-stage2-vstar-trigger-softforce-comparison

- Status: DONE.
- Question:
  - Does the diagnostic legacy bridge produce the same high-level trigger and
    append behavior on the fixed trigger-positive VStar sample?
- Baseline anchor:
  - Clean-native result:
    `EXP-20260626-032556-clean-native-stage2-vstar-trigger-softforce-smoke`.
- Intended diff:
  - Change only runner backend from `tgvf_stage2_qwen3_native` to
    `tgvf_stage2_qwen3_legacy`.
- Allowed changed variables:
  - Backend implementation path.
  - Output directory.
- Not allowed to change:
  - Checkpoint and processor identity.
  - Manifest sample id/hash.
  - Protocol: `protocol_c_tool_observation`.
  - Mode: `tgvf_softforce`.
  - Softforce prompt text: `Use focus tool.`
  - Forward mode: `kv_cache`.
  - Continuation: `natural_continue`.
  - Max image resolution: 512.
  - Scoring backend: `auto`.
- Code commit / worktree:
  - Commit: `f357d5df7727a2f7e67ca642961fe0bf1e35aee4`.
  - Worktree dirty only from ledger updates at launch.
- Stage1 checkpoint:
  - Not used directly by benchmark runner.
- Stage1 processor:
  - Not used directly by benchmark runner.
- Stage2 checkpoint/output:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt`
- Stage2 processor:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200`
- Train data:
  - Not used.
- Validation data:
  - Stage2 runtime identity file:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`
  - Diagnostic manifest:
    `revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_softforce_trigger_row10_20260626.json`
  - Manifest hash:
    `6421ea792f5db73555c895a7b28a4bacfb8eefa6046b1d35080b17b5081cfb44`
  - Sample count: 1.
- Benchmark output:
  - `outputs/clean_native_smoke/qwen3_stage2_legacy_vstar1_trigger_softforce_20260626_032925`
- Script / command:
  - `CUDA_VISIBLE_DEVICES=5 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.benchmark --run-id legacy_stage2_vstar1_trigger_softforce_20260626_032925 --checkpoint-path outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200 --mode tgvf_softforce --softforce-prompt-text "Use focus tool." --post-tgvf-forward-mode kv_cache --subset-id diagnostic_vstar_softforce_trigger_row10_20260626 --manifest-path revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_softforce_trigger_row10_20260626.json --manifest-hash 6421ea792f5db73555c895a7b28a4bacfb8eefa6046b1d35080b17b5081cfb44 --benchmark-root /home/dredvpn009/Flash_Storage/datasets/benchmarks --output-dir outputs/clean_native_smoke/qwen3_stage2_legacy_vstar1_trigger_softforce_20260626_032925 --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 64 --tgvf-protocol protocol_c_tool_observation --scoring-backend auto --runner-backend tgvf_stage2_qwen3_legacy --dtype bfloat16 --device cuda:0 --device-map cuda:0 --attn-implementation sdpa --stage2-checkpoint outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --stage2-eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl --stage2-d-condition correct_D --force-prefix-mode target_hint --execute`
- GPUs:
  - Planned: physical GPU 5 via `CUDA_VISIBLE_DEVICES=5`, runtime device
    `cuda:0`.
- tmux:
  - None planned unless foreground command exceeds interactive runtime.
- Started:
  - 2026-06-26T03:29:25+09:00.
- Finished:
  - 2026-06-26T03:32:06+09:00.
- Metrics:
  - Legacy:
    - n: 1.
    - accuracy: 1.0.
    - answer parse rate: 1.0.
    - trigger rate: 1.0.
    - focus valid rate: 1.0.
    - append success rate: 1.0.
    - row error: null.
    - parsed answer: `C`.
    - score: 1.0.
    - wall time for row: 21.69 seconds.
    - generated focus target:
      `small worker figure near the right side with helmet and nearby construction materials`.
    - FVT shape: `[234, 4096]`.
    - append path: `qwen3_visual_special_tokens_embedding_replace_lora_forward`.
  - Native comparison:
    - Same trigger/focus_valid/append_success/parsed_answer/score.
    - Same focus target.
    - Same raw output:
      `The focused view shows a bright yellow helmet on the worker.\n</think>\n(C) yellow<|im_end|>`.
    - Native append path:
      `clean_native_qwen3_visual_special_tokens_embedding_replace`.
- Analysis:
  - Legacy and clean-native agree on the high-level behavior for this fixed
    trigger-positive softforce sample: both trigger, append, and answer
    correctly.
  - The output text and focus target are identical. Debug names differ because
    native and legacy use different block/path labels.
  - Legacy row wall time is lower in this one-row run, but this is not a
    controlled performance comparison.
- Conclusion:
  - PASS for fixed trigger-positive clean-native vs legacy bridge comparison.
  - This gives a concrete anchor that the clean-native triggered softforce path
    is behaviorally aligned with the legacy bridge on at least one real sample.
- Comparable to baseline:
  - Diagnostic comparison only.
- Follow-up:
  - Move to a small fixed manifest, preferably
    `diagnostic_vstar_core_smoke_first_8_20260626` or a trigger-enriched
    8-sample manifest, before benchmark-scale claims.

### EXP-20260626-033554-clean-native-vs-legacy-vstar8-softforce

- Status: DONE.
- Question:
  - Do clean-native and legacy Stage2 softforce agree on a small fixed
    8-sample VStar diagnostic manifest?
- Baseline anchor:
  - Follows the one-sample trigger-positive comparison
    `EXP-20260626-032925-legacy-stage2-vstar-trigger-softforce-comparison`.
- Intended diff:
  - Expand from one trigger-positive sample to the committed 8-row VStar
    diagnostic manifest.
  - Compare `tgvf_stage2_qwen3_native` and `tgvf_stage2_qwen3_legacy`.
- Allowed changed variables:
  - Backend implementation path.
  - Sample count/manifest.
  - Output directories.
- Not allowed to change:
  - Checkpoint and processor identity.
  - Manifest sample ids/hash.
  - Protocol: `protocol_c_tool_observation`.
  - Mode: `tgvf_softforce`.
  - Softforce prompt text: `Use focus tool.`
  - Forward mode: `kv_cache`.
  - Continuation: `natural_continue`.
  - Max image resolution: 512.
  - Scoring backend: `auto`.
- Code commit / worktree:
  - Commit: `c65bd6276e18dd7ec812abe2e0eb9d8975e74f1b`.
  - Worktree dirty only from this RUNNING ledger update at launch.
- Stage1 checkpoint:
  - Not used directly by benchmark runner.
- Stage1 processor:
  - Not used directly by benchmark runner.
- Stage2 checkpoint/output:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt`
- Stage2 processor:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200`
- Train data:
  - Not used.
- Validation data:
  - Stage2 runtime identity file:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`
  - Diagnostic manifest:
    `revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_core_smoke_first_8_20260626.json`
  - Manifest hash:
    `3a6020b145c1543be3fc8a5541c17d00b4fc5d27aeddbecb90a10417d25c84f4`
  - Benchmark source:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks/vstar_bench/snapshot/test_questions.jsonl`
  - Sample count: 8.
- Benchmark output:
  - Native:
    `outputs/clean_native_smoke/qwen3_stage2_native_vstar8_softforce_20260626_033554`
  - Legacy:
    `outputs/clean_native_smoke/qwen3_stage2_legacy_vstar8_softforce_20260626_033554`
- Script / command:
  - Native:
    `CUDA_VISIBLE_DEVICES=4 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.benchmark --run-id clean_native_stage2_vstar8_softforce_20260626_033554 --checkpoint-path outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200 --mode tgvf_softforce --softforce-prompt-text "Use focus tool." --post-tgvf-forward-mode kv_cache --subset-id diagnostic_vstar_core_smoke_first_8_20260626 --manifest-path revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_core_smoke_first_8_20260626.json --manifest-hash 3a6020b145c1543be3fc8a5541c17d00b4fc5d27aeddbecb90a10417d25c84f4 --benchmark-root /home/dredvpn009/Flash_Storage/datasets/benchmarks --output-dir outputs/clean_native_smoke/qwen3_stage2_native_vstar8_softforce_20260626_033554 --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 64 --tgvf-protocol protocol_c_tool_observation --scoring-backend auto --runner-backend tgvf_stage2_qwen3_native --dtype bfloat16 --device cuda:0 --device-map cuda:0 --attn-implementation sdpa --stage2-checkpoint outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --stage2-eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl --stage2-d-condition correct_D --force-prefix-mode target_hint --execute`
  - Legacy:
    `CUDA_VISIBLE_DEVICES=5 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.benchmark --run-id legacy_stage2_vstar8_softforce_20260626_033554 --checkpoint-path outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200 --mode tgvf_softforce --softforce-prompt-text "Use focus tool." --post-tgvf-forward-mode kv_cache --subset-id diagnostic_vstar_core_smoke_first_8_20260626 --manifest-path revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_core_smoke_first_8_20260626.json --manifest-hash 3a6020b145c1543be3fc8a5541c17d00b4fc5d27aeddbecb90a10417d25c84f4 --benchmark-root /home/dredvpn009/Flash_Storage/datasets/benchmarks --output-dir outputs/clean_native_smoke/qwen3_stage2_legacy_vstar8_softforce_20260626_033554 --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 64 --tgvf-protocol protocol_c_tool_observation --scoring-backend auto --runner-backend tgvf_stage2_qwen3_legacy --dtype bfloat16 --device cuda:0 --device-map cuda:0 --attn-implementation sdpa --stage2-checkpoint outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --stage2-eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl --stage2-d-condition correct_D --force-prefix-mode target_hint --execute`
- GPUs:
  - Native: physical GPU 4 via `CUDA_VISIBLE_DEVICES=4`.
  - Legacy: physical GPU 5 via `CUDA_VISIBLE_DEVICES=5`.
- tmux:
  - None planned unless foreground command exceeds interactive runtime.
- Started:
  - 2026-06-26T03:35:54+09:00.
- Finished:
  - 2026-06-26T03:40:51+09:00.
- Metrics:
  - Native:
    - n: 8.
    - accuracy: 0.625.
    - answer parse rate: 1.0.
    - trigger rate: 0.25.
    - focus valid rate: 0.25.
    - append success rate: 1.0 over triggered rows.
    - malformed rate: 0.0.
    - row errors: 0.
    - triggered rows: 2/8.
    - correct rows: 5/8.
  - Legacy:
    - n: 8.
    - accuracy: 0.625.
    - answer parse rate: 1.0.
    - trigger rate: 0.25.
    - focus valid rate: 0.25.
    - append success rate: 1.0 over triggered rows.
    - malformed rate: 0.0.
    - row errors: 0.
    - triggered rows: 2/8.
    - correct rows: 5/8.
  - Per-row comparison:
    - Native and legacy matched on every sample for trigger decision,
      focus_valid, append_success, parsed_answer, score, focus target, and raw
      output.
    - Triggered sample ids:
      - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/87_000087`
      - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/53_000053`
    - Wrong sample ids for both:
      - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/9_000009`
      - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/145_000145`
      - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/65_000065`
- Analysis:
  - The 8-row diagnostic manifest exercises both branches:
    - 2 triggered softforce rows with successful D append;
    - 6 no-trigger direct rows.
  - Clean-native and legacy bridge are behaviorally identical on this fixed
    manifest at the row-output level, not only in aggregate metrics.
  - This is still a path-backed VStar diagnostic subset, not a benchmark table.
- Conclusion:
  - PASS for small fixed clean-native vs legacy Stage2 softforce comparison.
  - The clean-native backend is now validated beyond one-row smoke on a mixed
    trigger/no-trigger fixed manifest.
- Comparable to baseline:
  - Diagnostic comparison only.
- Follow-up:
  - Move to clean benchmark subset boundary and no-KV validation.
  - Run official small CoreSmoke slices only after the benchmark naming/sample
    rules are pinned in clean docs.

### EXP-20260626-034311-clean-native-vs-legacy-vstar8-softforce-nokv

- Status: DONE.
- Question:
  - Do clean-native and legacy Stage2 softforce agree on the same fixed 8-row
    VStar diagnostic manifest under `no_kv_full_sequence` post-D continuation?
- Baseline anchor:
  - KV comparison:
    `EXP-20260626-033554-clean-native-vs-legacy-vstar8-softforce`.
- Intended diff:
  - Change only `post_tgvf_forward_mode` from `kv_cache` to
    `no_kv_full_sequence`.
- Allowed changed variables:
  - Forward mode.
  - Output directories.
- Not allowed to change:
  - Checkpoint and processor identity.
  - Manifest sample ids/hash.
  - Protocol: `protocol_c_tool_observation`.
  - Mode: `tgvf_softforce`.
  - Softforce prompt text: `Use focus tool.`
  - Continuation: `natural_continue`.
  - Max image resolution: 512.
  - Scoring backend: `auto`.
- Code commit / worktree:
  - Commit: `92ccb82eb44eab9792aa86b205c35327c38c4509`.
  - Worktree dirty only from this RUNNING ledger update at launch.
- Stage1 checkpoint:
  - Not used directly by benchmark runner.
- Stage1 processor:
  - Not used directly by benchmark runner.
- Stage2 checkpoint/output:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt`
- Stage2 processor:
  - `outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200`
- Train data:
  - Not used.
- Validation data:
  - Stage2 runtime identity file:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`
  - Diagnostic manifest:
    `revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_core_smoke_first_8_20260626.json`
  - Manifest hash:
    `3a6020b145c1543be3fc8a5541c17d00b4fc5d27aeddbecb90a10417d25c84f4`
  - Benchmark source:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks/vstar_bench/snapshot/test_questions.jsonl`
  - Sample count: 8.
- Benchmark output:
  - Native:
    `outputs/clean_native_smoke/qwen3_stage2_native_vstar8_softforce_nokv_20260626_034311`
  - Legacy:
    `outputs/clean_native_smoke/qwen3_stage2_legacy_vstar8_softforce_nokv_20260626_034311`
- Script / command:
  - Native:
    `CUDA_VISIBLE_DEVICES=4 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.benchmark --run-id clean_native_stage2_vstar8_softforce_nokv_20260626_034311 --checkpoint-path outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200 --mode tgvf_softforce --softforce-prompt-text "Use focus tool." --post-tgvf-forward-mode no_kv_full_sequence --subset-id diagnostic_vstar_core_smoke_first_8_20260626 --manifest-path revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_core_smoke_first_8_20260626.json --manifest-hash 3a6020b145c1543be3fc8a5541c17d00b4fc5d27aeddbecb90a10417d25c84f4 --benchmark-root /home/dredvpn009/Flash_Storage/datasets/benchmarks --output-dir outputs/clean_native_smoke/qwen3_stage2_native_vstar8_softforce_nokv_20260626_034311 --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 64 --tgvf-protocol protocol_c_tool_observation --scoring-backend auto --runner-backend tgvf_stage2_qwen3_native --dtype bfloat16 --device cuda:0 --device-map cuda:0 --attn-implementation sdpa --stage2-checkpoint outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --stage2-eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl --stage2-d-condition correct_D --force-prefix-mode target_hint --execute`
  - Legacy:
    `CUDA_VISIBLE_DEVICES=5 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.benchmark --run-id legacy_stage2_vstar8_softforce_nokv_20260626_034311 --checkpoint-path outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/processor_step_1200 --mode tgvf_softforce --softforce-prompt-text "Use focus tool." --post-tgvf-forward-mode no_kv_full_sequence --subset-id diagnostic_vstar_core_smoke_first_8_20260626 --manifest-path revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_core_smoke_first_8_20260626.json --manifest-hash 3a6020b145c1543be3fc8a5541c17d00b4fc5d27aeddbecb90a10417d25c84f4 --benchmark-root /home/dredvpn009/Flash_Storage/datasets/benchmarks --output-dir outputs/clean_native_smoke/qwen3_stage2_legacy_vstar8_softforce_nokv_20260626_034311 --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 64 --tgvf-protocol protocol_c_tool_observation --scoring-backend auto --runner-backend tgvf_stage2_qwen3_legacy --dtype bfloat16 --device cuda:0 --device-map cuda:0 --attn-implementation sdpa --stage2-checkpoint outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_imend_open_answer_multifocus_focus_imend_from_2gpu_stage1_bidirectional_2gpu_bs16_accum4_focus80_value1_1200step_20260619_014148/checkpoint_step_1200.pt --stage2-eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl --stage2-d-condition correct_D --force-prefix-mode target_hint --execute`
- GPUs:
  - Native: physical GPU 4 via `CUDA_VISIBLE_DEVICES=4`.
  - Legacy: physical GPU 5 via `CUDA_VISIBLE_DEVICES=5`.
- tmux:
  - None planned unless foreground command exceeds interactive runtime.
- Started:
  - 2026-06-26T03:43:11+09:00.
- Finished:
  - 2026-06-26T03:52:28+09:00.
- Metrics:
  - Clean-native and legacy no-KV summaries matched exactly:
    - `n_rows=8`.
    - `n_scored=8`.
    - `accuracy=0.625`.
    - `answer_parse_rate=1.0`.
    - `trigger_rate=0.25`.
    - `focus_valid_rate=0.25`.
    - `append_success_rate=1.0` over triggered rows.
    - `malformed_rate=0.0`.
  - Triggered sample ids:
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/87_000087`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/53_000053`
  - Wrong sample ids:
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/9_000009`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/145_000145`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/65_000065`
  - Native-vs-legacy row diff:
    - `0/8` differences for sample id, trigger decision, focus validity,
      append success, parsed answer, score, raw output, focus target, error,
      malformed flag, and parse-success flag.
  - Compared to KV baseline
    `EXP-20260626-033554-clean-native-vs-legacy-vstar8-softforce`:
    - aggregate metrics were unchanged;
    - parsed answers and scores were unchanged;
    - one triggered row changed only explanation wording:
      `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/53_000053`.
- Analysis:
  - The clean-native `no_kv_full_sequence` path aligns with the diagnostic
    legacy bridge on this fixed VStar-8 softforce manifest.
  - The CLI manifest hash for this run is the clean stable manifest hash
    embedded in the manifest payload (`3a6020...`), not the raw file
    `sha256sum` (`afaf6427...`) that includes the embedded hash field itself.
  - Both outputs record DeepStack disabled and original-image scope off, so this
    validates clean-native parity for the current clean default only.
- Conclusion:
  - PASS for clean-native Stage2 Qwen3 softforce no-KV/full-sequence diagnostic
    parity against the legacy bridge on the fixed 8-row VStar manifest.
  - This is not a benchmark-scale effect claim.
- Comparable to baseline:
  - Diagnostic comparison only.
- Follow-up:
  - Use larger fixed clean subsets only after benchmark subset boundaries and
    scorer identity are pinned.

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

- Status: DONE.
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

### EXP-20260626-004135-clean-stage2-vstar8-smoke

- Status: DONE.
- Question:
  - Does the clean `tgvf_stage2_qwen3` bridge remain stable on a deterministic multi-row path-backed VStar manifest, beyond the one-row smoke?
- Baseline anchor:
  - Follows `EXP-20260626-002759-clean-stage2-vstar1-smoke`.
  - Uses the same historical checkpoint as `BASE-20260617-rowonly-original-stage2`.
- Intended diff:
  - Expand the diagnostic manifest from 1 fixed VStar row to 8 fixed VStar rows.
  - Keep mode, checkpoint, processor, protocol, max image resolution, DeepStack state, and continuation fixed.
- Allowed changed variables:
  - Manifest/sample count only.
- Not allowed to change:
  - Stage2 checkpoint.
  - Stage1 processor recorded in checkpoint.
  - TGVF protocol.
  - Mode: `tgvf_force`.
  - Max image resolution 512.
  - DeepStack state: clean default off.
  - Post-TGVF continuation: `natural_continue`.
  - Stage2 runtime validation jsonl.
- Code commit / worktree:
  - Planned from branch `clean/tgvf-clean-project-20260625`.
  - Pre-ledger code commit: `34a03d023bd33fdc94a1710b0c872f9c5b8d4e20`.
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
    `revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_core_smoke_first_8_20260626.json`
  - Manifest hash:
    `3a6020b145c1543be3fc8a5541c17d00b4fc5d27aeddbecb90a10417d25c84f4`
  - Selection rule:
    first eight VStar samples from committed `core_smoke_256_seed20260625`.
  - Sample count:
    `8`.
  - Categories:
    `direct_attributes=6`, `relative_position=2`.
  - Labels:
    `A=3`, `B=3`, `C=1`, `D=1`.
  - All eight materialized images exist under:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks/vstar_bench/snapshot/`.
- Benchmark output:
  - `outputs/clean_smokes/stage2_tgvf_force_vstar8_20260626_004135`
- Script / command:
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.benchmark --run-id clean_stage2_tgvf_force_vstar8_20260626_004135 --checkpoint-path outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617/train/processor_step_2000 --mode tgvf_force --post-tgvf-forward-mode kv_cache --tgvf-protocol protocol_c_tool_observation --population-id vstar_test_questions_191 --manifest-path revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_core_smoke_first_8_20260626.json --manifest-hash 3a6020b145c1543be3fc8a5541c17d00b4fc5d27aeddbecb90a10417d25c84f4 --benchmark-root /home/dredvpn009/Flash_Storage/datasets/benchmarks --output-dir outputs/clean_smokes/stage2_tgvf_force_vstar8_20260626_004135 --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 32 --execute --runner-backend tgvf_stage2_qwen3 --dtype bfloat16 --device cuda:0 --device-map cuda:0 --attn-implementation sdpa --stage2-checkpoint outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/checkpoint_step_1200.pt --stage2-eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl --stage2-d-condition correct_D --force-prefix-mode target_hint`
- GPUs:
  - Planned: GPU 0 only.
- tmux:
  - `clean_stage2_vstar8_20260626_004135`.
- Started:
  - 2026-06-26T00:44:03+09:00.
- Finished:
  - 2026-06-26T00:45:26+09:00.
- Metrics:
  - `n_rows=8`.
  - `n_scored=8`.
  - `accuracy=0.625`.
  - `answer_parse_rate=1.0`.
  - `trigger_rate=1.0`.
  - `focus_valid_rate=1.0`.
  - `append_success_rate=1.0`.
  - `malformed_rate=0.0`.
  - Scores by row:
    `[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]`.
  - Gold labels:
    `[A, B, C, D, A, B, A, B]`.
  - Parsed labels:
    `[A, B, C, A, B, A, A, B]`.
  - Wrong sample ids:
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/9_000009`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/145_000145`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/162_000162`
  - `target_answer_leakage_flag=true` on 8/8 legacy debug rows.
- Analysis:
  - The clean Stage2 bridge remained stable across all 8 rows: no errors, no malformed rows, all focus captures valid, and all D appends succeeded.
  - `run_config.txt`, `run_config.json`, `sample_manifest.json`, `rows.jsonl`, and `summary.json` were written.
  - Runtime code identity was recorded in `run_config.txt` as commit `c753a7206f8848cd4901f4c94c83c508583f9654` with `dirty_worktree=False`.
  - Output path:
    `outputs/clean_smokes/stage2_tgvf_force_vstar8_20260626_004135`.
  - The 62.5% accuracy is a diagnostic row result only; the manifest is deliberately tiny and not representative.
  - The universal legacy `target_answer_leakage_flag` means this smoke should not be used for focus-target quality claims.
- Conclusion:
  - Clean Stage2 multi-row force smoke passed.
  - The next clean-project risk is no longer basic Stage2 bridge executability; it is scorer parity and broader benchmark identity.
- Comparable to baseline:
  - No; diagnostic multi-row launch smoke only.
- Follow-up:
  - If stable, port/validate a native clean scorer wrapper or a larger CoreSmoke path-backed subset next.

### EXP-20260626-010550-clean-stage2-vstar32-smoke

- Status: DONE.
- Question:
  - Does the clean `tgvf_stage2_qwen3` bridge remain stable on the full VStar allocation from `CoreSmoke-256`, before moving toward larger clean benchmark subsets?
- Baseline anchor:
  - Follows `EXP-20260626-004135-clean-stage2-vstar8-smoke`.
  - Uses the same historical checkpoint as `BASE-20260617-rowonly-original-stage2`.
- Intended diff:
  - Expand the diagnostic path-backed VStar manifest from 8 rows to 32 rows.
  - Keep mode, checkpoint, processor, protocol, max image resolution, DeepStack state, scoring backend, and continuation fixed.
- Allowed changed variables:
  - Manifest/sample count only.
- Not allowed to change:
  - Stage2 checkpoint.
  - Stage1 processor recorded in checkpoint.
  - TGVF protocol.
  - Mode: `tgvf_force`.
  - Max image resolution 512.
  - DeepStack state: clean default off.
  - Scoring backend: `auto`.
  - Post-TGVF continuation: `natural_continue`.
  - Stage2 runtime validation jsonl.
- Code commit / worktree:
  - Planned from branch `clean/tgvf-clean-project-20260625`.
  - Pre-ledger code commit: `74707e4dc07385009b47a0168215e139f091b7be`.
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
    `revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_core_smoke_32_20260626.json`
  - Manifest hash:
    `d18563b8d2c1392295e80f7e3a8c4453cb7f725aebbba9b714fa58822be8ace7`
  - Selection rule:
    all 32 VStar samples from committed `core_smoke_256_seed20260625`.
  - Sample count:
    `32`.
  - Categories:
    `direct_attributes=22`, `relative_position=10`.
  - Labels:
    `A=11`, `B=11`, `C=5`, `D=5`.
  - Materialize/render/dry execute preflight:
    32 rows, all images exist, no `Choices:` duplication, dry accuracy 1.0.
- Benchmark output:
  - `outputs/clean_smokes/stage2_tgvf_force_vstar32_20260626_010550`
- Script / command:
  - `CUDA_VISIBLE_DEVICES=4 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.benchmark --run-id clean_stage2_tgvf_force_vstar32_20260626_010550 --checkpoint-path outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617/train/processor_step_2000 --mode tgvf_force --post-tgvf-forward-mode kv_cache --tgvf-protocol protocol_c_tool_observation --population-id vstar_test_questions_191 --manifest-path revisit_vlm_clean/benchmark_manifests/diagnostic_vstar_core_smoke_32_20260626.json --manifest-hash d18563b8d2c1392295e80f7e3a8c4453cb7f725aebbba9b714fa58822be8ace7 --benchmark-root /home/dredvpn009/Flash_Storage/datasets/benchmarks --output-dir outputs/clean_smokes/stage2_tgvf_force_vstar32_20260626_010550 --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 32 --execute --runner-backend tgvf_stage2_qwen3 --dtype bfloat16 --device cuda:0 --device-map cuda:0 --attn-implementation sdpa --stage2-checkpoint outputs/tgvf_v3_protocol_c/protocol_c_toolobs_stage2_v4data_clean_rowonly_gpu0_3_multifocus_focus_imend_from_clean_rowonly_focus_imend_stage1_bidirectional_4gpu_bs16_accum2_focus80_value1_1200step_20260617/checkpoint_step_1200.pt --stage2-eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl --stage2-d-condition correct_D --force-prefix-mode target_hint`
- GPUs:
  - Planned: GPU 4 only.
- tmux:
  - `clean_stage2_vstar32_20260626_010550`.
- Started:
  - 2026-06-26T01:07:55+09:00.
- Finished:
  - 2026-06-26T01:10:14+09:00.
- Metrics:
  - `n_rows=32`.
  - `n_scored=32`.
  - `accuracy=0.4375`.
  - `answer_parse_rate=1.0`.
  - `trigger_rate=1.0`.
  - `focus_valid_rate=1.0`.
  - `append_success_rate=1.0`.
  - `malformed_rate=0.0`.
  - Correct rows:
    `14/32`.
  - Wrong rows:
    `18/32`.
  - `target_answer_leakage_flag=true` on 30/32 legacy debug rows.
  - Wrong sample ids:
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/9_000009`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/145_000145`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/162_000162`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/12_000012`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/63_000063`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/155_000155`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/51_000051`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/41_000041`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/110_000110`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/168_000168`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/81_000081`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/18_000018`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/190_000190`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/80_000080`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/78_000078`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/15_000015`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/164_000164`
    - `vstar_test_questions_191/vstar_bench_snapshot_test_questions_jsonl/48_000048`
- Analysis:
  - The clean Stage2 bridge remained stable across all 32 rows: no errors, no malformed rows, all focus captures valid, and all D appends succeeded.
  - `run_config.txt`, `run_config.json`, `sample_manifest.json`, `rows.jsonl`, and `summary.json` were written.
  - Runtime code identity was recorded in `run_config.txt` as commit `25e239e68d441f1ea56ac78a4db8d77ea3765dec` with `dirty_worktree=False`.
  - Output path:
    `outputs/clean_smokes/stage2_tgvf_force_vstar32_20260626_010550`.
  - The 43.75% accuracy is a diagnostic row result only; this manifest is the VStar slice of CoreSmoke and is not a benchmark table.
  - The mostly universal legacy `target_answer_leakage_flag` means this smoke should not be used for focus-target quality claims.
- Conclusion:
  - Clean Stage2 larger path-backed force smoke passed.
  - The basic bridge now looks stable on the full VStar allocation of CoreSmoke; next work should move to remaining official scorer wrappers or explicit clean benchmark subset boundaries.
- Comparable to baseline:
  - No; diagnostic larger path-backed launch smoke only.
- Follow-up:
  - If stable, either port another official scorer wrapper or run clean CoreSmoke path-backed slices with explicit benchmark constraints.

### EXP-20260626-153326-clean-qwen3-deepstack-mask075-micro-smoke

- Status: DONE.
- Question:
  - What Stage1/Stage2 per-GPU micro-batch settings are feasible for the clean
    Qwen3 mainline before launching the new Stage1 -> Stage2 training run?
- Baseline anchor:
  - New clean training smoke. Uses clean project entrypoints and current branch
    state, not a benchmark-quality comparison.
- Intended diff:
  - Smoke only: run bounded clean optimizer-step probes instead of full
    2000/1200-step training.
  - Stage2 planned mainline uses `mask_original_image_after_tgvf_prob=0.75`
    and enabled DeepStack with `through_answer` scope.
  - Stage2 memory probe uses the 20260619 Stage1 checkpoint as a structural
    stand-in; final Stage2 training must bind the new Stage1 checkpoint from
    this training chain.
- Allowed changed variables:
  - `micro_batch_size` and corresponding `gradient_accumulation_steps`.
  - Probe output directories.
  - Stage2 follow-up candidates may increase to `micro_batch_size=8` and
    `micro_batch_size=16` while holding `global_batch_size=128` fixed, because
    `micro_batch_size=4` is only a conservative starting point.
- Not allowed to change:
  - Model family: Qwen3-VL-8B-Thinking.
  - Protocol: `protocol_c_tool_observation`.
  - Max image resolution: 512.
  - Stage1 global batch: 32.
  - Stage2 global batch: 128.
  - Stage2 mask scope: `through_answer`.
  - Stage2 mask probability: 0.75.
  - Stage2 DeepStack enabled/scope: enabled, `through_answer`.
- Code commit / worktree:
  - Planned on branch `clean/tgvf-clean-project-20260625`.
  - Plan artifact commit: `bf0c9ca38b29afae1bf918fb8c328113e1a784a3`.
  - RUNNING ledger commit: `d5ebe56`.
  - Smoke outputs are untracked experiment artifacts under `outputs/`.
- Stage1 checkpoint:
  - None for Stage1 smoke; Stage1 starts from base model.
- Stage1 processor:
  - Default processor for `Qwen/Qwen3-VL-8B-Thinking`.
- Stage2 checkpoint/output:
  - Stage2 smoke structural Stage1 checkpoint:
    `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_imend_bidirectional_2gpu_bs4_accum4_gbs32_2000step_20260619_014148/train/checkpoint_step_2000.pt`.
  - Stage2 smoke structural Stage1 processor:
    `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_imend_bidirectional_2gpu_bs4_accum4_gbs32_2000step_20260619_014148/train/processor_step_2000`.
- Train data:
  - Stage1:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl`.
  - Stage2:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl`.
- Validation data:
  - Stage1:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl`.
  - Stage2:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`.
- Benchmark output:
  - None. This is a training memory smoke only.
- Script / command:
  - Generate clean Stage1 plans under
    `outputs/clean_training_smoke/qwen3_deepstack_mask075_micro_probe_20260626_153326/stage1_*`.
  - Generate clean Stage2 plans under
    `outputs/clean_training_smoke/qwen3_deepstack_mask075_micro_probe_20260626_153326/stage2_*`.
  - Run `tgvf_train_stage*_executor --prepare-execution --audit-runtime --audit-optimizer-step`.
- GPUs:
  - Smoke GPU: `0` unless occupied at launch.
  - Full training target remains unset pending smoke outcome.
- tmux:
  - None planned for direct bounded probes.
- Started:
  - 2026-06-26T15:43:54+09:00.
- Finished:
  - 2026-06-26T15:49:23+09:00.
- Metrics:
  - Stage1 `micro_batch_size=8`, `world_size=4`,
    `gradient_accumulation_steps=1`, `global_batch_size=32`: PASS.
    - Output:
      `outputs/clean_training_smoke/qwen3_deepstack_mask075_micro_probe_20260626_153326/stage1_micro8`.
    - Actual forward/backward/optimizer/scheduler step ran.
    - `sample_count=3`.
    - `loss_total=3.574305295944214`, finite.
    - Stage1 matrix CE + visual-token manifold loss were active.
    - Stage1 position ids and original-image key mask were applied.
  - Stage2 `micro_batch_size=4`, `world_size=4`,
    `gradient_accumulation_steps=8`, `global_batch_size=128`: PASS.
    - Output:
      `outputs/clean_training_smoke/qwen3_deepstack_mask075_micro_probe_20260626_153326/stage2_deepstack_micro4`.
    - Actual forward/backward/optimizer/scheduler step ran.
    - `sample_count=4`.
    - `loss_total=4.03125`, finite.
    - `deepstack_training_enabled=true`.
    - `qwen3_deepstack_features_injected=true`.
    - Observed focus mask active rate on this small batch: `1.0`.
  - Stage2 `micro_batch_size=8`, `world_size=4`,
    `gradient_accumulation_steps=4`, `global_batch_size=128`: PASS.
    - Output:
      `outputs/clean_training_smoke/qwen3_deepstack_mask075_micro_probe_20260626_153326/stage2_deepstack_micro8`.
    - Actual forward/backward/optimizer/scheduler step ran.
    - `sample_count=8`.
    - `loss_total=4.03125`, finite.
    - `deepstack_training_enabled=true`.
    - `qwen3_deepstack_features_injected=true`.
    - `mask_original_image_after_tgvf_prob=0.75`.
    - `mask_original_image_after_tgvf_scope=through_answer`.
    - Observed focus mask active rate on this small batch: `0.7142857142857143`.
    - Gradient tensors with grad: `532`; nonfinite grad tensors: `0`.
  - Stage2 `micro_batch_size=16`, `world_size=4`,
    `gradient_accumulation_steps=2`, `global_batch_size=128`: FAILED/OOM.
    - Output:
      `outputs/clean_training_smoke/qwen3_deepstack_mask075_micro_probe_20260626_153326/stage2_deepstack_micro16`.
    - Failure happened during Qwen3 language forward on the LoRA `o_proj`
      path.
    - GPU 0 had `178.36 GiB` total, process used about `178.27 GiB`, and a
      further `196 MiB` allocation failed.
- Analysis:
  - Stage1 has reached the largest legal per-GPU micro batch for the chosen
    4-GPU, `global_batch_size=32` identity: `4 * 8 * 1 = 32`.
  - Stage2 `micro_batch_size=16` is too close to the 178 GiB card limit and is
    not robust enough for full training, especially with variable image/token
    lengths.
  - Stage2 `micro_batch_size=8` keeps the intended global batch while cutting
    accumulation from 8 to 4 compared with the conservative `micro=4` probe.
  - DeepStack training support was exercised in the Stage2 probes:
    original-image DeepStack was injected, and the plan/audit record
    `through_answer` scope masking.
  - Stage2 plan still has `loss.visual_token_manifold=0.0`; this smoke did not
    change that policy. If the next official run should increase Stage2
    manifold loss, set it explicitly in the formal training ledger entry.
- Conclusion:
  - Recommended full-training batch settings:
    - Stage1: `world_size=4`, `micro_batch_size=8`,
      `gradient_accumulation_steps=1`, `global_batch_size=32`.
    - Stage2: `world_size=4`, `micro_batch_size=8`,
      `gradient_accumulation_steps=4`, `global_batch_size=128`.
  - Do not use Stage2 `micro_batch_size=16` for the formal run under current
    DeepStack + mask-prob 0.75 settings.
- Comparable to baseline:
  - No; memory/configuration smoke only.
- Follow-up:
  - Write the formal Stage1 -> Stage2 training ledger entry using these batch
    settings, W&B project `tgvf-clean-qwen3-deepstack`, and Stage2
    `mask_original_image_after_tgvf_prob=0.75` + DeepStack `through_answer`.
  - Before formal launch, decide whether Stage2
    `loss_visual_token_manifold` should remain `0.0` or be explicitly raised.

### EXP-20260626-155246-clean-qwen3-stage12-deepstack-mask075-4gpu

- Status: STAGE1_COMPLETED_STAGE2_PENDING.
- Question:
  - Train the clean Qwen3 Stage1 -> Stage2 mainline with the smoke-selected
    4-GPU batch settings, then use the resulting chain for later benchmark
    evaluation.
- Baseline anchor:
  - Batch/memory smoke:
    `EXP-20260626-153326-clean-qwen3-deepstack-mask075-micro-smoke`.
- Intended diff:
  - Full training run instead of bounded smoke probes.
  - Stage1 runs first from the base Qwen3 model.
  - Stage2 will be planned and launched only after Stage1 produces its real
    checkpoint/processor, so the Stage2 checkpoint identity is file-backed.
- Allowed changed variables:
  - Full training duration/cadence instead of smoke `max_steps=2`.
  - Output directories and W&B run identities.
- Not allowed to change:
  - Model family: Qwen3-VL-8B-Thinking.
  - Protocol: `protocol_c_tool_observation`.
  - Max image resolution: 512.
  - Stage1 global batch: 32.
  - Stage1 global batch remains 32. Initial `micro_batch_size=8`,
    `gradient_accumulation_steps=1` OOMed on the first real DDP batch, so the
    active fallback is `world_size=4`, `micro_batch_size=4`,
    `gradient_accumulation_steps=2`.
  - Stage2 global batch: 128.
  - Stage2 batch identity: `world_size=4`, `micro_batch_size=8`,
    `gradient_accumulation_steps=4`.
  - Stage2 mask scope: `through_answer`.
  - Stage2 mask probability: 0.75.
  - Stage2 DeepStack enabled/scope: enabled, `through_answer`.
  - Stage2 weighted span defaults:
    `evidence_state=0.2`, `focus_target=1.5`, `evidence=1.0`,
    `value_span=1.0`, `answer=1.0`, `no_focus_evidence_state=0.2`,
    `no_focus_answer=1.0`.
  - Stage2 visual-token manifold loss remains the clean default `0.0` for
    this launch; changing it requires a separate named ablation.
- Code commit / worktree:
  - Planned on branch `clean/tgvf-clean-project-20260625`.
  - Pre-ledger commit: `a5a12d3`.
  - Initial formal plan commit: `28f50dc`.
  - Micro4 fallback ledger commit: `d435f86`.
  - Progress/W&B logging patch commit: `527d57b`.
  - Legacy-aligned component/debug logging patch: `69b01e3`.
  - Worktree expected clean except untracked `logs/` and `third_party/`.
- Stage1 checkpoint:
  - None for launch; Stage1 starts from `Qwen/Qwen3-VL-8B-Thinking`.
- Stage1 processor:
  - Default processor for `Qwen/Qwen3-VL-8B-Thinking`.
- Stage2 checkpoint/output:
  - Stage2 will use the Stage1 checkpoint produced by this run.
  - Exact Stage1 checkpoint/processor path: TBD after Stage1 completion.
- Train data:
  - Stage1:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl`
    (`39998` rows).
  - Stage2:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl`
    (`46883` rows).
- Validation data:
  - Stage2:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`
    (`1002` rows).
- Benchmark output:
  - None in this launch entry. Benchmarks require a separate eval ledger entry
    after Stage2 completes.
- Output root:
  - `outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_155246`.
- Script / command:
  - Stage1 plan command:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.train_stage1 --run-id clean_qwen3_stage1_4gpu_m8a1_20260626_155246 --train-file data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl --output-dir outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_155246/stage1 --max-image-resolution 512 --max-steps 2000 --save-every 2000 --world-size 4 --micro-batch-size 8 --global-batch 32 --wandb-project tgvf-clean-qwen3-deepstack --wandb-mode online --write-plan`.
  - Stage1 launch command:
    `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --nproc-per-node 4 -m revisit_vlm_clean.training.stage1_executor --plan outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_155246/stage1/training_plan.json --launch-training`.
  - Stage1 micro4 fallback plan command:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.train_stage1 --run-id clean_qwen3_stage1_4gpu_m4a2_20260626_155809 --train-file data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl --output-dir outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_155246/stage1_micro4 --max-image-resolution 512 --max-steps 2000 --save-every 2000 --world-size 4 --micro-batch-size 4 --global-batch 32 --wandb-project tgvf-clean-qwen3-deepstack --wandb-mode online --write-plan`.
  - Stage1 micro4 fallback launch command:
    `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --nproc-per-node 4 -m revisit_vlm_clean.training.stage1_executor --plan outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_155246/stage1_micro4/training_plan.json --launch-training`.
  - Stage1 micro4 relaunch plan command:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.train_stage1 --run-id clean_qwen3_stage1_4gpu_m4a2_wandb_20260626_170602 --train-file data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl --output-dir outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_170602/stage1_micro4 --max-image-resolution 512 --max-steps 2000 --save-every 2000 --world-size 4 --micro-batch-size 4 --global-batch 32 --wandb-project tgvf-clean-qwen3-deepstack --wandb-mode online --write-plan`.
  - Stage1 micro4 relaunch command:
    `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --nproc-per-node 4 -m revisit_vlm_clean.training.stage1_executor --plan outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_170602/stage1_micro4/training_plan.json --launch-training`.
  - Stage1 micro4 legacy-logging relaunch plan command:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.train_stage1 --run-id clean_qwen3_stage1_4gpu_m4a2_legacylog_20260626_172639 --train-file data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl --output-dir outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_172639/stage1_micro4 --max-image-resolution 512 --max-steps 2000 --save-every 2000 --world-size 4 --micro-batch-size 4 --global-batch 32 --wandb-project tgvf-clean-qwen3-deepstack --wandb-mode online --write-plan`.
  - Stage1 micro4 legacy-logging relaunch command:
    `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --nproc-per-node 4 -m revisit_vlm_clean.training.stage1_executor --plan outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_172639/stage1_micro4/training_plan.json --launch-training`.
  - Stage2 plan/launch command:
    TBD after Stage1 checkpoint exists.
- GPUs:
  - Stage1 planned: GPUs `0,1,2,3`.
  - Stage2 planned: GPUs `0,1,2,3`, after Stage1 finishes.
- tmux:
  - Failed initial Stage1 session:
    `clean_stage1_qwen3_mask075_4gpu_20260626_155246`.
  - Interrupted Stage1 fallback session:
    `clean_stage1_qwen3_mask075_4gpu_m4_20260626_155809`.
  - Stopped Stage1 relaunch session with incomplete component logging:
    `clean_stage1_qwen3_m4_wandb_20260626_170602`.
  - Active Stage1 legacy-logging relaunch session:
    `clean_stage1_qwen3_m4_legacylog_20260626_172639`.
- Started:
  - Initial Stage1 micro8 launch: 2026-06-26T15:55:33+09:00.
  - Stage1 micro4 fallback launch: 2026-06-26T15:59:57+09:00.
  - Stage1 micro4 relaunch with progress/W&B patch:
    2026-06-26T17:08:16+09:00.
  - Stage1 micro4 legacy-logging relaunch:
    2026-06-26T17:28:03+09:00.
- Finished:
  - Paused/interrupted at 2026-06-26T16:53:33+09:00 before checkpoint
    completion, to patch missing clean-native progress/W&B logging.
  - Stopped at 2026-06-26T17:25:03+09:00 before checkpoint completion,
    because clean progress/W&B logging recorded only `loss_total` plus optimizer
    state and did not inherit the legacy Stage1/Stage2 component/debug metric
    surface.
  - Stage1 micro4 legacy-logging relaunch completed at about
    2026-06-26T19:01+09:00.
- Metrics:
  - Initial Stage1 micro8 attempt: FAILED/OOM.
    - Log:
      `logs/clean_training/clean_stage1_qwen3_mask075_4gpu_20260626_155246.log`.
    - Failure: rank 2 root failure; rank 3 also reported CUDA OOM in Qwen3
      language-model MLP forward. GPU memory was effectively full
      (`178.29 GiB` in use on a `178.36 GiB` card).
  - Stage1 micro4 fallback: INTERRUPTED_FOR_CODE_FIX.
    - Log:
      `logs/clean_training/clean_stage1_qwen3_mask075_4gpu_m4_20260626_155809.log`.
    - As of 2026-06-26T16:04:04+09:00, tmux session is alive and all four
      worker processes are running on GPUs `0,1,2,3`.
    - Observed memory is about `80-84 GiB` per GPU.
    - GPU utilization is bursty rather than continuous, consistent with
      first-batch/batch-prep and per-rank length imbalance under non-bucketed
      Stage1 data.
    - This attempt did not produce a completed Stage1 checkpoint and must not
      be used as a Stage1 result.
  - Stage1 micro4 relaunch with progress/W&B patch:
    STOPPED_FOR_INCOMPLETE_LEGACY_LOGGING.
    - Output:
      `outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_170602/stage1_micro4`.
    - Log:
      `logs/clean_training/clean_stage1_qwen3_m4_wandb_20260626_170602.log`.
    - Expected progress file:
      `outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_170602/stage1_micro4/clean_training_execution/training_progress.jsonl`.
    - W&B:
      `https://wandb.ai/mio_nora/tgvf-clean-qwen3-deepstack/runs/mohvyqge`.
    - As of startup verification, progress logging reached `step=2/2000`
      with finite losses and `wandb_enabled=True`.
    - Last observed progress before stop: about `step=48/2000`.
    - This run is invalid for training diagnostics because it lacks Stage1
      component losses (`loss_gen`, `loss_visual_token_manifold`,
      `loss_same_image_negative`) and legacy debug metrics.
  - Stage1 micro4 legacy-logging relaunch: VERIFIED_RUNNING.
    - Output:
      `outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_172639/stage1_micro4`.
    - Plan:
      `outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_172639/stage1_micro4/training_plan.json`.
    - Log:
      `logs/clean_training/clean_stage1_qwen3_m4_legacylog_20260626_172639.log`.
    - Expected progress file:
      `outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_172639/stage1_micro4/clean_training_execution/training_progress.jsonl`.
    - W&B:
      `https://wandb.ai/mio_nora/tgvf-clean-qwen3-deepstack/runs/0yc1jt1f`.
    - Code commit embedded in plan: `69b01e3c46dc866130b69691aa2393d60d3dd19c`.
    - Plan dirty worktree: `False`.
    - Verification at about 2026-06-26T17:29+09:00:
      - tmux session alive.
      - progress reached at least `step=5/2000`.
      - latest checked scalar losses include
        `loss_total=3.743674874305725`, `loss_gen=2.21875`,
        `loss_same_image_negative=1.23046875`,
        `loss_visual_token_manifold=2.9445611238479614`.
      - latest checked legacy debug fields include `finite_rate=1.0`,
        `position_mode=native_source_grid`,
        `attention_mask_mode=weak_strict_original_image_keys_4d`,
        `visual_token_manifold_active=True`, `source_visual_token_count=234`,
        and `answer_token_count=15`.
  - Stage1 micro4 legacy-logging relaunch: COMPLETED.
    - Final checkpoint:
      `outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_172639/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt`.
    - Final checkpoint sha256:
      `1257d2fb21faa32937e5bc460b1ebcc250f27a7a64f4b3435210efd0dcba3d44`.
    - Final checkpoint size: `108317021` bytes.
    - Training status: `clean_distributed_training_completed`.
    - Final observed Stage1 losses at step 2000:
      `loss_total=2.0244152545928955`, `loss_gen=1.26953125`,
      `loss_same_image_negative=0.69140625`,
      `loss_visual_token_manifold=0.6347774565219879`.
- Analysis:
  - The single-process smoke under-sampled long first-batch examples; real DDP
    `micro_batch_size=8` is not robust.
  - Fallback keeps Stage1 global batch fixed: `4 * 4 * 2 = 32`.
  - Clean-native trainer launch initially recorded `wandb.mode=online` in the
    plan but did not actually call `wandb.init/log`; it also wrote step runtime
    only at the end. This observability gap is being patched before relaunch.
  - The first patch was still too narrow: it proved liveness/W&B upload but
    omitted legacy training diagnostics. Legacy Stage1 logs component losses,
    grad norm, peak memory, shape, attention, norm, mask/readout/position, and
    finite-rate diagnostics. Legacy Stage2 logs component losses, focus/no-focus
    counts, mask-active rates, value-span match rate, protocol boundary stats,
    grad norm, learning rates, peak memory, and batch identity.
- Conclusion:
  - Stage1 completed successfully under the legacy-aligned clean logging patch.
  - Stage2 remains pending and must bind the exact Stage1 checkpoint above.
- Comparable to baseline:
  - No; this is the formal training chain, not a benchmark result.
- Follow-up:
  - Run clean Stage1 internal diagnostics on the final checkpoint before Stage2.
  - After Stage1 completes, bind its exact checkpoint/processor and launch
    Stage2 with the recorded DeepStack/mask/batch settings.

### DIAG-20260626-clean-qwen3-stage1-internal-diagnostics

- Status: DONE.
- Question:
  - Does the clean Stage1 checkpoint pass the clean-native internal
    readout/query/distribution diagnostics before Stage2 launch?
- Parent training entry:
  - `EXP-20260626-155246-clean-qwen3-stage12-deepstack-mask075-4gpu`.
- Checkpoint:
  - `outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_172639/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt`.
  - sha256:
    `1257d2fb21faa32937e5bc460b1ebcc250f27a7a64f4b3435210efd0dcba3d44`.
- Eval data:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl`.
  - Rows: `867`.
- Code / worktree:
  - Code used by dry-run: `578aa6c make clean stage diagnostics native`.
  - Code used by execution plan: `d2658ef record clean stage1 diagnostics preflight`.
  - Worktree clean except untracked `logs/` and `third_party/`.
- Diagnostic entrypoint:
  - `python -m revisit_vlm_clean.cli.stage_diagnostics`.
  - Backend: `clean_native_stage_diagnostics`.
  - Eval family: `internal_diagnostic`.
  - Legacy bridge: `False`.
- Settings:
  - Stage: `stage1`.
  - Protocol: `protocol_c_tool_observation`.
  - Model: `Qwen/Qwen3-VL-8B-Thinking`.
  - Processor: checkpoint config / base processor when missing.
  - Max image resolution: `512`.
  - Position mode: `native_source_grid`.
  - Focus action im_end: `True`.
  - Tasks: `readout,query,distribution`.
  - D conditions:
    `correct_D`, `no_D`, `random_D`, `wrong_same_image_D`,
    `wrong_diff_image_D`.
  - Readout max samples: `200`.
  - Query max groups: `50`.
  - Query min targets per image: `3`.
  - Distribution max samples: `200`.
  - Device: `cuda:0`.
- Output:
  - `outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_172639/stage1_micro4/internal_diagnostics_step2000`.
- Command:
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.stage_diagnostics --run-id clean_qwen3_stage1_internal_diag_20260626_1901 --stage stage1 --checkpoint outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_172639/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt --eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl --output-dir outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_172639/stage1_micro4/internal_diagnostics_step2000 --max-image-resolution 512 --tasks all --readout-max-samples 200 --distribution-max-samples 200 --query-max-groups 50 --query-require-groups 0 --execute`.
- Verification before launch:
  - `PYTHONPATH=revisit_vlm_clean/src:src pytest -q revisit_vlm_clean/tests/test_stage_diagnostics.py`
    passed: `4 passed`.
  - Dry-run resolved `tasks=["readout","query","distribution"]` and expected
    report paths under the output directory above.
- Runtime:
  - tmux: `clean_stage1_internal_diag_20260626_1901`.
  - Started: 2026-06-26T19:12:43+09:00.
  - Completed: 2026-06-26T19:21:37+09:00.
  - Elapsed: about 9 minutes on one GPU.
  - Note: current clean diagnostics are single-process/single-device; the
    `eval_workers` argument is not yet a multi-GPU shard runner.
- Reports:
  - Readout:
    `outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_172639/stage1_micro4/internal_diagnostics_step2000/readout/readout_eval_report.json`.
  - Query:
    `outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_172639/stage1_micro4/internal_diagnostics_step2000/query_sensitivity/query_sensitivity_report.json`.
  - Distribution:
    `outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_172639/stage1_micro4/internal_diagnostics_step2000/fvt_distribution/fvt_distribution_report.json`.
- Metrics:
  - Samples/items built: `200`.
  - Readout:
    - `mean_nll_correct_D=1.64203125`.
    - `mean_nll_target_only=2.02021484375`.
    - `mean_nll_random_D=2.33296875`.
    - `mean_delta_correct_vs_target_only=0.37818359375`.
    - `mean_delta_correct_vs_random=0.6909375`.
    - `mean_delta_correct_vs_wrong_same=0.079609375`.
    - `mean_delta_correct_vs_wrong_diff=0.1239013671875`.
    - `pct_correct_D_beats_target_only=0.985`.
    - `pct_correct_D_beats_random=1.0`.
    - `pct_correct_D_beats_wrong_same=0.695`.
    - `pct_correct_D_beats_wrong_diff=0.875`.
  - Query sensitivity:
    - `num_groups_evaluated=46`.
    - `num_items_evaluated=200`.
    - `retrieval_top1=0.315`.
    - `retrieval_top2=0.55`.
    - `mrr=0.5574999999999999`.
    - `mean_diagonal_gap=-0.050205078125`.
    - `median_diagonal_gap=-0.0234375`.
  - FVT distribution:
    - `finite_rate=1.0`.
    - `manifold_active_rate=1.0`.
    - `avg_manifold_loss=0.5868046700954437`.
    - `median_manifold_loss=0.582190603017807`.
    - `avg_norm_D=51.86518354415894`.
    - `avg_norm_V_merge=20.683768496513366`.
    - `norm_ratio_D_to_Vmerge=2.5958012759847082`.
    - `collapse_near_identical_rate=0.0`.
    - `collapse_warning=False`.
- Analysis:
  - Stage1 readout signal is strong: correct D lowers evidence NLL versus no D
    on 98.5% of sampled items and versus random D on 100% of items.
  - Correct D also beats wrong same-image D on 69.5% and wrong different-image D
    on 87.5%, so D carries target-specific information, though same-image
    target specificity is weaker than the no-D/random comparisons.
  - Query retrieval is not yet strong (`top1=31.5%`, `mrr=0.5575`), matching the
    weaker same-image contrast signal.
  - FVT vectors are finite and non-collapsed, but D remains larger norm than
    V_merge (`2.60x` average), which should stay visible in later Stage2
    diagnostics.
- Follow-up:
  - Implement or test a multi-GPU sharded clean diagnostics runner; single-GPU
    200-item diagnostics took about 9 minutes.
  - Stage2 can proceed, but should bind this exact Stage1 checkpoint and keep
    Stage2 diagnostics/readout checks enabled.

### CODE-20260626-clean-stage1-same-image-sampler-fix

- Status: DONE.
- Timestamp: 2026-06-26T19:48:10+09:00.
- Question:
  - Why did the clean Stage1 run keep `loss_same_image_negative` near
    `ln(2)` even after 2000 steps, while legacy Stage1 runs reduced the same
    loss far below that range?
- Affected completed run:
  - `EXP-20260626-155246-clean-qwen3-stage12-deepstack-mask075-4gpu`.
  - Final checkpoint:
    `outputs/clean_training/qwen3_stage12_deepstack_mask075_4gpu_20260626_172639/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt`.
  - Final observed clean Stage1 `loss_same_image_negative=0.69140625`.
- Root cause:
  - The clean Stage1 single-process sample cursor used modulo fill for
    same-image batches. With `micro_batch_size=4`, an image group containing
    only two samples could become `[A,B,A,B]`.
  - Matrix CE then saw duplicate positive embeddings in different columns while
    only one column was the label. This creates an artificial lower bound near
    `ln(2)=0.693`, matching the observed clean Stage1 loss floor.
  - Legacy Stage1 assigns whole image groups to DDP ranks and drops incomplete
    same-image groups instead of duplicating samples within a batch.
- Change:
  - Clean Stage1 same-image grouping now keeps whole image groups together per
    rank using `sha256(image_key) % world_size`.
  - Incomplete same-image groups smaller than the Stage1 micro-batch are
    dropped.
  - Same-image batch construction no longer uses modulo duplicate fill; it
    raises if a complete unique batch cannot be produced.
  - Cursor summaries now record `same_image_drop_incomplete` and
    `same_image_group_owner` for audit visibility.
- Files:
  - `revisit_vlm_clean/src/revisit_vlm_clean/training/executor.py`.
  - `revisit_vlm_clean/tests/test_cli.py`.
- Verification:
  - `PYTHONPATH=revisit_vlm_clean/src:src python -m py_compile revisit_vlm_clean/src/revisit_vlm_clean/training/executor.py`.
  - `PYTHONPATH=revisit_vlm_clean/src:src pytest -q revisit_vlm_clean/tests/test_cli.py::test_stage1_same_image_cursor_drops_incomplete_groups_without_duplicate_fill revisit_vlm_clean/tests/test_cli.py::test_stage1_same_image_cursor_assigns_whole_image_groups_to_rank revisit_vlm_clean/tests/test_cli.py::test_stage1_training_executor_preflight_cli revisit_vlm_clean/tests/test_cli.py::test_stage1_training_executor_prepare_execution_cli revisit_vlm_clean/tests/test_cli.py::test_stage1_training_executor_runtime_audit_can_write_training_step_probe`.
  - Result: `5 passed in 2.33s`.
  - Real Stage1 train grouping check for
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl`
    under `world_size=4`, `micro_batch_size=4`:
    - Rank 0: `2061` usable same-image groups, `2061` full micro-batches.
    - Rank 1: `2062` usable same-image groups, `2062` full micro-batches.
    - Rank 2: `2072` usable same-image groups, `2072` full micro-batches.
    - Rank 3: `2014` usable same-image groups, `2014` full micro-batches.
    - Incomplete groups are present but safely dropped; all ranks have enough
      complete same-image groups for training.
- Conclusion:
  - The completed clean Stage1 checkpoint above is useful as a diagnostic
    artifact, but it should not be used as the Stage1 parent for the next clean
    Stage2 run.
  - The next valid experiment should rerun Stage1 after this sampler fix before
    evaluating manifold-weight or Stage2 changes.

### EXP-20260626-195348-clean-qwen3-stage1-samplerfix-4gpu

- Status: DONE.
- Question:
  - After fixing the clean Stage1 same-image sampler duplicate-fill bug, does
    Stage1 matrix CE escape the previous `ln(2)` floor while holding the
    previous formal Stage1 configuration fixed?
- Baseline / invalid parent:
  - `EXP-20260626-155246-clean-qwen3-stage12-deepstack-mask075-4gpu`.
  - Its completed Stage1 checkpoint is diagnostic-only because
    `loss_same_image_negative=0.69140625` matched the duplicate-positive
    `ln(2)` failure mode.
- Intended diff:
  - Code only: use commit `ed323848446deb8e858e3b3ae4770632cdc52ebd`,
    which includes `CODE-20260626-clean-stage1-same-image-sampler-fix`.
  - Keep model, data, seed, max resolution, batch, optimizer, loss weights,
    token row mode, position mode, and Stage1 mask behavior fixed.
- Code / worktree:
  - Branch: `clean/tgvf-clean-project-20260625`.
  - Commit: `ed323848446deb8e858e3b3ae4770632cdc52ebd`.
  - Plan dirty worktree: `False`.
  - Untracked local paths remain ignored for this run: `logs/`, `third_party/`.
- Plan:
  - `outputs/clean_training/qwen3_stage12_deepstack_mask075_samplerfix_4gpu_20260626_195348/stage1_micro4/training_plan.json`.
  - Plan sha256:
    `8858c4379dceacd7ead423381100165a5659cde0ad8bf6f36786a8d7b66c9d18`.
- Data:
  - Train file:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl`.
  - Rows: `39998`.
  - sha256:
    `c94a38b824b6603e555eed5ef3584c19cc903b76995d49c67ace36b18268443c`.
  - Runtime summary: all rows are focus rows; answer formats are
    `32203` multiple-choice and `7795` open.
- Model / processor:
  - Model: `Qwen/Qwen3-VL-8B-Thinking`.
  - Processor: model default / checkpoint config not applicable for Stage1
    from base model.
  - dtype: `bfloat16`.
  - Attention implementation: `sdpa`.
- Stage1 settings:
  - Protocol: `protocol_c_tool_observation`.
  - Variant: `tgvf_v2_bidirectional`.
  - Max image resolution: `512`.
  - Token row mode: `row_only`.
  - Capture mode: `teacher_forced`.
  - FVT position mode: `native_source_grid`.
  - Focus action im_end: `True`.
  - Mask original image after TGVF: `True`.
  - Readout context:
    - D append path: `native_qwen_visual_span`.
    - Original image placeholders: `replace_with_qwen_v_merge`.
    - Position ids: `real_qwen3_mrope_full_trajectory`.
    - Original image key blocking:
      `weak_strict_original_image_key_blocking_after_tgvf_append`.
  - Note: Stage1 clean planner has no DeepStack training toggle; DeepStack is a
    Stage2/eval first-class variable in the current clean interface.
- Training:
  - GPUs: `0,1,2,3`.
  - world size: `4`.
  - micro batch: `4`.
  - gradient accumulation: `2`.
  - global batch: `32`.
  - max steps: `2000`.
  - save every: `2000`.
  - seed: `20260525`.
  - optimizer: AdamW, learning rate `1e-4`, cosine schedule, warmup `100`,
    min LR ratio `0.1`, max grad norm `1.0`.
  - Loss weights:
    - `loss_gen=1.0`.
    - `loss_visual_token_manifold=0.1`.
    - `loss_same_image_negative=1.0`.
    - same-image negative mode: `matrix_ce`.
- W&B:
  - Project: `tgvf-clean-qwen3-deepstack`.
  - Mode: `online`.
- Preflight:
  - `stage1_executor --preflight-only`: passed; no blocking items.
  - `stage1_executor --prepare-execution`: passed; runner status
    `ready_for_explicit_distributed_launch`.
  - First materialized batch identity has `32` rows and `32` unique row keys.
- Command:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --nproc-per-node 4 -m revisit_vlm_clean.training.stage1_executor --plan outputs/clean_training/qwen3_stage12_deepstack_mask075_samplerfix_4gpu_20260626_195348/stage1_micro4/training_plan.json --launch-training`.
- Output:
  - `outputs/clean_training/qwen3_stage12_deepstack_mask075_samplerfix_4gpu_20260626_195348/stage1_micro4`.
- Launch:
  - Planned at 2026-06-26T19:54:43+09:00.
  - Started at 2026-06-26T19:55:27+09:00.
  - tmux: `clean_stage1_samplerfix_20260626_195348`.
  - Log:
    `logs/clean_training/clean_stage1_samplerfix_20260626_195348.log`.
  - W&B:
    `https://wandb.ai/mio_nora/tgvf-clean-qwen3-deepstack/runs/w77m0gy0`.
- Startup verification:
  - tmux session is alive.
  - GPUs `0,1,2,3` are allocated by the run.
  - `training_progress.jsonl` is being written.
  - W&B is online and syncing.
  - Training reached at least `step=2/2000`.
  - Early component losses:
    - step 1:
      `loss_total=3.614259123802185`,
      `loss_gen=2.0546875`,
      `loss_same_image_negative=1.3203125`,
      `loss_visual_token_manifold=2.392590880393982`,
      `grad_norm=5.625`.
    - step 2:
      `loss_total=3.957866668701172`,
      `loss_gen=2.328125`,
      `loss_same_image_negative=1.40234375`,
      `loss_visual_token_manifold=2.273978590965271`,
      `grad_norm=8.6875`.
  - Immediate interpretation:
    - The same-image loss is no longer pinned near `ln(2)` at startup, so the
      duplicate-fill lower-bound failure mode is not currently present.
- Finished:
  - Completed by 2026-06-26T21:51:48+09:00.
  - tmux session exited normally.
  - Training status: `clean_distributed_training_completed`.
  - Final checkpoint:
    `outputs/clean_training/qwen3_stage12_deepstack_mask075_samplerfix_4gpu_20260626_195348/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt`.
  - Final checkpoint sha256:
    `3fed9ec56335df1381772024f5469edaad8062a32000d04c0eec2587da2a7be4`.
  - Final checkpoint size: `108317021` bytes.
  - Runtime wrote `2002` progress records and completed `2000` optimizer
    steps / `4000` micro-steps.
- Final observed Stage1 losses near step 2000:
  - `loss_total=1.5284298658370972`.
  - `loss_gen=1.421875`.
  - `loss_same_image_negative=0.053466796875`.
  - `loss_visual_token_manifold=0.5308806300163269`.
  - `grad_norm=13.5`.
- Completion analysis:
  - The sampler fix resolved the hard matrix-CE floor: final
    `loss_same_image_negative=0.0535`, versus the previous invalid clean
    Stage1's `0.6914`.
  - The checkpoint is the current candidate Stage1 parent for subsequent
    Stage2, pending internal diagnostics below.

### DIAG-20260626-clean-qwen3-stage1-samplerfix-internal-diagnostics

- Status: DONE.
- Question:
  - Does the sampler-fixed clean Stage1 checkpoint pass the same internal
    readout/query/distribution diagnostics used for the previous clean Stage1
    checkpoint?
- Parent training entry:
  - `EXP-20260626-195348-clean-qwen3-stage1-samplerfix-4gpu`.
- Checkpoint:
  - `outputs/clean_training/qwen3_stage12_deepstack_mask075_samplerfix_4gpu_20260626_195348/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt`.
  - sha256:
    `3fed9ec56335df1381772024f5469edaad8062a32000d04c0eec2587da2a7be4`.
- Eval data:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl`.
- Code / worktree:
  - Git commit: `7cb7660d47aa6c4bb8207c9484c4b6412d5988a9`.
  - Worktree is dirty from unrelated Stage3/RL data-generation changes
    (`docs/TGVF_STAGE3_RL_WORKLOG.md`,
    `revisit_vlm_clean/src/revisit_vlm_clean/cli/generate_data.py`,
    and new Stage3/RL data files). These are recorded as dirty-worktree context
    and are not the intended diagnostic variable.
- Diagnostic entrypoint:
  - `python -m revisit_vlm_clean.cli.stage_diagnostics`.
  - Backend: `clean_native_stage_diagnostics`.
  - Eval family: `internal_diagnostic`.
  - Legacy bridge: `False`.
- Settings:
  - Stage: `stage1`.
  - Protocol: `protocol_c_tool_observation`.
  - Model: `Qwen/Qwen3-VL-8B-Thinking`.
  - Processor: checkpoint config / base processor when missing.
  - Max image resolution: `512`.
  - Position mode: `native_source_grid`.
  - Focus action im_end: `True`.
  - Tasks: `readout,query,distribution`.
  - D conditions:
    `correct_D`, `no_D`, `random_D`, `wrong_same_image_D`,
    `wrong_diff_image_D`.
  - Readout max samples: `200`.
  - Query max groups: `50`.
  - Query min targets per image: `3`.
  - Distribution max samples: `200`.
  - Device: `cuda:0`.
- Output:
  - `outputs/clean_training/qwen3_stage12_deepstack_mask075_samplerfix_4gpu_20260626_195348/stage1_micro4/internal_diagnostics_step2000`.
- Dry-run:
  - Completed at 2026-06-26T23:27:02+09:00.
  - Resolved tasks: `readout,query,distribution`.
  - Expected reports:
    - `readout/readout_eval_report.json`.
    - `query_sensitivity/query_sensitivity_report.json`.
    - `fvt_distribution/fvt_distribution_report.json`.
- Command:
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.stage_diagnostics --run-id clean_qwen3_stage1_samplerfix_internal_diag_20260626_2200 --stage stage1 --checkpoint outputs/clean_training/qwen3_stage12_deepstack_mask075_samplerfix_4gpu_20260626_195348/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt --eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl --output-dir outputs/clean_training/qwen3_stage12_deepstack_mask075_samplerfix_4gpu_20260626_195348/stage1_micro4/internal_diagnostics_step2000 --max-image-resolution 512 --tasks all --readout-max-samples 200 --distribution-max-samples 200 --query-max-groups 50 --query-require-groups 0 --execute`.
- Runtime:
  - tmux: `clean_stage1_samplerfix_diag_20260626_2327`.
  - Log:
    `logs/clean_training/clean_stage1_samplerfix_diag_20260626_2327.log`.
  - Started: 2026-06-26T23:27:57+09:00.
  - Completed: 2026-06-26T23:36:56+09:00.
  - Elapsed: about 9 minutes on one GPU.
- Reports:
  - Readout:
    `outputs/clean_training/qwen3_stage12_deepstack_mask075_samplerfix_4gpu_20260626_195348/stage1_micro4/internal_diagnostics_step2000/readout/readout_eval_report.json`.
  - Query:
    `outputs/clean_training/qwen3_stage12_deepstack_mask075_samplerfix_4gpu_20260626_195348/stage1_micro4/internal_diagnostics_step2000/query_sensitivity/query_sensitivity_report.json`.
  - Distribution:
    `outputs/clean_training/qwen3_stage12_deepstack_mask075_samplerfix_4gpu_20260626_195348/stage1_micro4/internal_diagnostics_step2000/fvt_distribution/fvt_distribution_report.json`.
- Metrics:
  - Samples/items built: `200`.
  - Readout:
    - `mean_nll_correct_D=1.656953125`.
    - `mean_nll_target_only=2.02021484375`.
    - `mean_nll_random_D=2.33296875`.
    - `mean_delta_correct_vs_target_only=0.36326171875`.
    - `mean_delta_correct_vs_random=0.676015625`.
    - `mean_delta_correct_vs_wrong_same=0.12716796875`.
    - `mean_delta_correct_vs_wrong_diff=0.1474609375`.
    - `pct_correct_D_beats_target_only=0.975`.
    - `pct_correct_D_beats_random=0.995`.
    - `pct_correct_D_beats_wrong_same=0.77`.
    - `pct_correct_D_beats_wrong_diff=0.875`.
  - Query sensitivity:
    - `num_groups_evaluated=46`.
    - `num_items_evaluated=200`.
    - `retrieval_top1=0.42`.
    - `retrieval_top2=0.635`.
    - `mrr=0.6312499999999998`.
    - `mean_diagonal_gap=-0.02484375`.
    - `median_diagonal_gap=-0.015625`.
  - FVT distribution:
    - `finite_rate=1.0`.
    - `manifold_active_rate=1.0`.
    - `avg_manifold_loss=0.4921232940256596`.
    - `median_manifold_loss=0.48194436728954315`.
    - `avg_norm_D=48.13942008972168`.
    - `avg_norm_V_merge=20.683768496513366`.
    - `norm_ratio_D_to_Vmerge=2.4010698315950645`.
    - `collapse_near_identical_rate=0.0`.
    - `collapse_warning=False`.
- Comparison with previous sampler-bug Stage1 diagnostic:
  - Stage1 training:
    - `loss_same_image_negative`: `0.69140625` -> `0.053466796875`.
  - Same-image readout specificity:
    - `pct_correct_D_beats_wrong_same`: `0.695` -> `0.77`.
    - `mean_delta_correct_vs_wrong_same`: `0.079609375` -> `0.12716796875`.
  - Query sensitivity:
    - `retrieval_top1`: `0.315` -> `0.42`.
    - `retrieval_top2`: `0.55` -> `0.635`.
    - `mrr`: `0.5575` -> `0.63125`.
    - `mean_diagonal_gap`: `-0.050205078125` -> `-0.02484375`.
  - Distribution:
    - `avg_manifold_loss`: `0.5868046700954437` ->
      `0.4921232940256596`.
    - `norm_ratio_D_to_Vmerge`: `2.5958012759847082` ->
      `2.4010698315950645`.
- Analysis:
  - The sampler fix materially improves the Stage1 target-specific signal:
    same-image discrimination and query retrieval both improved, which is the
    diagnostic surface most directly harmed by duplicate positives in matrix CE.
  - Broad readout versus target-only/random remains strong and comparable to
    the previous run. It is slightly lower on percentage terms
    (`target_only`: `0.985` -> `0.975`; `random`: `1.0` -> `0.995`), but the
    important same-image/query metrics improved.
  - D norm is still high relative to V_merge (`2.40x`), though better than the
    previous `2.60x`. This keeps the manifold/norm-loss follow-up active; the
    sampler bug was a real blocker but not the whole normalization story.
- Conclusion:
  - This checkpoint passes the current Stage1 internal diagnostic gate better
    than the sampler-bug checkpoint and is the preferred Stage1 parent for the
    next Stage2 run.

### EXP-20260627-000156-clean-qwen3-stage1-samplerfix-manifold001-4gpu

- Status: COMPLETED.
- Question:
  - Does restoring the historical Stage1 manifold weight
    `visual_token_manifold=0.01` recover the old same-image target specificity
    (`pct_correct_D_beats_wrong_same` around `0.9+`) while keeping the clean
    same-image sampler fix?
- Motivation:
  - The sampler-fixed clean Stage1 with `visual_token_manifold=0.1` resolved
    the matrix-CE duplicate-fill floor, but internal diagnostics remained weak:
    `pct_correct_D_beats_wrong_same=0.77`, `retrieval_top1=0.42`, and
    `norm_ratio_D_to_Vmerge=2.401`.
  - Historical Qwen3 Protocol-C Stage1 runs with the same train/test split and
    global batch used `visual_token_manifold=0.01` and achieved stronger
    target-specific diagnostics:
    - 20260617 row-only 4GPU:
      `pct_correct_D_beats_wrong_same=0.945`,
      `retrieval_top1=0.535`,
      `norm_ratio_D_to_Vmerge=5.174`.
    - 20260620 full_mask 2GPU:
      `pct_correct_D_beats_wrong_same=0.915`,
      `retrieval_top1=0.755`,
      `norm_ratio_D_to_Vmerge=4.983`.
  - Current hypothesis: raising the mean/std manifold term to `0.1` improved
    scale but over-regularized the target-specific D signal. A weaker manifold
    replay is needed before adding a separate explicit token-norm loss.
- Baselines:
  - Strong old same-split target-specific baseline:
    `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617`.
  - Current clean sampler-fixed `0.1` baseline:
    `EXP-20260626-195348-clean-qwen3-stage1-samplerfix-4gpu` and
    `DIAG-20260626-clean-qwen3-stage1-samplerfix-internal-diagnostics`.
- Intended diff:
  - Change only Stage1 loss weight:
    `loss_visual_token_manifold: 0.1 -> 0.01`.
  - Keep samplerfix code, data, model, seed, global batch, optimizer, mask,
    Protocol C, row-only token rows, and max image resolution fixed.
- Code / worktree:
  - Branch: `clean/tgvf-clean-project-20260625`.
  - Git commit: `da6def2d34ec35575009d83da7466a9587078e68`.
  - Training code includes samplerfix from commit
    `ed323848446deb8e858e3b3ae4770632cdc52ebd`.
  - Plan dirty worktree: `True`.
  - Dirty files are unrelated Stage3/RL work and are recorded as context:
    `docs/TGVF_STAGE3_RL_WORKLOG.md`,
    `revisit_vlm_clean/src/revisit_vlm_clean/cli/generate_data.py`,
    `revisit_vlm_clean/README_STAGE3_RL_DATA.md`,
    `revisit_vlm_clean/src/revisit_vlm_clean/stage3_rl_data/`,
    `revisit_vlm_clean/tests/test_stage3_rl_data.py`,
    plus untracked `logs/` and `third_party/`.
  - Import check found no `generate_data` or `stage3_rl_data` imports from the
    Stage1 training/diagnostics modules used by this run.
- Plan:
  - `outputs/clean_training/qwen3_stage12_samplerfix_manifold001_4gpu_20260627_000156/stage1_micro4/training_plan.json`.
  - Plan sha256:
    `ec70b83436320b5a3d2ac571ed934d13b4f984c4bb6fd097a0a3f07462c89c7a`.
- Data:
  - Train file:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl`.
  - Rows: `39998`.
  - sha256:
    `c94a38b824b6603e555eed5ef3584c19cc903b76995d49c67ace36b18268443c`.
- Model / processor:
  - Model: `Qwen/Qwen3-VL-8B-Thinking`.
  - Processor: model default / checkpoint config not applicable for Stage1
    from base model.
  - dtype: `bfloat16`.
  - Attention implementation: `sdpa`.
- Stage1 settings:
  - Protocol: `protocol_c_tool_observation`.
  - Variant: `tgvf_v2_bidirectional`.
  - Max image resolution: `512`.
  - Token row mode: `row_only`.
  - Capture mode: `teacher_forced`.
  - FVT position mode: `native_source_grid`.
  - Focus action im_end: `True`.
  - Mask original image after TGVF: `True`.
  - Same-image negative mode: `matrix_ce`.
- Training:
  - GPUs: `0,1,2,3`.
  - world size: `4`.
  - micro batch: `4`.
  - gradient accumulation: `2`.
  - global batch: `32`.
  - max steps: `2000`.
  - save every: `2000`.
  - seed: `20260525`.
  - optimizer: AdamW, learning rate `1e-4`, cosine schedule, warmup `100`,
    min LR ratio `0.1`, max grad norm `1.0`.
  - Loss weights:
    - `loss_gen=1.0`.
    - `loss_same_image_negative=1.0`.
    - `loss_visual_token_manifold=0.01`.
- W&B:
  - Project: `tgvf-clean-qwen3-deepstack`.
  - Mode: `online`.
- Preflight:
  - `stage1_executor --preflight-only`: passed; no blocking items.
  - `stage1_executor --prepare-execution`: passed; runner status
    `ready_for_explicit_distributed_launch`.
  - First materialized batch identity has `32` rows and `32` unique row keys.
- Command:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --nproc-per-node 4 -m revisit_vlm_clean.training.stage1_executor --plan outputs/clean_training/qwen3_stage12_samplerfix_manifold001_4gpu_20260627_000156/stage1_micro4/training_plan.json --launch-training`.
- Output:
  - `outputs/clean_training/qwen3_stage12_samplerfix_manifold001_4gpu_20260627_000156/stage1_micro4`.
- Launch:
  - Planned at 2026-06-27T00:01:32+09:00.
  - Started at 2026-06-27T00:03:44+09:00.
  - tmux: `clean_stage1_manifold001_20260627_000156`.
  - Log:
    `logs/clean_training/clean_stage1_manifold001_20260627_000156.log`.
  - W&B:
    `https://wandb.ai/mio_nora/tgvf-clean-qwen3-deepstack/runs/eb1r1pcx`.
- Startup verification:
  - tmux session is alive.
  - GPUs `0,1,2,3` are allocated by the run.
  - `training_progress.jsonl` is being written.
  - W&B is online and syncing.
  - Training reached at least `step=3/2000`.
  - Early component losses:
    - step 1:
      `loss_total=3.3852850198745728`,
      `loss_gen=2.04296875`,
      `loss_same_image_negative=1.31640625`,
      `loss_visual_token_manifold=2.5910078287124634`,
      `grad_norm=5.15625`.
    - step 2:
      `loss_total=3.677285313606262`,
      `loss_gen=2.28125`,
      `loss_same_image_negative=1.37109375`,
      `loss_visual_token_manifold=2.4941558837890625`,
      `grad_norm=5.1875`.
    - step 3:
      `loss_total=3.5562121868133545`,
      `loss_gen=2.109375`,
      `loss_same_image_negative=1.421875`,
      `loss_visual_token_manifold=2.496214985847473`,
      `grad_norm=5.3125`.
  - Immediate interpretation:
    - Raw manifold loss is in the historical `~2.5` range, but this run uses
      the historical `0.01` coefficient rather than the clean `0.1`
      coefficient.
- Training completion:
  - Completed at 2026-06-27T01:58:38+09:00.
  - Runtime: `6862.8s` (`1h54m23s`).
  - Final checkpoint:
    `outputs/clean_training/qwen3_stage12_samplerfix_manifold001_4gpu_20260627_000156/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt`.
  - Checkpoint sha256:
    `d021f9e2a7e07b85db5f31a46209ae40c081cb13e0d9a5fe30c4ff58e5482928`.
  - Final step losses:
    - `loss_total=1.5070979595184326`.
    - `loss_gen=1.4609375`.
    - `loss_same_image_negative=0.0216217041015625`.
    - raw `loss_visual_token_manifold=2.453878164291382`.
    - weighted manifold contribution: `0.02453878164291382`.
  - Final logged norm diagnostics:
    - `norm_ratio_mean=5.008244037628174`.
    - `d_norm_mean=102.59529113769531`.
    - `v_merge_norm_mean=20.48528289794922`.
- Internal diagnostics:
  - Status: COMPLETED.
  - Run id:
    `clean_qwen3_stage1_manifold001_internal_diag_20260627_022927`.
  - Purpose:
    - Compare this `visual_token_manifold=0.01` checkpoint to the clean
      sampler-fixed `0.1` run and old Qwen3 Protocol-C Stage1 baselines on the
      same internal diagnostic surfaces.
  - Fixed identity:
    - Checkpoint:
      `outputs/clean_training/qwen3_stage12_samplerfix_manifold001_4gpu_20260627_000156/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt`.
    - Eval JSONL:
      `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl`.
    - Max image resolution: `512`.
    - Tasks: `readout`, `query`, `distribution`.
    - Sample caps: readout `200`, distribution `200`, query groups `50`.
  - Command:
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.stage_diagnostics --run-id clean_qwen3_stage1_manifold001_internal_diag_20260627_022927 --stage stage1 --checkpoint outputs/clean_training/qwen3_stage12_samplerfix_manifold001_4gpu_20260627_000156/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt --eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl --output-dir outputs/clean_training/qwen3_stage12_samplerfix_manifold001_4gpu_20260627_000156/stage1_micro4/internal_diagnostics_step2000 --max-image-resolution 512 --tasks all --readout-max-samples 200 --distribution-max-samples 200 --query-max-groups 50 --query-require-groups 0 --execute`.
  - Output:
    - `outputs/clean_training/qwen3_stage12_samplerfix_manifold001_4gpu_20260627_000156/stage1_micro4/internal_diagnostics_step2000`.
  - Reports:
    - `readout/readout_eval_report.json`.
    - `query_sensitivity/query_sensitivity_report.json`.
    - `fvt_distribution/fvt_distribution_report.json`.
  - Results:
    - Completed at 2026-06-27T02:39:32+09:00.
    - Readout, `n=200`:
      - `pct_correct_D_beats_target_only=0.9`.
      - `pct_correct_D_beats_wrong_same=0.785`.
      - `pct_correct_D_beats_wrong_diff=0.75`.
      - `pct_correct_D_beats_random=0.99`.
      - `mean_delta_correct_vs_target_only=0.423056640625`.
      - `mean_delta_correct_vs_wrong_same=0.269921875`.
    - Query sensitivity, `groups=46`, `items=200`:
      - `retrieval_top1=0.425`.
      - `retrieval_top2=0.61`.
      - `mrr=0.6259999999999999`.
      - `mean_diagonal_gap=-0.09201171875`.
    - FVT distribution, `n=200`:
      - `avg_manifold_loss=2.4048784774541856`.
      - `norm_ratio_D_to_Vmerge=5.21002431395354`.
      - `avg_norm_D=104.1043255996704`.
      - `avg_norm_V_merge=20.683768496513366`.
      - `finite_rate=1.0`.
      - `collapse_warning=false`.
  - Comparison:
    - Clean samplerfix `0.1` baseline:
      - `wrong_same=0.77`, `retrieval_top1=0.42`, `mrr=0.63125`,
        `norm_ratio=2.401`.
    - This clean samplerfix `0.01` replay:
      - `wrong_same=0.785`, `retrieval_top1=0.425`, `mrr=0.626`,
        `norm_ratio=5.210`.
    - Old 20260617 row-only 4GPU reference:
      - `wrong_same=0.945`, `retrieval_top1=0.535`,
        `norm_ratio=5.174`.
    - Old 20260620 full_mask 2GPU reference:
      - `wrong_same=0.915`, `retrieval_top1=0.755`,
        `norm_ratio=4.983`.
  - Analysis:
    - Restoring `visual_token_manifold=0.01` restored the D/V_merge scale to
      the historical range (`norm_ratio ~= 5.2`), so the clean `0.1` run's
      smaller norm was indeed caused by the stronger manifold coefficient.
    - It did not restore the historical same-image target specificity:
      `wrong_same` only moved `0.77 -> 0.785`, and query retrieval is
      essentially unchanged (`0.42 -> 0.425`).
    - Therefore the main clean-vs-old Stage1 gap is not explained by manifold
      weight alone. The remaining difference must come from another training
      semantic/code-path/config/data behavior, or from the internal diagnostic
      path still not being perfectly comparable to the old reference.
  - Conclusion:
    - Treat this as a negative replay for the "just set manifold back to
      `0.01`" hypothesis.
    - Keep this checkpoint as a diagnostic artifact, but it is not yet a
      strong Stage1 parent candidate versus the historical Stage1 references.

### AUDIT-20260627-025608-clean-stage1-vs-legacy-sampler-semantics

- Status: COMPLETED.
- Question:
  - Why does clean Stage1 with the same visible config as the old 20260617
    row-only Stage1 fail to recover the old target-specific diagnostics?
- Compared artifacts:
  - Legacy strong reference:
    `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617`.
  - Clean replay:
    `outputs/clean_training/qwen3_stage12_samplerfix_manifold001_4gpu_20260627_000156/stage1_micro4`.
- Fixed/equivalent visible settings:
  - Train file path:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl`.
  - Test file path:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl`.
  - Stage1 Protocol C tool-observation, `row_only`, `matrix_ce`,
    `native_source_grid`, `teacher_forced`, `focus_action_im_end=true`,
    `mask_original_image_after_tgvf=true`, `max_image_resolution=512`,
    `world=4`, `micro=4`, `accum=2`, `global_batch=32`,
    `max_steps=2000`, `visual_token_manifold=0.01`.
- Confirmed implementation difference:
  - Legacy `scripts/train_tgvf_v3_stage1.py::SameImageBatchSampler` used:
    - image-group owner: `sha1(image_key) % world_size`;
    - per-epoch `random.Random(seed + epoch)` shuffle of group ids;
    - per-epoch shuffle of samples within each image group;
    - incomplete tail chunks dropped after shuffle.
  - Clean `_SingleProcessSampleCursor` used before this audit:
    - image-group owner: `sha256(image_key) % world_size`;
    - deterministic dataset-order group cycle;
    - deterministic group offsets;
    - for 5/6-item image groups, tail samples were never selected because the
      cursor reset when `offset + batch_size > group_size`.
- Coverage simulation over the actual Stage1 train JSONL with `world=4`,
  `micro=4`, `accum=2`, and `2000` optimizer steps:
  - Legacy `sha1 + shuffle`:
    - total draws: `64000`.
    - unique samples: `35312`.
    - draw-count histogram: `{2: 28688, 1: 6624}`.
    - open draws: `12120`; multiple-choice draws: `51880`.
  - Clean pre-fix `sha256 + ordered`:
    - total draws: `64000`.
    - unique samples: `32836`.
    - draw-count histogram: `{2: 31164, 1: 1672}`.
    - open draws: `9786`; multiple-choice draws: `54214`.
  - Clean with only `sha1` but still ordered:
    - unique samples: `32836`.
    - open draws: `9778`; multiple-choice draws: `54222`.
  - Clean with legacy shuffle semantics:
    - unique samples: `35338`.
    - open draws: `12156`; multiple-choice draws: `51844`.
  - Interpretation:
    - The meaningful difference is shuffle/epoch semantics, not the hash
      function itself.
    - Group size distribution for eligible groups is `{4: 4972, 5: 3193,
      6: 44}`. Ordered cycling permanently made `3281` tail samples
      inaccessible, including `1747` open samples.
    - The inaccessible tail samples are enriched for `ocr_text`, `counting`,
      `spatial_relation`, and `state_action`, which are exactly the kind of
      same-image target-specific distinctions Stage1 is supposed to learn.
- Code fix:
  - Updated `revisit_vlm_clean/src/revisit_vlm_clean/training/executor.py`
    `_SingleProcessSampleCursor` to match legacy same-image sampler semantics:
    `sha1(image_key)%world_size`, seeded per-epoch shuffle of groups, seeded
    per-epoch shuffle inside groups, and complete-batch tail dropping after
    shuffle.
  - Added a regression test that a 5-item same-image group can emit the tail
    item under legacy shuffle, instead of permanently repeating the first 4.
  - Updated the rank-owner unit test from `sha256` to `sha1`.
- Verification:
  - `PYTHONPATH=revisit_vlm_clean/src:src pytest -q revisit_vlm_clean/tests/test_cli.py -k 'stage1_same_image_cursor'`
    passed: `3 passed`.
  - `python -m py_compile revisit_vlm_clean/src/revisit_vlm_clean/training/executor.py`
    passed.
  - Direct clean cursor simulation after the fix over the real train JSONL:
    - total draws: `64000`.
    - unique samples: `35312`.
    - draw-count histogram: `{2: 28688, 1: 6624}`.
    - rank0 summary mode: `same_image_legacy_shuffle`.
- Remaining notes:
  - This audit does not prove the sampler difference is the only clean-vs-old
    gap. It proves one concrete, behaviorally significant difference and fixes
    it.
  - Other surface differences still recorded for later audit:
    - legacy `model_id` was the local path
      `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`, while the clean
      plan used `Qwen/Qwen3-VL-8B-Thinking`;
    - legacy checkpoint cadence was `save_every=500`, while the clean replay
      used `save_every=2000`.
    - checkpoint cadence should not affect gradients, but model-id resolution
      should be pinned to the local path in the next comparable run to remove
      any residual ambiguity.
- Conclusion:
  - The current clean Stage1 replay was not equivalent to the old 20260617
    Stage1 because its same-image sampler changed the effective training
    distribution. A new clean Stage1 replay is required before judging
    manifold, Stage2, or benchmark behavior against the old baseline.

### AUDIT-20260627-clean-stage1-executor-identity-followup

- Status: COMPLETED.
- Question:
  - After fixing the clean Stage1 same-image sampler, are there remaining
    executor/plan identity gaps that could make future clean-vs-legacy
    comparisons ambiguous?
- Confirmed gaps and fixes:
  - Stage1 training-step defaults:
    - Legacy `v3_stage1_training_step` receives explicit
      `same_image_negative_margin=1.0` and `readout_batch_size=4`.
    - Clean executor previously relied on callee defaults for those values.
    - Clean `train_stage1`, `Stage1LaunchConfig`, training plan, legacy
      reference command, and training-step probe now carry both values
      explicitly.
    - AST keyword check after the change showed the clean and legacy
      `v3_stage1_training_step(...)` calls have the same keyword set.
  - Stage1 dataset required fields:
    - Legacy `TGVFv3Stage1Dataset.required_fields` includes
      `evidence_description`.
    - Clean dataset runtime audit now requires
      `["image", "question", "target", "evidence_description"]` for Stage1.
  - `first_batch_identity.json`:
    - Clean executor previously recorded the first N JSONL rows from the file
      scanner. For Stage1, this did not prove the same-image sampler,
      rank-owner assignment, or accumulation-step materialization.
    - Clean executor now materializes the first Stage1 optimizer step with
      `TGVFv3Stage1Dataset(focus_only=True)` plus `_SingleProcessSampleCursor`
      over every rank and accumulation micro-step.
    - The artifact now records rank, micro-index, sample index, sampler mode,
      and sampler group owner for each materialized row.
- Verification:
  - `PYTHONPATH=revisit_vlm_clean/src:src pytest -q revisit_vlm_clean/tests/test_cli.py -k 'stage1'`
    passed: `15 passed`.
  - `PYTHONPATH=revisit_vlm_clean/src:src pytest -q revisit_vlm_clean/tests/test_cli.py -k 'training_plan or stage1_same_image_cursor or stage1_first_batch'`
    passed: `5 passed`.
  - `python -m py_compile revisit_vlm_clean/src/revisit_vlm_clean/cli/train_stage1.py revisit_vlm_clean/src/revisit_vlm_clean/training_plan.py revisit_vlm_clean/src/revisit_vlm_clean/training/executor.py revisit_vlm_clean/tests/test_cli.py`
    passed.
  - `git diff --check -- revisit_vlm_clean/src/revisit_vlm_clean/cli/train_stage1.py revisit_vlm_clean/src/revisit_vlm_clean/training_plan.py revisit_vlm_clean/src/revisit_vlm_clean/training/executor.py revisit_vlm_clean/tests/test_cli.py`
    passed.
- Interpretation:
  - These fixes do not explain the old clean Stage1 diagnostic gap by
    themselves; they remove ambiguity and prevent future runs from looking
    comparable while using an unproven executor identity.
  - Existing clean Stage1 outputs generated before these fixes remain
    diagnostic side results, not final equivalence results.
- Next required comparable run:
  - Re-run clean Stage1 from the corrected code with the legacy-comparable
    local Qwen3 model path, legacy-compatible sampler semantics, explicit
    Stage1 training-step defaults, and `visual_token_manifold=0.01`.
  - Only after that run should Stage2 and benchmark conclusions be compared
    against 20260617/20260620 legacy references.

### AUDIT-20260627-clean-stage1-tgvf-config-and-sampler-identity

- Status: COMPLETED.
- Question:
  - Besides sampler semantics and Stage1 training-step defaults, does clean
    Stage1 still omit identity fields that legacy Stage1 recorded and that
    Stage2/diagnostics may later depend on?
- Evidence inspected:
  - Legacy reference:
    `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617`.
  - Legacy `train/config.json` recorded:
    - `batch_sampling="same_image"`.
    - `drop_incomplete_same_image_batches=true`.
    - full `config.tgvf`, including `spatial_merge_size=2`,
      `encoder_adapter_layers=[8,16,24]`, `encoder_adapter_type="bidirectional"`,
      `preserve_llm_kv_cache=true`, and `second_full_llm_forward=false`.
  - Clean checkpoint from the earlier replay recorded only:
    - `config.tgvf={"variant":"tgvf_v2_bidirectional","num_foveated_tokens":null}`.
- Confirmed interpretation:
  - For the current `tgvf_v2_bidirectional` variant, the encoder-adapter
    options are not used by `build_tgvf_module`, so this omission is unlikely
    to directly explain the existing Stage1 diagnostic gap.
  - It is still an identity/checkpoint-contract gap: Stage2 and diagnostics
    reconstruct TGVF from Stage1 checkpoint config, so clean checkpoints
    should carry the same complete TGVF identity as legacy checkpoints.
  - Clean distributed training uses a native executor with manual gradient
    averaging after backward rather than wrapping the TGVF module in PyTorch
    DDP. This is expected to be mathematically equivalent for the current
    trainable parameter groups, but it remains a recorded implementation
    difference from the legacy script.
- Code fix:
  - Clean Stage1 `training_plan.json` now explicitly records:
    - `dataset.batch_sampling="same_image"`.
    - `dataset.drop_incomplete_same_image_batches=true`.
    - full requested `tgvf` config with canonical clean defaults.
  - Clean Stage1 executor preflight now validates those Stage1 sampler identity
    fields and validates the presence of Stage1 `tgvf` identity fields.
  - Stage1 runtime loader now resolves `spatial_merge_size="auto"` to the
    actual Qwen visual merge size and stores `loader.resolved_tgvf_config`.
  - Clean checkpoint config now writes resolved `config.tgvf` from the runtime
    loader when available, falling back to plan `tgvf` only if needed.
  - Stage2 runtime loader now also records the resolved TGVF config and passes
    `encoder_adapter_type` explicitly when rebuilding from a Stage1 checkpoint.
- Plancheck:
  - Generated a no-launch clean Stage1 plan with legacy-comparable identity:
    - model/processor:
      `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
    - train file:
      `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl`.
    - `world=4`, `micro=4`, `accum=2`, `global=32`.
    - `max_steps=2000`, `save_every=500`.
    - `visual_token_manifold=0.01`.
    - `batch_sampling=same_image`, `drop_incomplete_same_image_batches=true`.
    - requested `tgvf.spatial_merge_size="auto"`, to be resolved to actual
      `2` at runtime/checkpoint save.
- Verification:
  - `PYTHONPATH=revisit_vlm_clean/src:src pytest -q revisit_vlm_clean/tests/test_cli.py`
    passed: `49 passed`.
  - `python -m py_compile revisit_vlm_clean/src/revisit_vlm_clean/cli/train_stage1.py revisit_vlm_clean/src/revisit_vlm_clean/training_plan.py revisit_vlm_clean/src/revisit_vlm_clean/training/executor.py revisit_vlm_clean/tests/test_cli.py`
    passed.
  - `git diff --check -- revisit_vlm_clean/src/revisit_vlm_clean/cli/train_stage1.py revisit_vlm_clean/src/revisit_vlm_clean/training_plan.py revisit_vlm_clean/src/revisit_vlm_clean/training/executor.py revisit_vlm_clean/tests/test_cli.py`
    passed.
- Conclusion:
  - This fixes another concrete clean-vs-legacy identity gap. It does not
    validate any previously generated clean Stage1 result as comparable.
  - The next valid comparison still requires a fresh clean Stage1 run from
    code after this audit, then internal diagnostics on the resulting
    checkpoint.

### AUDIT-20260627-clean-stage1-forward-training-ddp-semantics

- Status: COMPLETED.
- Question:
  - For clean-vs-legacy Stage1 comparison, are the forward/backward training
    semantics actually aligned, not just the config/artifact identity?
- Correction to prior notes:
  - The earlier note in
    `AUDIT-20260627-clean-stage1-tgvf-config-and-sampler-identity` said clean
    manual gradient averaging was expected to be mathematically equivalent to
    legacy DDP for current trainable groups.
  - That was too weak for a comparable run. Legacy Stage1 wraps the TGVF module
    in `DistributedDataParallel`, which broadcasts rank0 initial parameters and
    uses DDP gradient synchronization with `no_sync()` during gradient
    accumulation. Clean Stage1 did not wrap TGVF in DDP.
- Confirmed non-equivalence before this fix:
  - Legacy `scripts/train_tgvf_v3_stage1.py`:
    - builds `raw_foveal_module`;
    - calls `raw_foveal_module.train()`;
    - wraps it with `DistributedDataParallel(...)` when `world_size > 1`;
    - uses `foveal_module.no_sync()` for non-final accumulation micro-steps;
    - manually averages protocol token row gradients separately.
  - Clean executor before this audit:
    - built one TGVF module per rank without DDP wrapping;
    - relied on manual all-reduce over optimizer gradients after accumulation;
    - therefore did not inherit DDP's initial parameter broadcast and did not
      match the legacy accumulation synchronization path.
- Code fix:
  - Clean distributed Stage1 launch now applies legacy-compatible distributed
    training semantics before trainable-parameter audit and optimizer
    construction:
    - wrap `modules["tgvf"]` with PyTorch `DistributedDataParallel`;
    - keep `find_unused_parameters=False`, matching the legacy default;
    - record loader metadata:
      `stage1_tgvf_wrapped_with_ddp=true`,
      `ddp_broadcast_initial_parameters=true`, and
      `gradient_sync=legacy_ddp_tgvf_plus_manual_protocol_rows`.
  - Clean trainer loop now uses the DDP module's `no_sync()` context for
    non-final accumulation micro-steps, matching legacy Stage1.
  - Clean manual gradient averaging now skips parameters owned by DDP modules,
    so TGVF gradients are not double-averaged; non-DDP trainables such as
    protocol token row parameters remain manually averaged.
  - Clean Stage1 checkpoint save/resume unwraps DDP before reading/writing
    `tgvf_module`, preserving legacy-compatible checkpoint keys without a
    `module.` prefix.
- What this does not claim:
  - This does not validate the old clean Stage1 outputs. Runs produced before
    this fix remain diagnostic side results.
  - This does not prove all remaining variables are aligned; the next
    comparable run must still pin the local Qwen3 model path and legacy-visible
    Stage1 settings.
- Verification:
  - `PYTHONPATH=revisit_vlm_clean/src:src python -m py_compile revisit_vlm_clean/src/revisit_vlm_clean/training/executor.py revisit_vlm_clean/tests/test_cli.py`
    passed.
  - `PYTHONPATH=revisit_vlm_clean/src:src pytest -q revisit_vlm_clean/tests/test_cli.py -k 'stage1_clean_ddp_helpers or stage1_same_image_cursor'`
    passed: `5 passed, 46 deselected`.
  - Added a direct fake-DDP unit check that
    `_apply_legacy_distributed_training_semantics(...)` wraps Stage1 TGVF before
    training and records the legacy distributed metadata without loading the
    full Qwen model.
  - `PYTHONPATH=revisit_vlm_clean/src:src pytest -q revisit_vlm_clean/tests/test_cli.py`
    passed: `52 passed, 2 warnings`.
- Conclusion:
  - This fixes the most important currently confirmed forward/training
    semantics gap in clean distributed Stage1.
  - A fresh clean Stage1 run after this commit is required before comparing
    clean Stage1 diagnostics against 20260617/20260620 legacy references.

### EXP-20260627-040717-clean-qwen3-stage1-legacyrepro-ddp-manifold001-4gpu

- Status: COMPLETED.
- Question:
  - After fixing clean Stage1 distributed training semantics to match legacy
    DDP/no_sync behavior, can the clean project reproduce the old Qwen3
    Protocol-C Stage1 diagnostics when `visual_token_manifold=0.01`?
- Baseline:
  - Old 20260617 row-only 4GPU Stage1:
    `outputs/tgvf_v3_protocol_c_stage1_8b/protocol_c_toolobs_stage1_v4data_clean_rowonly_gpu0_3_focus_imend_bidirectional_4gpu_bs4_accum2_gbs32_2000step_20260617`.
  - Key old internal diagnostics recorded earlier:
    `pct_correct_D_beats_wrong_same=0.945`, `retrieval_top1=0.535`,
    `norm_ratio_D_to_Vmerge=5.174`.
  - Previous clean `visual_token_manifold=0.01` replay
    `EXP-20260627-000156-clean-qwen3-stage1-samplerfix-manifold001-4gpu`
    is not the comparison target because it was run before the DDP wrapper /
    accumulation semantics fix.
- Intended diff:
  - Use current clean code after `AUDIT-20260627-clean-stage1-forward-training-ddp-semantics`.
  - Keep the historical Stage1 manifold coefficient:
    `loss_visual_token_manifold=0.01`.
  - Hold fixed: data, local Qwen3 model/processor path, Protocol-C
    tool-observation, row-only token rows, teacher-forced capture,
    `native_source_grid`, old mask behavior, `matrix_ce`, seed, LR/scheduler,
    max image resolution 512, and global batch 32.
- Code / worktree:
  - Branch: `clean/tgvf-clean-project-20260625`.
  - Git commit: `fc0fbf661df0943fca7c4f21f49a6566da511636`.
  - Dirty worktree: true, but dirty files are unrelated Stage3/RL work and are
    not part of this Stage1 training path:
    `docs/TGVF_STAGE3_RL_WORKLOG.md`,
    `revisit_vlm_clean/src/revisit_vlm_clean/cli/generate_data.py`,
    `revisit_vlm_clean/README_STAGE3_RL_DATA.md`,
    `revisit_vlm_clean/src/revisit_vlm_clean/stage3_rl_data/`,
    `revisit_vlm_clean/tests/test_stage3_rl_data.py`, plus untracked `logs/`
    and `third_party/`.
- Plan:
  - `outputs/clean_training/qwen3_stage1_legacyrepro_ddp_manifold001_4gpu_20260627_040717/stage1_micro4/training_plan.json`.
  - Plan sha256:
    `90bd9caee7b210023a6160a843483f82ae3d257be87371d5f26e5b3d52d13498`.
- Data:
  - Train file:
    `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl`.
  - Rows: `39998`.
  - sha256:
    `c94a38b824b6603e555eed5ef3584c19cc903b76995d49c67ace36b18268443c`.
  - Sampler: `same_image_legacy_shuffle`;
    group owner: `sha1(image_key)%world_size`;
    drop incomplete same-image batches: true.
- Model / processor:
  - Model: `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
  - Processor: `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
  - dtype: `bfloat16`.
  - Attention implementation: `sdpa`.
- Stage1 settings:
  - Protocol: `protocol_c_tool_observation`.
  - Variant: `tgvf_v2_bidirectional`.
  - Token row mode: `row_only`.
  - Capture mode: `teacher_forced`.
  - FVT position mode: `native_source_grid`.
  - Focus action im_end: true.
  - Mask original image after TGVF: true.
  - Same-image negative mode: `matrix_ce`.
  - DeepStack reencode compatibility: false.
  - Visual merger: frozen.
- Training:
  - GPUs: `0,1,2,3`.
  - world size: `4`.
  - micro batch: `4`.
  - gradient accumulation: `2`.
  - global batch: `32`.
  - max steps: `2000`.
  - save every: `500`.
  - seed: `20260525`.
  - optimizer: AdamW, learning rate `1e-4`, cosine schedule, warmup `100`,
    min LR ratio `0.1`, max grad norm `1.0`.
  - Loss weights:
    - `loss_gen=1.0`.
    - `loss_same_image_negative=1.0`.
    - `loss_visual_token_manifold=0.01`.
- W&B:
  - Project: `tgvf-clean-qwen3-deepstack`.
  - Mode: `online`.
- Preflight:
  - `stage1_executor --preflight-only`: passed; no blocking items.
  - `stage1_executor --prepare-execution`: passed; runner status
    `ready_for_explicit_distributed_launch`.
  - First materialized batch: requested global batch `32`, materialized batch
    `32`, batch sha256
    `eaec7f0c9a44edf04203051e3b5142ad152144c751adba0e26e90c8b33e638aa`.
- Command:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --nproc-per-node 4 -m revisit_vlm_clean.training.stage1_executor --plan outputs/clean_training/qwen3_stage1_legacyrepro_ddp_manifold001_4gpu_20260627_040717/stage1_micro4/training_plan.json --launch-training`.
- Output:
  - `outputs/clean_training/qwen3_stage1_legacyrepro_ddp_manifold001_4gpu_20260627_040717/stage1_micro4`.
- Launch:
  - Planned at 2026-06-27T04:09:23+09:00.
  - Started at 2026-06-27T04:10:34+09:00.
  - tmux: `clean_stage1_legacyrepro_20260627_040717`.
  - Log:
    `logs/clean_training/clean_stage1_legacyrepro_20260627_040717.log`.
  - W&B:
    `https://wandb.ai/mio_nora/tgvf-clean-qwen3-deepstack/runs/3djvsggm`.
- Startup verification:
  - tmux session is alive.
  - GPUs `0,1,2,3` are allocated by the run.
  - `training_progress.jsonl` is being written.
  - W&B is online and syncing.
  - Training reached at least `step=4/2000`.
  - Runtime progress reports `ddp_enabled=true` and `world_size=4`.
  - Trainable audit reports `28` tensors and `18,046,720` parameters:
    protocol token rows plus `tgvf.module.*`; visual merger remains frozen.
  - Debug field `qwen_frozen=false` is also present in the old 20260617 legacy
    train log, so it is not treated as a new clean-run anomaly; optimizer /
    trainable audit is the source of truth here.
  - Early component losses:
    - step 1:
      `loss_total=3.4725849628448486`,
      `loss_gen=2.15625`,
      `loss_same_image_negative=1.28515625`,
      raw `loss_visual_token_manifold=3.1178618669509888`,
      `grad_norm=6.25`.
    - step 4:
      `loss_total=3.61555016040802`,
      `loss_gen=2.234375`,
      `loss_same_image_negative=1.3515625`,
      raw `loss_visual_token_manifold=2.961265802383423`,
      `grad_norm=5.875`.
- Training completion:
  - Completed at approximately 2026-06-27T06:08:00+09:00.
  - Runtime from progress log: `7014.081746816635s` (`1h56m54s`).
  - Optimizer steps completed: `2000`.
  - Micro steps completed: `4000`.
  - Checkpoints saved at steps `500`, `1000`, `1500`, and `2000`.
  - Final checkpoint:
    `outputs/clean_training/qwen3_stage1_legacyrepro_ddp_manifold001_4gpu_20260627_040717/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt`.
  - Final checkpoint sha256:
    `7f702f05f5e38ca9261c2a4dc8abee663f3b2cade565760155a9e3e501f6e8c4`.
  - Final step losses:
    - `loss_total=1.3478202819824219`.
    - `loss_gen=1.3125`.
    - `loss_same_image_negative=0.00506591796875`.
    - raw `loss_visual_token_manifold=3.0254406929016113`.
    - weighted manifold contribution: `0.030254406929016115`.
  - Final logged norm diagnostics:
    - `norm_ratio_mean=5.325041770935059`.
    - `d_norm_mean=116.21395111083984`.
    - `v_merge_norm_mean=21.82404136657715`.
- Internal diagnostics:
  - Status: COMPLETED.
  - Run id:
    `clean_qwen3_stage1_legacyrepro_internal_diag_20260627_124111`.
  - Purpose:
    - Measure the fresh DDP-semantics clean Stage1 replay on the same internal
      readout/query/distribution surfaces used for previous Stage1 comparisons.
  - Fixed identity:
    - Checkpoint:
      `outputs/clean_training/qwen3_stage1_legacyrepro_ddp_manifold001_4gpu_20260627_040717/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt`.
    - Checkpoint sha256:
      `7f702f05f5e38ca9261c2a4dc8abee663f3b2cade565760155a9e3e501f6e8c4`.
    - Eval JSONL:
      `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl`.
    - Max image resolution: `512`.
    - Tasks: `readout`, `query`, `distribution`.
    - Sample caps: readout `200`, distribution `200`, query groups `50`.
    - Device: `cuda:0`.
    - use_fvt_cache: false.
  - Plan:
    `outputs/clean_training/qwen3_stage1_legacyrepro_ddp_manifold001_4gpu_20260627_040717/stage1_micro4/internal_diagnostics_step2000_20260627_124111/clean_stage_diagnostic_plan.json`.
  - Plan sha256:
    `135ccd1f720ec2197c385c5952c668e2c75cbf97f0d68c37c98b6f3e55248b46`.
  - Command:
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.stage_diagnostics --run-id clean_qwen3_stage1_legacyrepro_internal_diag_20260627_124111 --stage stage1 --checkpoint outputs/clean_training/qwen3_stage1_legacyrepro_ddp_manifold001_4gpu_20260627_040717/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt --eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl --output-dir outputs/clean_training/qwen3_stage1_legacyrepro_ddp_manifold001_4gpu_20260627_040717/stage1_micro4/internal_diagnostics_step2000_20260627_124111 --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --protocol protocol_c_tool_observation --focus-action-im-end --variant tgvf_v2_bidirectional --encoder-adapter-type bidirectional --max-image-resolution 512 --fvt-position-mode native_source_grid --dtype bfloat16 --attn-implementation sdpa --device cuda:0 --tasks all --readout-max-samples 200 --distribution-max-samples 200 --query-max-groups 50 --query-require-groups 0 --seed 20260525 --execute`.
  - Output:
    `outputs/clean_training/qwen3_stage1_legacyrepro_ddp_manifold001_4gpu_20260627_040717/stage1_micro4/internal_diagnostics_step2000_20260627_124111`.
  - Launch:
    - Started at 2026-06-27T12:42:00+09:00.
    - tmux: `clean_stage1_legacyrepro_diag_20260627_124111`.
    - Log:
      `logs/clean_training/clean_stage1_legacyrepro_diag_20260627_124111.log`.
    - Runtime status file reports `status=running`.
    - `runtime_config.json` confirms checkpoint global step `2000`, protocol
      token rows loaded from checkpoint, no Qwen LoRA in checkpoint, and
      `spatial_merge_size=2`.
  - Completed at 2026-06-27T12:50:00+09:00.
  - Reports:
    - `outputs/clean_training/qwen3_stage1_legacyrepro_ddp_manifold001_4gpu_20260627_040717/stage1_micro4/internal_diagnostics_step2000_20260627_124111/readout/readout_eval_report.json`.
    - `outputs/clean_training/qwen3_stage1_legacyrepro_ddp_manifold001_4gpu_20260627_040717/stage1_micro4/internal_diagnostics_step2000_20260627_124111/query_sensitivity/query_sensitivity_report.json`.
    - `outputs/clean_training/qwen3_stage1_legacyrepro_ddp_manifold001_4gpu_20260627_040717/stage1_micro4/internal_diagnostics_step2000_20260627_124111/fvt_distribution/fvt_distribution_report.json`.
  - Results:
    - Readout, `n=200`:
      - `pct_correct_D_beats_target_only=0.985`.
      - `pct_correct_D_beats_random=1.0`.
      - `pct_correct_D_beats_wrong_same=0.87`.
      - `pct_correct_D_beats_wrong_diff=0.90625`.
      - `mean_delta_correct_vs_target_only=0.58361328125`.
      - `mean_delta_correct_vs_wrong_same=0.260595703125`.
      - `mean_nll_correct_D=1.4366015625`.
      - `mean_nll_target_only=2.02021484375`.
    - Query sensitivity, `groups=46`, `items=200`:
      - `retrieval_top1=0.705`.
      - `retrieval_top2=0.88`.
      - `mrr=0.82825`.
      - `mean_diagonal_gap=0.0759375`.
      - `median_diagonal_gap=0.0546875`.
    - FVT distribution, `n=200`:
      - `avg_manifold_loss=2.9338350534439086`.
      - `median_manifold_loss=2.941246747970581`.
      - `avg_norm_D=113.53232604980468`.
      - `avg_norm_V_merge=20.683768496513366`.
      - `norm_ratio_D_to_Vmerge=5.676891053685856`.
      - `finite_rate=1.0`.
      - `collapse_warning=false`.
      - `manifold_active_rate=1.0`.
  - Comparison:
    - Previous clean `visual_token_manifold=0.01` replay before the DDP
      semantics fix:
      - `wrong_same=0.785`, `retrieval_top1=0.425`, `mrr=0.626`,
        `norm_ratio=5.210`.
    - This fresh DDP-semantics replay:
      - `wrong_same=0.87`, `retrieval_top1=0.705`, `mrr=0.82825`,
        `norm_ratio=5.677`.
    - Old 20260617 row-only 4GPU reference:
      - `wrong_same=0.945`, `retrieval_top1=0.535`,
        `norm_ratio=5.174`.
    - Clean samplerfix `0.1` run:
      - `wrong_same=0.77`, `retrieval_top1=0.42`, `norm_ratio=2.401`.
  - Analysis:
    - The DDP/no_sync clean-vs-legacy semantic fix had a large positive effect:
      `wrong_same` improved by `+0.085` over the previous `0.01` replay, and
      query `retrieval_top1` improved by `+0.28`.
    - Restoring `visual_token_manifold=0.01` keeps D scale in the historical
      high-norm range; this run is slightly above the old 20260617 norm ratio
      (`5.677` vs `5.174`).
    - The run still does not fully reproduce the old 20260617 readout
      `wrong_same=0.945`, but the query retrieval surface is now stronger than
      the old 20260617 reference (`0.705` vs `0.535`).
    - Current conclusion: the DDP semantics fix was a real missing piece. The
      remaining gap is narrower and is no longer the broad collapse seen in
      earlier clean Stage1 attempts.
  - User decision / backup status:
    - Keep this checkpoint as a backup and diagnostic reference:
      `outputs/clean_training/qwen3_stage1_legacyrepro_ddp_manifold001_4gpu_20260627_040717/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt`.
    - Do not treat it as the immediate Stage2 parent: D norm is still high
      (`norm_ratio_D_to_Vmerge=5.6769`), and the current mean/std manifold
      objective appears to trade off against target-specific D discrimination.
    - Across the recent clean Stage1 ablations, larger manifold weight reduced
      norm but hurt the useful D signal; weaker manifold recovered signal
      better but did not control norm enough.
    - Next training hypothesis: replace or demote the current mean/std
      manifold term and test a direct token-norm constraint, ideally as the
      primary scale control rather than increasing `visual_token_manifold`.
  - Code follow-up:
    - Added a Stage1 `visual_token_norm` loss switch for direct D-scale control.
    - Loss definition: mean squared log norm ratio between each D token norm and
      the detached mean norm of same-sample merged visual tokens.
    - Default remains off (`loss_visual_token_norm=0.0`) so old commands keep
      their meaning; the intended next replay setting is
      `loss_visual_token_manifold=0.0`, `loss_visual_token_norm=0.1`.
    - Clean Stage1 plan/CLI/executor now carry this loss into `LossWeights`,
      runtime audit, progress logging, and W&B loss fields.
    - Verification before commit:
      `python -m py_compile src/revisit_vlm/tgvf_training.py src/revisit_vlm/tgvf_v3_stage1.py scripts/train_tgvf_v3_stage1.py scripts/train_tgvf_fvt.py revisit_vlm_clean/src/revisit_vlm_clean/training_plan.py revisit_vlm_clean/src/revisit_vlm_clean/cli/train_stage1.py revisit_vlm_clean/src/revisit_vlm_clean/training/executor.py`;
      `PYTHONPATH=revisit_vlm_clean/src:src pytest -q revisit_vlm_clean/tests/test_cli.py`.
  - Conclusion:
    - This is the best clean Stage1 replay so far, but it is now archived as a
      backup/diagnostic reference rather than promoted directly to Stage2.

### EXP-20260627-133252-clean-qwen3-stage1-norm01-manifold0-4gpu

- Status: DONE.
- Question:
  - Test whether a direct D token-norm constraint can control Stage1 D scale
    without the target-discrimination damage seen from increasing the previous
    mean/std manifold loss.
- Baseline:
  - `EXP-20260627-040717-clean-qwen3-stage1-legacyrepro-ddp-manifold001-4gpu`.
- Intended diff:
  - Change Stage1 loss from `loss_visual_token_manifold=0.01` to
    `loss_visual_token_manifold=0.0`.
  - Enable direct norm loss with `loss_visual_token_norm=0.1`.
- Held fixed:
  - Qwen3 model/processor path, Protocol-C tool-observation, row-only token
    rows, teacher-forced capture, native source grid positions, old Stage1
    readout mask behavior, frozen Qwen visual merger, matrix CE same-image
    negative loss, seed, LR/scheduler, max image resolution 512, global batch
    32, 4-GPU DDP/no_sync clean training semantics.
- Code / worktree:
  - Branch: `clean/tgvf-clean-project-20260625`.
  - Git commit: `0ce3c7899ec789259a68c32cb513fc039729c20e`.
  - Dirty worktree: true; known unrelated dirty files are Stage3/RL/data
    generation work and untracked logs/data directories.
- Plan:
  - `outputs/clean_training/qwen3_stage1_norm01_manifold0_4gpu_20260627_133252/stage1_micro4/training_plan.json`.
  - Plan sha256:
    `2422fb1e832b03c9db9c76632348f23341985374d1911456ef5aac5228ab59c7`.
- Data:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl`.
  - Rows: `39998`.
  - sha256:
    `c94a38b824b6603e555eed5ef3584c19cc903b76995d49c67ace36b18268443c`.
- Training:
  - GPUs: `0,1,2,3`.
  - world size: `4`; micro batch: `4`; accumulation: `2`; global batch: `32`.
  - max steps: `2000`; save every: `500`.
  - W&B project: `tgvf-clean-qwen3-deepstack`; mode: `online`.
- Command:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --nproc-per-node 4 -m revisit_vlm_clean.training.stage1_executor --plan outputs/clean_training/qwen3_stage1_norm01_manifold0_4gpu_20260627_133252/stage1_micro4/training_plan.json --launch-training`.
- Preflight:
  - `stage1_executor --preflight-only`: passed, no blocking items.
  - `stage1_executor --prepare-execution`: passed,
    `ready_for_explicit_distributed_launch`.
- Launch:
  - Started at 2026-06-27T13:35+09:00.
  - tmux: `clean_stage1_norm01_m0_20260627_133252`.
  - Log:
    `logs/clean_training/clean_stage1_norm01_m0_20260627_133252.log`.
- Training completion:
  - Finished at approximately 2026-06-27T15:32+09:00.
  - Runtime from progress log: `7041.058387517929s` (`1h57m21s`).
  - Optimizer steps completed: `2000`; micro steps completed: `4000`.
  - Final checkpoint:
    `outputs/clean_training/qwen3_stage1_norm01_manifold0_4gpu_20260627_133252/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt`.
  - Final checkpoint sha256:
    `ab6bd554cfb405208f13270c298f7fd0ab01305c83ca07bf8eca667cbe1632a1`.
  - Final step losses:
    - `loss_total=1.3660708665847778`.
    - `loss_gen=1.3203125`.
    - `loss_same_image_negative=0.0016641616821289062`.
    - raw `loss_visual_token_manifold=0.44337937235832214`.
    - raw `loss_visual_token_norm=0.4409423768520355`.
  - Final logged norm diagnostics:
    - `norm_ratio_mean=1.9146325588226318`.
    - `d_norm_mean=41.7850227355957`.
    - `v_merge_norm_mean=21.82404136657715`.
- Internal diagnostics:
  - Status: COMPLETED.
  - Run id:
    `clean_qwen3_stage1_norm01_internal_diag_20260627_154418`.
  - Output:
    `outputs/clean_training/qwen3_stage1_norm01_manifold0_4gpu_20260627_133252/stage1_micro4/internal_diagnostics_step2000_20260627_154418`.
  - Reports:
    - `outputs/clean_training/qwen3_stage1_norm01_manifold0_4gpu_20260627_133252/stage1_micro4/internal_diagnostics_step2000_20260627_154418/readout/readout_eval_report.json`.
    - `outputs/clean_training/qwen3_stage1_norm01_manifold0_4gpu_20260627_133252/stage1_micro4/internal_diagnostics_step2000_20260627_154418/query_sensitivity/query_sensitivity_report.json`.
    - `outputs/clean_training/qwen3_stage1_norm01_manifold0_4gpu_20260627_133252/stage1_micro4/internal_diagnostics_step2000_20260627_154418/fvt_distribution/fvt_distribution_report.json`.
  - Readout, `n=200`:
    - `pct_correct_D_beats_target_only=0.995`.
    - `pct_correct_D_beats_random=1.0`.
    - `pct_correct_D_beats_wrong_same=0.9`.
    - `pct_correct_D_beats_wrong_diff=0.9375`.
    - `mean_nll_correct_D=1.40556640625`.
    - `mean_delta_correct_vs_target_only=0.6146484375`.
  - Query sensitivity, `groups=46`, `items=200`:
    - `retrieval_top1=0.7`.
    - `retrieval_top2=0.915`.
    - `mrr=0.8341666666666668`.
    - `mean_diagonal_gap=0.099765625`.
  - FVT distribution, `n=200`:
    - `avg_manifold_loss=0.4619758182764053`.
    - `avg_norm_D=42.55310848236084`.
    - `avg_norm_V_merge=20.683768496513366`.
    - `norm_ratio_D_to_Vmerge=2.12579250719686`.
    - `finite_rate=1.0`.
    - `collapse_warning=false`.
- Metrics:
  - Compared to `EXP-20260627-040717-clean-qwen3-stage1-legacyrepro-ddp-manifold001-4gpu`:
    - `wrong_same`: `0.87 -> 0.90`.
    - `wrong_diff`: `0.90625 -> 0.9375`.
    - `target_only`: `0.985 -> 0.995`.
    - `retrieval_top1`: `0.705 -> 0.700`.
    - `retrieval_top2`: `0.88 -> 0.915`.
    - `mrr`: `0.82825 -> 0.83417`.
    - `norm_ratio_D_to_Vmerge`: `5.6769 -> 2.1258`.
    - `avg_norm_D`: `113.53 -> 42.55`.
- Conclusion:
  - Direct norm loss with weight `0.1` substantially improves D scale and does
    not show the readout/query collapse seen when increasing the old mean/std
    manifold objective. This Stage1 checkpoint is a stronger Stage2 candidate
    than the previous `visual_token_manifold=0.01` clean replay on the internal
    diagnostics surface.

### EXP-20260626T1921-stage3-rl-data

- Status: DONE.
- Question: Build clean Stage3 RL source-QA pool manifest and reports.
- Baseline anchor: Current 50k SFT image pool excluded by manifest.
- Intended diff: Source-QA Stage3 RL pool, image-disjoint from SFT images; no training/eval/judge execution.
- Allowed changed variables: Data source selection, filtering, balancing, TargetSpec rule construction.
- Not allowed to change: SFT 50k data, benchmark eval data, GRPO/reward code, D generation.
- Code commit / worktree: `10c2e60e4bc751336c01aed37b32ad0d29c1d292`, dirty status not recorded by data builder.
- Train data: `/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/revisit_vlm_clean/data/stage3_rl/v0_teacher_triage_20k_20260627_041834/accepted_rl_prompts.jsonl`.
- Validation data: none.
- Benchmark output: none.
- Script / command: `tgvf_generate_data stage3-rl --plan /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/revisit_vlm_clean/data/stage3_rl/v0_teacher_triage_20k_20260627_041834/stage3_rl_data_plan.json --execute`.
- GPUs: none.
- Started: 2026-06-26T19:21:15.034357+00:00.
- Finished: 2026-06-26T19:21:15.039625+00:00.
- Metrics: accepted_qa=0, accepted_images=0, rejected_qa=0.
- Analysis: Data generation only; reports are under `/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/revisit_vlm_clean/data/stage3_rl/v0_teacher_triage_20k_20260627_041834`.
- Conclusion: Stage3 RL source pool generated; reward/judge/training remain non-goals.
- Comparable to baseline: Not a benchmark or training result.
- Follow-up: Run forced probes and GRPO reward design on this source pool separately.

### EXP-20260627T0244-stage3-rl-mixed20-smoke

- Status: SIDE_RESULT.
- Question: Check Stage3 RL GPT-5.4 teacher quality on a mixed 20-image smoke batch.
- Baseline anchor: Not a benchmark/training baseline; data-generation smoke only.
- Intended diff: Mixed source selection smoke after source-mix request preparation.
- Not allowed to change: Formal RL sample manifests, reward/judge/training code, benchmark eval data.
- Code commit / worktree: dirty local worktree.
- Output root:
  `/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/revisit_vlm_clean/data/stage3_rl/v0_teacher_triage_20k_20260627_041834`.
- Archive:
  `/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/revisit_vlm_clean/data/stage3_rl/v0_teacher_triage_20k_20260627_041834/archives/smoke_mixed20_20260627_batch_6a3f3295`.
- Batch id: `batch_6a3f32958870819087b273c79e521920`.
- Sample rule: 20 image requests, source-mix stratified as Visual Genome 8,
  TextVQA 6, DocVQA 4, ChartQA 2.
- API result: 20 completed, 0 failed, 0 parse errors.
- Runtime: about 9m42s OpenAI backend time; about 10m49s submit-to-local-parse.
- Quality summary: JSON/field format usable, but too many direct/no-tool items,
  too many raw items per image, and several target texts contain answer-adjacent
  judgments.
- Conclusion: Archived and excluded from formal samples. Prompt/schema/filter
  revised before any formal generation.
- Comparable to baseline: No.

### EXP-20260627T0301-stage3-rl-v1-direct20-smoke

- Status: SIDE_RESULT.
- Question: Check v1 Stage3 RL GPT-5.4 teacher quality and direct concurrent
  smoke speed on 20 mixed images.
- Baseline anchor: Previous mixed20 Batch API smoke
  `batch_6a3f32958870819087b273c79e521920`.
- Intended diff: Use `stage3_rl_gpt54_triage_v1`, max 4 items/image, direct
  concurrent API with 8 workers, no Batch queue.
- Not allowed to change: Formal RL sample manifests, reward/judge/training code,
  benchmark eval data.
- Output root:
  `/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/revisit_vlm_clean/data/stage3_rl/v1_direct_smoke20_20260627_030144`.
- Archive:
  `/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/revisit_vlm_clean/data/stage3_rl/v1_direct_smoke20_20260627_030144/archives/smoke_v1_direct20_20260627_030144`.
- Sample rule: 20 image requests, source-mix stratified as Visual Genome 8,
  TextVQA 6, DocVQA 4, ChartQA 2.
- API result: 20 completed direct requests, 0 API errors, 0 parse errors.
- Runtime: 25.9s runner elapsed, 27s wall-clock.
- Quality summary: 79 raw items, average 3.95/image; local deterministic filter
  valid 77/rejected 2; tool distribution improved with 44 useful_tool and 7
  likely_required.
- Conclusion: V1 direct smoke is substantially faster and cleaner than V0 Batch
  smoke. It remains archived and excluded from formal samples.
- Comparable to baseline: No benchmark comparison; data smoke only.

### EXP-20260627T0324-stage3-rl-v1-direct20k

- Status: DONE.
- Question: Generate the formal Stage3 RL v1 teacher-triage source pool for
  about 20k accepted QA prompts.
- Baseline anchor: V1 direct20 smoke
  `/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/revisit_vlm_clean/data/stage3_rl/v1_direct_smoke20_20260627_030144`.
- Intended diff: Run direct concurrent GPT-5.4 teacher generation on the v1
  8000-image source pool, then parse/finalize accepted prompts.
- Not allowed to change: SFT 50k data, benchmark eval data, GRPO/reward code,
  D generation, model training.
- Output root:
  `/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447`.
- Sample rule: source-mix stratified image pool with default weights:
  Visual Genome 0.40, TextVQA 0.30, DocVQA 0.20, ChartQA 0.10.
- Target accepted prompts: 20000.
- Candidate images: planned 8000.
- Exclusions: default SFT 50k manifests plus archived smoke custom/image ids
  from `smoke_mixed20_20260627_batch_6a3f3295` and
  `smoke_v1_direct20_20260627_030144`.
- Prompt/format: `stage3_rl_gpt54_triage_v1`, max 4 items/image,
  max_output_tokens=3000, open-answer.
- API mode: direct Responses API in 500-request chunks with 16 local workers.
- GPUs: none.
- Started: 2026-06-27T03:24:47Z.
- Command:
  - plan/execute through `revisit_vlm_clean.cli.generate_data stage3-rl`;
  - API generation through repeated `--run-api-direct --limit-requests 500
    --direct-workers 16 --skip-submitted-requests`.
- Validation plan: parse outputs, finalize teacher outputs, report accepted QA,
  source/tool/difficulty distributions, parse errors, API errors, and archive
  status of smoke roots.
- Finished: 2026-06-27T05:36:35Z.
- API outcome: 8000/8000 direct requests submitted, 16 raw response files,
  0 API errors. The first session was interrupted at chunk 11 before any chunk
  11 output was written; resume used the same plan with
  `--skip-submitted-requests` and completed chunks 11-16 without duplicate
  submission.
- API timing: 16 chunks, total direct chunk time 3887.395s, average 242.962s,
  min 233.470s, max 256.409s.
- Parse outcome: 8000 teacher outputs parsed, 0 parse errors, token usage
  input=32486913, output=8437698, total=40924611.
- Final data outcome: 20000 accepted QA prompts from 7849 images, 31487 QA
  candidates, 11487 rejected prompts. `target_spec` available for all accepted
  prompts.
- Source distribution: visual_genome=8000, textvqa=6000, docvqa=4000,
  chartqa=2000.
- Provenance distribution: source_qa_kept=1796, source_qa_rewritten=7847,
  teacher_generated_legacy_style=10357.
- Tool-need hint distribution: useful_tool=10510, optional_tool=4668,
  no_tool=3078, likely_required=1744.
- Difficulty distribution: local_medium=10671, local_easy=5356,
  direct_easy=2082, local_hard=1472, reasoning_hard=419.
- Filter outcome: 30573 valid after deterministic filters, 914 filter
  rejected; top filter rejection reasons were teacher_too_easy=409,
  target_leakage=318, low_teacher_confidence=141,
  sensitive_personal_info=40. Balance then selected 20000 accepted prompts.
- Balance warnings: none.
- Verification: `PYTHONPATH=revisit_vlm_clean/src pytest -q
  revisit_vlm_clean/tests/test_stage3_rl_data.py` passed, 8 tests.

### EXP-20260627-163250-clean-qwen3-stage2-norm01-mask075-deepstack

- Status: DONE.
- Question:
  - Start Stage2 preparation from the current clean Qwen3 Stage1 norm-only
    candidate, while keeping the remaining Stage1 uncertainty explicitly
    recorded.
- Baseline anchor:
  - Stage2 data/protocol follows `BASE-20260619-open-answer-rowonly`
    open-answer Protocol-C Stage2 split.
  - Stage1 is the new clean norm-only checkpoint, not the 20260619 baseline
    Stage1.
- Intended diff:
  - Use clean Stage1 checkpoint
    `clean_qwen3_stage1_norm01_manifold0_4gpu_20260627_133252`.
  - Enable Qwen3 DeepStack training semantics.
  - Use original-image mask tuple:
    `mask_original_image_after_tgvf=true`,
    `mask_scope=through_answer`,
    `mask_probability=0.75`.
- Allowed changed variables:
  - Stage1 checkpoint lineage.
  - Stage2 DeepStack enabled.
  - Stage2 mask probability.
- Not allowed to change:
  - Protocol: `protocol_c_tool_observation`.
  - Stage2 train/val data identity.
  - Max image resolution: `512`.
  - Weighted span losses:
    `evidence_state=0.2`, `focus_target=1.5`, `evidence=1.0`,
    `value_span=1.0`, `answer=1.0`,
    `no_focus_evidence_state=0.2`, `no_focus_answer=1.0`.
  - D remains a v-merge-level visual token span; no D DeepStack-like features.
- Code commit / worktree:
  - `0ce3c7899ec789259a68c32cb513fc039729c20e`.
  - Dirty worktree: true, due unrelated Stage3/RL/doc artifacts plus this
    ledger entry.
- Stage1 checkpoint:
  - `outputs/clean_training/qwen3_stage1_norm01_manifold0_4gpu_20260627_133252/stage1_micro4/clean_training_execution/checkpoint_step_2000.pt`.
  - sha256:
    `ab6bd554cfb405208f13270c298f7fd0ab01305c83ca07bf8eca667cbe1632a1`.
- Stage1 processor:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Stage2 checkpoint/output:
  - Output:
    `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4`.
  - Final checkpoint:
    `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - Final checkpoint sha256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Train data:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl`.
  - Rows: `46883`.
  - sha256:
    `b5027e72dda7601073ddb8bc9cf1853ec564fa415a3e9cb7d1e684cc7c0d733b`.
  - Focus/no-focus: `39655/7228`.
  - Answer format: `short_text=46883`.
- Validation data:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`.
  - Rows: `1002`.
  - sha256:
    `3b719e1bd03a09741cc05dac3ed85e423a9b2c29209b07546792a6795cdbbfc5`.
  - Focus/no-focus: `857/145`.
  - Answer format: `short_text=1002`.
- Benchmark output:
  - Not created yet.
- Training plan:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/training_plan.json`.
  - sha256:
    `a69cdc341cdbb7256b855f4d726e38b2d3ca20e7d88bacfe4d1aa85830b74cfa`.
- Execution bundle:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/clean_training_execution_bundle.json`.
  - sha256:
    `8df6bdc554e15b6560f21f82140e1943e5869f10e729bb793e91ae6759451a51`.
- Script / command:
  - Prepare:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.training.stage2_executor --plan outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/training_plan.json --prepare-execution`.
  - Clean launch command prepared:
    `torchrun --nproc-per-node 4 -m revisit_vlm_clean.training.stage2_executor --plan outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/training_plan.json --launch-training`.
  - Actual launch:
    `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --nproc-per-node 4 -m revisit_vlm_clean.training.stage2_executor --plan outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/training_plan.json --launch-training`.
- GPUs:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`.
  - Planned world size: `4`.
  - micro batch: `4`; accumulation: `8`; global batch: `128`.
- tmux:
  - `clean_stage2_norm01_mask075_ds_20260627_163250`.
- Log:
  - `logs/clean_training/clean_stage2_norm01_mask075_ds_20260627_163250.log`.
- W&B:
  - Project: `tgvf-clean-qwen3-deepstack`.
  - Run:
    `https://wandb.ai/mio_nora/tgvf-clean-qwen3-deepstack/runs/yxdq5cie`.
  - Local W&B path warning observed: planned output-local `wandb/` path was
    not writable, so W&B used `/tmp/wandb/run-20260627_165340-yxdq5cie`.
- Started:
  - 2026-06-27T16:52:22+09:00.
- Finished:
  - 2026-06-27T21:02:06+09:00.
- Prepared artifacts:
  - `stage2_executor --prepare-execution` passed.
  - Runner status: `ready_for_explicit_distributed_launch`.
  - Stage1 checkpoint contract: `validated`, `global_step=2000`,
    Protocol-C token rows present for 4 tokens.
  - Optimizer groups validated:
    `llm_lora lr=2e-5`, `tgvf_refiner lr=5e-6`,
    `fvt_calibration lr=1e-5`.
  - DeepStack plan:
    `enabled=true`, `execution_supported=true`,
    `original_image_scope=through_answer`, no blocking items,
    no D DeepStack-like features.
- Metrics:
  - Initial observed training line:
    `step=1/1200 loss=3.8515625 micro_steps=8 checkpoint=False validation=False`.
  - Training status: `clean_distributed_training_completed`.
  - Final observed training line:
    `step=1200/1200 loss=0.736328125 micro_steps=9600 checkpoint=True validation=True`.
  - Optimizer steps completed: `1200`.
  - Micro steps completed: `9600`.
  - Checkpoints saved at steps: `300`, `600`, `900`, `1200`.
  - Validation records: `4`.
  - Final W&B summary:
    - `train/loss_total=0.73633`.
    - `train/loss_focus=0.74561`.
    - `train/loss_no_focus=0.34509`.
    - `validation/loss_total=0.91016`.
    - `validation/loss_focus=0.93359`.
    - `validation/loss_no_focus=0.39453`.
    - `train/qwen3_deepstack_features_injected=1.0`.
    - `train/mask_original_image_after_tgvf_prob=0.75`.
- Analysis:
  - Stage2 completed successfully under clean distributed training.
  - It is not a clean ablation of any old Stage2 result because both Stage1
    lineage and DeepStack state differ from the historical baselines.
- Conclusion:
  - Stage2 checkpoint is ready for internal D/readout diagnostics and external
    benchmark comparison.
- Comparable to baseline:
  - Training loss is not directly comparable to benchmark baselines. Benchmark
    comparability will be decided by same-sample eval runs.
- Follow-up:
  - Run Stage1-style internal diagnostics on the final Stage2 checkpoint.
  - Run same-sample benchmark comparisons against the chosen baseline.

### EXP-20260627-223829-clean-qwen3-stage2-norm01-internal-diagnostics

- Status: DONE.
- Question:
  - Did the Stage2 trajectory/LoRA training degrade the Stage1-style D/readout,
    same-image retrieval, or FVT distribution diagnostics?
- Baseline anchor:
  - Compare primarily against the source Stage1 checkpoint diagnostics from
    `EXP-20260627-133252-clean-qwen3-stage1-norm01-manifold0-4gpu`:
    readout `wrong_same=0.9`, query `top1=0.7`, `MRR=0.83417`,
    distribution `norm_ratio_D_to_Vmerge=2.12579`.
- Intended diff:
  - Load the Stage2 checkpoint from
    `EXP-20260627-163250-clean-qwen3-stage2-norm01-mask075-deepstack`.
  - Use the same Stage1 focus eval JSONL and diagnostic sample counts as the
    Stage1 internal diagnostic run.
- Allowed changed variables:
  - Checkpoint stage: Stage2 checkpoint with Qwen LoRA loaded when present.
- Not allowed to change:
  - Eval JSONL, max image resolution, protocol, readout/query/distribution
    sample limits.
- Code commit / worktree:
  - `0ce3c7899ec789259a68c32cb513fc039729c20e`.
  - Dirty worktree: true.
- Stage2 checkpoint:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - sha256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Eval data:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl`.
- Output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/internal_diagnostics_step1200_20260627_223829`.
- Script / command:
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.stage_diagnostics --run-id clean_qwen3_stage2_norm01_internal_diag_20260627_223829 --stage stage2 --checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl --output-dir outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/internal_diagnostics_step1200_20260627_223829 --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --protocol protocol_c_tool_observation --focus-action-im-end --variant tgvf_v2_bidirectional --encoder-adapter-type bidirectional --max-image-resolution 512 --fvt-position-mode native_source_grid --dtype bfloat16 --attn-implementation sdpa --device cuda:0 --device-map cuda:0 --tasks all --readout-max-samples 200 --distribution-max-samples 200 --query-max-groups 50 --query-require-groups 0 --seed 20260525 --execute`.
- GPUs:
  - `CUDA_VISIBLE_DEVICES=0`.
- tmux:
  - `clean_stage2_norm01_diag_20260627_223829`.
  - Retry after serialization fix:
    `clean_stage2_norm01_diag_retry1_20260627_223829`.
  - Retry after PEFT/Qwen3 position-id fix:
    `clean_stage2_norm01_diag_retry2_20260627_223829`.
- Started:
  - 2026-06-27T22:38:29+09:00.
- Finished:
  - 2026-06-27T22:58:40+09:00.
- Metrics:
  - Readout, n=200:
    `target_only=0.870`, `random=0.405`, `wrong_same=0.330`,
    `wrong_diff=0.281`, `mean_nll_correct_D=1.62755`.
  - Query sensitivity, groups=46 / items=200:
    `retrieval_top1=0.220`, `retrieval_top2=0.455`, `MRR=0.48408`,
    `mean_diagonal_gap=-0.03080`.
  - FVT distribution, n=200:
    `avg_manifold_loss=0.45436`, `avg_norm_D=42.9330`,
    `avg_norm_V_merge=20.6838`, `norm_ratio_D_to_Vmerge=2.14251`,
    `finite_rate=1.0`.
  - Reports:
    `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/internal_diagnostics_step1200_20260627_223829/readout/readout_eval_report.json`,
    `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/internal_diagnostics_step1200_20260627_223829/query_sensitivity/query_sensitivity_report.json`,
    `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/internal_diagnostics_step1200_20260627_223829/fvt_distribution/fvt_distribution_report.json`.
- Analysis:
  - First attempt failed before readout/query/distribution execution while
    writing `runtime_config.json`: `_json_safe` did not serialize `set`.
  - Code patch added deterministic `set`/`frozenset` JSON conversion in
    `revisit_vlm_clean.stage_diagnostics._json_safe`.
  - Verification after patch:
    `PYTHONPATH=revisit_vlm_clean/src:src pytest -q revisit_vlm_clean/tests/test_stage_diagnostics.py`
    passed, `5 passed`.
  - Second attempt passed JSON writing but failed before readout execution:
    PEFT-wrapped Stage2 model did not expose Qwen3
    `compute_3d_position_ids` to the readout input builder.
  - Code patch made `_compute_qwen3_position_ids_for_sequence` unwrap
    PEFT-like models via `get_base_model()`/nested `.model` for position
    computation while still using the wrapper's input embeddings.
  - Verification after second patch:
    `PYTHONPATH=revisit_vlm_clean/src:src pytest -q tests/test_tgvf_v3_stage1.py revisit_vlm_clean/tests/test_stage_diagnostics.py`
    passed, `9 passed`.
  - Retrying the same diagnostic run id/output after both fixes.
  - The internal diagnostic path is Stage1-style and does not explicitly enable
    Qwen3 DeepStack injection. This is acceptable for D/readout/query
    regression checking, but benchmark evaluation must explicitly enable
    DeepStack with training-matched scope.
  - Compared with the source Stage1 diagnostics, Stage2 preserves the D/FVT
    distribution scale (`norm_ratio_D_to_Vmerge` 2.1425 vs 2.1258), but loses
    most of the Stage1 discrimination behavior: readout `wrong_same` falls
    from 0.900 to 0.330 and query `top1` from 0.700 to 0.220.
- Conclusion:
  - Stage2 significantly degrades the internal D readout/retrieval behavior
    under this Stage1-style diagnostic, despite keeping D norm/manifold scale
    close to the Stage1 checkpoint. Treat benchmark results from this Stage2
    checkpoint as suspect until compared against baseline with training-matched
    DeepStack enabled.
- Comparable to baseline:
  - Yes, intended to be comparable to the Stage1 internal diagnostics above.

### EXP-20260627-2220-stage3-grpo-native-smoke

- Status:
  - DONE.
- Question:
  - Does the new Stage3 GRPO `native_single_focus` path run a real Stage2/TGVF
    rollout, replay generated-token logprobs, perform one bounded GRPO optimizer
    step, and save a native Stage3 checkpoint?
- Baseline anchor:
  - No benchmark baseline. This is an implementation smoke for the Stage3 RL
    training loop.
- Intended diff:
  - Use the newly implemented clean Stage3 GRPO executor with
    `runtime_backend=native_single_focus`.
  - Use the clean Stage2 step1200 checkpoint as policy initialization.
  - Run only one prompt group with `group_size=2` and `max_tool_calls=1`.
- Allowed changed variables:
  - Stage3 GRPO smoke code path and output directory.
  - Sampling temperature/top-p for rollout smoke.
- Not allowed to change:
  - Stage3 RL data file.
  - Stage2 source checkpoint.
  - Protocol-C tool-observation contract.
- Code commit / worktree:
  - Base commit: `343bd730cbcaf235294621e264a94abc164068ef`.
  - Dirty worktree: Stage3 GRPO implementation plus native logprob plumbing.
- Stage1 checkpoint:
  - Indirectly from the Stage2 checkpoint config.
- Stage1 processor:
  - Indirectly from the Stage2 checkpoint config.
- Stage2 checkpoint/output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - sha256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
  - global_step: `1200`.
  - model/processor:
    `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
  - protocol: `protocol_c_tool_observation`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
  - Rows: `20000`.
  - sha256:
    `2e39a1dadcc020001bd3d763635f461d2b7dc6d94bfb3cfecdb9bb20240fa758`.
- Validation data:
  - Not used in this smoke.
- Benchmark output:
  - Not used in this smoke.
- Script / command:
  - Plan:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.train_stage3_grpo --write-plan --run-id stage3_grpo_native_smoke_clean_stage2_step1200_20260627 --rl-data-path revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl --policy-checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --output-dir outputs/stage3_grpo/native_smoke_clean_stage2_step1200_20260627 --runtime-backend native_single_focus --group-size 2 --per-device-prompt-batch-size 1 --max-tool-calls 1 --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 96 --max-new-tokens 160 --temperature 0.8 --top-p 0.95 --judge-mode cache_only --learning-rate 1e-6 --max-steps 1`.
  - Preflight:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/native_smoke_clean_stage2_step1200_20260627/stage3_grpo_training_plan.json --preflight-only`.
  - Launch:
    `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/native_smoke_clean_stage2_step1200_20260627/stage3_grpo_training_plan.json --launch-training`.
- GPUs:
  - Planned: `CUDA_VISIBLE_DEVICES=0`.
- tmux:
  - None. Direct smoke command.
- Started:
  - 2026-06-27T22:24:30+09:00.
- Finished:
  - 2026-06-27T22:29:10+09:00.
- Metrics:
  - Preflight: passed.
  - Launch status: `stage3_grpo_native_step_completed`.
  - Native update status: `native_grpo_update_completed`.
  - group_count: `1`.
  - rollout_count: `2`.
  - replayed_tokens: `44`.
  - token_count: `44`.
  - loss: `0.10806262493133545`.
  - policy_loss: `-0.0`.
  - kl: `5.403131484985352`.
  - clip_fraction: `0.04545454680919647`.
  - grad_norm: `0.25591158866882324`.
  - mean_reward: `-0.5`.
  - Rollout debug rows: `2`.
  - Reward breakdown rows: `2`.
  - Native checkpoint:
    `outputs/stage3_grpo/native_smoke_clean_stage2_step1200_20260627/checkpoint_step_1.pt`.
  - Checkpoint schema: `stage3_grpo_native_checkpoint_v0`.
  - Checkpoint size: about `2.0G`.
  - Saved state: `qwen_lora` has `504` keys; `tgvf_module` has `26` keys.
- Analysis:
  - First launch attempt failed during focus-segment logprob replay with Qwen3-VL
    `mm_token_type_ids` / `attention_mask` sequence-length mismatch.
  - Fixed Stage3 replay input construction so generated text ids extend
    `mm_token_type_ids` with language-token zeros and stale rope/cache fields
    are removed before replay.
  - Retry loaded the real Stage2 checkpoint, generated two sampled rollouts,
    replayed policy/reference logprobs, performed one optimizer step, and saved
    a native Stage3 checkpoint.
  - Judge caches were intentionally absent in cache-only mode, so focus/ground
    judge rewards were cache-miss rewards for this smoke.
- Conclusion:
  - The Stage3 GRPO native single-focus smoke path is now executable against the
    clean Stage2 step1200 checkpoint for a one-prompt, two-rollout bounded
    update.
- Comparable to baseline:
  - No. This is a code-path smoke, not a benchmark comparison.
- Follow-up:
  - If this passes, inspect rollout/reward/train metrics before scaling beyond
    a smoke.

### EXP-20260627-230746-clean-qwen3-original-coredev2511-baseline

- Status: FAILED_NO_OUTPUT.
- Question:
  - What is the clean original Qwen3-VL baseline on the CoreDev-2511 external
    benchmark subset before comparing the new Stage2 checkpoint?
- Baseline anchor:
  - This run is the baseline anchor for the upcoming clean Stage2 comparison.
  - It does not compare to historical BLINK-120 or legacy full-benchmark labels.
- Intended diff:
  - Run original Qwen3-VL-8B-Thinking only, without loading any TGVF checkpoint.
  - Use the clean benchmark runner, current V3 external parser/scorer, and the
    explicit CoreDev-2511 manifest.
- Allowed changed variables:
  - None; this is the original-model baseline measurement for this clean table.
- Not allowed to change:
  - Manifest path/hash, sample set, max image resolution, parser/scorer,
    runner backend, and model/processor.
- Code commit / worktree:
  - `27544f22c31773fd6d7eb297f680296d60b61885`.
  - Dirty worktree: false.
- Model / processor:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Checkpoint path:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Benchmark:
  - Subset id: `core_balanced_dev_2511_seed20260625`.
  - Human label: `CoreDev-2511`.
  - Manifest:
    `revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json`.
  - Manifest file sha256:
    `3a013b2bcc64316054d28239a3cea3f44211cadbfe19787be3b7f285620fa5c1`.
  - Manifest internal hash:
    `a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`.
  - Benchmark root:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks`.
  - Sample count: `2511`.
  - Allocation:
    `vstar_bench=191`, `blink=420`, `hr_bench_4k=200`,
    `mmmu_pro=300`, `mathvista=300`, `mathverse=500`,
    `ocrbench_v2=600`.
- Output:
  - `outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_20260627_230746`.
- Evaluation identity:
  - Eval family: `project_native_external`.
  - Mode: `original`.
  - Runner backend: `qwen3_original`.
  - TGVF protocol field: `protocol_c_tool_observation` (schema identity only;
    original mode does not use TGVF).
  - Post-TGVF continuation: `natural_continue`.
  - Post-TGVF forward mode: `kv_cache` (schema-required, not used by original
    backend).
  - DeepStack: disabled/noop for original backend.
  - Parser/scorer: `revisit_vlm_clean.scoring.parse_and_score:v3_external`,
    scoring backend `auto`.
  - Max image resolution: `512`.
  - Max answer tokens: `128`.
- Script / command:
  - Four shard commands, each with `--num-shards 4`, one `--shard-index`,
    `CUDA_VISIBLE_DEVICES=<gpu>`, `--device cuda:0`, `--device-map cuda:0`,
    and output under `shards/shard_<index>`.
  - Merge command after completion:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.merge_benchmark --output-dir outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_20260627_230746/merged --run-id clean_qwen3_original_coredev2511_4shard_20260627_230746 --expected-num-shards 4 --expected-source-manifest-hash a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579 outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_20260627_230746/shards/shard_0 outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_20260627_230746/shards/shard_1 outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_20260627_230746/shards/shard_2 outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_20260627_230746/shards/shard_3`.
- GPUs:
  - Planned: `0,1,2,3`, one shard per GPU.
- tmux:
  - `clean_qwen3_original_coredev2511_4shard_20260627_230746_s0`.
  - `clean_qwen3_original_coredev2511_4shard_20260627_230746_s1`.
  - `clean_qwen3_original_coredev2511_4shard_20260627_230746_s2`.
  - `clean_qwen3_original_coredev2511_4shard_20260627_230746_s3`.
- Started:
  - 2026-06-27T23:08:34+09:00.
  - Relaunched after preflight fixes: 2026-06-27T23:13:12+09:00.
- Finished:
  - 2026-06-27T23:59:27+09:00.
- Metrics:
  - No valid shard metrics.
  - `shards/shard_0` through `shards/shard_3` existed but contained no result
    files when checked after all tmux sessions had exited.
- Analysis:
  - Preflight dry-run passed with the expected manifest hash, parser/scorer,
    clean git commit, original mode, and `qwen3_original` backend.
  - First launch attempt exited before inference on all shards because the
    command incorrectly passed the manifest file sha256 as `--manifest-hash`.
    The runner expects the manifest's internal `manifest_hash` field instead.
    No benchmark rows or summaries were written by that failed attempt.
  - A second preflight with the internal manifest hash and clean benchmark root
    materialized all `2511` rows successfully; `benchmark_sources.json`
    reported all `24` source files present.
  - The relaunched shard tmux sessions exited without writing shard outputs.
    The shard launch did not tee stdout/stderr to persistent log files, so the
    exact shard failure reason is unavailable from the finished sessions.
  - This run is invalid for comparison and must be rerun with per-shard logs
    before it can serve as the CoreDev-2511 original baseline.
- Conclusion:
  - Invalid run; no benchmark rows were produced.
- Comparable to baseline:
  - No. It was intended to be the baseline but produced no shard outputs.

### EXP-20260627-234638-clean-qwen3-stage2-norm01-internal-baseqwen

- Status: DONE.
- Question:
  - Does the new clean Stage2 checkpoint preserve the old legacy-style internal
    D/readout/query behavior when only the Stage2 `tgvf_module` is evaluated
    with the base Qwen readout, without loading Stage2 `qwen_lora`?
- Baseline anchor:
  - Stage1 source diagnostics:
    `outputs/clean_training/qwen3_stage1_norm01_manifold0_4gpu_20260627_133252/stage1_micro4/internal_diagnostics_step2000_20260627_154418`.
  - Earlier Stage2 internal diagnostic with LoRA loaded:
    `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/internal_diagnostics_step1200_20260627_223829`.
  - Historical legacy readout behavior used Stage2 `tgvf_module` with base Qwen
    readout and did not load Stage2 LoRA.
- Intended diff:
  - Use the same new Stage2 checkpoint and same internal diagnostic samples as
    the previous Stage2 diagnostic.
  - Disable Stage2 LoRA loading via `--no-stage2-load-lora`, so capture/readout
    use base Qwen while the D/FVT module comes from the Stage2 checkpoint.
- Allowed changed variables:
  - Diagnostic-only code now exposes `--stage2-load-lora/--no-stage2-load-lora`.
  - Runtime semantic: `stage2_tgvf_module_with_base_qwen_readout`.
- Not allowed to change:
  - Checkpoint, processor, protocol, eval JSONL, sample order, sample limits,
    max image resolution, FVT position mode, dtype, or diagnostic metric code.
- Code commit / worktree:
  - Base commit: `27544f22c31773fd6d7eb297f680296d60b61885`.
  - Dirty worktree: true.
  - Dirty executable changes are limited to the diagnostic-only LoRA loading
    switch in `revisit_vlm_clean/src/revisit_vlm_clean/stage_diagnostics.py`,
    `revisit_vlm_clean/src/revisit_vlm_clean/cli/stage_diagnostics.py`, plus
    its targeted test.
  - Verification before launch:
    `PYTHONPATH=revisit_vlm_clean/src:src pytest -q revisit_vlm_clean/tests/test_stage_diagnostics.py`
    passed, `6 passed`.
- Checkpoint:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - sha256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Model / processor:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Eval data:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl`.
  - Rows: `867`.
  - sha256:
    `de61c731eb961825a77df587cd76c00eabfea75b5c6003096f3cc7f1a51dd82d`.
- Output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/internal_diagnostics_step1200_baseqwen_20260627_234638`.
- Evaluation identity:
  - Eval family: `internal_diagnostic`.
  - Stage: `stage2`.
  - Diagnostic kind: `stage1_style_fvt_readout_regression`.
  - Stage2 LoRA: disabled intentionally.
  - Forward semantics:
    `stage2_tgvf_module_with_base_qwen_readout; qwen_lora ignored by request`.
  - Tasks: `readout,query,distribution`.
  - Readout max samples: `200`.
  - Distribution max samples: `200`.
  - Query max groups: `50`.
  - Query min targets per image: `3`.
  - Query require groups: `0`.
  - Protocol: `protocol_c_tool_observation`.
  - Focus action im_end: true.
  - Max image resolution: `512`.
  - FVT position mode: `native_source_grid`.
  - Capture mode: `teacher_forced`.
  - DeepStack: not explicitly injected in this Stage1-style internal diagnostic.
- Script / command:
  - `CUDA_VISIBLE_DEVICES=7 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.stage_diagnostics --run-id clean_qwen3_stage2_norm01_internal_diag_baseqwen_20260627_234638 --stage stage2 --checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl --output-dir outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/internal_diagnostics_step1200_baseqwen_20260627_234638 --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --protocol protocol_c_tool_observation --variant tgvf_v2_bidirectional --num-foveated-tokens none --encoder-adapter-type bidirectional --max-image-resolution 512 --fvt-position-mode native_source_grid --dtype bfloat16 --attn-implementation sdpa --device cuda:0 --device-map cuda:0 --tasks readout,query,distribution --readout-max-samples 200 --distribution-max-samples 200 --query-max-groups 50 --query-require-groups 0 --query-min-targets-per-image 3 --eval-workers 1 --no-stage2-load-lora --seed 20260525 --focus-action-im-end --no-use-fvt-cache --execute`.
- GPUs:
  - Planned: `CUDA_VISIBLE_DEVICES=7`.
- tmux:
  - `clean_stage2_baseqwen_diag_20260627_234638`.
- Started:
  - 2026-06-27T23:48:26+09:00.
- Finished:
  - 2026-06-27T23:59:27+09:00.
- Metrics:
  - Runtime confirmed `stage2_load_lora=false`.
  - Runtime confirmed checkpoint contains `qwen_lora`, but it was not loaded:
    `available_in_checkpoint=true`, `loaded=false`,
    `reason=stage2_load_lora_disabled`.
  - Readout, n=`200`:
    - `mean_nll_correct_D=1.407724609375`.
    - `pct_correct_D_beats_target_only=0.995`.
    - `pct_correct_D_beats_random=1.0`.
    - `pct_correct_D_beats_wrong_same=0.905`.
    - `pct_correct_D_beats_wrong_diff=0.84375`.
  - Query sensitivity:
    - `retrieval_top1=0.700`.
    - `retrieval_top2=0.915`.
    - `MRR=0.8350000000000002`.
    - `mean_diagonal_gap=0.099326171875`.
  - Distribution:
    - `avg_manifold_loss=0.46140419349074363`.
    - `avg_norm_D=42.51886070251465`.
    - `avg_norm_V_merge=20.683768496513366`.
    - `norm_ratio_D_to_Vmerge=2.1240311511843672`.
    - `finite_rate=1.0`.
  - Comparison table:
    - Stage1 source norm01:
      `NLL=1.4056`, `beats_target=0.995`, `beats_random=1.000`,
      `beats_wrong_same=0.900`, `beats_wrong_diff=0.938`,
      `query_top1=0.700`, `query_top2=0.915`, `MRR=0.834`,
      `diag_gap=0.0998`, `norm_ratio=2.126`, `manifold=0.4620`.
    - Stage2 LoRA-loaded clean diagnostic:
      `NLL=1.6275`, `beats_target=0.870`, `beats_random=0.405`,
      `beats_wrong_same=0.330`, `beats_wrong_diff=0.281`,
      `query_top1=0.220`, `query_top2=0.455`, `MRR=0.484`,
      `diag_gap=-0.0308`, `norm_ratio=2.143`, `manifold=0.4544`.
    - Stage2 base-Qwen diagnostic from this run:
      `NLL=1.4077`, `beats_target=0.995`, `beats_random=1.000`,
      `beats_wrong_same=0.905`, `beats_wrong_diff=0.844`,
      `query_top1=0.700`, `query_top2=0.915`, `MRR=0.835`,
      `diag_gap=0.0993`, `norm_ratio=2.124`, `manifold=0.4614`.
- Analysis:
  - This run resolves the apparent Stage2 internal collapse as a diagnostic
    semantics issue, not as Stage2 `tgvf_module` destruction.
  - With Stage2 LoRA disabled, the new Stage2 checkpoint's TGVF module nearly
    matches the Stage1 source on readout and query:
    `wrong_same` improves from the LoRA-loaded `0.330` back to `0.905`, and
    query `top1` improves from `0.220` back to `0.700`.
  - The remaining drop in `wrong_diff` relative to Stage1 (`0.844` vs `0.938`)
    should be tracked, but it is not the catastrophic failure suggested by the
    LoRA-loaded diagnostic.
  - D/FVT scale remains aligned with Stage1 (`norm_ratio` `2.124` vs `2.126`).
- Conclusion:
  - The clean Stage2 checkpoint preserves the legacy-style internal
    Stage2-TGVF/base-Qwen D readout/query behavior.
  - The previous clean Stage2 internal diagnostic was not comparable to legacy
    readout because it loaded Stage2 LoRA into the readout path.
  - For future internal D-module regression checks, report both modes explicitly
    if needed: `stage2_load_lora=false` for legacy-comparable TGVF-module
    readout; `stage2_load_lora=true` for full Stage2-adapted model behavior.
- Comparable to baseline:
  - Yes, intended to be comparable to legacy Stage2 TGVF-module readout and to
    Stage1 source internal diagnostics, but not to the previous LoRA-loaded
    clean Stage2 internal diagnostic except as a semantic ablation.

### EXP-20260628-001951-clean-qwen3-original-coredev2511-baseline-logged-rerun

- Status: FAILED_SCORING_NO_ROWS.
- Question:
  - What is the clean original Qwen3-VL baseline on CoreDev-2511 after fixing
    the previous no-output benchmark launch by rerunning with persistent
    per-shard logs and exit-code files?
- Baseline anchor:
  - Replaces invalid no-output run:
    `EXP-20260627-230746-clean-qwen3-original-coredev2511-baseline`.
- Intended diff:
  - Same benchmark identity as the invalid run.
  - Add persistent per-shard `shard_<i>.log` and `exit_code.txt`.
  - Do not change model, processor, sample set, parser/scorer, mode, max
    resolution, or runner backend.
- Diagnostic precheck:
  - A one-sample side smoke from the same manifest succeeded and wrote
    `rows.jsonl`, `summary.json`, and `run_config.txt` under
    `outputs/clean_benchmarks/debug_one_sample_coredev_20260628_0013/run_registered`.
  - The first failed one-sample attempt used an unregistered debug subset id and
    failed before model loading; it is unrelated to the original benchmark
    failure.
- Code commit / worktree:
  - `253edac8645dcf2cded44614b41ab8cae0d79bc1`.
  - Dirty worktree before launch: true only because this ledger entry is being
    added.
- Model / processor:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Checkpoint path:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Benchmark:
  - Subset id: `core_balanced_dev_2511_seed20260625`.
  - Human label: `CoreDev-2511`.
  - Manifest:
    `revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json`.
  - Manifest file sha256:
    `3a013b2bcc64316054d28239a3cea3f44211cadbfe19787be3b7f285620fa5c1`.
  - Manifest internal hash:
    `a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`.
  - Benchmark root:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks`.
  - Sample count: `2511`.
  - Allocation:
    `vstar_bench=191`, `blink=420`, `hr_bench_4k=200`,
    `mmmu_pro=300`, `mathvista=300`, `mathverse=500`,
    `ocrbench_v2=600`.
- Output:
  - `outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_logged_20260628_001951`.
- Evaluation identity:
  - Eval family: `project_native_external`.
  - Mode: `original`.
  - Runner backend: `qwen3_original`.
  - Post-TGVF continuation: `natural_continue`.
  - Post-TGVF forward mode: `kv_cache` (schema-required, not used by original
    backend).
  - DeepStack: disabled/noop for original backend.
  - Parser/scorer: `revisit_vlm_clean.scoring.parse_and_score:v3_external`,
    scoring backend `auto`.
  - Max image resolution: `512`.
  - Max answer tokens: `128`.
- Script / command:
  - Four shard commands, one per GPU `0,1,2,3`, each with
    `CUDA_VISIBLE_DEVICES=<gpu>`, `--device cuda:0`, `--device-map cuda:0`,
    `--num-shards 4`, and one `--shard-index`.
  - Each shard writes under `shards/shard_<i>` and logs to
    `logs/shard_<i>.log`.
  - Each shard writes process status to `logs/shard_<i>.exit_code.txt`.
  - Merge command after completion:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.merge_benchmark --output-dir outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_logged_20260628_001951/merged --run-id clean_qwen3_original_coredev2511_4shard_logged_20260628_001951 --expected-num-shards 4 --expected-source-manifest-hash a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579 outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_logged_20260628_001951/shards/shard_0 outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_logged_20260628_001951/shards/shard_1 outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_logged_20260628_001951/shards/shard_2 outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_logged_20260628_001951/shards/shard_3`.
- GPUs:
  - Planned: `0,1,2,3`, one shard per GPU.
- tmux:
  - `clean_qwen3_original_coredev2511_logged_20260628_001951_s0`.
  - `clean_qwen3_original_coredev2511_logged_20260628_001951_s1`.
  - `clean_qwen3_original_coredev2511_logged_20260628_001951_s2`.
  - `clean_qwen3_original_coredev2511_logged_20260628_001951_s3`.
- Started:
  - 2026-06-28T00:19:51+09:00.
- Finished:
  - 2026-06-28T01:00:26+09:00.
- Metrics:
  - No valid shard rows were written.
  - All four shard processes exited with code `1`.
  - Logs were preserved:
    - `logs/shard_0.log`.
    - `logs/shard_1.log`.
    - `logs/shard_2.log`.
    - `logs/shard_3.log`.
  - Each shard completed model inference and reached OCRBench-v2 official
    scoring before failing.
- Analysis:
  - The prior no-output benchmark failure was reproduced with persistent logs.
  - Root cause was not model loading or generation. The crash happened after
    inference, inside OCRBench-v2 official scoring.
  - Shards `0`, `1`, and `3` failed because NLTK `wordnet` was missing for
    OCRBench-v2 METEOR computation.
  - Shard `2` additionally exposed a clean adapter bug: OCRBench-v2 parquet
    rows do not contain an `eval` column, but the official script requires
    `data_item["eval"]` for `type == "text counting en"`.
  - The clean runner also had a robustness bug: it only wrote benchmark rows
    after scoring, so a scoring exception discarded all generated rows.
  - Fixes implemented after this failed run:
    - Downloaded NLTK `wordnet` and `omw-1.4` into
      `/home/dredvpn009/nltk_data`.
    - Added clean OCRBench-v2 `text counting en` eval-method inference:
      numeric-only answers use `regression`; mixed textual answers use
      `exact match`.
    - Added benchmark runner progress logging.
    - Added scoring-exception fallback so generated rows survive as
      non-comparable outputs if scoring fails.
    - Added targeted tests for OCRBench-v2 text counting and scoring failure
      row preservation.
- Conclusion:
  - Invalid run; excluded from comparison.
  - A corrected rerun is required from the fixed code commit.
- Comparable to baseline:
  - No. It produced no shard rows and is retained only as a failure diagnosis.

### EXP-20260628-010859-clean-qwen3-original-coredev2511-baseline-scoringfix

- Status: DONE.
- Question:
  - What is the valid clean original Qwen3-VL baseline on CoreDev-2511 after
    fixing OCRBench-v2 official scoring and benchmark failure-output handling?
- Baseline anchor:
  - Replaces invalid no-output runs:
    - `EXP-20260627-230746-clean-qwen3-original-coredev2511-baseline`.
    - `EXP-20260628-001951-clean-qwen3-original-coredev2511-baseline-logged-rerun`.
- Intended diff:
  - Same benchmark identity as the prior attempts.
  - Use fixed code commit `a48ef3f9850b4dc4b01fdce74af2136eb5b0ffff`.
  - NLTK `wordnet` and `omw-1.4` are installed under
    `/home/dredvpn009/nltk_data` for OCRBench-v2 official METEOR.
  - Runner logs progress and preserves generated rows as non-comparable output
    if scoring fails.
- Not allowed to change:
  - Model, processor, sample set, manifest hash, max image resolution,
    mode, runner backend, parser/scorer, or benchmark root.
- Code commit / worktree:
  - `a48ef3f9850b4dc4b01fdce74af2136eb5b0ffff`.
  - Dirty worktree before launch: true only because this ledger entry is being
    added.
  - Verification before launch:
    `PYTHONPATH=revisit_vlm_clean/src:src pytest -q revisit_vlm_clean/tests/test_scoring.py revisit_vlm_clean/tests/test_benchmark_data.py`
    passed, `38 passed`.
  - Real OCRBench-v2 official scoring smoke passed for `text counting en` and
    `full-page OCR en`.
- Model / processor:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Checkpoint path:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Benchmark:
  - Subset id: `core_balanced_dev_2511_seed20260625`.
  - Human label: `CoreDev-2511`.
  - Manifest:
    `revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json`.
  - Manifest file sha256:
    `3a013b2bcc64316054d28239a3cea3f44211cadbfe19787be3b7f285620fa5c1`.
  - Manifest internal hash:
    `a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`.
  - Benchmark root:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks`.
  - Sample count: `2511`.
  - Allocation:
    `vstar_bench=191`, `blink=420`, `hr_bench_4k=200`,
    `mmmu_pro=300`, `mathvista=300`, `mathverse=500`,
    `ocrbench_v2=600`.
- Output:
  - `outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_scoringfix_20260628_010859`.
- Evaluation identity:
  - Eval family: `project_native_external`.
  - Mode: `original`.
  - Runner backend: `qwen3_original`.
  - Post-TGVF continuation: `natural_continue`.
  - Post-TGVF forward mode: `kv_cache` (schema-required, not used by original
    backend).
  - DeepStack: disabled/noop for original backend.
  - Parser/scorer: `revisit_vlm_clean.scoring.parse_and_score:v3_external`,
    scoring backend `auto`.
  - Max image resolution: `512`.
  - Max answer tokens: `128`.
- Script / command:
  - Four shard commands, one per GPU `0,1,2,3`, each with
    `CUDA_VISIBLE_DEVICES=<gpu>`, `--device cuda:0`, `--device-map cuda:0`,
    `--num-shards 4`, and one `--shard-index`.
  - Each shard writes under `shards/shard_<i>` and logs to
    `logs/shard_<i>.log`.
  - Each shard writes process status to `logs/shard_<i>.exit_code.txt`.
  - Merge command after completion:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.merge_benchmark --output-dir outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_scoringfix_20260628_010859/merged --run-id clean_qwen3_original_coredev2511_4shard_scoringfix_20260628_010859 --expected-num-shards 4 --expected-source-manifest-hash a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579 outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_scoringfix_20260628_010859/shards/shard_0 outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_scoringfix_20260628_010859/shards/shard_1 outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_scoringfix_20260628_010859/shards/shard_2 outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_scoringfix_20260628_010859/shards/shard_3`.
- GPUs:
  - Planned: `0,1,2,3`, one shard per GPU.
- tmux:
  - `clean_qwen3_original_coredev2511_scoringfix_20260628_010859_s0`.
  - `clean_qwen3_original_coredev2511_scoringfix_20260628_010859_s1`.
  - `clean_qwen3_original_coredev2511_scoringfix_20260628_010859_s2`.
  - `clean_qwen3_original_coredev2511_scoringfix_20260628_010859_s3`.
- Started:
  - 2026-06-28T01:10:33+09:00.
- Finished:
  - 2026-06-28T01:57:09+09:00.
- Metrics:
  - Shard exits:
    - `shard_0`: `exit=0`, rows=`628`.
    - `shard_1`: `exit=0`, rows=`628`.
    - `shard_2`: `exit=0`, rows=`628`.
    - `shard_3`: `exit=0`, rows=`627`.
  - Merge:
    - Output:
      `outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_scoringfix_20260628_010859/merged`.
    - Rows: `2511`.
    - Scored rows: `2511`.
    - Comparable: true.
    - Comparability note: `deterministically merged clean benchmark shards`.
    - Manifest hash:
      `a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`.
    - Scoring errors: `0`.
    - Malformed rows: `0`.
  - Overall:
    - Accuracy: `0.19372487579257566` (`19.37%`).
    - Answer parse rate: `0.8403026682596575`.
    - Trigger rate: `0.0` (original mode).
  - By benchmark:
    - `vstar_bench`: n=`191`, acc=`0.5026178010471204`,
      parse=`0.93717277486911`.
    - `blink`: n=`420`, acc=`0.1976190476190476`,
      parse=`0.2523809523809524`.
    - `hr_bench_4k`: n=`200`, acc=`0.395`, parse=`0.625`.
    - `mmmu_pro`: n=`300`, acc=`0.17333333333333334`,
      parse=`1.0`.
    - `mathvista`: n=`300`, acc=`0.19666666666666666`,
      parse=`1.0`.
    - `mathverse`: n=`500`, acc=`0.008`, parse=`1.0`.
    - `ocrbench_v2`: n=`600`, acc=`0.18907193852526274`,
      parse=`1.0`.
  - Official scoring:
    - Official/scorer rows: `2320`.
    - Official-compatible rows: `200`.
    - Scorers:
      `official_blink_exact_match=420`,
      `official_compatible_hrbench4k_mc=200`,
      `official_mmmu_pro=300`,
      `official_mathvista=300`,
      `official_mathverse=500`,
      `official_ocrbench_v2=600`,
      `project_choice_exact_match=191`.
- Analysis:
  - The benchmark failure was fixed end to end. The prior no-output behavior
    was caused by scoring exceptions after generation, plus the runner only
    writing rows after scoring.
  - This rerun confirms the OCRBench-v2 fixes: all shards exited `0`,
    OCRBench-v2 scored all `600` rows, and merged rows contain no
    `scoring_error`.
  - The low BLINK parse rate (`0.2524`) is a property of the original Qwen3
    output/parser interaction under this clean table, not a runner failure.
    Inspecting row examples should come before drawing model conclusions.
  - MathVerse accuracy is very low (`0.008`) under the current official
    disabled-LLM scorer path; treat it as a baseline measurement for this clean
    setup and inspect rows before comparing against historical tables.
- Conclusion:
  - Valid CoreDev-2511 original Qwen3 baseline is available and comparable
    within the clean benchmark framework.
  - Use this run as the original baseline for subsequent clean Stage2
    CoreDev-2511 comparisons.
- Comparable to baseline:
  - This is the valid original baseline for CoreDev-2511.

### EXP-20260628-0216-clean-qwen3-original-coredev2511-maxans512

- Status: DONE.
- Question:
  - Does the low clean original Qwen3 CoreDev-2511 accuracy/parse rate come from
    the clean run using `max_answer_tokens=128`, rather than from a parser/scorer
    regression?
- Baseline anchor:
  - `EXP-20260628-010859-clean-qwen3-original-coredev2511-baseline-scoringfix`.
- Intended diff:
  - Change only original answer generation budget:
    `max_answer_tokens=128 -> 512`.
  - Keep the same model, processor, sample manifest/order, max image resolution,
    parser/scorer, scoring backend, mode, runner backend, benchmark root, and
    GPUs as the 128-token clean baseline.
  - Match the legacy fullbench original token budget used by
    `scripts/run_tgvf_v3_protocol_c_all_official_benchmarks_0_3.sh`
    (`ORIGINAL_MAX_TOKENS=512`).
- Not allowed to change for the generation run:
  - Parser/scorer code or settings at launch.
  - Prompt suffix / extra prompt.
  - Model, processor, benchmark root, manifest, max image resolution, scoring
    backend, runner backend, or DeepStack state.
- Code commit / worktree:
  - `1a5ad48a7db9a039fdee191697e7cba4558318e6`.
  - Dirty worktree before launch: false.
  - Post-run scorer fix:
    - `revisit_vlm_clean/src/revisit_vlm_clean/scoring.py` now supplies an
      empty OCRBench-v2 GT `bbox` for `VQA with position en` rows when the
      parquet metadata lacks one, preventing the official scorer from aborting
      the whole shard with `KeyError: 'bbox'`.
    - Covered by
      `test_score_output_rows_ocrbench_v2_missing_position_bbox_does_not_abort`.
    - This changes scoring robustness only; model generation outputs are
      unchanged.
- Model / processor:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
  - Processor: same path via `processor_id=null`.
- Checkpoint path:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Benchmark:
  - Subset id: `core_balanced_dev_2511_seed20260625`.
  - Human label: `CoreDev-2511`.
  - Manifest:
    `revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json`.
  - Manifest file sha256:
    `3a013b2bcc64316054d28239a3cea3f44211cadbfe19787be3b7f285620fa5c1`.
  - Manifest internal hash:
    `a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`.
  - Benchmark root:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks`.
  - Sample count: `2511`.
  - Sample overlap with baseline:
    `2511/2511`, same order.
  - Allocation:
    `vstar_bench=191`, `blink=420`, `hr_bench_4k=200`,
    `mmmu_pro=300`, `mathvista=300`, `mathverse=500`,
    `ocrbench_v2=600`.
- Output:
  - `outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_maxans512_20260628_0216`.
  - Initial merged output:
    `outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_maxans512_20260628_0216/merged`.
  - Final comparable rescored output:
    `outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_maxans512_20260628_0216_rescored_bboxfix_20260628_035759/merged`.
- Evaluation identity:
  - Eval family: `project_native_external`.
  - Mode: `original`.
  - Runner backend: `qwen3_original`.
  - Post-TGVF continuation: `natural_continue`.
  - Post-TGVF forward mode: `kv_cache` (schema-required, not used by original
    backend).
  - DeepStack: disabled/noop for original backend.
  - Parser/scorer: `revisit_vlm_clean.scoring.parse_and_score:v3_external`,
    scoring backend `auto`.
  - Prompt suffix / softforce prompt: empty.
  - Max image resolution: `512`.
  - Max answer tokens: `512`.
- Script / command:
  - Four shard commands, one per GPU `0,1,2,3`, each with
    `CUDA_VISIBLE_DEVICES=<gpu>`, `--device cuda:0`, `--device-map cuda:0`,
    `--num-shards 4`, and one `--shard-index`.
  - Base command:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.benchmark --run-id clean_qwen3_original_coredev2511_maxans512_20260628_0216_s<shard> --checkpoint-path /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --mode original --runner-backend qwen3_original --post-tgvf-forward-mode kv_cache --subset-id core_balanced_dev_2511_seed20260625 --manifest-path revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json --manifest-hash a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579 --benchmark-root /home/dredvpn009/Flash_Storage/datasets/benchmarks --output-dir outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_maxans512_20260628_0216/shards/shard_<shard> --max-image-resolution 512 --max-answer-tokens 512 --scoring-backend auto --dtype bfloat16 --device cuda:0 --device-map cuda:0 --attn-implementation sdpa --num-shards 4 --shard-index <shard> --execute`.
  - Merge command after completion:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.merge_benchmark --output-dir outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_maxans512_20260628_0216/merged --run-id clean_qwen3_original_coredev2511_maxans512_20260628_0216 --expected-num-shards 4 --expected-source-manifest-hash a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579 outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_maxans512_20260628_0216/shards/shard_0 outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_maxans512_20260628_0216/shards/shard_1 outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_maxans512_20260628_0216/shards/shard_2 outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_maxans512_20260628_0216/shards/shard_3`.
- GPUs:
  - Planned: `0,1,2,3`, one shard per GPU.
  - Preflight: GPUs `0-7` showed `0 MiB` used and `0%` utilization.
- tmux:
  - Planned orchestrator session:
    `clean_qwen3_original_coredev2511_maxans512_20260628_0216`.
- Started:
  - 2026-06-28T02:17:16+09:00.
- Finished:
  - Generation shards:
    - `shard_0`: 2026-06-28T03:41:06+09:00, `exit=0`, rows `628/628`.
    - `shard_1`: 2026-06-28T03:51:20+09:00, `exit=0`, rows `628/628`.
    - `shard_2`: 2026-06-28T03:45:43+09:00, `exit=0`, rows `628/628`.
    - `shard_3`: 2026-06-28T03:51:33+09:00, `exit=0`, rows `627/627`.
  - Initial merge: 2026-06-28T03:51:34+09:00.
  - Rescored merge: 2026-06-28T03:57:59+09:00 output family.
- Metrics:
  - Initial merged output was not used for final comparison:
    - It had `score=None` on `352/2511` rows because OCRBench-v2 official
      scoring raised `KeyError: 'bbox'` and the shard-level scorer catch left
      partially scored rows.
    - Its summary accuracy `0.302061` used `n_scored=2159`, so it was not
      denominator-comparable.
  - Final comparable rescored output:
    - `n_rows=2511`, `n_scored=2511`, `score_none=0`,
      `scoring_error=0`.
    - Overall: accuracy `0.308553`, answer parse rate `0.941458`,
      trigger rate `0.0`, malformed rate `0.0`.
    - By benchmark:
      - `blink`: n `420`, acc `0.464286`, parse `0.709524`.
      - `hr_bench_4k`: n `200`, acc `0.520000`, parse `0.890000`.
      - `mathverse`: n `500`, acc `0.048000`, parse `1.000000`.
      - `mathvista`: n `300`, acc `0.410000`, parse `1.000000`.
      - `mmmu_pro`: n `300`, acc `0.343333`, parse `1.000000`.
      - `ocrbench_v2`: n `600`, acc `0.204628`, parse `1.000000`.
      - `vstar_bench`: n `191`, acc `0.539267`, parse `0.984293`.
  - Baseline 128-token clean original:
    - Overall: accuracy `0.193725`, answer parse rate `0.840303`.
    - By benchmark:
      - `blink`: acc `0.197619`, parse `0.252381`.
      - `hr_bench_4k`: acc `0.395000`, parse `0.625000`.
      - `mathverse`: acc `0.008000`, parse `1.000000`.
      - `mathvista`: acc `0.196667`, parse `1.000000`.
      - `mmmu_pro`: acc `0.173333`, parse `1.000000`.
      - `ocrbench_v2`: acc `0.189072`, parse `1.000000`.
      - `vstar_bench`: acc `0.502618`, parse `0.937173`.
    - In-memory rescore with the fixed scorer changed `0/2511` baseline scores,
      confirming the fix does not move the 128-token anchor.
  - Truncation / parse diagnostics:
    - 128-token baseline hit max tokens on `2139/2511` rows; all `401`
      parse failures hit the 128-token cap.
    - 512-token run hit max tokens on `1536/2511` rows; all `147` parse
      failures hit the 512-token cap.
- Analysis:
  - The low clean original baseline was substantially caused by the 128-token
    answer cap. Raising the cap to 512 improves overall accuracy from
    `19.37%` to `30.86%` and parse rate from `84.03%` to `94.15%`.
  - The strongest improvements are on tasks where the Thinking model needs
    room to finish its final answer: BLINK, HRBench, MathVista, MMMU-Pro, and
    MathVerse.
  - OCRBench remains mostly unchanged, so its limitation is not mainly final
    answer truncation.
  - VStar improves modestly after correct rescoring; the initial 512 summary
    under-reported parse because the shard-level OCRBench scorer failure
    contaminated unrelated rows in the same shard.
- Conclusion:
  - Use `max_answer_tokens=512` for Qwen3 original Thinking baselines.
  - Treat the initial `.../merged` result from this run as a side artifact only;
    use the `..._rescored_bboxfix_20260628_035759/merged` result for tables.
  - Clean scorer now needs this OCRBench missing-bbox fix before future
    benchmark runs.
- Comparable to baseline:
  - Comparable to
    `EXP-20260628-010859-clean-qwen3-original-coredev2511-baseline-scoringfix`
    with only `max_answer_tokens` changed.

### EXP-20260628-0407-clean-qwen3-stage2-norm01-free-coredev2511

- Status: SIDE_RESULT / INVALID_FOR_BASELINE.
- Question:
  - After confirming the Qwen3 original parser/scorer baseline is healthy with
    `max_answer_tokens=512`, evaluate the current clean Stage2 method on the
    same CoreDev-2511 benchmark sample set.
- Baseline anchor:
  - Original baseline:
    `EXP-20260628-0216-clean-qwen3-original-coredev2511-maxans512`.
  - Baseline output:
    `outputs/clean_benchmarks/qwen3_original_coredev2511_4shard_maxans512_20260628_0216_rescored_bboxfix_20260628_035759/merged`.
- Intended diff:
  - Replace original Qwen3 generation with clean Stage2 TGVF free-router
    evaluation using the current Stage2 checkpoint.
  - Keep benchmark manifest/order, benchmark root, max image resolution,
    max answer tokens, parser/scorer, scoring backend, and no-extra-prompt
    policy fixed.
  - Enable Qwen3 DeepStack benchmark execution with training-matched
    `original_image_scope=through_answer`.
- Not allowed to change:
  - Benchmark sample set/order.
  - Prompt suffix / softforce prompt; this is `tgvf_free`.
  - Parser/scorer identity or scoring backend.
  - Stage2 checkpoint.
- Code commit / worktree:
  - Launch commit: `d17e84f58b3f63470fccdef9388fce1cc5618eaf`.
  - Dirty worktree before ledger entry: false.
- Model / processor:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Stage2 checkpoint:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - sha256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
  - Source training entry:
    `EXP-20260627-163250-clean-qwen3-stage2-norm01-mask075-deepstack`.
- Stage2 runtime eval JSONL:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`.
  - Rows: `1002`; focus/no-focus: `857/145`.
  - sha256:
    `3b719e1bd03a09741cc05dac3ed85e423a9b2c29209b07546792a6795cdbbfc5`.
- Benchmark:
  - Subset id: `core_balanced_dev_2511_seed20260625`.
  - Human label: `CoreDev-2511`.
  - Manifest:
    `revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json`.
  - Manifest file sha256:
    `3a013b2bcc64316054d28239a3cea3f44211cadbfe19787be3b7f285620fa5c1`.
  - Manifest internal hash:
    `a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`.
  - Benchmark root:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks`.
  - Sample count: `2511`.
  - Sample overlap/order vs original 512 baseline:
    `2511/2511`, same order.
  - Allocation:
    `vstar_bench=191`, `blink=420`, `hr_bench_4k=200`,
    `mmmu_pro=300`, `mathvista=300`, `mathverse=500`,
    `ocrbench_v2=600`.
- Output:
  - Smoke:
    `outputs/clean_benchmarks/qwen3_stage2_norm01_free_coredev2511_deepstack512_smoke1_20260628_0407`.
  - Full:
    `outputs/clean_benchmarks/qwen3_stage2_norm01_free_coredev2511_deepstack512_4shard_20260628_0407`.
- Evaluation identity:
  - Eval family: `project_native_external`.
  - Mode: `tgvf_free`.
  - Runner backend: `tgvf_stage2_qwen3_native`.
  - Post-TGVF continuation: `natural_continue`.
  - Post-TGVF forward mode: `no_kv_full_sequence`.
  - DeepStack: enabled; `original_image_scope=through_answer`;
    D DeepStack-like features disabled.
  - Stage2 D condition: `correct_D`.
  - Force prefix mode: `target_hint` (runtime default; only relevant for force).
  - Parser/scorer: `revisit_vlm_clean.scoring.parse_and_score:v3_external`,
    scoring backend `auto`.
  - Prompt suffix / softforce prompt: empty.
  - Max image resolution: `512`.
  - Max action tokens: `64`.
  - Max answer tokens: `512`.
- Script / command:
  - Smoke command uses one modulo shard with `--num-shards 2511 --shard-index 0`
    to run one real sample before the full benchmark.
  - Full benchmark uses four shard commands, one per GPU `0,1,2,3`, each with
    `CUDA_VISIBLE_DEVICES=<gpu>`, `--device cuda:0`, `--device-map cuda:0`,
    `--num-shards 4`, and one `--shard-index`.
  - Base full command:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.benchmark --run-id clean_qwen3_stage2_norm01_free_coredev2511_ds512_20260628_0407_s<shard> --checkpoint-path outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --mode tgvf_free --runner-backend tgvf_stage2_qwen3_native --post-tgvf-forward-mode no_kv_full_sequence --subset-id core_balanced_dev_2511_seed20260625 --manifest-path revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json --manifest-hash a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579 --benchmark-root /home/dredvpn009/Flash_Storage/datasets/benchmarks --output-dir outputs/clean_benchmarks/qwen3_stage2_norm01_free_coredev2511_deepstack512_4shard_20260628_0407/shards/shard_<shard> --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 512 --scoring-backend auto --dtype bfloat16 --device cuda:0 --device-map cuda:0 --attn-implementation sdpa --stage2-checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --stage2-eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl --stage2-d-condition correct_D --force-prefix-mode target_hint --deepstack-enabled --deepstack-original-image-scope through_answer --num-shards 4 --shard-index <shard> --execute`.
- GPUs:
  - Planned full benchmark: `0,1,2,3`.
  - Preflight at 2026-06-28T04:07:16+09:00:
    GPUs `0-7` showed `0 MiB` used and `0%` utilization.
- tmux:
  - Smoke: foreground preflight, completed successfully.
  - Full:
    `clean_qwen3_stage2_norm01_free_coredev2511_ds512_20260628_0407`.
- Started:
  - Smoke: 2026-06-28T04:09:xx+09:00.
  - Full: 2026-06-28T04:12:00+09:00.
- Finished:
  - Full generation/merge completed:
    - `shard_0`: 2026-06-28T04:28:06+09:00, `exit=0`, rows `628/628`.
    - `shard_1`: 2026-06-28T04:31:12+09:00, `exit=0`, rows `628/628`.
    - `shard_2`: 2026-06-28T04:31:19+09:00, `exit=0`, rows `628/628`.
    - `shard_3`: 2026-06-28T04:21:56+09:00, `exit=0`, rows `627/627`.
    - Merge: 2026-06-28T04:31:20+09:00.
- Metrics:
  - Smoke:
    - Output:
      `outputs/clean_benchmarks/qwen3_stage2_norm01_free_coredev2511_deepstack512_smoke1_20260628_0407`.
    - Rows: `1`; score `1.0`; parse `1.0`; trigger `0.0`.
    - Run config recorded `git_commit=d17e84f58b3f63470fccdef9388fce1cc5618eaf`,
      `dirty_worktree=false`, DeepStack enabled with
      `original_image_scope=through_answer`.
    - Smoke did not naturally trigger focus on its single VStar sample, so it
      validates model/runtime/scoring startup but not a triggered append case.
  - Full merged output:
    `outputs/clean_benchmarks/qwen3_stage2_norm01_free_coredev2511_deepstack512_4shard_20260628_0407/merged`.
  - Invalid summary:
    - `n_rows=2511`, `n_scored=738`, `score_none=1773`, `row_error=1773`.
    - Overall among scored rows only: accuracy `0.288618`, parse `0.293907`,
      trigger `0.047790`, focus-valid `0.159151`, append success `0.866667`,
      malformed `0.706093`.
    - Error distribution from `merged/rows.jsonl`:
      - `1520` rows: `ValueError: clean-native Stage2 requires path-backed image media`.
      - `246` rows: `RuntimeError: Expected mha_graph.execute(...)`.
      - `4` rows: `RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR`.
      - `3` rows: CUDA OOM.
    - Media failures covered all BLINK `420/420`, HRBench `200/200`,
      OCRBench-v2 `600/600`, and MMMU-Pro `300/300` rows.
    - Scored/non-error rows were mainly VStar, MathVista, and MathVerse, so the
      benchmark table is not denominator-comparable to the original baseline.
  - Follow-up smoke/fix validation after the invalid run:
    - HRBench embedded-base64 sample `index=191`: output
      `outputs/clean_benchmarks/qwen3_stage2_norm01_free_coredev2511_mediafix_smoke_hr_20260628`,
      `error=null`, parse `1.0`, score `1.0`.
    - BLINK multi-image direct sample `index=391`: output
      `outputs/clean_benchmarks/qwen3_stage2_norm01_free_coredev2511_mediafix_smoke_blink_multi_20260628`,
      `error=null`, parse `1.0`, score `1.0`, debug image count `3`.
    - BLINK multi-image forced append sample `index=391`: final validated output
      `outputs/clean_benchmarks/qwen3_stage2_norm01_force_coredev2511_mediafix_smoke_blink_multi_append_20260628_rerun3`,
      `append_success=true`, `D_shape=[719,4096]`,
      `fvt_position_mode=multi_image_inherit_source_visual_positions`,
      DeepStack used, parse `1.0`, score `1.0`, `error=null`.
    - Former MathVista `mha_graph.execute` sample `index=1803`: output
      `outputs/clean_benchmarks/qwen3_stage2_norm01_free_coredev2511_mediafix_smoke_mathvista_mha_20260628`,
      `error=null`, parse `1.0`; it did not naturally append.
- Analysis:
  - The full run is invalid because clean-native Stage2 inherited the legacy
    path-backed-image assumption while CoreDev-2511 includes parquet
    `image_struct`, embedded bytes, embedded base64, and multi-image samples.
  - A second triggered-append issue was exposed for multi-image samples:
    Qwen3 native `compute_3d_position_ids` cannot interpret one continuous FVT
    visual block as three separate appended image grids. The clean-native fix
    keeps single-image native position compute unchanged and falls back only for
    multi-image append to: native base positions for the original multi-image
    prefix plus inherited source visual positions for the FVT span.
  - The `mha_graph.execute` errors were not reproduced by the single formerly
    failing MathVista smoke after the media/append fixes. Keep `sdpa` for the
    rerun, but inspect full rerun row errors before reporting final metrics.
- Conclusion:
  - Do not use this output in result tables.
  - Rerun the same CoreDev-2511 Stage2 free benchmark after the clean-native
    media materialization and multi-image FVT position fallback patch.
- Comparable to baseline:
  - Not comparable. Same manifest/order and intended configuration, but
    `1773/2511` rows were unscored runtime failures.

### EXP-20260628-0510-clean-qwen3-stage2-norm01-free-coredev2511-mediafix

- Status: SIDE_RESULT / INVALID_FOR_BASELINE.
- Question:
  - Rerun the current clean Stage2 Qwen3 free method on the exact same
    CoreDev-2511 sample set after fixing clean-native benchmark media handling
    and multi-image FVT position fallback.
- Baseline anchor:
  - Original baseline:
    `EXP-20260628-0216-clean-qwen3-original-coredev2511-maxans512`.
  - Invalid predecessor:
    `EXP-20260628-0407-clean-qwen3-stage2-norm01-free-coredev2511`.
- Intended diff:
  - Same checkpoint, processor, manifest/order, parser/scorer, max image
    resolution, max answer tokens, no-extra-prompt free mode, DeepStack
    through-answer semantics, and `sdpa` attention as the invalid predecessor.
  - Code now includes clean-native Stage2 media materialization for path,
    parquet image bytes, embedded base64, and multi-image inputs; multi-image
    full-sequence append falls back to inherited source visual positions for
    the FVT span.
- Not allowed to change:
  - Benchmark sample set/order.
  - Stage2 checkpoint.
  - Parser/scorer identity.
  - Prompt suffix / softforce prompt; this is still `tgvf_free`.
- Code commit / worktree:
  - Fix commit: `2364aee113b4970b368a4a4c6d4aa8069a1712dc`.
  - Launch commit: `49db0636796dba4786ba39c5bb525e9a45b7fa89`.
  - Dirty worktree before ledger entry: false.
- Model / processor:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Stage2 checkpoint:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - sha256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Stage2 runtime eval JSONL:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`.
  - Rows: `1002`; focus/no-focus: `857/145`.
  - sha256:
    `3b719e1bd03a09741cc05dac3ed85e423a9b2c29209b07546792a6795cdbbfc5`.
- Benchmark:
  - Subset id: `core_balanced_dev_2511_seed20260625`.
  - Manifest:
    `revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json`.
  - Manifest internal hash:
    `a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`.
  - Benchmark root:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks`.
  - Sample count/order vs original baseline: `2511/2511`, same order.
- Output:
  - Full:
    `outputs/clean_benchmarks/qwen3_stage2_norm01_free_coredev2511_deepstack512_mediafix_4shard_20260628_0510`.
- Evaluation identity:
  - Eval family: `project_native_external`.
  - Mode: `tgvf_free`.
  - Runner backend: `tgvf_stage2_qwen3_native`.
  - Post-TGVF continuation: `natural_continue`.
  - Post-TGVF forward mode: `no_kv_full_sequence`.
  - DeepStack: enabled; `original_image_scope=through_answer`;
    D DeepStack-like features disabled.
  - Stage2 D condition: `correct_D`.
  - Prompt suffix / softforce prompt: empty.
  - Max image resolution: `512`.
  - Max action tokens: `64`.
  - Max answer tokens: `512`.
  - Attention implementation: `sdpa`.
- Smoke validation before launch:
  - `PYTHONPATH=revisit_vlm_clean/src:src pytest -q revisit_vlm_clean/tests/test_runner_backend.py revisit_vlm_clean/tests/test_scoring.py`
    passed: `41 passed`.
  - HRBench embedded-base64 free smoke passed with `error=null`.
  - BLINK multi-image direct free smoke passed with `error=null`.
  - BLINK multi-image forced append smoke passed with `append_success=true`,
    `D_shape=[719,4096]`, `fvt_position_mode=multi_image_inherit_source_visual_positions`,
    DeepStack used, and `error=null`.
  - Former MathVista `mha_graph.execute` sample smoke passed with `error=null`.
- Script / command:
  - Four shard commands, one per GPU `0,1,2,3`, each with
    `CUDA_VISIBLE_DEVICES=<gpu>`, `--device cuda:0`, `--device-map cuda:0`,
    `--num-shards 4`, and one `--shard-index`.
  - Base full command:
    `REVISIT_VLM_CLEAN_STAGE2_MEDIA_CACHE=outputs/clean_media_cache/stage2_native_coredev2511 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.benchmark --run-id clean_qwen3_stage2_norm01_free_coredev2511_ds512_mediafix_20260628_0510_s<shard> --checkpoint-path outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --mode tgvf_free --runner-backend tgvf_stage2_qwen3_native --post-tgvf-forward-mode no_kv_full_sequence --subset-id core_balanced_dev_2511_seed20260625 --manifest-path revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json --manifest-hash a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579 --benchmark-root /home/dredvpn009/Flash_Storage/datasets/benchmarks --output-dir outputs/clean_benchmarks/qwen3_stage2_norm01_free_coredev2511_deepstack512_mediafix_4shard_20260628_0510/shards/shard_<shard> --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 512 --scoring-backend auto --dtype bfloat16 --device cuda:0 --device-map cuda:0 --attn-implementation sdpa --stage2-checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --stage2-eval-jsonl data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl --stage2-d-condition correct_D --force-prefix-mode target_hint --deepstack-enabled --deepstack-original-image-scope through_answer --num-shards 4 --shard-index <shard> --execute`.
- GPUs:
  - Planned full benchmark: `0,1,2,3`.
- tmux:
  - Running:
    `clean_qwen3_stage2_norm01_free_coredev2511_ds512_mediafix_20260628_0510`.
- Started:
  - 2026-06-28T05:10:38+09:00.
- Finished:
  - 2026-06-28T05:11:07+09:00.
- Metrics:
  - Output:
    `outputs/clean_benchmarks/qwen3_stage2_norm01_free_coredev2511_deepstack512_mediafix_4shard_20260628_0510/merged`.
  - Invalid summary:
    - `n_rows=2511`, `n_scored=0`, parse `0.0`, malformed `1.0`.
    - Every row has `error="ModuleNotFoundError: No module named 'torch'"`.
    - Runtime never loaded the model (`heavy_runtime_loaded=false`).
- Analysis:
  - Launch bug: the tmux session was started with a non-login shell, so the
    project conda/Python environment was not active. This is not a model,
    parser, checkpoint, or media-materialization result.
  - A direct tmux validation with `bash -lc 'python -c "import torch; ..."'`
    succeeded, so the rerun must use a login shell / `bash -lc` inside tmux.
- Conclusion:
  - Do not use this output in result tables.
  - Rerun the same command in a tmux `bash -lc` environment.

### EXP-20260628-0520-clean-qwen3-stage2-norm01-free-coredev2511-mediafix-login

- Status: SIDE_RESULT / INVALID_FOR_BASELINE.
- Question:
  - Rerun `EXP-20260628-0510` with the same evaluation identity, but fix the
    tmux launch environment by running the orchestrator inside `bash -lc`.
- Baseline anchor:
  - Original baseline:
    `EXP-20260628-0216-clean-qwen3-original-coredev2511-maxans512`.
  - Invalid predecessor:
    `EXP-20260628-0510-clean-qwen3-stage2-norm01-free-coredev2511-mediafix`.
- Intended diff:
  - Only launch environment changes: tmux uses `bash -lc`, intended to make
    `torch` and the revisit-vlm environment available.
  - Checkpoint, model/processor, manifest/order, parser/scorer, no-extra-prompt
    free mode, DeepStack state, max resolution/tokens, and `sdpa` remain fixed.
- Code commit / worktree:
  - Fix commit: `2364aee113b4970b368a4a4c6d4aa8069a1712dc`.
  - Prior launch-record commit: `49db0636796dba4786ba39c5bb525e9a45b7fa89`.
  - Dirty worktree before ledger entry: false except this ledger update.
- Model / processor:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Stage2 checkpoint:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - sha256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Stage2 runtime eval JSONL:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`.
  - sha256:
    `3b719e1bd03a09741cc05dac3ed85e423a9b2c29209b07546792a6795cdbbfc5`.
- Benchmark:
  - Subset id: `core_balanced_dev_2511_seed20260625`.
  - Manifest:
    `revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json`.
  - Manifest internal hash:
    `a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`.
  - Benchmark root:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks`.
  - Sample count/order vs original baseline: `2511/2511`, same order.
- Output:
  - Full:
    `outputs/clean_benchmarks/qwen3_stage2_norm01_free_coredev2511_deepstack512_mediafix_login_4shard_20260628_0520`.
- Evaluation identity:
  - Eval family: `project_native_external`.
  - Mode: `tgvf_free`.
  - Runner backend: `tgvf_stage2_qwen3_native`.
  - Post-TGVF continuation: `natural_continue`.
  - Post-TGVF forward mode: `no_kv_full_sequence`.
  - DeepStack: enabled; `original_image_scope=through_answer`;
    D DeepStack-like features disabled.
  - Stage2 D condition: `correct_D`.
  - Prompt suffix / softforce prompt: empty.
  - Max image resolution: `512`.
  - Max action tokens: `64`.
  - Max answer tokens: `512`.
  - Attention implementation: `sdpa`.
- tmux:
  - Session:
    `clean_qwen3_stage2_norm01_free_coredev2511_ds512_mediafix_login_20260628_0520`.
- Started:
  - `2026-06-28T05:15:35+09:00`.
  - Runtime commit recorded by orchestrator:
    `3f67d8934cfbfcca84f118d7e9414869a33acca7`.
- Finished:
  - `2026-06-28T05:16:02+09:00`.
- Metrics:
  - `n_rows=2511`, `n_scored=0`, parse `0.0`, malformed `1.0`.
  - Every row has `error="ModuleNotFoundError: No module named 'torch'"`.
  - Shard process exit codes were all `0` because the runner catches row-level
    backend errors and writes malformed rows.
- Analysis:
  - `bash -lc` was insufficient because the tmux server retained an older/base
    environment. The top-level torch probe failed, but the launch script did
    not use `set -e`, so it continued into shard execution and produced a fake
    complete result.
- Conclusion:
  - Do not use this output in result tables.
  - Next rerun must explicitly activate the `revisit-vlm` conda environment
    inside tmux and fail fast on the torch probe.

### EXP-20260628-0530-clean-qwen3-stage2-norm01-free-coredev2511-mediafix-conda

- Status: SIDE_RESULT / INVALID_FOR_BASELINE.
- Question:
  - Rerun `EXP-20260628-0520` with the same evaluation identity, but fix the
    tmux environment deterministically by explicit conda activation and fail
    fast on `torch` import.
- Baseline anchor:
  - Original baseline:
    `EXP-20260628-0216-clean-qwen3-original-coredev2511-maxans512`.
  - Invalid predecessors:
    `EXP-20260628-0510-clean-qwen3-stage2-norm01-free-coredev2511-mediafix`
    and
    `EXP-20260628-0520-clean-qwen3-stage2-norm01-free-coredev2511-mediafix-login`.
- Intended diff:
  - Only launch environment changes:
    `source /home/dredvpn009/Flash_Storage/anaconda3/etc/profile.d/conda.sh`
    then `conda activate revisit-vlm`, with `set -euo pipefail`.
  - Checkpoint, model/processor, manifest/order, parser/scorer, no-extra-prompt
    free mode, DeepStack state, max resolution/tokens, and `sdpa` remain fixed.
- Code commit / worktree:
  - Ledger/planned-entry commit:
    `7a679dba65e722910a6ff8f7f6897ffdc2b547fb`.
  - Runtime launch commit:
    `7a679dba65e722910a6ff8f7f6897ffdc2b547fb`.
  - Dirty worktree before ledger entry: false except this ledger update.
- Model / processor:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Stage2 checkpoint:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - sha256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Stage2 runtime eval JSONL:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`.
  - sha256:
    `3b719e1bd03a09741cc05dac3ed85e423a9b2c29209b07546792a6795cdbbfc5`.
- Benchmark:
  - Subset id: `core_balanced_dev_2511_seed20260625`.
  - Manifest:
    `revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json`.
  - Manifest internal hash:
    `a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`.
  - Benchmark root:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks`.
  - Sample count/order vs original baseline: `2511/2511`, same order.
- Output:
  - Full:
    `outputs/clean_benchmarks/qwen3_stage2_norm01_free_coredev2511_deepstack512_mediafix_conda_4shard_20260628_0530`.
- Evaluation identity:
  - Eval family: `project_native_external`.
  - Mode: `tgvf_free`.
  - Runner backend: `tgvf_stage2_qwen3_native`.
  - Post-TGVF continuation: `natural_continue`.
  - Post-TGVF forward mode: `no_kv_full_sequence`.
  - DeepStack: enabled; `original_image_scope=through_answer`;
    D DeepStack-like features disabled.
  - Stage2 D condition: `correct_D`.
  - Prompt suffix / softforce prompt: empty.
  - Max image resolution: `512`.
  - Max action tokens: `64`.
  - Max answer tokens: `512`.
  - Attention implementation: `sdpa`.
- tmux:
  - Shard sessions:
    `clean_qwen3_stage2_norm01_free_coredev2511_ds512_mediafix_conda_20260628_0530_s0`,
    `clean_qwen3_stage2_norm01_free_coredev2511_ds512_mediafix_conda_20260628_0530_s1`,
    `clean_qwen3_stage2_norm01_free_coredev2511_ds512_mediafix_conda_20260628_0530_s2`,
    `clean_qwen3_stage2_norm01_free_coredev2511_ds512_mediafix_conda_20260628_0530_s3`.
  - Launch implementation note: after a heredoc-based tmux wrapper exited before
    writing logs, the actual run was launched as four explicit shard sessions
    using absolute Python
    `/home/dredvpn009/Flash_Storage/anaconda3/envs/revisit-vlm/bin/python`.
- Started:
  - `2026-06-28T05:25:11+09:00`.
  - Torch/env probe:
    `/home/dredvpn009/Flash_Storage/anaconda3/envs/revisit-vlm/bin/python`,
    torch `2.11.0+cu128`.
  - Early runtime check: four Python processes are on GPUs 0-3; shard 0 has
    loaded weights and written first progress row.
- Finished:
  - Stopped manually at `2026-06-28T05:52:13+09:00` after shard 0 and shard 2
    showed the same OOM -> CUDA/MHA error cascade.
- Metrics:
  - No merged result was produced; this run is partial and invalid.
  - Completed shards:
    - shard 0: `628/628` rows, `334` scored, `294` malformed.
      First error at row `267`:
      `OutOfMemoryError: CUDA out of memory`; then `293` rows with
      `RuntimeError: Expected mha_graph.execute(...).is_good()`.
    - shard 2: `628/628` rows, `384` scored, `244` malformed.
      First error at row `299`:
      `OutOfMemoryError: CUDA out of memory`; then `243` rows with
      `RuntimeError: Expected mha_graph.execute(...).is_good()`.
  - Stopped shards:
    - shard 1: stopped after progress `325/628`; no rows file was written.
    - shard 3: stopped after progress `525/627`; no rows file was written.
- Analysis:
  - The explicit Python/torch environment fix worked: model weights loaded and
    shards reached real benchmark progress on GPUs 0-3.
  - The run is invalid because row-level error handling continued after a CUDA
    OOM, leaving the CUDA/MHA state poisoned and producing hundreds of
    malformed rows instead of a clean fail-fast.
  - Parser/scorer audit performed during this run:
    - Legacy external benchmark scripts use `SCORING_BACKEND=auto` and
      `OFFICIAL_LLM_MODE=disabled`.
    - Legacy BLINK/HR use official-compatible choice scorers; MMMU-Pro,
      MathVista, MathVerse, and OCRBench-v2 use local official scorers when
      available.
    - Clean `score_output_rows()` exactly reproduced stored parsed answers and
      scores for the clean original CoreDev-2511 run, for legacy BLINK original
      n=120, and for legacy HR original n=300.
    - The broader legacy fallback parser can extract extra bare letters from
      long reasoning text, but that was not the effective BLINK/HR official
      scorer behavior and is too permissive for the clean default.
- Conclusion:
  - Do not use this output in result tables.
  - Parser/scorer settings are aligned with the legacy effective benchmark
    path; the immediate blocker is Stage2 clean-native eval memory/error
    recovery under DeepStack + max answer 512, not parser mismatch.
  - Before rerunning this benchmark, add fail-fast or model reload after CUDA
    OOM, and/or reduce eval memory pressure as a named non-comparable
    diagnostic.

### EXP-20260628-0600-clean-qwen3-stage2-norm01-oom-recovery-smoke

- Status: COMPLETE.
- Question:
  - Verify the clean-native Stage2 eval recovery fix from commit
    `e182a01` on the first OOM sample observed in
    `EXP-20260628-0530`, plus the following sample.
- Baseline anchor:
  - Invalid predecessor:
    `EXP-20260628-0530-clean-qwen3-stage2-norm01-free-coredev2511-mediafix-conda`.
  - Original parser/scorer baseline remains
    `EXP-20260628-0216-clean-qwen3-original-coredev2511-maxans512`.
- Intended diff:
  - Code now clears per-sample native Stage2 vision/deepstack caches after each
    row and unloads/reloads the native runtime after CUDA-fatal row errors.
  - Checkpoint, model/processor, parser/scorer, mode, DeepStack state, max
    image resolution, max answer tokens, and prompt settings remain fixed.
- Code commit / worktree:
  - Recovery fix commit:
    `e182a01`.
  - Runtime launch commit:
    `756b22f0b687e09c22b9359151b7b6def0234fad`.
  - Dirty worktree before ledger entry: false except this ledger update.
- Model / processor:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Stage2 checkpoint:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - sha256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Stage2 runtime eval JSONL:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`.
  - sha256:
    `3b719e1bd03a09741cc05dac3ed85e423a9b2c29209b07546792a6795cdbbfc5`.
- Benchmark:
  - Smoke manifest:
    `outputs/clean_smoke_manifests/coredev2511_oom_recovery_2_20260628.json`.
  - Manifest hash:
    `cce8dcb15df66d318cb639c691b7a8b9a827b28927d2ef3cf853949e41848e35`.
  - Source full manifest:
    `revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json`.
  - Sample count: `2`.
  - Sample ids:
    - `ocrbench_v2_data_test_10000/ocrbench_v2_snapshot_data_test_00002_of_00004_parquet/5666_000666`.
    - `ocrbench_v2_data_test_10000/ocrbench_v2_snapshot_data_test_00002_of_00004_parquet/5869_000869`.
- Output:
  - `outputs/clean_benchmarks/qwen3_stage2_norm01_oom_recovery_smoke2_20260628_0600`.
- Evaluation identity:
  - Eval family: `project_native_external`.
  - Mode: `tgvf_free`.
  - Runner backend: `tgvf_stage2_qwen3_native`.
  - Post-TGVF continuation: `natural_continue`.
  - Post-TGVF forward mode: `no_kv_full_sequence`.
  - DeepStack: enabled; `original_image_scope=through_answer`.
  - Stage2 D condition: `correct_D`.
  - Prompt suffix / softforce prompt: empty.
  - Max image resolution: `512`.
  - Max action tokens: `64`.
  - Max answer tokens: `512`.
  - Attention implementation: `sdpa`.
- GPU:
  - Planned: GPU `0`.
- Started:
  - `2026-06-28T06:00` approximate local launch.
- Finished:
  - Completed normally; output files written.
- Metrics:
  - Output:
    `outputs/clean_benchmarks/qwen3_stage2_norm01_oom_recovery_smoke2_20260628_0600`.
  - `n_rows=2`, `n_scored=1`, `answer_parse_rate=0.5`,
    `malformed_rate=0.5`, `trigger_rate=0.5`.
  - Row 1:
    `OutOfMemoryError: CUDA out of memory`, `malformed=true`,
    `append_success=false`.
  - Row 1 recovery metadata:
    `vision_cache_entries_cleared=1`,
    `deepstack_cache_entries_cleared=1`,
    `fatal_cuda_error=true`, `runtime_unloaded=true`.
  - Row 2:
    `error=null`, `malformed=false`, `answer_parse_success=true`;
    no MHA cascade occurred after the row-1 OOM.
- Analysis:
  - The first selected sample still exceeds memory as a single sample under the
    current DeepStack/full-sequence settings, so the fix does not make every
    sample fit.
  - The recovery fix works for the benchmark-runner failure mode: after the
    OOM row, the runtime was unloaded/reloaded and the next row completed
    normally instead of producing `mha_graph.execute` errors.
  - GPU memory returned to zero after completion.
- Conclusion:
  - Use the recovery fix for the next full CoreDev-2511 rerun.

### EXP-20260628-0615-clean-qwen3-stage2-norm01-free-coredev2511-recovery

- Status: COMPLETE.
- Question:
  - Rerun the same clean-native Qwen3 Stage2 free CoreDev-2511 benchmark after
    the OOM recovery fix validated by
    `EXP-20260628-0600-clean-qwen3-stage2-norm01-oom-recovery-smoke`.
- Baseline anchor:
  - Original baseline:
    `EXP-20260628-0216-clean-qwen3-original-coredev2511-maxans512`.
  - Invalid predecessor:
    `EXP-20260628-0530-clean-qwen3-stage2-norm01-free-coredev2511-mediafix-conda`.
- Intended diff:
  - Code adds per-row native Stage2 vision/deepstack cache cleanup and CUDA
    fatal-error runtime unload/reload.
  - Checkpoint, model/processor, manifest/order, parser/scorer, no-extra-prompt
    free mode, DeepStack state, max resolution/tokens, and `sdpa` remain fixed.
- Code commit / worktree:
  - Runtime launch commit recorded by orchestrator:
    `164836886906472c1603b45c482f208e33fc0a5e`.
  - Recovery code fix commit:
    `e182a01b9b770a33262f48951733b9f90124d24c`.
  - Recovery smoke record commit:
    `8c92e6f7e7b1a201ad0c884a46015e58bcf5e950`.
  - Dirty worktree before ledger entry: false except this ledger update.
- Model / processor:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Stage2 checkpoint:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - sha256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Stage2 runtime eval JSONL:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`.
  - sha256:
    `3b719e1bd03a09741cc05dac3ed85e423a9b2c29209b07546792a6795cdbbfc5`.
- Benchmark:
  - Subset id: `core_balanced_dev_2511_seed20260625`.
  - Manifest:
    `revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json`.
  - Manifest internal hash:
    `a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`.
  - Benchmark root:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks`.
  - Sample count/order vs original baseline: `2511/2511`, same order.
- Output:
  - Full:
    `outputs/clean_benchmarks/qwen3_stage2_norm01_free_coredev2511_deepstack512_recovery_4shard_20260628_0615`.
- Evaluation identity:
  - Eval family: `project_native_external`.
  - Mode: `tgvf_free`.
  - Runner backend: `tgvf_stage2_qwen3_native`.
  - Post-TGVF continuation: `natural_continue`.
  - Post-TGVF forward mode: `no_kv_full_sequence`.
  - DeepStack: enabled; `original_image_scope=through_answer`;
    D DeepStack-like features disabled.
  - Stage2 D condition: `correct_D`.
  - Prompt suffix / softforce prompt: empty.
  - Max image resolution: `512`.
  - Max action tokens: `64`.
  - Max answer tokens: `512`.
  - Attention implementation: `sdpa`.
- GPUs:
  - Planned: `0,1,2,3`, one shard per GPU.
- Started:
  - `2026-06-28T06:12:26+09:00`.
  - Shard sessions:
    `clean_qwen3_stage2_norm01_free_coredev2511_ds512_recovery_20260628_0615_s0`,
    `clean_qwen3_stage2_norm01_free_coredev2511_ds512_recovery_20260628_0615_s1`,
    `clean_qwen3_stage2_norm01_free_coredev2511_ds512_recovery_20260628_0615_s2`,
    `clean_qwen3_stage2_norm01_free_coredev2511_ds512_recovery_20260628_0615_s3`.
  - Early runtime check: all four shards loaded weights and reached
    benchmark progress row `1`.
- Finished:
  - `2026-06-28T07:12:30+09:00`.
  - All four shards exited with `exit=0`; merged with manifest hash
    verification.
- Metrics:
  - Output merged:
    `outputs/clean_benchmarks/qwen3_stage2_norm01_free_coredev2511_deepstack512_recovery_4shard_20260628_0615/merged`.
  - Same sample order as original baseline: yes, `2511/2511`.
  - Overall:
    - Original baseline accuracy: `0.308553`; parse: `0.941458`.
    - TGVF free accuracy: `0.320675`; parse: `0.987256`.
    - Delta: `+0.012122`.
    - TGVF trigger/focus-valid rate: `0.186380`.
    - Append success among focus-valid rows: `0.967949`.
    - Malformed/OOM rows: `15/2511 = 0.005974`.
  - By benchmark, accuracy delta vs original baseline:
    - `blink`: `0.464286 -> 0.554762`, delta `+0.090476`;
      trigger `0.080952`, parse `0.709524 -> 0.976190`.
    - `hr_bench_4k`: `0.520000 -> 0.540000`, delta `+0.020000`;
      trigger `0.370000`, parse `0.890000 -> 0.975000`.
    - `mathverse`: `0.048000 -> 0.048193`, delta `+0.000193`;
      trigger `0.240000`, parse `1.000000 -> 0.996000`.
    - `mathvista`: `0.410000 -> 0.419463`, delta `+0.009463`;
      trigger `0.110000`, parse `1.000000 -> 0.993333`.
    - `mmmu_pro`: `0.343333 -> 0.281879`, delta `-0.061454`;
      trigger `0.143333`, parse `1.000000 -> 0.993333`.
    - `ocrbench_v2`: `0.204628 -> 0.222342`, delta `+0.017714`;
      trigger `0.220000`, parse `1.000000 -> 0.981667`.
    - `vstar_bench`: `0.539267 -> 0.497382`, delta `-0.041885`;
      trigger `0.167539`, parse `0.984293 -> 1.000000`.
  - Pairwise scored rows:
    - Overall: method higher on `313`, baseline higher on `252`, equal on
      `1931`, unscored method rows `15`; paired score delta sum `+25.7771`.
    - `blink`: method higher `84`, baseline higher `46`, equal `290`.
    - `hr_bench_4k`: method higher `27`, baseline higher `23`, equal `150`.
    - `mathverse`: method higher `22`, baseline higher `22`, equal `454`,
      unscored `2`.
    - `mathvista`: method higher `44`, baseline higher `42`, equal `212`,
      unscored `2`.
    - `mmmu_pro`: method higher `39`, baseline higher `58`, equal `201`,
      unscored `2`.
    - `ocrbench_v2`: method higher `77`, baseline higher `33`, equal `481`,
      unscored `9`.
    - `vstar_bench`: method higher `20`, baseline higher `28`, equal `143`.
  - Fatal CUDA/OOM recovery:
    - `15` rows hit real OOM and were marked malformed/unscored.
    - Recovery unloaded/reloaded runtime on all `15`; no MHA cascade and all
      shards completed.
- Analysis:
  - The recovery patch fixes the prior invalid-run failure mode: large-sample
    OOMs are isolated to their own rows instead of poisoning subsequent rows.
  - This is a valid comparable run against the original baseline because the
    manifest/order, model/processor, parser/scorer, max resolution/tokens,
    free/no-extra-prompt mode, DeepStack state, and checkpoint are fixed.
  - TGVF free improves the overall CoreDev-2511 score modestly (`+1.21pt`).
    Gains concentrate on BLINK and OCRBench-v2; MMMU-Pro and V* regress.
  - Triggered rows have a positive aggregate paired effect, but trigger quality
    is benchmark-dependent. The method helps HR, MathVista/MMMU triggered
    subsets, and OCRBench slightly; BLINK triggered rows regress even though
    BLINK overall rises, so BLINK's gain is not only from successful focus
    appends.
  - The remaining reliability gap is long-sample memory pressure under
    max-resolution-512, DeepStack-enabled, full-sequence Stage2 eval.
- Conclusion:
  - Parser/scorer alignment is not the blocker for this table.
  - Clean-native Stage2 benchmark is now runnable end-to-end with bounded OOM
    isolation.
  - The current Qwen3 Stage2 checkpoint is better than original baseline on
    this balanced subset overall, but not uniformly; MMMU-Pro and V* need
    row-level mechanism analysis before treating the method as broadly solved.

### EXP-20260628-1027-clean-qwen3-stage2-norm01-softforce-coredev2511

- Status: COMPLETE.
- Question:
  - Complete the paired soft-force benchmark for the same clean Qwen3 Stage2
    checkpoint and CoreDev-2511 manifest, so the comparison table contains
    original, `tgvf_free`, and `tgvf_softforce` with trigger rates.
- Baseline anchors:
  - Original baseline:
    `EXP-20260628-0216-clean-qwen3-original-coredev2511-maxans512`.
  - Free paired run:
    `EXP-20260628-0615-clean-qwen3-stage2-norm01-free-coredev2511-recovery`.
- Intended diff:
  - Change mode from `tgvf_free` to `tgvf_softforce`.
  - Add clean soft-force prompt text: `Use focus tool.`
  - Keep checkpoint, model/processor, manifest/order, parser/scorer,
    DeepStack state, max resolution/tokens, Stage2 D condition, forward mode,
    attention implementation, and shard rule fixed.
- Code / worktree:
  - Runtime launch commit:
    `dc28ab1cc9044ca2ef6e55ac0561a909b3480ac0`.
  - Launch from clean branch
    `clean/tgvf-clean-project-20260625`.
  - No executable code change in this entry; workflow rule updated so future
    method tables include original/free/softforce trigger rates.
- Model / processor:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Stage2 checkpoint:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - sha256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Stage2 runtime eval JSONL:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`.
  - sha256:
    `3b719e1bd03a09741cc05dac3ed85e423a9b2c29209b07546792a6795cdbbfc5`.
- Benchmark:
  - Subset id: `core_balanced_dev_2511_seed20260625`.
  - Manifest:
    `revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json`.
  - Manifest internal hash:
    `a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579`.
  - Benchmark root:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks`.
  - Sample count/order: `2511`, same manifest as original and free runs.
  - Shard rule: fixed manifest order, `row_index % 4`.
- Output:
  - Full:
    `outputs/clean_benchmarks/qwen3_stage2_norm01_softforce_coredev2511_deepstack512_recovery_4shard_20260628_102713`.
- Evaluation identity:
  - Eval family: `project_native_external`.
  - Mode: `tgvf_softforce`.
  - Soft-force prompt text: `Use focus tool.`
  - Runner backend: `tgvf_stage2_qwen3_native`.
  - Post-TGVF continuation: `natural_continue`.
  - Post-TGVF forward mode: `no_kv_full_sequence`.
  - DeepStack: enabled; `original_image_scope=through_answer`;
    D DeepStack-like features disabled.
  - Stage2 D condition: `correct_D`.
  - Max image resolution: `512`.
  - Max action tokens: `64`.
  - Max answer tokens: `512`.
  - Attention implementation: `sdpa`.
- GPUs:
  - Planned: `0,1,2,3`, one shard per GPU.
- Started:
  - `2026-06-28T10:29:42+09:00`.
  - Shard sessions:
    `clean_qwen3_stage2_norm01_softforce_coredev2511_ds512_recovery_20260628_102713_s0`,
    `clean_qwen3_stage2_norm01_softforce_coredev2511_ds512_recovery_20260628_102713_s1`,
    `clean_qwen3_stage2_norm01_softforce_coredev2511_ds512_recovery_20260628_102713_s2`,
    `clean_qwen3_stage2_norm01_softforce_coredev2511_ds512_recovery_20260628_102713_s3`.
- Finished:
  - `2026-06-28T11:34:39+09:00`.
  - All four shards exited with `exit=0`; merged with manifest hash
    verification.
- Metrics:
  - Output merged:
    `outputs/clean_benchmarks/qwen3_stage2_norm01_softforce_coredev2511_deepstack512_recovery_4shard_20260628_102713/merged`.
  - Same sample order as original and free runs: yes, `2511/2511`.
  - Overall comparison:
    - Original: accuracy `0.308553`, parse `0.941458`, trigger `0.000000`,
      malformed `0.000000`.
    - Free: accuracy `0.320675`, parse `0.987256`, trigger `0.186380`,
      append success `0.967949`, malformed `0.005974`.
    - Soft-force: accuracy `0.313802`, parse `0.980884`, trigger
      `0.340104`, append success `0.971897`, malformed `0.009558`.
  - By benchmark, `original / free / soft-force` accuracy and trigger:
    - `blink`, n=`420`: acc `0.464286 / 0.554762 / 0.535714`;
      trigger `0.000000 / 0.080952 / 0.233333`.
    - `hr_bench_4k`, n=`200`: acc `0.520000 / 0.540000 / 0.510000`;
      trigger `0.000000 / 0.370000 / 0.720000`.
    - `mathverse`, n=`500`: acc `0.048000 / 0.048193 / 0.054217`;
      trigger `0.000000 / 0.240000 / 0.274000`.
    - `mathvista`, n=`300`: acc `0.410000 / 0.419463 / 0.417508`;
      trigger `0.000000 / 0.110000 / 0.183333`.
    - `mmmu_pro`, n=`300`: acc `0.343333 / 0.281879 / 0.277592`;
      trigger `0.000000 / 0.143333 / 0.180000`.
    - `ocrbench_v2`, n=`600`: acc `0.204628 / 0.222342 / 0.218944`;
      trigger `0.000000 / 0.220000 / 0.406667`.
    - `vstar_bench`, n=`191`: acc `0.539267 / 0.497382 / 0.481675`;
      trigger `0.000000 / 0.167539 / 0.638743`.
  - Pairwise free vs soft-force:
    - Overall: soft-force higher on `133`, free higher on `156`, equal on
      `2186`, unpaired/unscored `36`.
    - `blink`: soft higher `24`, free higher `32`, equal `364`.
    - `hr_bench_4k`: soft higher `10`, free higher `16`, equal `174`.
    - `mathverse`: soft higher `12`, free higher `9`, equal `475`,
      unpaired `4`.
    - `mathvista`: soft higher `15`, free higher `16`, equal `265`,
      unpaired `4`.
    - `mmmu_pro`: soft higher `17`, free higher `18`, equal `262`,
      unpaired `3`.
    - `ocrbench_v2`: soft higher `42`, free higher `49`, equal `484`,
      unpaired `25`.
    - `vstar_bench`: soft higher `13`, free higher `16`, equal `162`.
  - Fatal CUDA/OOM recovery:
    - `24` soft-force rows hit real OOM and were marked malformed/unscored.
    - Recovery unloaded/reloaded runtime on all `24`; no MHA cascade and all
      shards completed.
- Analysis:
  - Soft-force raises the trigger rate substantially (`18.64% -> 34.01%`)
    but reduces overall accuracy relative to free (`32.07% -> 31.38%`).
  - The extra triggers are not uniformly useful: HR, V*, BLINK, MMMU-Pro, and
    OCRBench-v2 all drop versus free despite higher trigger rates; only
    MathVerse improves slightly over free.
  - Soft-force also increases malformed/OOM rows (`15 -> 24`), concentrated in
    OCRBench-v2 (`18`), plus MathVerse (`2`), MathVista (`3`), and MMMU-Pro
    (`1`). The recovery path still works, but soft-force creates more
    expensive post-D rows.
  - This confirms the table should always include trigger rate together with
    accuracy: higher trigger is not automatically better under the current
    Stage2 checkpoint.
- Conclusion:
  - For this Qwen3 clean Stage2 checkpoint on CoreDev-2511, `tgvf_free` is the
    best of the three tested modes overall: original `30.86`, free `32.07`,
    soft-force `31.38`.
  - Soft-force is still useful diagnostically because it stresses router/focus
    behavior, but it should not be reported alone as the method score.

### EXP-20260628-1226-stage3-grpo-native-2step-smoke

- Status:
  - DONE.
- Question:
  - Does the Stage3 GRPO `native_single_focus` launch path now run more than one
    configured optimizer step against the latest clean Stage2 checkpoint?
- Baseline anchor:
  - Implementation smoke, not a benchmark comparison.
- Intended diff:
  - Use the Stage3 GRPO multi-step launch code added after one-step smoke.
  - Run `max_steps=2`, `save_steps=1`, `group_size=2`,
    `per_device_prompt_batch_size=1`, `gradient_accumulation_steps=1`.
- Allowed changed variables:
  - Stage3 GRPO code path and smoke output directory.
- Not allowed to change:
  - RL data file.
  - Stage2 source checkpoint.
  - Protocol-C tool-observation contract.
- Code commit / worktree:
  - Base commit: `0da19d0`.
  - Dirty worktree: Stage3 multi-step GRPO loop implementation in progress.
- Stage1 checkpoint:
  - Indirectly from the Stage2 checkpoint config.
- Stage1 processor:
  - Indirectly from the Stage2 checkpoint config.
- Stage2 checkpoint/output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - sha256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
  - Rows: `20000`.
  - sha256:
    `2e39a1dadcc020001bd3d763635f461d2b7dc6d94bfb3cfecdb9bb20240fa758`.
- Validation data:
  - Not used in this smoke.
- Benchmark output:
  - Not used in this smoke.
- Script / command:
  - Plan:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.train_stage3_grpo --write-plan --run-id stage3_grpo_native_2step_smoke_clean_stage2_step1200_20260628 --rl-data-path revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl --policy-checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --output-dir outputs/stage3_grpo/native_2step_smoke_clean_stage2_step1200_20260628 --runtime-backend native_single_focus --group-size 2 --per-device-prompt-batch-size 1 --gradient-accumulation-steps 1 --max-steps 2 --save-steps 1 --max-tool-calls 1 --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 96 --max-new-tokens 160 --temperature 0.8 --top-p 0.95 --judge-mode cache_only --learning-rate 1e-6 --wandb-mode disabled`.
  - Preflight:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/native_2step_smoke_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --preflight-only`.
  - Launch:
    `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/native_2step_smoke_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --launch-training`.
- GPUs:
  - Planned: `CUDA_VISIBLE_DEVICES=0`.
- tmux:
  - None. Direct smoke command.
- Started:
  - 2026-06-28 12:31:54 JST.
- Finished:
  - 2026-06-28 12:35:39 JST.
- Metrics:
  - Status: `stage3_grpo_training_completed`.
  - Output:
    `outputs/stage3_grpo/native_2step_smoke_clean_stage2_step1200_20260628`.
  - Global steps: `2`.
  - Checkpoints:
    `checkpoint_step_1.pt`, `checkpoint_step_2.pt`.
  - Latest checkpoint:
    `outputs/stage3_grpo/native_2step_smoke_clean_stage2_step1200_20260628/checkpoint_step_2.pt`.
  - Rollout rows: `4`.
  - Reward rows: `4`.
  - Train metric rows: `2`.
  - Latest step metrics: `loss=0.0410798005759716`,
    `kl=2.053990125656128`, `grad_norm=0.16426536440849304`,
    `replayed_tokens=115.0`, `mean_reward=1.5`.
  - Aggregate: `answer_accuracy=0.5`, `tool_trigger_rate=0.0`,
    `malformed_rate=0.0`, `avg_tool_calls=0.0`.
  - Checkpoint inspection: native checkpoints include `qwen_lora`
    (`504` keys), `tgvf_module` (`26` keys), and optimizer state.
- Analysis:
  - The clean Stage3 GRPO `native_single_focus` path successfully loaded the
    latest Stage2 checkpoint, sampled real rollouts, replayed policy/reference
    logprobs, ran two optimizer steps, and saved native Stage3 checkpoints.
  - Judge was `cache_only`; missing judge scores used the configured cache-miss
    behavior, so this smoke verifies training plumbing rather than reward
    quality.
- Conclusion:
  - Stage3 GRPO is ready for a larger controlled native smoke or first formal
    short run after deciding the judge/probe cache policy.
- Comparable to baseline:
  - No. This is a code-path smoke.

### EXP-20260628-1249-stage3-grpo-formal-answer-tool-100step

- Status:
  - STOPPED.
- Question:
  - Can the clean Stage3 GRPO native training path run a first formal
    non-smoke RL job from the latest clean Stage2 checkpoint with tracked
    artifacts, checkpoints, and W&B-compatible logging?
- Baseline anchor:
  - Implementation/operation run, not a benchmark comparison.
  - Builds on `EXP-20260628-1226-stage3-grpo-native-2step-smoke`.
- Intended diff:
  - Run `max_steps=100`, `save_steps=25`, `group_size=4`,
    `per_device_prompt_batch_size=1`, `gradient_accumulation_steps=1`.
  - Use answer correctness and tool-decision reward for the active optimizer
    signal: `w_answer=2.0`, `w_tool=1.0`, `w_focus=0.0`, `w_ground=0.0`.
  - Keep judge cache/pending enabled with `judge_model=qwen3_vl_32b_thinking`;
    focus/ground rewards are intentionally zero-weighted for this first formal
    run because online/local judge reward is not yet in the train loop.
  - Use W&B offline mode because this machine has no configured W&B API key;
    the run still writes W&B-compatible local logs/artifacts for later sync.
- Allowed changed variables:
  - Stage3 GRPO output directory.
  - Stage3 RL optimizer steps and save cadence.
  - Reward weights for this first formal answer/tool run.
  - W&B mode: `offline`.
- Not allowed to change:
  - RL data file.
  - Stage2 source checkpoint.
  - Protocol-C tool-observation contract.
  - Stage1/Stage2 training code or data.
  - Benchmark eval data.
- Code commit / worktree:
  - Commit: `ba1da3ab36fad601b5f725eb5632fc83e62b2721`
    (`ba1da3a Implement multi-step Stage3 GRPO training loop`).
  - Worktree before launch: clean.
- Stage1 checkpoint:
  - Indirectly from the Stage2 checkpoint config.
- Stage1 processor:
  - Indirectly from the Stage2 checkpoint config.
- Stage2 checkpoint/output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - sha256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
  - Preflight status: `passed`; checkpoint `global_step=1200`;
    protocol `protocol_c_tool_observation`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
  - Rows: `20000`.
  - sha256:
    `2e39a1dadcc020001bd3d763635f461d2b7dc6d94bfb3cfecdb9bb20240fa758`.
  - Source distribution: `visual_genome=8000`, `textvqa=6000`,
    `docvqa=4000`, `chartqa=2000`.
- Validation data:
  - Not used in this training launch.
- Benchmark output:
  - Not used in this training launch.
- Judge model:
  - `qwen3_vl_32b_thinking`.
  - Local path ready:
    `/nvmesv/dredvpn009/models/hf/Qwen3-VL-32B-Thinking`.
- Script / command:
  - Plan:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.train_stage3_grpo --write-plan --run-id stage3_grpo_formal_answer_tool_100step_clean_stage2_step1200_20260628 --rl-data-path revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl --policy-checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --output-dir outputs/stage3_grpo/formal_answer_tool_100step_clean_stage2_step1200_20260628 --runtime-backend native_single_focus --group-size 4 --per-device-prompt-batch-size 1 --gradient-accumulation-steps 1 --max-steps 100 --save-steps 25 --max-tool-calls 1 --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 96 --max-new-tokens 160 --temperature 1.0 --top-p 0.95 --probe-enabled --missing-probe-policy teacher_hint --hint-label-weight 0.5 --w-answer 2.0 --w-tool 1.0 --w-focus 0.0 --w-ground 0.0 --lambda-call 0.05 --judge-enabled --judge-mode cache_only --judge-model qwen3_vl_32b_thinking --judge-cache-miss-reward 0.0 --learning-rate 5e-7 --kl-coef 0.02 --clip-range 0.2 --max-grad-norm 1.0 --seed 20260628 --wandb-project tgvf-stage3-grpo --wandb-mode offline --wandb-run-name stage3_grpo_formal_answer_tool_100step_clean_stage2_step1200_20260628 --wandb-group stage3-grpo-formal --wandb-tags stage3,grpo,formal,answer-tool,cache-only,clean-stage2-step1200 --no-wandb-log-checkpoint-artifact`.
  - Preflight:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/formal_answer_tool_100step_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --preflight-only`.
  - Launch:
    `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/formal_answer_tool_100step_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --launch-training`.
- GPUs:
  - Planned: `CUDA_VISIBLE_DEVICES=0` on NVIDIA B200.
  - Other B200 GPUs were idle at preflight.
- tmux:
  - `stage3_grpo_formal_answer_tool_100step`.
- Started:
  - 2026-06-28 12:49:10 JST.
- Finished:
  - 2026-06-28 12:55:11 JST.
- Metrics:
  - Stopped at `global_step=3`.
  - `train_metrics.jsonl`: `3` rows.
  - `reward_breakdown.jsonl`: `12` rows.
  - `rollout_debug.jsonl`: `12` rows.
  - `judge_pending.jsonl`: `8` rows.
  - Latest metrics: `loss=0.07115457206964493`,
    `kl=3.5577285289764404`, `grad_norm=0.21515274047851562`,
    `replayed_tokens=99.0`, `mean_reward=-0.5`.
- Analysis:
  - This single-GPU launch was intentionally stopped after user correction:
    formal RL should use GPUs `0-3` and larger per-step rollout volume to avoid
    wasting B200 capacity.
  - It is retained only as a side operational trace; it should not be treated
    as the formal run result.
- Conclusion:
  - Replaced by a 4-GPU launch plan.
- Comparable to baseline:
  - No. This is the first formal Stage3 GRPO training run, not benchmark eval.

### EXP-20260628-1259-stage3-grpo-formal-answer-tool-4gpu-g8pb2ga2

- Status:
  - STOPPED.
- Question:
  - Can the first formal Stage3 answer/tool GRPO run use GPUs `0-3` together
    and increase per-step rollout volume enough to better use B200 memory?
- Baseline anchor:
  - Replaces stopped single-GPU launch
    `EXP-20260628-1249-stage3-grpo-formal-answer-tool-100step`.
- Intended diff:
  - Use `CUDA_VISIBLE_DEVICES=0,1,2,3`.
  - Increase per-step rollout volume to `group_size=8`,
    `per_device_prompt_batch_size=2`, `gradient_accumulation_steps=2`
    (`32` free rollouts per optimizer step).
  - Run `max_steps=100`, `save_steps=20`.
  - Active reward remains answer/tool only:
    `w_answer=2.0`, `w_tool=1.0`, `w_focus=0.0`, `w_ground=0.0`.
  - Keep `judge_model=qwen3_vl_32b_thinking` and judge pending output for
    later 32B offline scoring.
- Allowed changed variables:
  - GPU visibility and auto device map placement.
  - Per-step rollout volume.
  - Output directory and W&B offline run name.
- Not allowed to change:
  - RL data file.
  - Stage2 source checkpoint.
  - Protocol-C tool-observation contract.
  - Stage1/Stage2 training code or data.
  - Benchmark eval data.
- Code commit / worktree:
  - Commit: `ae22922dc8816435b82f4ef7b33b19f4e1368262`
    (`ae22922 Prepare Stage3 formal GRPO launch`).
  - Plan was generated from a clean tracked worktree.
- Stage1 checkpoint:
  - Indirectly from the Stage2 checkpoint config.
- Stage1 processor:
  - Indirectly from the Stage2 checkpoint config.
- Stage2 checkpoint/output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - sha256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
  - Preflight status: `passed`; checkpoint `global_step=1200`;
    protocol `protocol_c_tool_observation`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
  - Rows: `20000`.
  - sha256:
    `2e39a1dadcc020001bd3d763635f461d2b7dc6d94bfb3cfecdb9bb20240fa758`.
- Validation data:
  - Not used in this training launch.
- Benchmark output:
  - Not used in this training launch.
- Judge model:
  - `qwen3_vl_32b_thinking`.
  - Local path ready:
    `/nvmesv/dredvpn009/models/hf/Qwen3-VL-32B-Thinking`.
- Script / command:
  - Plan:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.train_stage3_grpo --write-plan --run-id stage3_grpo_formal_answer_tool_4gpu_g8pb2ga2_100step_clean_stage2_step1200_20260628 --rl-data-path revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl --policy-checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --output-dir outputs/stage3_grpo/formal_answer_tool_4gpu_g8pb2ga2_100step_clean_stage2_step1200_20260628 --runtime-backend native_single_focus --group-size 8 --per-device-prompt-batch-size 2 --gradient-accumulation-steps 2 --max-steps 100 --save-steps 20 --max-tool-calls 1 --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 96 --max-new-tokens 160 --temperature 1.0 --top-p 0.95 --probe-enabled --missing-probe-policy teacher_hint --hint-label-weight 0.5 --w-answer 2.0 --w-tool 1.0 --w-focus 0.0 --w-ground 0.0 --lambda-call 0.05 --judge-enabled --judge-mode cache_only --judge-model qwen3_vl_32b_thinking --judge-cache-miss-reward 0.0 --learning-rate 5e-7 --kl-coef 0.02 --clip-range 0.2 --max-grad-norm 1.0 --seed 20260628 --wandb-project tgvf-stage3-grpo --wandb-mode offline --wandb-run-name stage3_grpo_formal_answer_tool_4gpu_g8pb2ga2_100step_clean_stage2_step1200_20260628 --wandb-group stage3-grpo-formal --wandb-tags stage3,grpo,formal,answer-tool,cache-only,clean-stage2-step1200,4gpu,g8,pb2,ga2 --no-wandb-log-checkpoint-artifact`.
  - Preflight:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/formal_answer_tool_4gpu_g8pb2ga2_100step_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --preflight-only`.
  - Launch:
    `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/formal_answer_tool_4gpu_g8pb2ga2_100step_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --launch-training`.
- GPUs:
  - Planned: `CUDA_VISIBLE_DEVICES=0,1,2,3` on NVIDIA B200.
- tmux:
  - `stage3_grpo_formal_4gpu_g8pb2ga2`.
- Started:
  - 2026-06-28 12:59:23 JST.
- Finished:
  - 2026-06-28 13:08 JST.
- Metrics:
  - Stopped before first optimizer metric.
  - `reward_breakdown.jsonl`: `32` rows.
  - `rollout_debug.jsonl`: `32` rows.
  - `judge_pending.jsonl`: `24` rows.
  - `train_metrics.jsonl`: `0` rows.
- Analysis:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3` with `device_map=auto` did place the model
    across GPUs `0-3`, but it did not increase useful memory pressure: static
    memory stayed around `8-10GB` per GPU and utilization was low after rollout
    generation.
  - The larger `32`-rollout step produced rollout/reward rows, then spent too
    long before the first optimizer metric. This is not the desired formal
    configuration.
  - Conclusion: Stage3 needs a real multi-process/data-parallel or rollout
    parallel launch path rather than relying on single-process auto-sharding.
- Conclusion:
  - Stopped and replaced by engineering work toward a proper GPUs `0-3`
    training launch.
- Comparable to baseline:
  - No. This is a Stage3 training launch, not benchmark eval.

### EXP-20260628-1319-stage3-grpo-native-dist-4gpu-2step-smoke

- Status:
  - STOPPED.
- Question:
  - Does the new Stage3 torchrun distributed native path run on GPUs `0-3`
    with one full model per rank, manual gradient all-reduce, rank0 checkpoint
    writing, and per-rank debug outputs?
- Baseline anchor:
  - Replaces failed single-process auto-shard attempt
    `EXP-20260628-1259-stage3-grpo-formal-answer-tool-4gpu-g8pb2ga2`.
- Intended diff:
  - Use `torchrun --nproc_per_node=4`.
  - Use `train.world_size=4`.
  - Run `max_steps=2`, `save_steps=1`, `group_size=2`,
    `per_device_prompt_batch_size=1`, `gradient_accumulation_steps=1`.
  - Expected global free rollouts per optimizer step: `8`.
- Allowed changed variables:
  - Stage3 distributed execution path and output directory.
- Not allowed to change:
  - RL data file.
  - Stage2 source checkpoint.
  - Protocol-C tool-observation contract.
- Code commit / worktree:
  - Commit: `a254987b2a7576c0ae5eb36a3754358355d4f4bb`
    (`a254987 Add distributed Stage3 GRPO launch support`).
- Stage2 checkpoint/output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - sha256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
  - Rows: `20000`.
  - sha256:
    `2e39a1dadcc020001bd3d763635f461d2b7dc6d94bfb3cfecdb9bb20240fa758`.
- Script / command:
  - Plan:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.train_stage3_grpo --write-plan --run-id stage3_grpo_native_dist_4gpu_2step_smoke_clean_stage2_step1200_20260628 --rl-data-path revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl --policy-checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --model-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --processor-id /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking --output-dir outputs/stage3_grpo/native_dist_4gpu_2step_smoke_clean_stage2_step1200_20260628 --runtime-backend native_single_focus --world-size 4 --group-size 2 --per-device-prompt-batch-size 1 --gradient-accumulation-steps 1 --max-steps 2 --save-steps 1 --max-tool-calls 1 --max-image-resolution 512 --max-action-tokens 64 --max-answer-tokens 96 --max-new-tokens 160 --temperature 0.8 --top-p 0.95 --probe-enabled --missing-probe-policy teacher_hint --hint-label-weight 0.5 --w-answer 2.0 --w-tool 1.0 --w-focus 0.0 --w-ground 0.0 --judge-enabled --judge-mode cache_only --judge-model qwen3_vl_32b_thinking --learning-rate 5e-7 --wandb-mode disabled`.
  - Preflight:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/native_dist_4gpu_2step_smoke_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --preflight-only`.
  - Launch:
    `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --standalone --nproc_per_node=4 -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/native_dist_4gpu_2step_smoke_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --launch-training`.
- GPUs:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- tmux:
  - Direct command, no tmux.
- Started:
  - 2026-06-28 13:19:54 JST.
- Finished:
  - 2026-06-28 13:33:00 JST.
- Metrics:
  - Stopped before first optimizer metric.
  - Per-rank reward rows: rank0=`2`, rank1=`2`, rank2=`2`, rank3=`2`.
  - `train_metrics.jsonl`: `0` rows.
- Analysis:
  - The distributed launch did use GPUs `0-3` with one process per GPU:
    pmon showed each GPU at approximately `99-100%` SM utilization and about
    `25-28GB` memory per GPU.
  - The run reached reward writing on all ranks, then spent too long before
    first optimizer metrics. It was stopped so the Stage3 distributed path can
    add per-rank progress instrumentation around replay/backward/all-reduce.
- Conclusion:
  - Distributed GPU placement works, but the update path needs instrumentation
    before it can be used as the formal run.
- Comparable to baseline:
  - No. This is a distributed code-path smoke.

### EXP-20260628-1336-stage3-grpo-native-dist-4gpu-1step-debug

- Status:
  - STOPPED.
- Question:
  - With per-rank progress instrumentation, where does the Stage3 native
    distributed path spend time during the first optimizer step?
- Baseline anchor:
  - Follows stopped `EXP-20260628-1319-stage3-grpo-native-dist-4gpu-2step-smoke`.
- Intended diff:
  - Same 4-GPU torchrun native path, reduced to `max_steps=1`.
  - Progress events written to each rank's `progress.jsonl`.
- Code commit / worktree:
  - Commit: `518e8a6bcb2911dae4a5b829a90250186e939b16`
    (`518e8a6 Instrument distributed Stage3 GRPO progress`).
- Stage2 checkpoint/output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
- Script / command:
  - Launch:
    `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --standalone --nproc_per_node=4 -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/native_dist_4gpu_1step_debug_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --launch-training`.
- GPUs:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- Started:
  - 2026-06-28 13:36:08 JST.
- Finished:
  - 2026-06-28 13:39 JST.
- Metrics:
  - Stopped before optimizer metric after instrumentation identified the slow
    region.
  - Progress showed rollout, replay, backward, and gradient all-reduce all
    completed on all ranks.
- Analysis:
  - The slow point was after `gradient_allreduce_done` and after
    `optimizer_step_start`.
  - The trainer was building `optimizer.state_dict()` on every rank every step
    immediately after `optimizer.step`; this is unnecessary except when rank0
    saves a checkpoint.
- Conclusion:
  - Patch Stage3 to collect optimizer state only during checkpoint saving, then
    rerun distributed native smoke.

### EXP-20260628-1342-stage3-grpo-native-dist-4gpu-1step-debug2

- Status:
  - STOPPED.
- Question:
  - After removing per-step optimizer state materialization, does the 4-GPU
    native distributed Stage3 path complete one optimizer step and rank0
    checkpoint writing?
- Baseline anchor:
  - Follows `EXP-20260628-1336-stage3-grpo-native-dist-4gpu-1step-debug`.
- Intended diff:
  - Same 4-GPU native distributed debug, with code commit `9e58147`.
  - `max_steps=1`, `world_size=4`, global rollouts per step `8`.
- Code commit / worktree:
  - Commit: `9e581479dab17d4378ab6f0ebc4f50bf19555557`
    (`9e58147 Avoid per-step Stage3 optimizer state materialization`).
- Script / command:
  - Launch:
    `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --standalone --nproc_per_node=4 -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/native_dist_4gpu_1step_debug2_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --launch-training`.
- GPUs:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- Started:
  - 2026-06-28 13:42:01 JST.
- Finished:
  - 2026-06-28 13:50 JST.
- Metrics:
  - Stopped before first optimizer metric.
  - Progress showed optimizer.step itself finished on ranks 0/2/3 after the
    previous optimizer-state fix.
- Analysis:
  - Rank1 reached `gradient_allreduce_done` but did not reach
    `optimizer_step_start`, so the remaining stall is in the PyTorch
    `clip_grad_norm_` call between all-reduce and optimizer.step.
- Conclusion:
  - Replace `torch.nn.utils.clip_grad_norm_` with a simple local grad-norm/clip
    helper and rerun.

### EXP-20260628-1353-stage3-grpo-native-dist-4gpu-1step-debug3

- Status:
  - STOPPED.
- Question:
  - Does the 4-GPU native distributed Stage3 path complete one optimizer step
    after replacing PyTorch `clip_grad_norm_` with explicit local clipping?
- Baseline anchor:
  - Follows `EXP-20260628-1342-stage3-grpo-native-dist-4gpu-1step-debug2`.
- Intended diff:
  - Same 4-GPU native distributed debug, with code commit `89e34c9`.
- Code commit / worktree:
  - Commit: `89e34c95845a064e7e9bfddfa78d0f113c76e1c2`
    (`89e34c9 Use explicit Stage3 gradient clipping`).
- Script / command:
  - Launch:
    `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --standalone --nproc_per_node=4 -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/native_dist_4gpu_1step_debug3_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --launch-training`.
- GPUs:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- Started:
  - 2026-06-28 13:53:06 JST.
- Finished:
  - 2026-06-28 14:01 JST, stopped manually after repeated rank1 clip stall.
- Metrics:
  - No optimizer metric written.
  - Rollout/reward files were written on all ranks.
  - Rank0/2/3 reached `optimizer_step_torch_done`.
  - Rank1 reached `gradient_allreduce_done` and `grad_clip_start`, but did not
    reach `grad_clip_done`.
  - Observed GPU use before stop: GPUs 0-3 active, about 25-32 GiB each and
    100% utilization during the stall.
- Analysis:
  - Explicit local clipping still forces per-gradient norm computation and a
    CUDA synchronization. Rank1 reproducibly stalls in that local norm path.
  - `train.max_grad_norm=0` was already accepted by schema but still computed
    the norm, so it did not truly disable clipping.
- Conclusion:
  - Patch `_stage3_clip_grad_norm` so `max_grad_norm <= 0` fully skips norm and
    clip, then rerun a 4-GPU one-step smoke with `--max-grad-norm 0`.

### EXP-20260628-1404-stage3-grpo-native-dist-4gpu-1step-debug4-noclip

- Status:
  - STOPPED.
- Question:
  - Does the 4-GPU native distributed Stage3 path complete one optimizer step
    and write metrics/checkpoint when gradient clipping is explicitly disabled?
- Baseline anchor:
  - Follows `EXP-20260628-1353-stage3-grpo-native-dist-4gpu-1step-debug3`.
- Intended diff:
  - Use commit `63f70a6` and `--max-grad-norm 0`; otherwise keep the same
    4-GPU one-step native distributed debug shape.
- Code commit / worktree:
  - Commit: `f314830 Record Stage3 debug4 noclip plan`.
  - Training code commit includes `63f70a6 Allow disabling Stage3 gradient clipping`.
  - Worktree dirty only for this RUNNING ledger update at launch.
- Stage2 checkpoint/output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
- Script / command:
  - Plan:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.train_stage3_grpo --write-plan --run-id stage3_grpo_native_dist_4gpu_1step_debug4_noclip_clean_stage2_step1200_20260628 --rl-data-path revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl --policy-checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --output-dir outputs/stage3_grpo/native_dist_4gpu_1step_debug4_noclip_clean_stage2_step1200_20260628 --runtime-backend native_single_focus --world-size 4 --group-size 2 --per-device-prompt-batch-size 1 --gradient-accumulation-steps 1 --max-steps 1 --save-steps 1 --max-grad-norm 0 --max-tool-calls 1 --w-focus 0 --w-ground 0 --judge-mode cache_only --judge-model qwen3_vl_32b_thinking --wandb-mode disabled`.
  - Launch:
    `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --standalone --nproc_per_node=4 -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/native_dist_4gpu_1step_debug4_noclip_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --launch-training`.
- GPUs:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- Started:
  - 2026-06-28 14:06:30 JST.
- Finished:
  - 2026-06-28 14:12 JST, stopped manually after rank2 optimizer step stall.
- Metrics:
  - No optimizer metric written.
  - All ranks passed `gradient_allreduce_done`, `grad_clip_done`, and entered
    `optimizer_step_start`.
  - Rank0/1/3 reached `optimizer_step_torch_done`.
  - Rank2 remained at `optimizer_step_start`.
  - Observed GPU use during stall: GPUs 0-3 active, about 25-28 GiB each and
    100% utilization.
- Analysis:
  - The `max_grad_norm=0` patch successfully bypassed the previous clip/norm
    stall.
  - The next bottleneck is inside default `torch.optim.AdamW.step()` on one rank,
    likely the default foreach/fused CUDA optimizer path or a similar
    multi-tensor kernel.
- Conclusion:
  - Switch native Stage3 AdamW construction to explicit non-foreach/non-fused
    mode and rerun a 4-GPU one-step smoke.

### EXP-20260628-1414-stage3-grpo-native-dist-4gpu-1step-debug5-safe-adamw

- Status:
  - STOPPED.
- Question:
  - Does the 4-GPU native distributed Stage3 path complete one optimizer step
    and write metrics/checkpoint with clipping disabled and AdamW forced onto
    the non-foreach/non-fused path?
- Baseline anchor:
  - Follows `EXP-20260628-1404-stage3-grpo-native-dist-4gpu-1step-debug4-noclip`.
- Intended diff:
  - Use commit `c43e680`; native optimizer is `AdamW(foreach=False,
    fused=False)`.
  - Keep `--max-grad-norm 0` and the same 4-GPU one-step debug shape.
- Code commit / worktree:
  - Commit: `4e324b3 Record Stage3 debug5 safe AdamW plan`.
  - Training code commit includes `c43e680 Use stable AdamW path for Stage3 native GRPO`.
  - Worktree dirty only for this RUNNING ledger update at launch.
- Stage2 checkpoint/output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
- Script / command:
  - Plan:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.train_stage3_grpo --write-plan --run-id stage3_grpo_native_dist_4gpu_1step_debug5_safe_adamw_clean_stage2_step1200_20260628 --rl-data-path revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl --policy-checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --output-dir outputs/stage3_grpo/native_dist_4gpu_1step_debug5_safe_adamw_clean_stage2_step1200_20260628 --runtime-backend native_single_focus --world-size 4 --group-size 2 --per-device-prompt-batch-size 1 --gradient-accumulation-steps 1 --max-steps 1 --save-steps 1 --max-grad-norm 0 --max-tool-calls 1 --w-focus 0 --w-ground 0 --judge-mode cache_only --judge-model qwen3_vl_32b_thinking --wandb-mode disabled`.
  - Launch:
    `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --standalone --nproc_per_node=4 -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/native_dist_4gpu_1step_debug5_safe_adamw_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --launch-training`.
- GPUs:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- Started:
  - 2026-06-28 14:15:53 JST.
- Finished:
  - 2026-06-28 14:19 JST, stopped manually after rank1 optimizer step stall.
- Metrics:
  - No optimizer metric written.
  - All ranks passed `gradient_allreduce_done`, `grad_clip_done`, and entered
    `optimizer_step_start`.
  - Rank0/2/3 reached `optimizer_step_torch_done`.
  - Rank1 remained at `optimizer_step_start`.
- Analysis:
  - Forcing AdamW to non-foreach/non-fused made several ranks step quickly but
    did not eliminate rank-local optimizer step stalls.
  - The remaining risk is inside `torch.optim.AdamW.step()` itself or its CUDA
    update kernels, not in GRPO loss, backward, all-reduce, or clipping.
- Conclusion:
  - Add a native Stage3 `manual_sgd` optimizer mode that applies the averaged
    gradients directly with `param.add_(grad, alpha=-lr)`, then rerun 4-GPU
    one-step smoke.

### EXP-20260628-1423-stage3-grpo-native-dist-4gpu-1step-debug6-manual-sgd

- Status:
  - STOPPED.
- Question:
  - Does the 4-GPU native distributed Stage3 path complete one optimizer step
    and write metrics/checkpoint using manual SGD updates?
- Baseline anchor:
  - Follows `EXP-20260628-1414-stage3-grpo-native-dist-4gpu-1step-debug5-safe-adamw`.
- Intended diff:
  - Use commit `ad0b7c1`; set `--optimizer manual_sgd`.
  - Keep `--max-grad-norm 0` and the same 4-GPU one-step debug shape.
- Code commit / worktree:
  - Commit: `672c96d Record Stage3 debug6 manual SGD plan`.
  - Training code commit includes `ad0b7c1 Add manual Stage3 native optimizer path`.
  - Worktree dirty only for this RUNNING ledger update at launch.
- Stage2 checkpoint/output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
- Script / command:
  - Plan:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.train_stage3_grpo --write-plan --run-id stage3_grpo_native_dist_4gpu_1step_debug6_manual_sgd_clean_stage2_step1200_20260628 --rl-data-path revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl --policy-checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --output-dir outputs/stage3_grpo/native_dist_4gpu_1step_debug6_manual_sgd_clean_stage2_step1200_20260628 --runtime-backend native_single_focus --optimizer manual_sgd --world-size 4 --group-size 2 --per-device-prompt-batch-size 1 --gradient-accumulation-steps 1 --max-steps 1 --save-steps 1 --max-grad-norm 0 --max-tool-calls 1 --w-focus 0 --w-ground 0 --judge-mode cache_only --judge-model qwen3_vl_32b_thinking --wandb-mode disabled`.
  - Launch:
    `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --standalone --nproc_per_node=4 -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/native_dist_4gpu_1step_debug6_manual_sgd_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --launch-training`.
- GPUs:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- Started:
  - 2026-06-28 14:24:32 JST.
- Finished:
  - 2026-06-28 14:30 JST, stopped manually after post-step CUDA stall.
- Metrics:
  - No optimizer metric written.
  - All ranks passed `gradient_allreduce_done`, `grad_clip_done`,
    `optimizer_step_start`, and `optimizer_step_torch_done`.
  - No rank reached `optimizer_step_done`; the next code path is distributed
    metric summary.
  - GPUs 0-3 remained at 100% utilization for several minutes after manual SGD
    step events were written.
- Analysis:
  - `manual_sgd` removed the previous optimizer API stall.
  - The sustained post-step GPU work suggests the Stage3 native trainable
    parameter set is too large, likely because the loaded policy model has many
    parameters still marked `requires_grad=True`.
  - Before scaling, inspect and constrain trainable parameters to the intended
    LoRA/TGVF surface.
- Conclusion:
  - Audit native model preparation / `requires_grad` selection, log trainable
    parameter counts, freeze unintended base-model parameters, then rerun a
    4-GPU smoke.

### EXP-20260628-1433-stage3-grpo-native-dist-4gpu-1step-debug7-lora-only

- Status:
  - DONE.
- Question:
  - Does the 4-GPU native distributed Stage3 path complete one optimizer step
    and write metrics/checkpoint when Stage3 trains only policy LoRA/adapter
    parameters and freezes the foveal/TGVF module?
- Baseline anchor:
  - Follows `EXP-20260628-1423-stage3-grpo-native-dist-4gpu-1step-debug6-manual-sgd`.
- Intended diff:
  - Use commit `7be02a1`; Stage3 native update restricts trainables to policy
    LoRA/adapter markers and freezes foveal/TGVF parameters.
  - Keep `--optimizer manual_sgd`, `--max-grad-norm 0`, and the same 4-GPU
    one-step debug shape.
- Code commit / worktree:
  - Commit: `9dca809 Record Stage3 debug7 LoRA-only plan`.
  - Training code commit includes `7be02a1 Restrict Stage3 native trainable parameters`.
  - Worktree dirty only for this RUNNING ledger update at launch.
- Stage2 checkpoint/output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
- Script / command:
  - Plan:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.train_stage3_grpo --write-plan --run-id stage3_grpo_native_dist_4gpu_1step_debug7_lora_only_clean_stage2_step1200_20260628 --rl-data-path revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl --policy-checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --output-dir outputs/stage3_grpo/native_dist_4gpu_1step_debug7_lora_only_clean_stage2_step1200_20260628 --runtime-backend native_single_focus --optimizer manual_sgd --world-size 4 --group-size 2 --per-device-prompt-batch-size 1 --gradient-accumulation-steps 1 --max-steps 1 --save-steps 1 --max-grad-norm 0 --max-tool-calls 1 --w-focus 0 --w-ground 0 --judge-mode cache_only --judge-model qwen3_vl_32b_thinking --wandb-mode disabled`.
  - Launch:
    `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --standalone --nproc_per_node=4 -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/native_dist_4gpu_1step_debug7_lora_only_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --launch-training`.
- GPUs:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- Started:
  - 2026-06-28 14:36:46 JST.
- Finished:
  - 2026-06-28 14:39 JST.
- Metrics:
  - Status: `stage3_grpo_training_completed`.
  - Checkpoint: `outputs/stage3_grpo/native_dist_4gpu_1step_debug7_lora_only_clean_stage2_step1200_20260628/checkpoint_step_1.pt`
    (701 MiB).
  - Train metrics: one row per rank.
  - Reward rows: 2 per rank, 8 global rollouts total.
  - Distributed metrics from rank0:
    - `distributed_rollout_count=8`.
    - `distributed_group_count=4`.
    - `distributed_mean_reward=0.74375`.
    - `distributed_loss_mean=0.18225767649710178`.
    - `distributed_policy_loss_mean=0.10812820494174957`.
    - `distributed_kl_mean=3.7064733803272247`.
    - `distributed_replayed_tokens=335`.
  - Trainable summary:
    - Policy total parameters: 8,939,557,104.
    - Policy trainable LoRA/adapter parameters: 174,587,904 across 504 tensors.
    - Foveal/TGVF module parameters: 18,013,952 total, 0 trainable.
  - Peak observed small-smoke GPU memory: about 24-27 GiB per GPU during update.
- Analysis:
  - Restricting Stage3 trainables to policy LoRA/adapter parameters fixed the
    previous post-step stall.
  - The one-step native distributed path now completes rollout, reward, replay,
    GRPO loss, gradient all-reduce, update, metric summary, checkpoint, and
    process-group teardown on GPUs 0-3.
  - Foveal/TGVF remains frozen, matching the Stage3 non-goal of not retraining
    the D extractor / visual focusing module.
- Conclusion:
  - 4-GPU native Stage3 GRPO smoke is operational. Next run can scale group size
    and/or prompt batch to improve VRAM utilization for a formal training run.

### EXP-20260628-1444-stage3-grpo-formal-4gpu-g8pb2-lora-only

- Status:
  - STOPPED_SIDE_RESULT.
- Question:
  - Run the first formal 4-GPU Stage3 GRPO training job from the latest Stage2
    checkpoint on the 20k RL QA data, using a higher per-rank workload to use
    more GPU memory.
- Baseline anchor:
  - Follows successful smoke `EXP-20260628-1433-stage3-grpo-native-dist-4gpu-1step-debug7-lora-only`.
- Intended diff:
  - Scale from debug `group_size=2`, `per_device_prompt_batch_size=1`,
    `max_steps=1` to formal `group_size=8`,
    `per_device_prompt_batch_size=2`, `max_steps=20`.
  - Keep `world_size=4`, LoRA-only trainables, frozen foveal/TGVF module,
    `optimizer=manual_sgd`, and `max_grad_norm=0`.
  - Enable W&B offline logging with full config/artifact metadata; checkpoint
    artifact upload remains disabled because checkpoints are large.
- Code commit / worktree:
  - Commit: `cc01e06 Record formal Stage3 GRPO run plan`.
  - Training code commit includes `7be02a1 Restrict Stage3 native trainable parameters`.
  - Worktree dirty only for this RUNNING ledger update at launch.
- Stage2 checkpoint/output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - SHA256: `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
  - Rows: 20,000.
  - SHA256: `2e39a1dadcc020001bd3d763635f461d2b7dc6d94bfb3cfecdb9bb20240fa758`.
- Script / command:
  - Plan:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.train_stage3_grpo --write-plan --run-id stage3_grpo_formal_4gpu_g8pb2_manualsgd_loraonly_step1200_20260628 --rl-data-path revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl --policy-checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --output-dir outputs/stage3_grpo/formal_4gpu_g8pb2_manualsgd_loraonly_clean_stage2_step1200_20260628 --runtime-backend native_single_focus --optimizer manual_sgd --world-size 4 --group-size 8 --per-device-prompt-batch-size 2 --gradient-accumulation-steps 1 --max-steps 20 --save-steps 5 --max-grad-norm 0 --max-tool-calls 1 --w-answer 2 --w-tool 1 --w-focus 0 --w-ground 0 --judge-mode cache_only --judge-model qwen3_vl_32b_thinking --wandb-project tgvf-stage3 --wandb-mode offline --wandb-run-name stage3_grpo_formal_4gpu_g8pb2_manualsgd_loraonly_step1200_20260628 --wandb-group stage3-formal --wandb-tags stage3,grpo,formal,4gpu,manual-sgd,lora-only --no-wandb-log-checkpoint-artifact`.
  - Launch:
    `tmux new-session -d -s stage3_grpo_formal_4gpu_g8pb2 -- bash -lc 'cd /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --standalone --nproc_per_node=4 -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/formal_4gpu_g8pb2_manualsgd_loraonly_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --launch-training 2>&1 | tee outputs/stage3_grpo/formal_4gpu_g8pb2_manualsgd_loraonly_clean_stage2_step1200_20260628/torchrun_stdout.log'`.
- GPUs:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- tmux:
  - Planned session: `stage3_grpo_formal_4gpu_g8pb2`.
- Started:
  - 2026-06-28 14:47:10 JST.
- Finished:
  - 2026-06-28 15:00 JST, intentionally stopped after `checkpoint_step_5.pt`
    because the run was stable but under-filled GPU memory for the formal job.
- Metrics:
  - Completed 5 optimizer steps before stop.
  - Checkpoint:
    `outputs/stage3_grpo/formal_4gpu_g8pb2_manualsgd_loraonly_clean_stage2_step1200_20260628/checkpoint_step_5.pt`
    (734,585,995 bytes).
  - Train metrics: 5 rows per rank.
  - Distributed metrics at step 5 from rank0:
    - `distributed_rollout_count=64`.
    - `distributed_group_count=8`.
    - `distributed_mean_reward=1.7734375`.
    - `distributed_loss_mean=0.07011652458459139`.
    - `distributed_policy_loss_mean=-0.002690044231712818`.
    - `distributed_kl_mean=3.640328526496887`.
    - `distributed_replayed_tokens=2066`.
  - Peak observed GPU memory during this run:
    - GPU0 about 88 GiB.
    - GPU1 about 79 GiB.
    - GPU2 about 97 GiB.
    - GPU3 about 81 GiB.
- Analysis:
  - The run validated the formal 4-GPU path beyond the one-step smoke:
    rollout, reward, replay, backward, all-reduce, manual SGD update, W&B
    offline logging, and checkpoint writing all worked through step 5.
  - Memory use was higher than debug7 but still substantially below the
    183 GiB available on each selected GPU.
- Conclusion:
  - Treat this as a clean side result / capacity probe. Superseded for the
    formal run by the larger `g8pb4` launch below.

### EXP-20260628-1503-stage3-grpo-formal-4gpu-g8pb4-lora-only

- Status:
  - STOPPED_SIDE_RESULT.
- Question:
  - Run the formal 4-GPU Stage3 GRPO job from the latest Stage2 checkpoint on
    the 20k RL QA data with higher per-rank workload so GPUs 0-3 are used more
    fully.
- Baseline anchor:
  - Supersedes capacity probe
    `EXP-20260628-1444-stage3-grpo-formal-4gpu-g8pb2-lora-only`.
- Intended diff:
  - Increase `per_device_prompt_batch_size` from 2 to 4 while keeping
    `group_size=8`, `world_size=4`, `max_steps=20`, `save_steps=5`,
    LoRA-only trainables, frozen foveal/TGVF module, `optimizer=manual_sgd`,
    and `max_grad_norm=0`.
  - This changes global rollouts per step from 64 to 128.
  - Keep W&B offline logging enabled with full config/artifact metadata;
    checkpoint artifact upload remains disabled because checkpoints are large.
- Code commit / worktree:
  - Plan generated from commit `23e3d84`.
  - Training code commit includes `7be02a1 Restrict Stage3 native trainable parameters`.
  - Worktree dirty only for this RUNNING ledger update at launch.
- Stage2 checkpoint/output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - SHA256: `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
  - Rows: 20,000.
  - SHA256: `2e39a1dadcc020001bd3d763635f461d2b7dc6d94bfb3cfecdb9bb20240fa758`.
- Preflight:
  - Report:
    `outputs/stage3_grpo/formal_4gpu_g8pb4_manualsgd_loraonly_clean_stage2_step1200_20260628/stage3_grpo_preflight_report.json`.
  - Status: `passed`.
  - Plan SHA256:
    `a556d5d6fe1096bd3dde9ff2e5c88e7b6d61a3da3c92da4b234ad7f57db2f717`.
  - Global rollouts per step: 128.
- Script / command:
  - Plan:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.train_stage3_grpo --write-plan --run-id stage3_grpo_formal_4gpu_g8pb4_manualsgd_loraonly_step1200_20260628 --rl-data-path revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl --policy-checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --output-dir outputs/stage3_grpo/formal_4gpu_g8pb4_manualsgd_loraonly_clean_stage2_step1200_20260628 --runtime-backend native_single_focus --optimizer manual_sgd --world-size 4 --group-size 8 --per-device-prompt-batch-size 4 --gradient-accumulation-steps 1 --max-steps 20 --save-steps 5 --max-grad-norm 0 --max-tool-calls 1 --w-answer 2 --w-tool 1 --w-focus 0 --w-ground 0 --judge-mode cache_only --judge-model qwen3_vl_32b_thinking --wandb-project tgvf-stage3 --wandb-mode offline --wandb-run-name stage3_grpo_formal_4gpu_g8pb4_manualsgd_loraonly_step1200_20260628 --wandb-group stage3-formal --wandb-tags stage3,grpo,formal,4gpu,manual-sgd,lora-only,pb4 --no-wandb-log-checkpoint-artifact`.
  - Launch:
    `tmux new-session -d -s stage3_grpo_formal_4gpu_g8pb4 -- bash -lc 'cd /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --standalone --nproc_per_node=4 -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/formal_4gpu_g8pb4_manualsgd_loraonly_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --launch-training 2>&1 | tee outputs/stage3_grpo/formal_4gpu_g8pb4_manualsgd_loraonly_clean_stage2_step1200_20260628/torchrun_stdout.log'`.
- GPUs:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- tmux:
  - Planned session: `stage3_grpo_formal_4gpu_g8pb4`.
- Started:
  - 2026-06-28 15:03 JST.
- Finished:
  - 2026-06-28 15:29 JST, intentionally stopped after step 4 failed to make
    progress under near-full GPU memory pressure.
- Metrics:
  - Completed 3 optimizer steps before stop.
  - Train metrics: 3 rows per rank.
  - Distributed metrics at step 3 from rank0:
    - `distributed_rollout_count=128`.
    - `distributed_group_count=16`.
    - `distributed_mean_reward=1.161328125`.
    - `distributed_loss_mean=0.05562916491180658`.
    - `distributed_policy_loss_mean=-0.01146760699339211`.
    - `distributed_kl_mean=3.354838728904724`.
    - `distributed_replayed_tokens=4204`.
  - Peak observed GPU memory:
    - GPU0 about 139 GiB.
    - GPU1 about 176 GiB.
    - GPU2 about 182.6 GiB out of 183.4 GiB.
    - GPU3 about 156.7 GiB.
- Analysis:
  - `g8pb4` successfully proved that 128-rollout steps can complete several
    updates, but it is too close to the memory limit for a stable formal run.
  - At step 4 rank2 remained in replay while other ranks waited; GPU2 reached
    near-full memory and the job made no progress for over 10 minutes.
- Conclusion:
  - Treat as a capacity side result. Superseded by `g8pb3`, which keeps a
    higher workload than `g8pb2` while restoring memory headroom.

### EXP-20260628-1530-stage3-grpo-formal-4gpu-g8pb3-lora-only

- Status:
  - STOPPED_SIDE_RESULT.
- Question:
  - Run the formal 4-GPU Stage3 GRPO job from the latest Stage2 checkpoint on
    the 20k RL QA data with a stable high-utilization workload.
- Baseline anchor:
  - Supersedes capacity probes:
    - `EXP-20260628-1444-stage3-grpo-formal-4gpu-g8pb2-lora-only`.
    - `EXP-20260628-1503-stage3-grpo-formal-4gpu-g8pb4-lora-only`.
- Intended diff:
  - Use `per_device_prompt_batch_size=3` with `group_size=8` and
    `world_size=4`, giving 96 global rollouts per update step.
  - Keep `max_steps=20`, `save_steps=5`, LoRA-only trainables, frozen
    foveal/TGVF module, `optimizer=manual_sgd`, and `max_grad_norm=0`.
  - Keep W&B offline logging enabled with full config/artifact metadata;
    checkpoint artifact upload remains disabled because checkpoints are large.
- Code commit / worktree:
  - Plan generated from commit `af7dc0a`.
  - Training code commit includes `7be02a1 Restrict Stage3 native trainable parameters`.
  - Worktree dirty only for this RUNNING ledger update at launch.
- Stage2 checkpoint/output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - SHA256: `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
  - Rows: 20,000.
  - SHA256: `2e39a1dadcc020001bd3d763635f461d2b7dc6d94bfb3cfecdb9bb20240fa758`.
- Preflight:
  - Report:
    `outputs/stage3_grpo/formal_4gpu_g8pb3_manualsgd_loraonly_clean_stage2_step1200_20260628/stage3_grpo_preflight_report.json`.
  - Status: `passed`.
  - Plan SHA256:
    `6d3f98eb7edf5c9871585d169e0c430264ce1f4529f9d6f8cc4df3d908349162`.
  - Global rollouts per step: 96.
- Script / command:
  - Plan:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.train_stage3_grpo --write-plan --run-id stage3_grpo_formal_4gpu_g8pb3_manualsgd_loraonly_step1200_20260628 --rl-data-path revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl --policy-checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --output-dir outputs/stage3_grpo/formal_4gpu_g8pb3_manualsgd_loraonly_clean_stage2_step1200_20260628 --runtime-backend native_single_focus --optimizer manual_sgd --world-size 4 --group-size 8 --per-device-prompt-batch-size 3 --gradient-accumulation-steps 1 --max-steps 20 --save-steps 5 --max-grad-norm 0 --max-tool-calls 1 --w-answer 2 --w-tool 1 --w-focus 0 --w-ground 0 --judge-mode cache_only --judge-model qwen3_vl_32b_thinking --wandb-project tgvf-stage3 --wandb-mode offline --wandb-run-name stage3_grpo_formal_4gpu_g8pb3_manualsgd_loraonly_step1200_20260628 --wandb-group stage3-formal --wandb-tags stage3,grpo,formal,4gpu,manual-sgd,lora-only,pb3 --no-wandb-log-checkpoint-artifact`.
  - Launch:
    `tmux new-session -d -s stage3_grpo_formal_4gpu_g8pb3 -- bash -lc 'cd /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --standalone --nproc_per_node=4 -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/formal_4gpu_g8pb3_manualsgd_loraonly_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --launch-training 2>&1 | tee outputs/stage3_grpo/formal_4gpu_g8pb3_manualsgd_loraonly_clean_stage2_step1200_20260628/torchrun_stdout.log'`.
- GPUs:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- tmux:
  - Planned session: `stage3_grpo_formal_4gpu_g8pb3`.
- Started:
  - 2026-06-28 15:30 JST.
- Finished:
  - 2026-06-28 15:52 JST, intentionally stopped after step 6 showed the same
    near-full-memory stall pattern as `g8pb4`.
- Metrics:
  - Completed 5 optimizer steps before stop.
  - Checkpoint:
    `outputs/stage3_grpo/formal_4gpu_g8pb3_manualsgd_loraonly_clean_stage2_step1200_20260628/checkpoint_step_5.pt`
    (734,585,995 bytes).
  - Train metrics: 5 rows per rank.
  - Distributed metrics at step 5 from rank0:
    - `distributed_rollout_count=96`.
    - `distributed_group_count=12`.
    - `distributed_mean_reward=0.78125`.
    - `distributed_loss_mean=-0.019163240678608418`.
    - `distributed_policy_loss_mean=-0.08346463227644563`.
    - `distributed_kl_mean=3.2150697112083435`.
    - `distributed_replayed_tokens=3554`.
  - Peak observed GPU memory:
    - GPU0 about 125 GiB.
    - GPU1 about 128 GiB.
    - GPU2 about 142 GiB.
    - GPU3 reached about 182.6 GiB at step 6.
- Analysis:
  - `g8pb3` is better than `g8pb4` and successfully wrote the step-5
    checkpoint, but random long rollout/replay groups can still push a rank to
    the memory limit.
  - At step 6 GPU3 reached near-full memory and the job stopped making progress
    in optimizer/update synchronization.
- Conclusion:
  - Treat as a capacity side result. The stable formal 20-step run should use
    `g8pb2` unless token-budget batching or stricter rollout length caps are
    added.

### EXP-20260628-1553-stage3-grpo-formal-4gpu-g8pb2-stable20-lora-only

- Status:
  - DONE.
- Question:
  - Complete a stable formal 20-step 4-GPU Stage3 GRPO run from the latest
    Stage2 checkpoint on the 20k RL QA data.
- Baseline anchor:
  - Uses the highest workload that proved stable through checkpoint writing:
    `g8pb2`, after `g8pb3` and `g8pb4` capacity probes showed near-full-memory
    stalls on long sampled groups.
- Intended diff:
  - Use `per_device_prompt_batch_size=2` with `group_size=8` and
    `world_size=4`, giving 64 global rollouts per update step.
  - Keep `max_steps=20`, `save_steps=5`, LoRA-only trainables, frozen
    foveal/TGVF module, `optimizer=manual_sgd`, and `max_grad_norm=0`.
  - Keep W&B offline logging enabled with full config/artifact metadata;
    checkpoint artifact upload remains disabled because checkpoints are large.
- Code commit / worktree:
  - Plan generated from commit `7f28b9c`.
  - Training code commit includes `7be02a1 Restrict Stage3 native trainable parameters`.
  - Worktree dirty only for this RUNNING ledger update at launch.
- Stage2 checkpoint/output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - SHA256: `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
  - Rows: 20,000.
  - SHA256: `2e39a1dadcc020001bd3d763635f461d2b7dc6d94bfb3cfecdb9bb20240fa758`.
- Preflight:
  - Report:
    `outputs/stage3_grpo/formal_4gpu_g8pb2_stable20_manualsgd_loraonly_clean_stage2_step1200_20260628/stage3_grpo_preflight_report.json`.
  - Status: `passed`.
  - Plan SHA256:
    `e5bfac3f8c73a3a6f0b6d31f0e473b54e0b4840f5350a0276a18c4c74222cae0`.
  - Global rollouts per step: 64.
- Script / command:
  - Plan:
    `PYTHONPATH=revisit_vlm_clean/src:src python -m revisit_vlm_clean.cli.train_stage3_grpo --write-plan --run-id stage3_grpo_formal_4gpu_g8pb2_stable20_manualsgd_loraonly_step1200_20260628 --rl-data-path revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl --policy-checkpoint outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt --output-dir outputs/stage3_grpo/formal_4gpu_g8pb2_stable20_manualsgd_loraonly_clean_stage2_step1200_20260628 --runtime-backend native_single_focus --optimizer manual_sgd --world-size 4 --group-size 8 --per-device-prompt-batch-size 2 --gradient-accumulation-steps 1 --max-steps 20 --save-steps 5 --max-grad-norm 0 --max-tool-calls 1 --w-answer 2 --w-tool 1 --w-focus 0 --w-ground 0 --judge-mode cache_only --judge-model qwen3_vl_32b_thinking --wandb-project tgvf-stage3 --wandb-mode offline --wandb-run-name stage3_grpo_formal_4gpu_g8pb2_stable20_manualsgd_loraonly_step1200_20260628 --wandb-group stage3-formal --wandb-tags stage3,grpo,formal,4gpu,manual-sgd,lora-only,pb2,stable20 --no-wandb-log-checkpoint-artifact`.
  - Launch:
    `tmux new-session -d -s stage3_grpo_formal_4gpu_g8pb2_stable20 -- bash -lc 'cd /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src:src torchrun --standalone --nproc_per_node=4 -m revisit_vlm_clean.training.stage3_grpo_executor --plan outputs/stage3_grpo/formal_4gpu_g8pb2_stable20_manualsgd_loraonly_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json --launch-training 2>&1 | tee outputs/stage3_grpo/formal_4gpu_g8pb2_stable20_manualsgd_loraonly_clean_stage2_step1200_20260628/torchrun_stdout.log'`.
- GPUs:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- tmux:
  - Planned session: `stage3_grpo_formal_4gpu_g8pb2_stable20`.
- Started:
  - 2026-06-28 15:53 JST.
- Finished:
  - 2026-06-28 16:34 JST.
- Metrics:
  - Status: `stage3_grpo_training_completed`.
  - Train metrics: 20 rows per rank, 80 rows total.
  - Checkpoints:
    - `outputs/stage3_grpo/formal_4gpu_g8pb2_stable20_manualsgd_loraonly_clean_stage2_step1200_20260628/checkpoint_step_5.pt`
      (734,586,059 bytes).
    - `outputs/stage3_grpo/formal_4gpu_g8pb2_stable20_manualsgd_loraonly_clean_stage2_step1200_20260628/checkpoint_step_10.pt`
      (734,586,595 bytes).
    - `outputs/stage3_grpo/formal_4gpu_g8pb2_stable20_manualsgd_loraonly_clean_stage2_step1200_20260628/checkpoint_step_15.pt`
      (734,586,595 bytes).
    - `outputs/stage3_grpo/formal_4gpu_g8pb2_stable20_manualsgd_loraonly_clean_stage2_step1200_20260628/checkpoint_step_20.pt`
      (734,586,595 bytes).
  - Latest checkpoint pointer:
    `outputs/stage3_grpo/formal_4gpu_g8pb2_stable20_manualsgd_loraonly_clean_stage2_step1200_20260628/LATEST_CHECKPOINT.txt`.
  - Final step-20 distributed metrics from rank0/root logs:
    - `distributed_rollout_count=64`.
    - `distributed_group_count=8`.
    - `distributed_mean_reward=0.57421875`.
    - `distributed_loss_mean=0.07808398269116879`.
    - `distributed_policy_loss_mean=0.016010917723178864`.
    - `distributed_kl_mean=3.1036531925201416`.
    - `distributed_replayed_tokens=2202`.
  - 20-step means from rank0/root `train_metrics.jsonl`:
    - mean `distributed_mean_reward=1.143515625`.
    - mean `distributed_kl_mean=3.438732349872589`.
    - mean `distributed_loss_mean=0.049926025234162806`.
    - mean `distributed_policy_loss_mean=-0.01884862000897556`.
    - mean `distributed_replayed_tokens=2283.1`.
  - W&B:
    - Offline run directory:
      `outputs/stage3_grpo/formal_4gpu_g8pb2_stable20_manualsgd_loraonly_clean_stage2_step1200_20260628/wandb/wandb/offline-run-20260628_155555-khqo0njf`.
    - Checkpoint artifact upload disabled by plan.
  - Peak observed GPU memory:
    - Around 144 GiB on the highest rank during this stable run.
- Analysis:
  - The full native Stage3 GRPO path completed 20 optimizer steps on GPUs 0-3:
    free rollouts, TGVF tool protocol, reward computation, replay, GRPO loss,
    backward, gradient all-reduce, manual SGD update, checkpointing, W&B offline
    logging, and clean process teardown.
  - `g8pb3` and `g8pb4` are useful capacity probes but not stable long-run
    settings because random long rollout/replay groups can push a rank to the
    memory limit. `g8pb2` is the stable formal setting until token-budget
    batching or stricter rollout length caps are added.
- Conclusion:
  - Stage3 GRPO training framework is operational end to end for a formal
    4-GPU 20-step run. Next scaling step should improve batching by token budget
    or rollout length control before increasing prompt batch again.

### EXP-20260628-1726-stage3-grpo-probe-judge-reward-smoke

- Status:
  - DONE.
- Question:
  - Verify that Stage3 GRPO can consume real forced-probe `Delta_tool` labels and
    real visual judge rewards, instead of relying on teacher-hint tool labels or
    zeroed judge weights.
- Baseline anchor:
  - `EXP-20260628-1553-stage3-grpo-formal-4gpu-g8pb2-stable20-lora-only`
    completed native GRPO training, but used `w_focus=0`, `w_ground=0`, and no
    probe cache, so it did not validate judge reward or forced-probe
    ToolDecision reward.
- Intended diff:
  - Use a small smoke subset from the 20k RL data so probe cache and training
    samples exactly overlap.
  - Precompute forced OFF / forced ON-clean probes with `num_off=1`,
    `num_on_clean=1` to populate `delta_tool`.
  - Generate rollout judge pending rows, score them with a local VLM judge, then
    run one 4-GPU native GRPO update with `w_focus=1`, `w_ground=1`,
    `judge_mode=cache_only`, and judge caches loaded.
  - Add reward/logging fix so judge cache misses use
    `judge.cache_miss_reward` and metrics report judge hit rate plus
    `tool_label_source` counts.
- Allowed changed variables:
  - Smoke sample count, group size, max steps, probe repetitions, judge caches,
    reward weights, and reward logging.
- Not allowed to change:
  - Stage2 checkpoint, Stage3 RL source data identity, native TGVF/free rollout
    path, LoRA-only trainable scope, D generation, D-condition audit, benchmark
    eval, or long training schedule.
- Code commit / worktree:
  - Base commit: `b0cb8c7`.
  - Worktree dirty by this DONE ledger update and code fixes:
    - judge cache miss now uses `judge.cache_miss_reward`;
    - W&B/aggregate metrics report judge hit rate and `tool_label_source`;
    - trainer seeds Python/Torch/NumPy so rollout-only and first training step
      can share cache keys;
    - judge runner uses JSON-only system prompting, conservative text fallback,
      and supports single-GPU `--device-map cuda:0`.
- Stage2 checkpoint/output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - SHA256: `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Train data:
  - Initial 8-sample ChartQA smoke subset:
    `outputs/stage3_grpo/probe_judge_reward_smoke_4gpu_g2pb1_clean_stage2_step1200_20260628/smoke_prompts_8.jsonl`.
  - Rows: 8.
  - SHA256: `607372523a35b9fa1b3ba51341ebbf80d96daca41f374b19a82185297e57c671`.
  - Final trigger-smoke subset:
    `outputs/stage3_grpo/probe_judge_reward_trigger_smoke_4gpu_g8pb1_clean_stage2_step1200_20260628/trigger_smoke_prompts_4.jsonl`.
  - Rows: 4.
  - SHA256: `5c8efec0ea893668d89488339d9d1c24e9ac53090dbf610d99f806148a26dbec`.
  - Parent data:
    `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`,
    rows 20,000, SHA256
    `2e39a1dadcc020001bd3d763635f461d2b7dc6d94bfb3cfecdb9bb20240fa758`.
- Validation data:
  - None; this is a reward-path smoke, not benchmark validation.
- Benchmark output:
  - None.
- Script / command:
  - Final plan:
    `outputs/stage3_grpo/probe_judge_reward_trigger_smoke_4gpu_g8pb1_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json`.
  - Final run id:
    `stage3_grpo_probe_judge_reward_trigger_smoke_4gpu_g8pb1_8bjudge_step1200_20260628`.
  - Final config summary:
    `runtime_backend=native_single_focus`, `world_size=4`, `group_size=8`,
    `per_device_prompt_batch_size=1`, `max_steps=1`, `optimizer=manual_sgd`,
    `max_grad_norm=0`, `w_answer=2`, `w_tool=1`, `w_focus=1`, `w_ground=1`,
    `judge_model=qwen3_vl_8b_thinking`, `judge_mode=cache_only`.
  - Main commands:
    - forced probes:
      `python -m revisit_vlm_clean.training.stage3_grpo_executor --plan .../stage3_grpo_training_plan.json --precompute-probes --probe-output .../probe_cache.jsonl`.
    - rollout-only pending generation:
      `torchrun --standalone --nproc_per_node=4 -m revisit_vlm_clean.training.stage3_grpo_executor --plan .../stage3_grpo_training_plan.json --rollout-only`.
    - judge cache:
      `CUDA_VISIBLE_DEVICES=0 python -m revisit_vlm_clean.cli.stage3_grpo_judge --pending-path .../judge_pending_all_ranks.jsonl --output-dir .../offline_judge_8b --backend local_qwen_vl --model-preset qwen3_vl_8b_thinking --device cuda:0 --device-map cuda:0 --max-new-tokens 192 --execute`.
    - training:
      `torchrun --standalone --nproc_per_node=4 -m revisit_vlm_clean.training.stage3_grpo_executor --plan .../stage3_grpo_training_plan.json --launch-training`.
- GPUs:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- tmux:
  - None planned; run foreground and monitor time/memory directly.
- Started:
  - 2026-06-28 17:30 JST.
- Preflight:
  - Initial 8-sample plan passed, but rollout-only produced no tool calls.
  - Final trigger-smoke preflight after judge cache:
    `outputs/stage3_grpo/probe_judge_reward_trigger_smoke_4gpu_g8pb1_clean_stage2_step1200_20260628/stage3_grpo_preflight_after_judge8b.json`.
  - Status: `passed`.
  - Final plan SHA256:
    `deb1cd76e595c6b10e849a05be838b2de6bb624a0a15022aa4a5c54fea0420c2`.
  - Global rollouts per step: 32.
  - Judge cache status: ready, focus rows 30, grounding rows 24.
- Finished:
  - 2026-06-28 18:28 JST.
- Metrics:
  - Probe:
    - Final trigger-smoke forced probes: 4 samples, 8 trajectories,
      wall time 265s.
    - `mean_delta_tool=0.25`; deltas were three `0.0` and one `1.0`.
    - Peak probe memory: about 10.2GB on the highest GPU; utilization low
      (max about 18%).
  - Judge:
    - 32B judge preflight passed, but 32B thinking generation was too slow for
      this smoke and initially failed JSON parsing before the JSON-only/fallback
      fix.
    - 8B single-GPU judge scored 54/62 pending rows in 320s:
      30 focus rows and 24 grounding rows; 8 rows failed parser/fallback.
    - Score distribution: focus `2` x30; grounding `2` x16 and `0` x8.
    - Parse fallback used on 5 scored rows.
    - Peak judge memory: GPU0 about 20.3GB; other GPUs idle.
  - Training:
    - Status: `stage3_grpo_training_completed`.
    - Wall time: 223s.
    - Global rollouts: 32; distributed groups: 4.
    - Checkpoint:
      `outputs/stage3_grpo/probe_judge_reward_trigger_smoke_4gpu_g8pb1_clean_stage2_step1200_20260628/checkpoint_step_1.pt`
      (about 701MB).
    - Distributed metrics:
      - `distributed_mean_reward=1.5765625`.
      - `distributed_loss_mean=0.0461840453`.
      - `distributed_policy_loss_mean=-0.0219766728`.
      - `distributed_kl_mean=3.4080360532`.
      - `distributed_replayed_tokens=2028`.
    - Combined reward breakdown across 4 ranks:
      - `tool_label_source`: `forced_probe` x32.
      - `used_tool`: 31/32 rollouts.
      - focus judge hit: 30/31 used-tool rollouts.
      - grounding judge hit: 24/31 used-tool rollouts.
      - mean rewards: total 1.57656, answer 0.21875, tool -0.04844,
        focus 0.9375, ground 0.25, protocol 0.0.
      - malformed/protocol penalty: 0.
    - Peak training memory: GPU0 66.1GB, GPU1 82.5GB, GPU2 66.8GB,
      GPU3 62.3GB.
  - W&B:
    - Offline run:
      `outputs/stage3_grpo/probe_judge_reward_trigger_smoke_4gpu_g8pb1_clean_stage2_step1200_20260628/wandb/wandb/offline-run-20260628_182559-ic7ckwn5`.
    - Logged `reward/focus_judge_hit_rate`,
      `reward/grounding_judge_hit_rate`,
      `reward/tool_label_source_count/forced_probe`, reward means, rollout
      rates, distributed loss/KL/replayed-token metrics, and full config.
- Analysis:
  - D-condition audit is intentionally excluded for this run.
  - Forced probe reward path is real and no longer falls back to teacher hints:
    all 32 training rewards had `tool_label_source=forced_probe`.
  - Judge reward path is real: focus/ground judge caches were consumed during
    reward calculation and affected `reward_focus`/`reward_ground`.
  - The initial 8 ChartQA smoke set had zero free tool calls even at
    `group_size=8`, so it was unsuitable for judge reward verification; the
    trigger-smoke set was built from samples known to trigger in the prior
    formal run.
  - Probe generation is much too slow in the current single-process form
    (4-sample smoke took 265s; 8-sample ChartQA probe took 392s), so full-scale
    probes must be offline and parallelized.
  - 32B thinking judge is not yet practical in this runner without further
    generation tuning; 8B single-GPU judge is usable for plumbing and cache
    validation.
- Conclusion:
  - Stage3 reward plumbing is now verified end to end with forced-probe
    `Delta_tool` labels and nonzero cached VLM judge rewards entering a real
    4-GPU GRPO update. The next production step is probe/judge cache scaling and
    judge reliability work, not D-condition audit.
- Comparable to baseline:
  - Only comparable as reward-path plumbing/capacity validation, not as a
    model-quality run.
- Follow-up:
  - Parallelize forced probe precompute.
  - Improve judge JSON compliance / retry failed judge rows before large runs.
  - Keep 8B single-GPU judge for smoke; revisit 32B only after generation config
    is optimized.

### EXP-20260628-194148-stage3-judge-thinking-ab

- Status: DONE.
- Question:
  - For the production 32B Qwen3-VL judge, does disabling thinking reduce
    offline judge latency and improve JSON cache usability?
- Baseline anchor:
  - Diagnostic side result anchored to
    `EXP-20260628-stage3-probe-judge-reward-trigger-smoke`.
- Intended diff:
  - A/B the Stage3 offline judge with 32B thinking-on vs no-thinking.
  - After raw output inspection showed prompt-only no-thinking still generated
    `<think>`, update the no-thinking path to:
    - strip hardcoded `<think>` from the generation prompt;
    - ban `<think>` and `</think>` token ids during generation;
    - prefill the assistant response with `{` so the model directly completes a
      JSON object.
- Allowed changed variables:
  - Judge thinking prompt switch and the no-thinking JSON enforcement needed to
    make the switch real.
- Not allowed to change:
  - Pending rows, 32B judge model preset, max image resolution, max new tokens,
    backend, GPU.
- Code commit / worktree:
  - Base `7a478b9`.
  - Current worktree includes this ledger update plus judge runner/CLI/tests for
    JSON prefill and `<think>` token suppression.
- Stage1 checkpoint:
  - N/A.
- Stage1 processor:
  - N/A.
- Stage2 checkpoint/output:
  - N/A for judge diagnostic.
- Train data:
  - N/A.
- Validation data:
  - Pending judge rows:
    `outputs/stage3_grpo/probe_judge_reward_trigger_smoke_4gpu_g8pb1_clean_stage2_step1200_20260628/judge_pending_all_ranks.jsonl`.
  - 32B comparison used first 4 rows by file order.
  - Accidental/side 8B no-thinking run used first 8 rows and is not used for
    production judge choice.
- Benchmark output:
  - 32B:
    `outputs/stage3_grpo/judge_thinking_ab_32b_limit4_20260628`.
  - 8B side result:
    `outputs/stage3_grpo/judge_thinking_ab_8b_limit8_20260628`.
- Script / command:
  - 32B thinking-on:
    `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.stage3_grpo_judge --pending-path outputs/stage3_grpo/probe_judge_reward_trigger_smoke_4gpu_g8pb1_clean_stage2_step1200_20260628/judge_pending_all_ranks.jsonl --output-dir outputs/stage3_grpo/judge_thinking_ab_32b_limit4_20260628/thinking --backend local_qwen_vl --model-preset qwen3_vl_32b_thinking --device cuda:0 --device-map cuda:0 --dtype bfloat16 --attn-implementation sdpa --max-image-resolution 512 --max-new-tokens 128 --limit 4 --no-skip-existing --no-append --judge-enable-thinking --execute`.
  - 32B no-thinking effective:
    `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.stage3_grpo_judge --pending-path outputs/stage3_grpo/probe_judge_reward_trigger_smoke_4gpu_g8pb1_clean_stage2_step1200_20260628/judge_pending_all_ranks.jsonl --output-dir outputs/stage3_grpo/judge_thinking_ab_32b_limit4_20260628/no_thinking_jsonprefill --backend local_qwen_vl --model-preset qwen3_vl_32b_thinking --device cuda:0 --device-map cuda:0 --dtype bfloat16 --attn-implementation sdpa --max-image-resolution 512 --max-new-tokens 128 --limit 4 --no-skip-existing --no-append --execute`.
- GPUs:
  - GPU 0 only.
- tmux:
  - None.
- Started:
  - 2026-06-28T19:41:48+09:00.
- Finished:
  - 2026-06-28T19:52:52+09:00.
- Metrics:
  - Prompt-only 32B no-thinking before JSON prefill:
    - 4 rows, 80.82s, 3/4 scored, 1 failed.
    - Raw outputs still contained `<think>` in 3/3 scored rows, so this was not
      effective no-thinking.
  - 32B thinking-on:
    - 4 rows, 77.78s, 3/4 scored, 1 failed.
    - Raw outputs were long rationale-like text, not compact JSON.
  - 32B effective no-thinking with JSON prefill and `<think>` token ban:
    - 4 rows, 39.44s, 4/4 scored, 0 failed.
    - Raw outputs had 0 rows containing `<think>`.
    - Raw outputs were compact JSON-like objects, lengths 54-60 chars.
  - 8B side result:
    - 8 rows, 75.43s, 8/8 scored, but not used for production judge choice.
- Analysis:
  - The initial template-only switch was insufficient for Qwen3-VL-32B-Thinking:
    the model can self-emit `<think>` even when the prompt does not end with the
    thinking tag.
  - Effective no-thinking requires generation-level blocking plus JSON prefill.
  - On this small 32B diagnostic, effective no-thinking is about 2.0x faster
    than thinking-on including model load, and has better cache usability
    (4/4 vs 3/4 scored).
- Conclusion:
  - Use 32B judge for production, with thinking disabled, `<think>` tokens
    banned, and `{` JSON response prefill enabled by default.
- Comparable to baseline:
  - Diagnostic only, not a benchmark/training comparison.
- Follow-up:
  - For larger 32B judge cache fills, keep the model resident per worker to
    amortize load time and shard pending rows across available GPUs.

### EXP-20260628-203232-stage3-all5-32bjudge-trigger-pilot

- Status: DONE.
- Question:
  - Can Stage3 GRPO run with all five reward components active using 32B VLM
    judge caches: Answer, ToolDecision from forced probes, FocusEvidence,
    GroundedReasoning, and ProtocolGate?
- Baseline anchor:
  - `EXP-20260628-stage3-probe-judge-reward-trigger-smoke`.
- Intended diff:
  - Replace the prior 8B judge cache with the production-intended
    `Qwen3-VL-32B-Thinking` judge.
  - Use effective no-thinking judge generation: `<think>` token ban plus `{`
    JSON response prefill.
  - Keep the same 4-prompt trigger subset and one 4GPU GRPO update so judge
    cache keys can be generated offline before training and then hit exactly.
- Allowed changed variables:
  - Judge model/cache, run id/output dir, and regenerated cache artifacts.
- Not allowed to change:
  - Stage2 checkpoint, RL trigger subset rows, protocol, world size, group
    size, per-device prompt batch, reward weights, GRPO algorithm, optimizer,
    LoRA-only trainable scope, D generation path, and DeepStack/mask behavior.
- Code commit / worktree:
  - `5a6b849` plus this PLANNED ledger entry.
- Stage1 checkpoint:
  - Encoded in the Stage2 checkpoint lineage.
- Stage1 processor:
  - Encoded in the Stage2 checkpoint lineage.
- Stage2 checkpoint/output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
- Train data:
  - `outputs/stage3_grpo/probe_judge_reward_trigger_smoke_4gpu_g8pb1_clean_stage2_step1200_20260628/trigger_smoke_prompts_4.jsonl`.
  - Parent full RL data:
    `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
- Validation data:
  - None; this is an all-reward training-path pilot, not benchmark eval.
- Benchmark output:
  - None.
- Script / command:
  - Plan:
    `outputs/stage3_grpo/all5_32bjudge_trigger_4gpu_g8pb1_step1_clean_stage2_step1200_20260628/stage3_grpo_training_plan.json`.
  - Pipeline:
    1. write plan with all reward weights active;
    2. preflight;
    3. precompute forced probes;
    4. rollout-only on 4 GPUs to generate judge pending rows;
    5. run offline 32B no-thinking JSON judge;
    6. preflight after judge cache;
    7. launch one 4GPU GRPO update.
- GPUs:
  - Training/probe/rollout: GPUs 0,1,2,3.
  - Offline 32B judge: GPU 0.
- tmux:
  - None; foreground run monitored directly.
- Started:
  - 2026-06-28T20:34:40+09:00.
- Finished:
  - 2026-06-28T20:50:20+09:00.
- Metrics:
  - Plan:
    - `stage3_grpo_training_plan.json` SHA256
      `ffbadb808aa74eb6d94d6e61a26c760cbb4270a037435db75b2c6c9ade201a1e`.
    - Trigger subset rows: 4.
    - Global rollouts per training step: 32.
  - Preflight:
    - Initial preflight passed with expected judge-cache-missing warnings.
    - After 32B judge cache, preflight passed with judge cache status `ready`.
    - Focus cache rows: 31.
    - Grounding cache rows: 31.
  - Forced probes:
    - Wall time: 271.96s.
    - Rows: 4.
    - `mean_delta_tool=0.5`.
    - Probe rows: two samples `delta_tool=0.0`, two samples `delta_tool=1.0`.
  - Rollout-only cache-generation pass:
    - Wall time: 108.39s.
    - 4 ranks x 8 rollouts = 32 free rollouts.
    - Judge pending rows: 62 total, 31 focus and 31 grounding.
  - 32B judge:
    - Model: `Qwen/Qwen3-VL-32B-Thinking`.
    - Generation: no-thinking, `{` JSON prefill, `<think>` token ban.
    - Wall time: 115.18s (`judge_run_summary` wall time 113.77s).
    - Scored: 62/62, failed 0.
    - Focus scores: `1` x28, `2` x3.
    - Grounding scores: `0` x14, `2` x17.
  - Training:
    - Wall time: 225.67s.
    - Status: `stage3_grpo_training_completed`.
    - Distributed rollouts: 32.
    - Distributed groups: 4.
    - Distributed replayed tokens: 2028.
    - Distributed mean reward: 1.0140625.
    - Distributed loss mean: 0.05968846.
    - Distributed policy loss mean: -0.00847226.
    - Distributed KL mean: 3.40803605.
    - Checkpoint:
      `outputs/stage3_grpo/all5_32bjudge_trigger_4gpu_g8pb1_step1_clean_stage2_step1200_20260628/checkpoint_step_1.pt`
      (about 701MB).
  - Combined reward breakdown across 4 ranks:
    - Reward rows: 32.
    - Rollout rows: 32.
    - Used-tool rollouts: 31/32.
    - Focus judge hit: 31/31 used-tool rollouts.
    - Grounding judge hit: 31/31 used-tool rollouts.
    - Answer correct: 7/32.
    - Protocol penalty count: 0.
    - `tool_label_source`: `forced_probe` x32.
    - `tool_label`: `unknown` x32.
    - Mean rewards:
      - total 1.0140625.
      - answer 0.21875.
      - tool -0.0484375.
      - focus 0.53125.
      - ground 0.09375.
      - protocol 0.0.
  - W&B:
    - Offline run:
      `outputs/stage3_grpo/all5_32bjudge_trigger_4gpu_g8pb1_step1_clean_stage2_step1200_20260628/wandb/wandb/offline-run-20260628_204553-u5dgvbms`.
- Analysis:
  - The all-reward path ran end to end with real native rollouts, forced-probe
    cache, 32B focus/grounding judge cache, GRPO replay/update, W&B offline
    logging, and checkpoint save.
  - The focus/grounding judge path is materially better than the earlier 8B
    smoke: 32B scored all 62 rows with 0 parse failures after the no-thinking
    JSON generation fix.
  - Important caveat: although forced-probe labels were used for all 32 rewards,
    the weighted sampler selected only two of the four trigger-subset samples for
    this single step, and those two had `delta_tool=0.0`; therefore
    `tool_label=unknown` for all training rollouts. The ToolDecision reward
    component was active, but in this step it contributed the extra-call
    efficiency penalty rather than a `tool_needed`/`tool_unnecessary` positive
    or negative label.
  - The sampled groups covered only 2 unique sample ids. This is acceptable for
    the all-reward path pilot but too narrow for a training-quality run.
  - No GPU peak monitor was attached for this run; post-run GPUs returned to 0
    MiB used.
- Conclusion:
  - The five-component Stage3 GRPO reward/training path is now runnable with
    production-intended 32B judge caches. The next blocker for real multi-step
    all-5 training is not reward plumbing; it is scheduler/sampling: we need
    stepwise rollout -> 32B judge cache fill -> train update, and prompt
    sampling should use forced-probe labels so tool-needed/tool-unnecessary
    samples are actually represented.
- Comparable to baseline:
  - Comparable only as an all-reward path pilot; not a model-quality or
    benchmark-comparable run.
- Follow-up:
  - Add a stepwise all-5 training scheduler that pauses between rollout and
    update to fill 32B judge caches.
  - Add probe-label-aware prompt sampling or construct per-step prompt shards so
    `tool_needed`, `tool_unnecessary`, and `unknown` labels are represented.
  - For the immediate next pilot, force a small `delta_tool=1.0` subset to
    verify nonzero ToolDecision classification reward, not just efficiency
    penalty.

### EXP-20260628-205812-stage3-formal-all5-hint-stepwise

- Status: FAILED_OOM_BEFORE_METRICS.
- Question:
  - Prepare formal Stage3 all-5 GRPO training using dataset `tool_need_hint` for
    ToolDecision labels instead of forced-probe `Delta_tool`.
- Baseline anchor:
  - `EXP-20260628-203232-stage3-all5-32bjudge-trigger-pilot`.
- Intended diff:
  - Disable forced probes for reward labels.
  - Use `missing_probe_policy=teacher_hint` with `hint_label_weight=1.0`.
  - Keep Answer, ProtocolGate, FocusEvidence, and GroundedReasoning active.
  - Use production-intended 32B judge cache generation.
  - Use full 20k RL prompt pool instead of the 4-prompt trigger subset.
- Allowed changed variables:
  - ToolDecision label source, train data size, seed, and per-step output/cache
    directory.
- Not allowed to change:
  - Stage2 source checkpoint, protocol, world size, group size, reward weights,
    optimizer, LoRA-only trainable scope, D generation path, and 32B judge model.
- Code commit / worktree:
  - `026a711`; worktree clean before plan generation.
- Stage1 checkpoint:
  - Encoded in the Stage2 checkpoint lineage.
- Stage1 processor:
  - Encoded in the Stage2 checkpoint lineage.
- Stage2 checkpoint/output:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
  - Rows: 20,000.
  - SHA256: `2e39a1dadcc020001bd3d763635f461d2b7dc6d94bfb3cfecdb9bb20240fa758`.
  - Hint distribution:
    - `useful_tool`: 10,510.
    - `likely_required`: 1,744.
    - `no_tool`: 3,078.
    - `optional_tool`: 4,668.
- Validation data:
  - None for the training launch itself; eval must be a separate ledger entry.
- Benchmark output:
  - None.
- Script / command:
  - Prepared step-000001 plan:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g8pb1_stepwise_20260628/step_000001/stage3_grpo_training_plan.json`.
  - Prepared preflight:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g8pb1_stepwise_20260628/step_000001/stage3_grpo_preflight_report.json`.
- GPUs:
  - Planned training: GPUs 0,1,2,3.
  - Planned 32B judge: one B200 GPU per judge shard; current prepared command
    assumes one shard.
- tmux:
  - Not launched.
- Started:
  - Pending launch.
- Finished:
  - Pending.
- Metrics:
  - Prepared plan passed preflight.
  - Global rollouts per step: 32.
  - Probe config: `enabled=false`, `missing_policy=teacher_hint`,
    `hint_label_weight=1.0`.
  - Judge cache status before rollout/judge fill: expected `cache_missing`.
- Analysis:
  - This config will produce ToolDecision labels from hints:
    `useful_tool` and `likely_required` -> `tool_needed`;
    `no_tool` -> `tool_unnecessary`; `optional_tool` -> `unknown`.
  - With the current cache-only judge design, formal multi-step all-5 training
    should run stepwise: rollout-only -> 32B judge cache fill -> one training
    update -> next checkpoint/seed/step directory.
  - Measured pilot timings for 32 rollouts/step:
    - rollout-only pending generation: about 108s.
    - 32B judge for 62 rows: about 115s.
    - one 4GPU train update: about 226s.
    - total current single-shard estimate: about 7.5 minutes per step.
  - If judge pending rows are sharded across 4 available B200 GPUs, expected
    per-step wall time is roughly 6-6.5 minutes, dominated by rollout/replay.
- Conclusion:
  - Formal training is ready at the one-step plan/preflight level, but launch
    should use a stepwise driver or manual loop so every step gets fresh 32B
    judge caches before reward computation.
- Comparable to baseline:
  - Not launched yet.
- Follow-up:
  - Decide step count for first formal run: recommended 100-step run before
    committing to 500-1000 steps.
  - Add or run a stepwise launcher that updates policy checkpoint and seed after
    each one-step update.

### EXP-20260628-213900-stage3-formal-all5-hint-200step

- Status: SIDE_RESULT_STOPPED_FOR_RECONFIG.
- Question:
  - Run the first formal Stage3 all-5 GRPO training for 200 optimizer steps with
    resume-safe, non-repeating RL prompts.
- Baseline anchor:
  - `EXP-20260628-205812-stage3-formal-all5-hint-stepwise`.
- Intended diff:
  - Add a deterministic global sample schedule and stepwise runner.
  - Bind 200 steps x 4 ranks x 1 prompt/rank = 800 unique training prompts.
  - Require no repeated `sample_id` and no repeated `stable_image_uid` within
    this 200-step schedule.
  - Run each step as rollout-only -> 32B judge cache fill -> one 4GPU GRPO
    update -> state/checkpoint update.
- Code commit / worktree:
  - Base commit before local launch changes: `ca79aed`.
  - Launch uses the committed Stage3 schedule/stepwise-runner code from this
    ledger update; the regenerated step plan records the exact git commit.
- Source checkpoint:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - SHA256: `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
  - Rows: 20,000.
  - SHA256: `2e39a1dadcc020001bd3d763635f461d2b7dc6d94bfb3cfecdb9bb20240fa758`.
- Sample schedule:
  - `outputs/stage3_grpo/formal_all5_hint_4gpu_g8pb1_200step_20260628/stage3_grpo_sample_schedule.jsonl`.
  - Rows: 800.
  - SHA256: `603a0e2900185a054b910121c595447a9bf56c393f0bcedc5dced72f944de89c`.
  - Duplicate sample ids: 0.
  - Duplicate image uids: 0.
  - Tool buckets: `tool_helpful=392`, `tool_unnecessary=256`,
    `uncertain=152`.
- Stepwise state:
  - `outputs/stage3_grpo/formal_all5_hint_4gpu_g8pb1_200step_20260628/stage3_grpo_stepwise_state.json`.
  - Resume rule: use `current_checkpoint` and `next_step`; failed steps rerun
    the same scheduled sample ids and do not advance state.
- Reward / rollout:
  - Group size: 8.
  - Global prompts per step: 4.
  - Global rollouts per step: 32.
  - `max_tool_calls=1`.
  - Reward weights: answer=2.0, tool=1.0, focus=1.0, ground=1.0,
    protocol gate=-1.0.
  - Probe labels: forced probes disabled for this run; use teacher hint with
    `hint_label_weight=1.0`.
  - Judge: local Qwen3-VL-32B-Thinking, no-thinking JSON mode, 4 shards on
    GPUs 0,1,2,3.
- W&B:
  - Project: `tgvf-stage3`.
  - Mode: offline.
  - Group: `formal_all5_hint_200step`.
  - Checkpoint artifact upload: disabled by default due checkpoint size.
- Planned command:
  - `PYTHONPATH=revisit_vlm_clean/src python -m revisit_vlm_clean.cli.stage3_grpo_stepwise --template-plan outputs/stage3_grpo/formal_all5_hint_4gpu_g8pb1_200step_20260628/step_000001/stage3_grpo_training_plan.json --output-root outputs/stage3_grpo/formal_all5_hint_4gpu_g8pb1_200step_20260628 --state-path outputs/stage3_grpo/formal_all5_hint_4gpu_g8pb1_200step_20260628/stage3_grpo_stepwise_state.json --target-step 200 --max-new-steps 200 --judge-devices cuda:0,cuda:1,cuda:2,cuda:3 --execute`
- Started:
  - 2026-06-28 21:47:38 JST.
  - tmux session: `stage3_grpo_200step`.
  - Runner log:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g8pb1_200step_20260628/stepwise_runner_stdout.log`.
- Finished:
  - Stopped at 2026-06-28 JST after user rejected the low-VRAM conservative
    configuration.
- Metrics:
  - Preflight status: passed.
  - Expected runtime from current pilot: roughly 20-25 hours depending on judge
    shard utilization and rollout/replay variance.
  - Stopped after 13 completed stepwise updates.
  - Last checkpoint before stop:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g8pb1_200step_20260628/step_000013/checkpoint_step_1.pt`.
  - Reason:
    - The launch used `group_size=8`, `max_image_resolution=512`,
      LoRA-only policy training, and serial rollout/replay. Policy stages used
      far below the desired B200 memory target; user requested a higher-throughput
      configuration closer to 120GB/GPU rather than a conservative ~30-50GB/GPU
      profile.
  - Conclusion:
    - Do not treat this as the formal Stage3 checkpoint lineage.
    - Next run should be a VRAM calibration and then a fresh formal schedule.

### EXP-20260628-223500-stage3-vram-calib-g16-res768

- Status: COMPLETED.
- Question:
  - Calibrate a less conservative Stage3 GRPO configuration that uses B200
    memory more effectively than the stopped `G=8`, `res=512` run.
- Intended diff:
  - `group_size=16`.
  - `max_image_resolution=768`.
  - `max_new_tokens=256`, `max_action_tokens=128`, `max_answer_tokens=160`.
  - Keep 4GPU, manual SGD, LoRA-only, all-5 reward plumbing, and teacher-hint
    ToolDecision.
- Output:
  - `outputs/stage3_grpo/vram_calib_g16_res768_4gpu_step014_20260628`.
- Source checkpoint:
  - Side-result checkpoint from stopped run:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g8pb1_200step_20260628/step_000013/checkpoint_step_1.pt`.
  - This checkpoint was used only for VRAM calibration, not as a formal lineage.
- Metrics:
  - One step completed successfully.
  - Global rollouts per step: 64.
  - Distributed rollout count: 64.
  - Distributed replayed tokens: 2515.
  - GPU memory trace peak on physical GPUs 0-3:
    - GPU0: 117,878 MiB.
    - GPU1: 124,642 MiB.
    - GPU2: 62,756 MiB.
    - GPU3: 119,672 MiB.
  - The low GPU2 peak appears sample/trajectory-length driven; rollout-only
    still uses about 23GB because rollout generation remains serial.
- Conclusion:
  - `G=16`, `res=768` is the first acceptable high-VRAM configuration. It
    reaches the requested ~120GB/GPU profile during train/replay on ranks with
    nontrivial trajectory length, while staying well under B200 capacity.
  - For an even stricter all-rank floor, consider `G=20` or `res=896`, but that
    likely pushes long-rank peaks toward 150GB+.

### EXP-20260628-224500-stage3-formal-all5-g16-res768-200step

- Status: SIDE_RESULT_STOPPED_FOR_WANDB_ONLINE_RESTART.
- Question:
  - Relaunch formal Stage3 all-5 GRPO from the clean Stage2 checkpoint using the
    high-VRAM calibrated `G=16`, `res=768` profile.
- Intended diff from stopped run:
  - Start again from the original Stage2 step-1200 checkpoint, not the stopped
    side-result checkpoint.
  - Use a fresh deterministic 200-step schedule.
  - Use `group_size=16`, `max_image_resolution=768`,
    `max_new_tokens=256`, `max_action_tokens=128`,
    `max_answer_tokens=160`.
- Source checkpoint:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - SHA256: `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
  - SHA256: `2e39a1dadcc020001bd3d763635f461d2b7dc6d94bfb3cfecdb9bb20240fa758`.
- Planned output:
  - `outputs/stage3_grpo/formal_all5_hint_4gpu_g16_res768_200step_20260628`.
- Sample schedule:
  - `outputs/stage3_grpo/formal_all5_hint_4gpu_g16_res768_200step_20260628/stage3_grpo_sample_schedule.jsonl`.
  - Rows: 800.
  - SHA256: `e0a1695b3634a95fbc48acd8147a366274422a590ca332b4114c38150c522324`.
  - Duplicate sample ids: 0.
  - Duplicate image uids: 0.
- Runtime:
  - Started: 2026-06-28 22:42:25 JST.
  - tmux session: `stage3_g16_res768_200step`.
  - W&B mode: offline.
  - Runner log:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g16_res768_200step_20260628/stepwise_runner_stdout.log`.
  - GPU memory trace:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g16_res768_200step_20260628/gpu_mem_trace.csv`.
- Initial metrics:
  - Step 1 completed successfully and advanced state to `next_step=2`.
  - Step 1 checkpoint:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g16_res768_200step_20260628/step_000001/checkpoint_step_1.pt`.
  - Step 1 distributed rollout count: 64.
  - Step 1 distributed replayed tokens: 1850.
  - Observed GPU memory trace peak on physical GPUs 0-3 after step 1:
    - GPU0: 125,154 MiB.
    - GPU1: 133,704 MiB.
    - GPU2: 118,098 MiB.
    - GPU3: 120,540 MiB.
  - This satisfies the requested high-VRAM profile; rollout-only phases still
    drop to about 23GB because rollout generation is serial, while train/replay
    reaches the intended 120GB range.
- Stop / side-result update:
  - Stopped after user requested cloud W&B upload for the formal run.
  - Completed steps before stop: 1-19.
  - Current checkpoint at stop:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g16_res768_200step_20260628/step_000019/checkpoint_step_1.pt`.
  - State file status:
    `stopped_for_wandb_online_restart`.
  - Reason this is a side result:
    W&B mode was `offline`; it does not satisfy the formal logging requirement.

### EXP-20260629-000300-stage3-formal-all5-g16-res768-200step-wandb-online

- Status: FAILED.
- Question:
  - Relaunch the formal Stage3 all-5 GRPO 200-step run with W&B cloud upload
    enabled, while keeping checkpoint artifacts disabled.
- Repository / code identity:
  - Git commit: `76378830e48ba6208fda95666ce273bf5b55d137`.
  - Worktree at plan time: clean.
- Source checkpoint:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - SHA256: `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
  - Rows: 20,000.
  - SHA256: `2e39a1dadcc020001bd3d763635f461d2b7dc6d94bfb3cfecdb9bb20240fa758`.
- Planned output:
  - `outputs/stage3_grpo/formal_all5_hint_4gpu_g16_res768_200step_wandb_online_20260629`.
- Plan / preflight:
  - Plan:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g16_res768_200step_wandb_online_20260629/step_000001/stage3_grpo_training_plan.json`.
  - Preflight:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g16_res768_200step_wandb_online_20260629/step_000001/stage3_grpo_preflight_report.json`.
  - Preflight status: `passed`.
- Sample schedule:
  - `outputs/stage3_grpo/formal_all5_hint_4gpu_g16_res768_200step_wandb_online_20260629/stage3_grpo_sample_schedule.jsonl`.
  - Rows: 800 = 200 steps x 4 ranks x 1 prompt.
  - SHA256: `27cbfd4f44715b3f33aeff981327596e22ec9410528e60c0545e99f8d2a7c474`.
  - Duplicate sample ids: 0.
  - Duplicate image uids: 0.
  - Source distribution:
    chartqa 85, docvqa 153, textvqa 237, visual_genome 325.
  - Tool bucket distribution:
    tool_helpful 387, tool_unnecessary 248, uncertain 165.
- Runtime config:
  - GPUs: physical 0-3 via `CUDA_VISIBLE_DEVICES=0,1,2,3`.
  - World size: 4.
  - Per-device prompt batch size: 1.
  - Group size: 16.
  - Global rollouts per step: 64.
  - `max_image_resolution=768`, `max_new_tokens=256`,
    `max_action_tokens=128`, `max_answer_tokens=160`.
  - Reward: answer/tool/focus/ground/protocol all enabled.
  - Tool decision reward uses teacher hint labels; probe cache is not required
    for this launch.
  - Judge mode: `cache_only`; judge cache misses receive configured cache-miss
    reward and are logged for offline judging.
- W&B:
  - Project: `tgvf-stage3`.
  - Mode: `online`.
  - Group: `formal_all5_g16_res768_200step_online`.
  - `log_artifacts=true`.
  - `log_checkpoint_artifact=false`.
  - Check immediately before launch still showed `wandb status` as
    `api_key=null`, but `~/.netrc` contains a W&B login entry and the user
    confirmed the machine is logged in, so the online launch is proceeding.
  - If W&B auth is still not visible to the training subprocess, this run should
    fail fast in step 1 rather than silently becoming offline.
  - Online auth confirmed in step 1 training log:
    logged in as `mio_mi0 (mio_nora)`.
  - Step 1 W&B run:
    `https://wandb.ai/mio_nora/tgvf-stage3/runs/0rwxtdpl`.
- Runtime:
  - Started: 2026-06-29 00:13 JST.
  - tmux session: `stage3_g16_res768_wandb_online_200step`.
  - Runner log:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g16_res768_200step_wandb_online_20260629/stepwise_runner_stdout.log`.
  - GPU memory trace:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g16_res768_200step_wandb_online_20260629/gpu_mem_trace.csv`.
- Launch command:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=revisit_vlm_clean/src python -u -m revisit_vlm_clean.cli.stage3_grpo_stepwise --template-plan outputs/stage3_grpo/formal_all5_hint_4gpu_g16_res768_200step_wandb_online_20260629/step_000001/stage3_grpo_training_plan.json --output-root outputs/stage3_grpo/formal_all5_hint_4gpu_g16_res768_200step_wandb_online_20260629 --state-path outputs/stage3_grpo/formal_all5_hint_4gpu_g16_res768_200step_wandb_online_20260629/stage3_grpo_stepwise_state.json --target-step 200 --max-new-steps 200 --judge-devices cuda:0,cuda:1,cuda:2,cuda:3 --judge-max-image-resolution 768 --execute`.
- Failure:
  - Failed during step 1 launch-training before `train_metrics.jsonl` had any
    rows, so no `train/loss` or reward scalar metrics were uploaded.
  - Root cause: rank2 CUDA OOM during replay `_append_visual_d`.
  - Error excerpt:
    `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 28.00 MiB.
    GPU 2 ... 10.12 MiB is free ... this process has 178.34 GiB memory in use`.
  - W&B uploaded/created only the step-1 run config and environment metadata for
    run `0rwxtdpl`; it did not receive loss metrics or output artifacts because
    artifact logging happens after a step completes.
  - State file:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g16_res768_200step_wandb_online_20260629/stage3_grpo_stepwise_state.json`.
  - State status: `failed`, completed steps: 0.

### EXP-20260629-002900-stage3-formal-all5-g12-res768-200step-wandb-online

- Status: STOPPED_INVALID_FOR_TOOL_LEARNING.
- Question:
  - Relaunch formal Stage3 all-5 GRPO after the G16/res768 online run OOMed
    before metrics, keeping W&B online upload and checkpoint artifact exclusion.
- Intended diff:
  - Reduce `group_size` from 16 to 12.
  - Keep `max_image_resolution=768`.
  - Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the launcher.
- Repository / code identity:
  - Git commit: `6847d8d91927a011440e9a5a6a7e4ba7130f3ed7`.
- Source checkpoint:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - SHA256: `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Train data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
  - Rows: 20,000.
  - SHA256: `2e39a1dadcc020001bd3d763635f461d2b7dc6d94bfb3cfecdb9bb20240fa758`.
- Output:
  - `outputs/stage3_grpo/formal_all5_hint_4gpu_g12_res768_200step_wandb_online_20260629`.
- Sample schedule:
  - `outputs/stage3_grpo/formal_all5_hint_4gpu_g12_res768_200step_wandb_online_20260629/stage3_grpo_sample_schedule.jsonl`.
  - Rows: 800 = 200 steps x 4 ranks x 1 prompt.
  - SHA256: `268384d6d1c4f06e73ed313cdb0202f01e8bc7f3e05d705a0d6377a4bc6dddf5`.
  - Duplicate sample ids: 0.
  - Duplicate image uids: 0.
- Plan / preflight:
  - Plan:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g12_res768_200step_wandb_online_20260629/step_000001/stage3_grpo_training_plan.json`.
  - Stepwise preflight: passed, warnings: none.
- Runtime config:
  - GPUs: physical 0-3 via `CUDA_VISIBLE_DEVICES=0,1,2,3`.
  - World size: 4.
  - Per-device prompt batch size: 1.
  - Group size: 12.
  - Global rollouts per step: 48.
  - `max_image_resolution=768`, `max_new_tokens=256`,
    `max_action_tokens=128`, `max_answer_tokens=160`.
  - Reward: answer/tool/focus/ground/protocol all enabled.
  - Judge mode: `cache_only`; offline 32B judge shards run before training.
- W&B:
  - Project: `tgvf-stage3`.
  - Mode: `online`.
  - Group: `formal_all5_g12_res768_200step_online`.
  - `log_artifacts=true`.
  - `log_checkpoint_artifact=false`.
- Runtime:
  - Started: 2026-06-29 00:29 JST.
  - tmux session: `stage3_g12_res768_wandb_online_200step`.
  - Runner log:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g12_res768_200step_wandb_online_20260629/stepwise_runner_stdout.log`.
  - GPU memory trace:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g12_res768_200step_wandb_online_20260629/gpu_mem_trace.csv`.
- Crash / recovery:
  - Failed at global step 14 on 2026-06-29 01:33 JST.
  - Completed steps before crash: 13.
  - Last good checkpoint:
    `outputs/stage3_grpo/formal_all5_hint_4gpu_g12_res768_200step_wandb_online_20260629/step_000013/checkpoint_step_1.pt`.
  - Step 14 wrote no `train_metrics.jsonl` rows and uploaded no loss metrics.
  - Root cause from logs: not an OOM. Rank0/1/2 entered gradient
    all-reduce, while rank3 hit the trainer failure path before
    `rollout_reward_done` and then entered the final barrier in `finally`.
    NCCL timed out because ranks were in different collectives.
  - Code recovery patch:
    - log `free_rollout_failed` / `run_training_failed` with traceback;
    - do not enter the final distributed barrier when a rank is unwinding from
      an exception;
    - all-reduce zero gradients for trainable params with missing `grad` so all
      ranks execute the same collective sequence.
  - Verification before resume:
    `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests/test_stage3_grpo.py`
    passed, 33 tests.
  - Recovery commit: `eb047d8`.
  - Failed step directory will be preserved as
    `step_000014_failed_nccl_20260629_013341` before retrying step 14.
  - Resume note:
    - A first resume shell command exited before training because `PYTHONPATH`
      was scoped only to the background memory tracer; it consumed no step and
      did not change the state file.
    - Corrected resume session: `stage3_g12_res768_wandb_online_200step_resume2`.
    - Step 14 retry succeeded and advanced state to `next_step=15`.
    - Step 14 checkpoint:
      `outputs/stage3_grpo/formal_all5_hint_4gpu_g12_res768_200step_wandb_online_20260629/step_000014/checkpoint_step_1.pt`.
    - Step 14 W&B online run:
      `https://wandb.ai/mio_nora/tgvf-stage3/runs/2aykbhba`.
    - Step 14 rank0 loss: `0.09086516499519348`;
      distributed mean loss: `0.15043716318905354`.
  - Pause / stepwise logging fix:
    - User noticed the stepwise implementation created one W&B run per step and
      one checkpoint per step. The run was stopped before continuing.
    - State at pause: completed through step 30, next step 31, current
      checkpoint
      `outputs/stage3_grpo/formal_all5_hint_4gpu_g12_res768_200step_wandb_online_20260629/step_000030/checkpoint_step_1.pt`.
    - Code change planned/applied: parent stepwise runner owns one W&B run;
      child per-step training subprocesses set `wandb.mode=disabled`; parent
      logs `train/*`, `reward/*`, rollout, and checkpoint retention metrics at
      true global step.
    - Checkpoint retention: keep last 2 plus every 25 steps by default.
    - Existing checkpoints cleaned with this policy: kept steps 25, 29, 30;
      deleted 27 old checkpoint files; output directory reduced to about 2.1G.
    - Verification:
      `PYTHONPATH=revisit_vlm_clean/src pytest -q revisit_vlm_clean/tests/test_stage3_grpo.py`
      passed, 35 tests.
    - Fix commit: `5e3b725`.
    - Resume2 launch command:
      `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/revisit_vlm_clean/src:/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/src python -u -m revisit_vlm_clean.cli.stage3_grpo_stepwise --template-plan outputs/stage3_grpo/formal_all5_hint_4gpu_g12_res768_200step_wandb_online_20260629/step_000001/stage3_grpo_training_plan.json --output-root outputs/stage3_grpo/formal_all5_hint_4gpu_g12_res768_200step_wandb_online_20260629 --state-path outputs/stage3_grpo/formal_all5_hint_4gpu_g12_res768_200step_wandb_online_20260629/stage3_grpo_stepwise_state.json --start-step 31 --target-step 200 --max-new-steps 170 --checkpoint-keep-last 2 --checkpoint-keep-every 25 --judge-devices cuda:0,cuda:1,cuda:2,cuda:3 --judge-max-image-resolution 768 --execute`.
    - Resume2 was stopped after the user requested the W&B/ckpt policy fix to
      be verified before continuing the formal run.
    - Resume3 tmux session:
      `stage3_g12_res768_wandb_online_200step_resume3`.
    - Resume3 parent W&B run:
      `https://wandb.ai/mio_nora/tgvf-stage3/runs/wczsw7l6`.
    - Verified child step plans set `wandb.mode=disabled`,
      `log_artifacts=false`, and `log_checkpoint_artifact=false`; grep of
      `step_000031/torchrun_train_stdout.log` showed no child W&B run init.
    - Step 31 completed successfully and advanced state to `next_step=32`.
    - After step 31 retention kept steps 25, 30, 31 and deleted the prior
      non-milestone step 29 checkpoint.
    - Resume3 completed through step 32, then failed at step 33 during model
      load on rank 1. It was not an OOM and not a W&B/ckpt failure. The error
      was a transformers/hub lookup for
      `Qwen/Qwen3-VL-8B-Thinking/model-00002-of-00004.safetensors` even though
      the shard exists in the local HuggingFace cache.
    - Local-only cache validation succeeded for config, processor,
      `model.safetensors.index.json`, and
      `model-00002-of-00004.safetensors`.
    - Resume4 plan: continue from step 33 using checkpoint
      `step_000032/checkpoint_step_1.pt`, with
      `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to avoid hub/cache false
      negatives during repeated multi-process model loads.
    - Failed step 33 directory archived as
      `step_000033_failed_hf_cache_20260629_031927`.
    - Resume4 tmux session:
      `stage3_g12_res768_wandb_online_200step_resume4`.
    - Resume4 parent W&B run:
      `https://wandb.ai/mio_nora/tgvf-stage3/runs/48tqt1mk`.
    - Step 33 retry succeeded with HF offline cache, wrote
      `step_000033/checkpoint_step_1.pt`, and advanced state to
      `next_step=34`.
    - User stopped the run on 2026-06-29 after observing average tool calls near
      zero. The run was killed during step 146 launch-training and GPUs were
      verified idle afterward.
    - Completed steps at stop: 145. Current checkpoint:
      `outputs/stage3_grpo/formal_all5_hint_4gpu_g12_res768_200step_wandb_online_20260629/step_000145/checkpoint_step_1.pt`.
    - Step 146 is an interrupted partial directory with empty rollout/reward/
      train metrics and no checkpoint.
    - Diagnostic aggregation through step 141 found all-rank `used_tool` =
      1/6816 rollouts, with many textual `<tool_response>` / `<tool_call>`
      emissions but almost no legal `<|focus_start|>...<|focus_end|>` actions.
      This run is invalid for judging Stage3 tool-use learning.
  - Stage3 RL 20k train data direct/focus distribution:
    - `tool_need_hint`: useful_tool 10,510; likely_required 1,744;
      optional_tool 4,668; no_tool 3,078.
    - Strict buckets: focus/useful_or_required 12,254; optional 4,668;
      direct/no_tool 3,078.
    - If optional is grouped with focus-like local/reasoning samples:
      focus/useful_or_required 16,876; direct/no_tool 3,124.

### DIAG-20260629-094323-stage3-original-ckpt-toolmark

- Status: DONE.
- Question:
  - After stopping the invalid Stage3 GRPO run with near-zero tool calls, test
    whether the original Stage2 checkpoint can generate legal Stage3 tool marks
    under the current Stage3 rollout code.
- Baseline / context:
  - Invalid stopped run:
    `EXP-20260629-002900-stage3-formal-all5-g12-res768-200step-wandb-online`.
  - Stopped at completed step 145; step 146 interrupted.
- Planned checkpoint / processor:
  - Policy checkpoint:
    `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - SHA256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
  - Checkpoint config model/processor:
    `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Data:
  - `revisit_vlm_clean/data/stage3_rl/v1_direct_20k_20260627_032447/accepted_rl_prompts.jsonl`.
  - SHA256:
    `2e39a1dadcc020001bd3d763635f461d2b7dc6d94bfb3cfecdb9bb20240fa758`.
- Planned output:
  - `outputs/stage3_grpo/diagnostics/original_ckpt_toolmark_current_stage3_20260629_094323`.
- Planned diagnostic:
  - Current code commit: `915c8dddbcfdf47a798f1e2eb32079e00b642c15`.
  - 4 GPUs, rollout-only, no training update, no W&B.
  - `runtime_backend=native_single_focus`.
  - `protocol=protocol_c_tool_observation`.
  - `group_size=12`, `world_size=4`, `per_device_prompt_batch_size=1`,
    `max_steps=1`, yielding 48 free rollouts.
  - `max_image_resolution=768`, `max_action_tokens=128`,
    `max_answer_tokens=160`, `temperature=1.0`, `top_p=0.95`.
  - DeepStack state: clean `RunConfig` default is disabled,
    `original_image_scope=off`, `d_features_enabled=false`.
- Results:
  - 4-GPU rollout-only command completed, but the direct executor output
    retained only rank-0 rollout rows in root output; it produced 12 root
    rollouts for one prompt and no legal tool marks:
    `used_tool=0/12`, `<|focus_start|>` count 0.
  - Follow-up single-rank batch-4 diagnostic:
    `outputs/stage3_grpo/diagnostics/original_ckpt_toolmark_current_stage3_20260629_094323_single_rank_pb4`.
  - Single-rank batch-4 result: 48 rollouts across four prompts,
    `used_tool=1/48`, `<|focus_start|>` count 1,
    `<|focus_end|>` count 1, `<tool_call>` count 0,
    `<tool_response>` count 0.
  - The one legal tool call came from sample
    `e80b6dd7f757f309deef6627` (`chartqa`, `optional_tool`,
    question: "What was Tanzania's fertility rate in 2019?").
  - Example legal action:
    `<|focus_start|>the 2019 bar with its value label and nearby year tick on the x-axis<|focus_end|>`.
  - The diagnostic confirms the original Stage2 checkpoint can generate legal
    Stage3 focus marks under the current code, but the spontaneous rate is very
    low. This matches the stopped formal run's near-zero tool-call average and
    means the current GRPO setup lacks enough tool-action exploration.

### DIAG-20260629-102455-clean-softforce-hr200-kv-vs-nokv

- Status: BLOCKED_FOR_KV_DEEPSTACK_COMPARISON.
- Question:
  - On a small high-trigger benchmark, test whether `kv_cache` and
    `no_kv_full_sequence` have matching accuracy and whether `kv_cache` is
    faster once DeepStack state is controlled.
- Macro plan:
  - Repository/worktree strategy: diagnostic run on the current clean branch;
    no executable code changes.
  - What changes now: run two fresh HR-Bench-4K 200-row soft-force benchmark
    evaluations, one with `post_tgvf_forward_mode=no_kv_full_sequence` and one
    with `post_tgvf_forward_mode=kv_cache`.
  - What will not be touched: training, parser/scorer, checkpoint, protocol,
    benchmark adapter code, Stage3/RL code, and existing benchmark outputs.
  - Verification: compare sample manifests, run configs, wall time, accuracy,
    parse rate, trigger/focus-valid rate, append success, malformed/OOM rows,
    and row-level answer agreement.
- Baseline anchors:
  - Full clean soft-force CoreDev run:
    `EXP-20260628-1027-clean-qwen3-stage2-norm01-softforce-coredev2511`.
  - Baseline output:
    `outputs/clean_benchmarks/qwen3_stage2_norm01_softforce_coredev2511_deepstack512_recovery_4shard_20260628_102713/merged`.
  - HR slice in that output: `200` rows, accuracy `0.510000`, trigger
    `0.720000`, parse `0.995000`.
- Model / processor:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Stage2 checkpoint:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - SHA256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Stage2 runtime eval JSONL:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`.
  - SHA256:
    `3b719e1bd03a09741cc05dac3ed85e423a9b2c29209b07546792a6795cdbbfc5`.
- Benchmark:
  - Diagnostic manifest:
    `outputs/clean_benchmarks/diagnostic_hr200_softforce_kv_vs_nokv_20260629/preflight/hr_core200_from_coredev2511_softforce_manifest.json`.
  - Manifest id:
    `diagnostic_hr_core200_from_coredev2511_seed20260625`.
  - Manifest hash:
    `705b29fc3c380d97a31947b54f3423f3c4da94da7fa79826c05ad8b2df297e76`.
  - Source rule: filter `benchmark == hr_bench_4k` from the completed
    CoreDev-2511 soft-force merged sample manifest, preserving order.
  - Sample identity check: `200/200` overlap with old HR rows, same order.
  - Benchmark root:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks`.
- Evaluation identity:
  - Eval family: `project_native_external`.
  - Mode: `tgvf_softforce`.
  - Soft-force prompt: `Use focus tool.`
  - Runner backend: `tgvf_stage2_qwen3_native`.
  - Protocol: `protocol_c_tool_observation`.
  - Stage2 D condition: `correct_D`.
  - DeepStack: enabled; `original_image_scope=through_answer`;
    D DeepStack-like features disabled.
  - Max image resolution: `512`.
  - Max action tokens: `64`.
  - Max answer tokens: `512`.
  - Scoring backend: `auto`.
  - Attention implementation: `sdpa`.
- Planned outputs:
  - No-KV:
    `outputs/clean_benchmarks/diagnostic_hr200_softforce_kv_vs_nokv_20260629/nokv`.
  - KV:
    `outputs/clean_benchmarks/diagnostic_hr200_softforce_kv_vs_nokv_20260629/kv`.
- GPUs:
  - Planned: physical GPUs `4,5,6,7`, one shard per GPU.
  - Current GPUs `0-7` were idle at preflight; using `4-7` to avoid any
    future overlap with Stage3/RL conventions on `0-3`.
- Launch command:
  - Planned ledger commit:
    `6712f40611c0e14e7f1816e6ef88bf42822c0c67`.
  - Started: `2026-06-29 10:27 JST`.
  - tmux session:
    `diag_hr200_softforce_kv_vs_nokv_20260629`.
  - Command:
    `cd /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm && outputs/clean_benchmarks/diagnostic_hr200_softforce_kv_vs_nokv_20260629/run_kv_vs_nokv.sh`.
- Results:
  - No-KV output:
    `outputs/clean_benchmarks/diagnostic_hr200_softforce_kv_vs_nokv_20260629/nokv/merged`.
  - No-KV wall time: `2026-06-29T10:27:59+09:00` to
    `2026-06-29T10:33:55+09:00`, about `5m56s` on 4 GPUs.
  - No-KV metrics:
    - n_rows `200`, n_scored `200`.
    - accuracy `0.510000`.
    - answer_parse_rate `0.995000`.
    - trigger/focus_valid `0.720000`.
    - append_success_rate `1.000000` among triggered appends.
    - malformed_rate `0.000000`.
    - row wall time sum `1232.50s`, mean `6.16s`, p50 `5.01s`.
  - No-KV reproducibility against the full CoreDev soft-force HR slice:
    - same sample ids/order: `200/200`.
    - same final output: `200/200`.
    - same parsed answer: `200/200`.
    - same score: `200/200`.
  - KV output:
    `outputs/clean_benchmarks/diagnostic_hr200_softforce_kv_vs_nokv_20260629/kv`.
  - KV result: invalid, no rows written.
  - KV failure:
    all four shards failed in `backend.prepare()` before model loading with
    `NotImplementedError`.
  - Failure reason:
    current clean benchmark backend only reports implemented DeepStack hooks for
    `post_tgvf_forward_mode=no_kv_full_sequence`. For
    `post_tgvf_forward_mode=kv_cache` with DeepStack enabled and
    `original_image_scope=through_answer`, the runtime correctly blocks the run
    because these hooks are not ported:
    `capture_original_image_deepstack_features`,
    `carry_original_image_deepstack_through_post_tgvf_append`,
    `apply_post_tgvf_deepstack_scope_mask`.
- Conclusion:
  - This diagnostic confirms the fresh no-KV HR200 path is stable and exactly
    reproduces the old full CoreDev soft-force HR slice.
  - It does not answer whether KV and no-KV are accuracy-equivalent under
    controlled DeepStack, because the clean KV+DeepStack execution path is
    intentionally blocked as unimplemented.
  - A `DeepStack off` KV-vs-no-KV diagnostic would test cache mechanics only,
    but it would not answer the requested controlled-DeepStack question and
    should be a separately named ablation if run.

### DIAG-20260629-111358-clean-kv-deepstack-implementation-smoke

- Status: RUNNING_AFTER_TAIL_ALIGN_FIX.
- Question:
  - Implement and validate clean Stage2 benchmark support for
    `kv_cache + DeepStack` so cached post-D continuation can be compared to
    `no_kv_full_sequence` under the same DeepStack scope.
- Macro plan:
  - Repository/worktree strategy: implement on the current clean branch and
    keep legacy code unchanged.
  - What changes now:
    - Add a cached multi-token original-image key-block attention mask helper.
    - Mark clean Qwen3 native KV+DeepStack hooks as supported.
    - Apply the cached chunk mask in `_append_visual_d` and propagate
      original-image key-block metadata into continuation.
    - Add focused unit tests for the gate, mask semantics, and fake KV append.
  - What will not be touched: training code, benchmark manifests, parser/scorer,
    legacy evaluation bridges, and Stage3/RL.
  - Verification:
    - Unit tests for DeepStack masks and Stage2 runner/backend identity.
    - Real HR-Bench-4K 200-row soft-force KV+DeepStack run on the same manifest
      used by `DIAG-20260629-102455`.
    - Compare KV rows against the prior no-KV HR200 rows for accuracy, parse
      rate, trigger rate, answer/score agreement, and wall time.
- Implementation tests already run before launch:
  - `PYTHONPATH=revisit_vlm_clean/src:src pytest -q revisit_vlm_clean/tests/test_deepstack.py revisit_vlm_clean/tests/test_runner_backend.py::test_stage2_native_backend_accepts_supported_kv_deepstack revisit_vlm_clean/tests/test_runner_backend.py::test_stage2_native_backend_accepts_supported_full_sequence_deepstack revisit_vlm_clean/tests/test_runner_backend.py::test_stage2_native_backend_accepts_supported_full_sequence_evidence_only_deepstack revisit_vlm_clean/tests/test_runner_backend.py::test_deepstack_execution_plan_records_scope_semantics revisit_vlm_clean/tests/test_runner_backend.py::test_stage2_legacy_backend_rejects_unported_deepstack_execution revisit_vlm_clean/tests/test_runner_backend.py::test_stage2_native_kv_deepstack_append_uses_cached_chunk_mask revisit_vlm_clean/tests/test_runner_backend.py::test_stage2_native_deepstack_evidence_only_restores_attention_after_answer_boundary`
    passed: `12 passed`.
  - `PYTHONPATH=revisit_vlm_clean/src:src pytest -q revisit_vlm_clean/tests/test_runner_backend.py revisit_vlm_clean/tests/test_benchmark_data.py revisit_vlm_clean/tests/test_cli.py::test_stage2_deepstack_plan_disables_legacy_command revisit_vlm_clean/tests/test_cli.py::test_stage2_deepstack_prepare_writes_training_plan`
    passed: `44 passed`.
- Baseline / comparison:
  - Prior no-KV HR200:
    `outputs/clean_benchmarks/diagnostic_hr200_softforce_kv_vs_nokv_20260629/nokv/merged`.
  - Prior no-KV metrics: accuracy `0.510000`, parse `0.995000`, trigger
    `0.720000`, malformed `0.000000`, wall about `5m56s`.
- Model / processor:
  - `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking`.
- Stage2 checkpoint:
  - `outputs/clean_training/qwen3_stage2_norm01_stage1_mask075_deepstack_4gpu_20260627_163250/stage2_micro4/clean_training_execution/checkpoint_step_1200.pt`.
  - SHA256:
    `50245a11c27ad9755eb815b5f008af50a659fa07a4043f0f9427f5bbea3c0236`.
- Stage2 runtime eval JSONL:
  - `data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl`.
  - SHA256:
    `3b719e1bd03a09741cc05dac3ed85e423a9b2c29209b07546792a6795cdbbfc5`.
- Benchmark:
  - Diagnostic manifest:
    `outputs/clean_benchmarks/diagnostic_hr200_softforce_kv_vs_nokv_20260629/preflight/hr_core200_from_coredev2511_softforce_manifest.json`.
  - Manifest hash:
    `705b29fc3c380d97a31947b54f3423f3c4da94da7fa79826c05ad8b2df297e76`.
  - Sample count/order: `200`, same as prior no-KV HR200.
  - Benchmark root:
    `/home/dredvpn009/Flash_Storage/datasets/benchmarks`.
- Evaluation identity:
  - Eval family: `project_native_external`.
  - Mode: `tgvf_softforce`.
  - Soft-force prompt: `Use focus tool.`
  - Runner backend: `tgvf_stage2_qwen3_native`.
  - Protocol: `protocol_c_tool_observation`.
  - Stage2 D condition: `correct_D`.
  - Post-TGVF forward mode to validate: `kv_cache`.
  - DeepStack: enabled; `original_image_scope=through_answer`;
    D DeepStack-like features disabled.
  - Max image resolution: `512`.
  - Max action tokens: `64`.
  - Max answer tokens: `512`.
  - Scoring backend: `auto`.
  - Attention implementation: `sdpa`.
- Planned output:
  - `outputs/clean_benchmarks/diagnostic_hr200_softforce_kv_deepstack_equiv_20260629_1114/kv`.
- GPUs:
  - Planned: physical GPUs `4,5,6,7`, one shard per GPU.
- Launch command:
  - Implementation / launch commit:
    `6a638c15ccea5fd986e63619b7202b9cc0c52b5a`.
  - Started: `2026-06-29 11:17 JST`.
  - tmux session:
    `diag_hr200_softforce_kv_deepstack_20260629`.
  - Command:
    `cd /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm && outputs/clean_benchmarks/diagnostic_hr200_softforce_kv_deepstack_equiv_20260629_1114/run_kv.sh`.
- First launch result:
  - Output:
    `outputs/clean_benchmarks/diagnostic_hr200_softforce_kv_deepstack_equiv_20260629_1114/kv/merged`.
  - Status: INVALID_FOR_ACCURACY_COMPARISON.
  - Metrics:
    - n_rows `200`, n_scored `56`.
    - reported accuracy `0.642857`, but this is only over untriggered/direct
      rows and must not be compared to no-KV.
    - answer_parse_rate `0.275000`.
    - trigger/focus_valid `0.720000`.
    - append_success_rate `0.000000`.
    - malformed_rate `0.720000`.
  - Failure reason:
    all triggered KV+DeepStack appends failed with a one-token key-length
    mismatch in the cached chunk 4D attention mask, for example target key
    length `656` vs mask key length `657`.
  - Code-level diagnosis:
    generated `past_key_values.get_seq_length()` can be one token shorter than
    `capture.input_ids.shape[-1]`. The fix is not to trim the mask; that would
    drop the last captured token from the cached-prefix semantics. The fix is
    to prefill any missing tail token(s) into the cache before appending D.
- Tail-align implementation:
  - Fix commit:
    `e5ce88227da20cebbdc9f091e8e906540ca34a7a`.
  - Added `_align_capture_cache_to_input_ids()` in clean native Stage2:
    when DeepStack KV append is requested, it compares cache seq length to
    captured `input_ids`, prefills missing tail tokens through the Qwen3 decode
    helper, then builds the cached D-chunk original-image key-block mask.
  - Added row debug fields:
    `kv_cache_input_len`, `kv_cache_initial_seq_len`,
    `kv_cache_tail_prefill_tokens`, `kv_cache_aligned_seq_len`,
    `kv_cache_tail_prefill_used`.
  - Additional tests:
    - `PYTHONPATH=revisit_vlm_clean/src:src pytest -q revisit_vlm_clean/tests/test_deepstack.py revisit_vlm_clean/tests/test_runner_backend.py::test_stage2_native_kv_deepstack_append_uses_cached_chunk_mask revisit_vlm_clean/tests/test_runner_backend.py::test_stage2_native_kv_deepstack_prefills_generate_cache_tail revisit_vlm_clean/tests/test_runner_backend.py::test_stage2_native_deepstack_evidence_only_restores_attention_after_answer_boundary`
      passed: `8 passed`.
    - `PYTHONPATH=revisit_vlm_clean/src:src pytest -q revisit_vlm_clean/tests/test_runner_backend.py revisit_vlm_clean/tests/test_benchmark_data.py revisit_vlm_clean/tests/test_cli.py::test_stage2_deepstack_plan_disables_legacy_command revisit_vlm_clean/tests/test_cli.py::test_stage2_deepstack_prepare_writes_training_plan`
      passed: `45 passed`.
- Tail-align rerun:
  - Planned output:
    `outputs/clean_benchmarks/diagnostic_hr200_softforce_kv_deepstack_equiv_tailfix_20260629/kv`.
  - Intended diff from first launch: code commit only; same checkpoint,
    processor/model, HR200 manifest, sample order, prompt, parser/scorer,
    DeepStack scope, max resolution, action/answer token budgets, and GPUs.
  - Launch commit:
    `e5ce88227da20cebbdc9f091e8e906540ca34a7a`.
  - Started:
    `2026-06-29 11:36 JST`.
  - tmux session:
    `diag_hr200_softforce_kv_deepstack_tailfix_20260629`.
  - Command:
    `cd /nvmesv/dredvpn009/projects/r-vlm/revisit_vlm && outputs/clean_benchmarks/diagnostic_hr200_softforce_kv_deepstack_equiv_tailfix_20260629/run_kv_tailfix.sh`.
