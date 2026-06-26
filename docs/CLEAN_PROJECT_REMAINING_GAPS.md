# Clean Project Remaining Gaps

Last updated: 2026-06-26.

This file tracks implementation gaps that affect clean-native TGVF training,
evaluation, or experiment operations. It is intentionally separate from the
experiment ledger: the ledger records individual runs; this file records
remaining engineering work.

## Fixed In Current Patch

- Clean training observability:
  - rank0 now writes `training_progress.jsonl` during the real trainer loop;
  - rank0 prints one compact stdout line per optimizer step;
  - rank0 initializes W&B and logs step metrics when `wandb.project` is set and
    `wandb.mode != disabled`;
  - non-rank0 processes do not initialize W&B or duplicate progress rows.
- The progress/W&B behavior is covered by
  `revisit_vlm_clean/tests/test_cli.py::test_stage2_training_executor_can_launch_single_process_training_loop`.

## Remaining Gaps

1. Restart formal Stage1 after the observability patch.
   - The interrupted Stage1 micro4 run did not produce a completed checkpoint.
   - Relaunch with `world_size=4`, `micro_batch_size=4`,
     `gradient_accumulation_steps=2`, `global_batch_size=32`.

2. W&B artifact upload is not yet implemented in clean training.
   - Metrics and summaries are now logged.
   - Checkpoint artifacts are still only local files.

3. Stage2 launch must remain chained to the new Stage1 checkpoint.
   - Do not generate the final Stage2 plan until the relaunched Stage1 writes
     its actual checkpoint and processor identity.

4. Clean benchmark evaluation with DeepStack remains incomplete.
   - Training supports Stage2 DeepStack scope semantics.
   - Benchmark eval still rejects unported DeepStack execution unless the backend
     can prove the hooks are implemented.
   - Relevant guard:
     `revisit_vlm_clean/src/revisit_vlm_clean/runner.py::_reject_unported_deepstack_execution`.

5. The clean benchmark runner still has a diagnostic legacy bridge.
   - `tgvf_stage2_qwen3_legacy` is restricted to `internal_diagnostic`.
   - Final benchmark tables should use clean-native backend paths, not the
     diagnostic bridge, unless explicitly labeled as diagnostic.

6. ValKit is first-class but externally configured.
   - The clean ValKit surface exists, but executable runs require explicit
     `--execute`, `--valkit-root`, and `--valkit-model-name`.
   - Results from ValKit and project-native external benchmarks are not
     interchangeable unless sample identity and scorer identity are matched.

7. Stage1/Stage2 throughput is not optimized.
   - Stage1 DDP utilization is bursty because batches are not length/visual-token
     bucketed across ranks.
   - This is an efficiency gap, not a correctness blocker.

8. Stage2 visual-token manifold loss is still an experiment decision.
   - Clean Stage2 default remains `loss_visual_token_manifold=0.0`.
   - Raising it should be a separately named ablation, not a silent default
     change.

9. Mid-run resume policy is minimal.
   - Clean checkpoint save/load contracts exist.
   - The current formal Stage1 plan saves only at final step 2000, so an
     interruption before then requires relaunch from scratch.

