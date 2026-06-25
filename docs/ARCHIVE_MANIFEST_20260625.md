# Archive Manifest 2026-06-25

This archive records the state before starting the clean TGVF project skeleton.

## Intended Commit Scope

- Tracked planning / workflow / experiment changes.
- Clean-project planning documents.
- Small local backup source files under `backups/chat_issue_20260618_181349/`.
- Small benchmark launcher script:
  `scripts/run_qwen3_two_ckpts_other_benchmarks_nokv_4gpu.sh`.

## Excluded Local Assets

The following local assets are intentionally not committed in the archive
snapshot:

- `logs/`: local run logs and pid files.
- `logs/tgvf_v4_teacher_50k.pid`: stale/local process id file.
- `third_party/VLMEvalKit/`: local third-party checkout, kept as an external
  dependency/reference instead of vendoring into this repository snapshot.

## Follow-Up

Implementation work should start from a separate clean implementation branch
after this archive commit exists.
