# Archive Commit Notes 2026-06-25

This archive branch preserves the current experimental code state before project normalization.

## Included

- Current tracked code modifications.
- Experiment ledger and workflow documents.
- TGVF experiment workflow Codex skill.
- New/modified training, eval, chat, probe, and benchmark scripts.
- Reports and small summary artifacts.

## Excluded

- `outputs/`: large generated checkpoints/evaluation outputs; already ignored and about 1TB.
- `logs/`: runtime logs; paths and conclusions are represented in the ledger.
- `backups/`: local backup snapshots, not canonical source.
- `third_party/VLMEvalKit`: nested git repository. It remains on disk as a local external dependency, but is not committed as an unusable gitlink in this archive commit.

## Cleanup Policy After Archive

No option, protocol, or script should be deleted solely because it looks old.
Future cleanup should classify each item as:

- active mainline;
- active RL/stage3;
- diagnostic side branch;
- archived baseline reference;
- unknown, requiring user decision.
