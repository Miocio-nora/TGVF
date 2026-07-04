#!/usr/bin/env python3
"""Generate or launch a sequential clean D-DeepStack ablation queue.

The queue is deliberately thin. It runs Stage1 for branches that do not have a
checkpoint yet, then delegates Stage1 diagnostics, Stage2, and CoreDev
benchmarks to ``run_clean_ddeepstack_ablation_pipeline.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHONPATH = "revisit_vlm_clean/src:src"
DEFAULT_STAGE1_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
STAGE1_TRAIN_JSONL = (
    "data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/"
    "splits/tgvf_v4_teacher_stage1_protocol_c_focus.train.jsonl"
)
STAGE1_EVAL_JSONL = (
    "data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend/"
    "splits/tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl"
)
STAGE2_TRAIN_JSONL = (
    "data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/"
    "splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl"
)
STAGE2_VAL_JSONL = (
    "data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/"
    "splits/tgvf_v4_teacher_stage2_protocol_c.test.jsonl"
)
COREDEV_MANIFEST = "revisit_vlm_clean/benchmark_manifests/core_balanced_dev_2511_seed20260625.json"
MCE_SIZE3_FREE_SUMMARY = (
    "outputs/clean_benchmarks/"
    "qwen3_stage2_ddeepstack_mce_size3_coredev2511_dynamic_maxtok512_flash2_"
    "ddeepstack_gpu0_1_2_3_4_5_6_7_20260703_224753/tgvf_free/summary.json"
)
MCE_SIZE3_SOFT_SUMMARY = (
    "outputs/clean_benchmarks/"
    "qwen3_stage2_ddeepstack_mce_size3_coredev2511_dynamic_maxtok512_flash2_"
    "ddeepstack_gpu0_1_2_3_4_5_6_7_20260703_224753/tgvf_softforce/summary.json"
)


@dataclass(frozen=True)
class BranchSpec:
    branch_id: str
    run_slug: str
    matrix_ce_setting: str
    stage1_micro_batch_size: int
    stage1_gradient_accumulation_steps: int
    stage1_global_batch: int
    loss_same_image_negative: float
    same_image_negative_mode: str = "matrix_ce"

    def stage1_output_dir(self, timestamp: str) -> Path:
        return (
            Path("outputs/clean_training")
            / f"qwen3_stage1_ddeepstack_{self.run_slug}_8gpu_{timestamp}"
            / f"stage1_micro{self.stage1_micro_batch_size}"
        )

    def stage1_smoke_output_dir(self, timestamp: str) -> Path:
        return (
            Path("outputs/clean_training")
            / f"qwen3_stage1_ddeepstack_{self.run_slug}_8gpu_{timestamp}_smoke"
            / f"stage1_micro{self.stage1_micro_batch_size}"
        )


BRANCHES = {
    "mce_size2": BranchSpec(
        branch_id="mce_size2",
        run_slug="size2",
        matrix_ce_setting="enabled, size 2; 2x2 Matrix CE / pairwise same-image contrast",
        stage1_micro_batch_size=2,
        stage1_gradient_accumulation_steps=2,
        stage1_global_batch=32,
        loss_same_image_negative=1.0,
    ),
    "mce_off_size1": BranchSpec(
        branch_id="mce_off_size1",
        run_slug="mce_off_size1",
        matrix_ce_setting="off; same-image negative loss weight 0.0",
        stage1_micro_batch_size=4,
        stage1_gradient_accumulation_steps=1,
        stage1_global_batch=32,
        loss_same_image_negative=0.0,
    ),
}


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _cmd(args: list[object]) -> str:
    return shlex.join([str(arg) for arg in args])


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _file_identity(path: str) -> dict[str, object]:
    target = REPO_ROOT / path
    identity: dict[str, object] = {
        "path": path,
        "exists": target.exists(),
    }
    if not target.exists():
        return identity
    digest = hashlib.sha256()
    line_count = 0
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            line_count += chunk.count(b"\n")
    identity.update(
        {
            "sha256": digest.hexdigest(),
            "size_bytes": target.stat().st_size,
            "line_count": line_count,
        }
    )
    return identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate/launch the queued clean D-DeepStack Matrix CE ablations."
    )
    parser.add_argument("--branches", default="mce_size2,mce_off_size1")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--wait-session", default="pipeline_mce_size3_20260703_224753")
    parser.add_argument("--wait-poll-seconds", type=int, default=300)
    parser.add_argument(
        "--wait-success-file",
        action="append",
        default=None,
        help="File that must exist after the wait session exits before queued branches start.",
    )
    parser.add_argument("--driver-output-dir", default=None)
    parser.add_argument("--wandb-project", default="tgvf-clean-qwen3-deepstack")
    parser.add_argument("--benchmark-batch-size", type=int, default=1)
    parser.add_argument("--write-driver", action="store_true")
    parser.add_argument("--launch-tmux", action="store_true")
    parser.add_argument("--tmux-session", default=None)
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    branch_ids = _parse_csv(args.branches)
    unknown = [branch_id for branch_id in branch_ids if branch_id not in BRANCHES]
    if unknown:
        raise ValueError(f"unknown branch ids: {unknown}; valid={sorted(BRANCHES)}")
    gpus = _parse_csv(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    if len(gpus) != 8:
        raise ValueError("this queued ablation is intentionally bound to 8 GPUs")
    wait_success_files = args.wait_success_file or [
        MCE_SIZE3_FREE_SUMMARY,
        MCE_SIZE3_SOFT_SUMMARY,
    ]
    queue_dir = Path(
        args.driver_output_dir or f"outputs/clean_pipeline/queued_mce_size2_size1_{timestamp}"
    )
    branches = [BRANCHES[branch_id] for branch_id in branch_ids]
    plan = _build_plan(args, timestamp, branches, gpus, wait_success_files, queue_dir)
    driver = _build_driver(args, plan, branches, gpus, wait_success_files)
    driver_path = queue_dir / "run_queue.sh"
    plan_path = queue_dir / "queue_plan.json"

    if args.write_driver or args.launch_tmux:
        queue_dir.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        driver_path.write_text(driver, encoding="utf-8")
        driver_path.chmod(0o755)

    if args.launch_tmux:
        session = args.tmux_session or f"queue_mce_size2_size1_{timestamp}"
        subprocess.check_call(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                session,
                f"cd {shlex.quote(str(REPO_ROOT))}; bash {shlex.quote(str(driver_path))}",
            ]
        )
        plan["tmux_session"] = session

    payload = {**plan, "driver_path": str(driver_path), "plan_path": str(plan_path)}
    if args.print_json or not (args.write_driver or args.launch_tmux):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "driver_path": str(driver_path),
                    "plan_path": str(plan_path),
                    **({"tmux_session": plan["tmux_session"]} if "tmux_session" in plan else {}),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


def _build_plan(
    args: argparse.Namespace,
    timestamp: str,
    branches: list[BranchSpec],
    gpus: list[str],
    wait_success_files: list[str],
    queue_dir: Path,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "queue_dir": str(queue_dir),
        "wait_session": args.wait_session,
        "wait_poll_seconds": args.wait_poll_seconds,
        "wait_success_files": wait_success_files,
        "gpus": gpus,
        "data": {
            "stage1_train": _file_identity(STAGE1_TRAIN_JSONL),
            "stage1_eval": _file_identity(STAGE1_EVAL_JSONL),
            "stage2_train": _file_identity(STAGE2_TRAIN_JSONL),
            "stage2_val": _file_identity(STAGE2_VAL_JSONL),
            "coredev_manifest": _file_identity(COREDEV_MANIFEST),
        },
        "branches": [_branch_plan(branch, timestamp) for branch in branches],
    }


def _branch_plan(branch: BranchSpec, timestamp: str) -> dict[str, object]:
    stage1_output_dir = branch.stage1_output_dir(timestamp)
    stage1_smoke_output_dir = branch.stage1_smoke_output_dir(timestamp)
    stage1_checkpoint = stage1_output_dir / "clean_training_execution" / "checkpoint_step_2000.pt"
    post_pipeline_dir = Path("outputs/clean_pipeline") / f"{branch.branch_id}_{timestamp}"
    stage2_dir = (
        Path("outputs/clean_training")
        / f"qwen3_stage2_ddeepstack_norm01_from_stage1_ddeepstack_{branch.branch_id}_8gpu_{timestamp}"
        / "stage2_micro4"
    )
    bench_base = (
        Path("outputs/clean_benchmarks")
        / f"qwen3_stage2_ddeepstack_{branch.branch_id}_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu0_1_2_3_4_5_6_7_{timestamp}"
    )
    return {
        "branch_id": branch.branch_id,
        "matrix_ce_setting": branch.matrix_ce_setting,
        "stage1_micro_batch_size": branch.stage1_micro_batch_size,
        "stage1_gradient_accumulation_steps": branch.stage1_gradient_accumulation_steps,
        "stage1_global_batch": branch.stage1_global_batch,
        "stage1_loss_same_image_negative": branch.loss_same_image_negative,
        "stage1_same_image_negative_mode": branch.same_image_negative_mode,
        "stage1_output_dir": str(stage1_output_dir),
        "stage1_smoke_output_dir": str(stage1_smoke_output_dir),
        "stage1_checkpoint": str(stage1_checkpoint),
        "post_stage1_pipeline_dir": str(post_pipeline_dir),
        "post_stage1_driver": str(post_pipeline_dir / "run_pipeline.sh"),
        "stage2_checkpoint": str(
            stage2_dir / "clean_training_execution" / "checkpoint_step_1200.pt"
        ),
        "coredev_free_summary": str(bench_base / "tgvf_free" / "summary.json"),
        "coredev_softforce_summary": str(bench_base / "tgvf_softforce" / "summary.json"),
    }


def _build_driver(
    args: argparse.Namespace,
    plan: dict[str, object],
    branches: list[BranchSpec],
    gpus: list[str],
    wait_success_files: list[str],
) -> str:
    timestamp = str(plan["timestamp"])
    gpus_csv = ",".join(gpus)
    world_size = len(gpus)
    chunks: list[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {_q(REPO_ROOT)}",
        f"export PYTHONPATH={_q(PYTHONPATH)}",
        "",
        "pick_port() { shuf -i 40000-59999 -n 1; }",
        "need_file() { if [[ ! -f \"$1\" ]]; then echo \"[missing] $1\" >&2; exit 1; fi; }",
        "step_header() { echo; echo \"========== $1 ==========\"; }",
        "run_checked() { echo \"+ $*\"; \"$@\"; }",
        "",
        "step_header wait_for_mce_size3",
        f"if tmux has-session -t {_q(args.wait_session)} 2>/dev/null; then",
        f"  echo '[wait] session {_q(args.wait_session)} is still running'",
        f"  while tmux has-session -t {_q(args.wait_session)} 2>/dev/null; do",
        f"    sleep {int(args.wait_poll_seconds)}",
        "    date '+[wait] %Y-%m-%d %H:%M:%S %Z'",
        "  done",
        "fi",
    ]
    for success_file in wait_success_files:
        chunks.append(f"need_file {_q(success_file)}")
    chunks.append("")

    for branch in branches:
        chunks.extend(_branch_driver_chunks(branch, timestamp, gpus_csv, world_size, args))

    chunks.extend(
        [
            "echo",
            "echo '[done] queued clean D-DeepStack ablation branches finished'",
            "",
        ]
    )
    return "\n".join(chunks)


def _branch_driver_chunks(
    branch: BranchSpec,
    timestamp: str,
    gpus_csv: str,
    world_size: int,
    args: argparse.Namespace,
) -> list[str]:
    stage1_dir = branch.stage1_output_dir(timestamp)
    smoke_dir = branch.stage1_smoke_output_dir(timestamp)
    stage1_plan = stage1_dir / "training_plan.json"
    smoke_plan = smoke_dir / "training_plan.json"
    stage1_ckpt = stage1_dir / "clean_training_execution" / "checkpoint_step_2000.pt"
    smoke_ckpt = smoke_dir / "clean_training_execution" / "checkpoint_step_1.pt"
    stage1_log = stage1_dir / "logs" / "train.log"
    smoke_log = smoke_dir / "logs" / "train.log"
    post_pipeline_dir = Path("outputs/clean_pipeline") / f"{branch.branch_id}_{timestamp}"
    post_driver = post_pipeline_dir / "run_pipeline.sh"
    chunks = [
        f"step_header {branch.branch_id}_stage1_smoke",
        f"if [[ -f {_q(smoke_ckpt)} ]]; then",
        f"  echo '[skip] Stage1 smoke checkpoint exists: {_q(smoke_ckpt)}'",
        "else",
        f"  mkdir -p {_q(smoke_log.parent)}",
        "  "
        + _cmd(
            _stage1_plan_cmd(
                branch,
                smoke_dir,
                f"clean_qwen3_stage1_ddeepstack_{branch.run_slug}_8gpu_smoke_{timestamp}",
                max_steps=1,
                save_every=1,
                wandb_mode="disabled",
                wandb_project=None,
            )
        ),
        f"  MASTER_PORT=$(pick_port) CUDA_VISIBLE_DEVICES={_q(gpus_csv)} "
        + f"torchrun --nproc-per-node {world_size} -m revisit_vlm_clean.training.stage1_executor --plan {_q(smoke_plan)} --launch-training 2>&1 | tee {_q(smoke_log)}",
        "fi",
        f"need_file {_q(smoke_ckpt)}",
        "",
        f"step_header {branch.branch_id}_stage1_train",
        f"if [[ -f {_q(stage1_ckpt)} ]]; then",
        f"  echo '[skip] Stage1 checkpoint exists: {_q(stage1_ckpt)}'",
        "else",
        f"  mkdir -p {_q(stage1_log.parent)}",
        "  "
        + _cmd(
            _stage1_plan_cmd(
                branch,
                stage1_dir,
                f"clean_qwen3_stage1_ddeepstack_{branch.run_slug}_8gpu_{timestamp}",
                max_steps=2000,
                save_every=500,
                wandb_mode="online",
                wandb_project=args.wandb_project,
            )
        ),
        f"  MASTER_PORT=$(pick_port) CUDA_VISIBLE_DEVICES={_q(gpus_csv)} "
        + f"torchrun --nproc-per-node {world_size} -m revisit_vlm_clean.training.stage1_executor --plan {_q(stage1_plan)} --launch-training 2>&1 | tee {_q(stage1_log)}",
        "fi",
        f"need_file {_q(stage1_ckpt)}",
        "",
        f"step_header {branch.branch_id}_post_stage1_pipeline",
        _cmd(
            [
                "python",
                "scripts/run_clean_ddeepstack_ablation_pipeline.py",
                "--branch-id",
                branch.branch_id,
                "--timestamp",
                timestamp,
                "--stage1-checkpoint",
                stage1_ckpt,
                "--stage1-output-dir",
                stage1_dir,
                "--gpus",
                gpus_csv,
                "--steps",
                "stage1_diag,stage2_smoke,stage2_train,bench_smoke,bench_free,bench_softforce",
                "--batch-size",
                str(args.benchmark_batch_size),
                "--wandb-project",
                args.wandb_project,
                "--write-driver",
            ]
        ),
        f"bash {_q(post_driver)}",
        "",
    ]
    return chunks


def _stage1_plan_cmd(
    branch: BranchSpec,
    output_dir: Path,
    run_id: str,
    *,
    max_steps: int,
    save_every: int,
    wandb_mode: str,
    wandb_project: str | None,
) -> list[object]:
    cmd: list[object] = [
        "python",
        "-m",
        "revisit_vlm_clean.cli.train_stage1",
        "--run-id",
        run_id,
        "--train-file",
        STAGE1_TRAIN_JSONL,
        "--output-dir",
        output_dir,
        "--model-id",
        DEFAULT_STAGE1_MODEL,
        "--protocol",
        "protocol_c_tool_observation",
        "--max-image-resolution",
        "512",
        "--max-steps",
        str(max_steps),
        "--save-every",
        str(save_every),
        "--seed",
        "20260525",
        "--dtype",
        "bfloat16",
        "--attn-implementation",
        "sdpa",
        "--variant",
        "tgvf_v2_bidirectional",
        "--d-deepstack-enabled",
        "--d-deepstack-branch-layers",
        "8,16,24",
        "--token-row-mode",
        "row_only",
        "--capture-mode",
        "teacher_forced",
        "--fvt-position-mode",
        "native_source_grid",
        "--focus-action-im-end",
        "--mask-original-image-after-tgvf",
        "--learning-rate",
        "1e-4",
        "--lr-scheduler",
        "cosine",
        "--warmup-steps",
        "100",
        "--min-lr-ratio",
        "0.1",
        "--max-grad-norm",
        "1.0",
        "--loss-gen",
        "1.0",
        "--loss-visual-token-manifold",
        "0.0",
        "--loss-visual-token-norm",
        "0.1",
        "--loss-same-image-negative",
        str(branch.loss_same_image_negative),
        "--same-image-negative-margin",
        "1.0",
        "--same-image-negative-mode",
        branch.same_image_negative_mode,
        "--readout-batch-size",
        "4",
        "--world-size",
        "8",
        "--micro-batch-size",
        str(branch.stage1_micro_batch_size),
        "--gradient-accumulation-steps",
        str(branch.stage1_gradient_accumulation_steps),
        "--global-batch",
        str(branch.stage1_global_batch),
        "--wandb-mode",
        wandb_mode,
    ]
    if wandb_project:
        cmd.extend(["--wandb-project", wandb_project])
    cmd.append("--write-plan")
    return cmd


if __name__ == "__main__":
    raise SystemExit(main())
