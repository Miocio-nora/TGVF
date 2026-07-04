#!/usr/bin/env python3
"""Generate or launch the clean D-DeepStack ablation pipeline driver.

This is intentionally a thin orchestration layer.  The individual clean CLIs
remain the source of truth for training, diagnostics, and benchmarks.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHONPATH = "revisit_vlm_clean/src:src"
DEFAULT_LOCAL_MODEL = "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking"
DEFAULT_STAGE1_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
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
COREDEV_HASH = "a461d9b482b7165b42b9bbb0fbf0ea6aff31fde0a838c13d953f070e770b0579"
BENCHMARK_ROOT = "/home/dredvpn009/Flash_Storage/datasets/benchmarks"


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _cmd(args: list[object]) -> str:
    return shlex.join([str(arg) for arg in args])


def _parse_steps(raw: str) -> list[str]:
    steps = [item.strip() for item in raw.split(",") if item.strip()]
    valid = {
        "stage1_diag",
        "stage2_smoke",
        "stage2_train",
        "bench_smoke",
        "bench_free",
        "bench_softforce",
    }
    unknown = [item for item in steps if item not in valid]
    if unknown:
        raise ValueError(f"unknown steps: {unknown}; valid={sorted(valid)}")
    return steps


def _gpu_list(raw: str) -> list[str]:
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must include at least one GPU id")
    return gpus


def _default_stage1_output_dir(checkpoint: Path) -> Path:
    parent = checkpoint.parent
    if parent.name == "clean_training_execution":
        return parent.parent
    return parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate/launch the clean D-DeepStack ablation pipeline driver."
    )
    parser.add_argument("--branch-id", required=True, help="Ablation branch id, e.g. mce_size3.")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage1-output-dir", default=None)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--diagnostic-gpu", default=None)
    parser.add_argument(
        "--steps",
        default="stage1_diag,stage2_smoke,stage2_train,bench_smoke,bench_free,bench_softforce",
        help="Comma-separated subset of stage1_diag,stage2_smoke,stage2_train,bench_smoke,bench_free,bench_softforce.",
    )
    parser.add_argument("--driver-output-dir", default=None)
    parser.add_argument("--stage1-model-id", default=DEFAULT_STAGE1_MODEL)
    parser.add_argument("--model-id", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--processor-id", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--stage1-eval-jsonl", default=STAGE1_EVAL_JSONL)
    parser.add_argument("--stage2-train-file", default=STAGE2_TRAIN_JSONL)
    parser.add_argument("--stage2-val-file", default=STAGE2_VAL_JSONL)
    parser.add_argument("--benchmark-root", default=BENCHMARK_ROOT)
    parser.add_argument("--wandb-project", default="tgvf-clean-qwen3-deepstack")
    parser.add_argument("--batch-size", type=int, default=1, help="Dynamic benchmark generation batch size.")
    parser.add_argument("--write-driver", action="store_true")
    parser.add_argument("--launch-tmux", action="store_true")
    parser.add_argument("--tmux-session", default=None)
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    stage1_checkpoint = Path(args.stage1_checkpoint)
    stage1_output_dir = (
        Path(args.stage1_output_dir)
        if args.stage1_output_dir
        else _default_stage1_output_dir(stage1_checkpoint)
    )
    gpus = _gpu_list(args.gpus)
    diagnostic_gpu = args.diagnostic_gpu or gpus[0]
    gpu_token = "_".join(gpus)
    steps = _parse_steps(args.steps)
    driver_output_dir = Path(
        args.driver_output_dir or f"outputs/clean_pipeline/{args.branch_id}_{timestamp}"
    )
    stage1_diag_dir = stage1_output_dir / f"internal_diagnostics_step2000_{timestamp}"
    stage2_dir = Path(
        "outputs/clean_training"
    ) / f"qwen3_stage2_ddeepstack_norm01_from_stage1_ddeepstack_{args.branch_id}_8gpu_{timestamp}" / "stage2_micro4"
    stage2_smoke_dir = Path(
        "outputs/clean_training"
    ) / f"qwen3_stage2_ddeepstack_norm01_from_stage1_ddeepstack_{args.branch_id}_8gpu_{timestamp}_smoke" / "stage2_micro4"
    bench_base = Path(
        "outputs/clean_benchmarks"
    ) / f"qwen3_stage2_ddeepstack_{args.branch_id}_coredev2511_dynamic_maxtok512_flash2_ddeepstack_gpu{gpu_token}_{timestamp}"
    bench_smoke_dir = Path(
        "outputs/clean_benchmarks"
    ) / f"qwen3_stage2_ddeepstack_{args.branch_id}_coredev2511_dynamic_smoke_free_maxtok512_flash2_ddeepstack_gpu{diagnostic_gpu}_{timestamp}"
    stage2_checkpoint = stage2_dir / "clean_training_execution" / "checkpoint_step_1200.pt"

    plan = {
        "branch_id": args.branch_id,
        "timestamp": timestamp,
        "steps": steps,
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage1_output_dir": str(stage1_output_dir),
        "stage1_diag_dir": str(stage1_diag_dir),
        "stage2_smoke_dir": str(stage2_smoke_dir),
        "stage2_dir": str(stage2_dir),
        "stage2_checkpoint": str(stage2_checkpoint),
        "bench_smoke_dir": str(bench_smoke_dir),
        "bench_free_dir": str(bench_base / "tgvf_free"),
        "bench_softforce_dir": str(bench_base / "tgvf_softforce"),
        "gpus": gpus,
        "diagnostic_gpu": diagnostic_gpu,
    }
    driver = _build_driver(args, plan)
    driver_path = driver_output_dir / "run_pipeline.sh"
    plan_path = driver_output_dir / "pipeline_plan.json"

    if args.write_driver or args.launch_tmux:
        driver_output_dir.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        driver_path.write_text(driver, encoding="utf-8")
        driver_path.chmod(0o755)

    if args.launch_tmux:
        session = args.tmux_session or f"pipeline_{args.branch_id}_{timestamp}"
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

    if args.print_json or not (args.write_driver or args.launch_tmux):
        print(json.dumps({**plan, "driver_path": str(driver_path), "plan_path": str(plan_path)}, indent=2))
    else:
        print(json.dumps({"driver_path": str(driver_path), "plan_path": str(plan_path), **({"tmux_session": plan["tmux_session"]} if "tmux_session" in plan else {})}, indent=2))
    return 0


def _build_driver(args: argparse.Namespace, plan: dict[str, object]) -> str:
    steps = set(plan["steps"])  # type: ignore[arg-type]
    gpus = ",".join(plan["gpus"])  # type: ignore[arg-type]
    world_size = len(plan["gpus"])  # type: ignore[arg-type]
    diagnostic_gpu = str(plan["diagnostic_gpu"])
    stage1_checkpoint = str(plan["stage1_checkpoint"])
    stage1_diag_dir = str(plan["stage1_diag_dir"])
    stage2_smoke_dir = str(plan["stage2_smoke_dir"])
    stage2_dir = str(plan["stage2_dir"])
    stage2_checkpoint = str(plan["stage2_checkpoint"])
    bench_smoke_dir = str(plan["bench_smoke_dir"])
    bench_free_dir = str(plan["bench_free_dir"])
    bench_softforce_dir = str(plan["bench_softforce_dir"])
    branch_id = str(plan["branch_id"])
    timestamp = str(plan["timestamp"])

    chunks: list[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {_q(REPO_ROOT)}",
        f"export PYTHONPATH={_q(PYTHONPATH)}",
        "",
        "pick_port() { shuf -i 40000-59999 -n 1; }",
        "need_file() { if [[ ! -f \"$1\" ]]; then echo \"[missing] $1\" >&2; exit 1; fi; }",
        "step_header() { echo; echo \"========== $1 ==========\"; }",
        f"need_file {_q(stage1_checkpoint)}",
        "",
    ]

    if "stage1_diag" in steps:
        diag_log = Path(stage1_diag_dir) / "logs" / "diagnostics.log"
        diag_done = Path(stage1_diag_dir) / "clean_stage_diagnostic_status.json"
        chunks.extend(
            [
                "step_header stage1_diag",
                f"if [[ -f {_q(diag_done)} ]]; then",
                f"  echo '[skip] stage1 diagnostics already exists: {_q(diag_done)}'",
                "else",
                f"  mkdir -p {_q(diag_log.parent)}",
                "  "
                + _env_cmd(
                    {"CUDA_VISIBLE_DEVICES": diagnostic_gpu},
                    _stage1_diag_cmd(args, stage1_checkpoint, stage1_diag_dir, branch_id, timestamp),
                )
                + f" 2>&1 | tee {_q(diag_log)}",
                "fi",
                "",
            ]
        )

    if "stage2_smoke" in steps:
        smoke_plan = Path(stage2_smoke_dir) / "training_plan.json"
        smoke_ckpt = Path(stage2_smoke_dir) / "clean_training_execution" / "checkpoint_step_1.pt"
        smoke_log = Path(stage2_smoke_dir) / "logs" / "train.log"
        chunks.extend(
            [
                "step_header stage2_smoke",
                f"if [[ -f {_q(smoke_ckpt)} ]]; then",
                f"  echo '[skip] stage2 smoke checkpoint already exists: {_q(smoke_ckpt)}'",
                "else",
                f"  mkdir -p {_q(smoke_log.parent)}",
                _cmd(_stage2_plan_cmd(args, stage1_checkpoint, stage2_smoke_dir, branch_id, timestamp, smoke=True, world_size=world_size)),
                f"  MASTER_PORT=$(pick_port) CUDA_VISIBLE_DEVICES={_q(gpus)} "
                + f"torchrun --nproc-per-node {world_size} -m revisit_vlm_clean.training.stage2_executor --plan {_q(smoke_plan)} --launch-training 2>&1 | tee {_q(smoke_log)}",
                "fi",
                f"need_file {_q(smoke_ckpt)}",
                "",
            ]
        )

    if "stage2_train" in steps:
        train_plan = Path(stage2_dir) / "training_plan.json"
        train_log = Path(stage2_dir) / "logs" / "train.log"
        chunks.extend(
            [
                "step_header stage2_train",
                f"if [[ -f {_q(stage2_checkpoint)} ]]; then",
                f"  echo '[skip] stage2 checkpoint already exists: {_q(stage2_checkpoint)}'",
                "else",
                f"  mkdir -p {_q(train_log.parent)}",
                _cmd(_stage2_plan_cmd(args, stage1_checkpoint, stage2_dir, branch_id, timestamp, smoke=False, world_size=world_size)),
                f"  MASTER_PORT=$(pick_port) CUDA_VISIBLE_DEVICES={_q(gpus)} "
                + f"torchrun --nproc-per-node {world_size} -m revisit_vlm_clean.training.stage2_executor --plan {_q(train_plan)} --launch-training 2>&1 | tee {_q(train_log)}",
                "fi",
                f"need_file {_q(stage2_checkpoint)}",
                "",
            ]
        )

    if any(step.startswith("bench_") for step in steps):
        chunks.append(f"need_file {_q(stage2_checkpoint)}")
        chunks.append("")

    if "bench_smoke" in steps:
        chunks.extend(_benchmark_step(args, "bench_smoke", "tgvf_free", bench_smoke_dir, stage2_checkpoint, diagnostic_gpu, "4", branch_id, timestamp))
    if "bench_free" in steps:
        chunks.extend(_benchmark_step(args, "bench_free", "tgvf_free", bench_free_dir, stage2_checkpoint, gpus, None, branch_id, timestamp))
    if "bench_softforce" in steps:
        chunks.extend(_benchmark_step(args, "bench_softforce", "tgvf_softforce", bench_softforce_dir, stage2_checkpoint, gpus, None, branch_id, timestamp))

    chunks.append("echo")
    chunks.append("echo '[done] clean ablation pipeline driver finished'")
    chunks.append("")
    return "\n".join(chunks)


def _env_cmd(env: dict[str, str], args: list[object]) -> str:
    prefix = " ".join(f"{key}={_q(value)}" for key, value in env.items())
    return f"{prefix} {_cmd(args)}"


def _stage1_diag_cmd(
    args: argparse.Namespace,
    checkpoint: str,
    output_dir: str,
    branch_id: str,
    timestamp: str,
) -> list[object]:
    return [
        "python",
        "-m",
        "revisit_vlm_clean.cli.stage_diagnostics",
        "--run-id",
        f"clean_qwen3_stage1_ddeepstack_{branch_id}_internal_diag_{timestamp}",
        "--stage",
        "stage1",
        "--checkpoint",
        checkpoint,
        "--eval-jsonl",
        args.stage1_eval_jsonl,
        "--output-dir",
        output_dir,
        "--model-id",
        args.stage1_model_id,
        "--protocol",
        "protocol_c_tool_observation",
        "--focus-action-im-end",
        "--variant",
        "tgvf_v2_bidirectional",
        "--encoder-adapter-type",
        "bidirectional",
        "--max-image-resolution",
        "512",
        "--fvt-position-mode",
        "native_source_grid",
        "--dtype",
        "bfloat16",
        "--attn-implementation",
        "sdpa",
        "--device",
        "cuda:0",
        "--device-map",
        "cuda:0",
        "--tasks",
        "all",
        "--readout-max-samples",
        "200",
        "--distribution-max-samples",
        "200",
        "--query-max-groups",
        "50",
        "--query-require-groups",
        "0",
        "--seed",
        "20260525",
        "--execute",
    ]


def _stage2_plan_cmd(
    args: argparse.Namespace,
    stage1_checkpoint: str,
    output_dir: str,
    branch_id: str,
    timestamp: str,
    *,
    smoke: bool,
    world_size: int,
) -> list[object]:
    suffix = "smoke" if smoke else "main"
    return [
        "python",
        "-m",
        "revisit_vlm_clean.cli.train_stage2",
        "--run-id",
        f"clean_qwen3_stage2_ddeepstack_norm01_from_stage1_ddeepstack_{branch_id}_{suffix}_{timestamp}",
        "--train-file",
        args.stage2_train_file,
        "--val-file",
        args.stage2_val_file,
        "--stage1-checkpoint",
        stage1_checkpoint,
        "--output-dir",
        output_dir,
        "--model-id",
        args.model_id,
        "--processor-id",
        args.processor_id,
        "--protocol",
        "protocol_c_tool_observation",
        "--max-image-resolution",
        "512",
        "--max-seq-len",
        "2048",
        "--max-steps",
        "1" if smoke else "1200",
        "--save-every",
        "1" if smoke else "300",
        "--eval-every",
        "1" if smoke else "300",
        "--seed",
        "20260525",
        "--dtype",
        "bfloat16",
        "--attn-implementation",
        "sdpa",
        "--variant",
        "tgvf_v2_bidirectional",
        "--use-stage1-tgvf-config",
        "--fast-batched-stage2",
        "--fvt-position-mode",
        "native_source_grid",
        "--target-focus-ratio",
        "0.8",
        "--mask-original-image-after-tgvf",
        "--mask-original-image-after-tgvf-prob",
        "0.75",
        "--mask-original-image-after-tgvf-scope",
        "through_answer",
        "--deepstack-enabled",
        "--deepstack-original-image-scope",
        "through_answer",
        "--d-deepstack-enabled",
        "--lora-rank",
        "64",
        "--lora-alpha",
        "256",
        "--lora-dropout",
        "0.05",
        "--lora-bias",
        "none",
        "--lora-target-modules",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "--lr-lora",
        "2e-5",
        "--lr-tgvf",
        "5e-6",
        "--lr-calibration",
        "1e-5",
        "--lr-scheduler",
        "cosine",
        "--warmup-ratio",
        "0.03",
        "--warmup-steps",
        "100",
        "--min-lr-ratio",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.95",
        "--adam-eps",
        "1e-8",
        "--weight-decay",
        "0.01",
        "--max-grad-norm",
        "1.0",
        "--loss-visual-token-manifold",
        "0.0",
        "--global-batch",
        "128",
        "--world-size",
        str(world_size),
        "--micro-batch-size",
        "4",
        "--gradient-accumulation-steps",
        "4",
        "--wandb-mode",
        "disabled" if smoke else "online",
        *(["--wandb-project", args.wandb_project] if not smoke else []),
        "--write-plan",
    ]


def _benchmark_step(
    args: argparse.Namespace,
    step_name: str,
    mode: str,
    output_dir: str,
    checkpoint: str,
    gpus: str,
    max_samples: str | None,
    branch_id: str,
    timestamp: str,
) -> list[str]:
    summary = Path(output_dir) / "summary.json"
    log_path = Path(output_dir) / "logs" / "benchmark.log"
    softforce = mode == "tgvf_softforce"
    cmd = [
        "python",
        "-m",
        "revisit_vlm_clean.cli.dynamic_benchmark",
        "--run-id",
        f"clean_qwen3_stage2_ddeepstack_{branch_id}_coredev2511_dynamic_{step_name}_{mode}_maxtok512_flash2_{timestamp}",
        "--checkpoint-path",
        checkpoint,
        "--model-id",
        args.model_id,
        "--processor-id",
        args.processor_id,
        "--mode",
        mode,
        "--runner-backend",
        "tgvf_stage2_qwen3_native",
        "--post-tgvf-forward-mode",
        "kv_cache",
        "--subset-id",
        "core_balanced_dev_2511_seed20260625",
        "--manifest-path",
        COREDEV_MANIFEST,
        "--manifest-hash",
        COREDEV_HASH,
        "--benchmark-root",
        args.benchmark_root,
        "--output-dir",
        output_dir,
        "--max-image-resolution",
        "512",
        "--max-tokens",
        "512",
        "--scoring-backend",
        "auto",
        "--dtype",
        "bfloat16",
        "--device",
        "cuda:0",
        "--device-map",
        "cuda:0",
        "--attn-implementation",
        "flash_attention_2",
        "--stage2-checkpoint",
        checkpoint,
        "--stage2-eval-jsonl",
        args.stage2_val_file,
        "--stage2-d-condition",
        "correct_D",
        "--force-prefix-mode",
        "target_hint",
        "--deepstack-enabled",
        "--deepstack-original-image-scope",
        "no_block",
        "--d-deepstack-enabled",
        "--gpus",
        gpus,
        "--batch-size",
        str(args.batch_size),
        "--progress-every",
        "1",
    ]
    if softforce:
        cmd.extend(["--softforce-prompt-text", "Use focus tool."])
    if max_samples:
        cmd.extend(["--max-samples", max_samples])
    return [
        f"step_header {step_name}",
        f"if [[ -f {_q(summary)} ]]; then",
        f"  echo '[skip] benchmark summary already exists: {_q(summary)}'",
        "else",
        f"  mkdir -p {_q(log_path.parent)}",
        f"  {_cmd(cmd)} 2>&1 | tee {_q(log_path)}",
        "fi",
        f"need_file {_q(summary)}",
        "",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
