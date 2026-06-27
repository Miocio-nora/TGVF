"""Clean data-generation planning CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from revisit_vlm_clean.cli.common import exit_not_implemented, print_json
from revisit_vlm_clean.data_generation import (
    DataGenerationConfig,
    DataGenerationStage,
    DataGenerationTransform,
    build_data_generation_plan,
    execute_data_generation,
    write_data_generation_plan,
)
from revisit_vlm_clean.defaults import DEFAULT_PROTOCOL
from revisit_vlm_clean.stage3_rl_data.api_runner import (
    archive_teacher_outputs,
    parse_teacher_batch_outputs,
    poll_teacher_batches,
    prepare_teacher_batch_files,
    run_teacher_requests_direct,
    submit_teacher_batches,
)
from revisit_vlm_clean.stage3_rl_data.pipeline import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MAX_QA_PER_IMAGE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
    DEFAULT_SPLIT_ALLOWLIST,
    DEFAULT_SPLIT_BLOCKLIST,
    DEFAULT_TARGET_ACCEPTED_PROMPTS,
    Stage3RLBuildOptions,
    SUPPORTED_QA_GENERATION_MODES,
    build_stage3_rl_plan,
    execute_stage3_rl_plan,
    finalize_teacher_outputs_stage3_rl_plan,
    preflight_stage3_rl_plan,
    write_stage3_rl_plan,
)
from revisit_vlm_clean.tgvf_protocol import SUPPORTED_PROTOCOLS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean TGVF data-generation planner.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage", choices=[item.value for item in DataGenerationStage], required=True
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-root", default=".")
    parser.add_argument("--input-files", nargs="*", default=[])
    parser.add_argument("--protocol", choices=SUPPORTED_PROTOCOLS, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--transform",
        choices=[item.value for item in DataGenerationTransform],
        default=DataGenerationTransform.NONE.value,
    )
    parser.add_argument("--source-manifest-path", default=None)
    parser.add_argument("--source-manifest-hash", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--split-policy", default="preserve_input")
    parser.add_argument(
        "--field-weight", action="append", default=[], help="Field loss weight as name=value."
    )
    parser.add_argument(
        "--mask-policy", action="append", default=[], help="Mask policy entry as name=value."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print resolved identity plan without writing files."
    )
    parser.add_argument("--write-plan", action="store_true", help="Write identity-only plan files.")
    parser.add_argument(
        "--execute", action="store_true", help="Execute a ported deterministic data transform."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    if raw_argv and raw_argv[0] == "stage3-rl":
        return _main_stage3_rl(raw_argv[1:])
    args = build_parser().parse_args(raw_argv)
    git_commit, dirty_worktree = _git_identity()
    config = DataGenerationConfig(
        run_id=args.run_id,
        stage=DataGenerationStage(args.stage),
        output_dir=args.output_dir,
        input_root=args.input_root,
        input_files=tuple(args.input_files),
        protocol=args.protocol,
        transform=DataGenerationTransform(args.transform),
        source_manifest_path=args.source_manifest_path,
        source_manifest_hash=args.source_manifest_hash,
        source_run_id=args.source_run_id,
        split_policy=args.split_policy,
        field_weights=_parse_float_mapping(args.field_weight),
        mask_policy=_parse_string_mapping(args.mask_policy),
        git_commit=git_commit,
        dirty_worktree=dirty_worktree,
    )
    config.validate()
    if args.dry_run:
        print_json(build_data_generation_plan(config))
        return 0
    if args.write_plan:
        print_json(write_data_generation_plan(args.output_dir, config=config))
        return 0
    if args.execute:
        print_json(execute_data_generation(config))
        return 0
    return exit_not_implemented("use --dry-run, --write-plan, or --execute with a ported transform")


def build_stage3_rl_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build clean Stage3 RL source-QA pool.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--plan", default=None, help="Existing stage3_rl_data_plan.json.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-config", default=None)
    parser.add_argument("--exclude-manifest", action="append", default=[])
    parser.add_argument(
        "--target-accepted-prompts", type=int, default=DEFAULT_TARGET_ACCEPTED_PROMPTS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--extend-from", default=None)
    parser.add_argument("--max-new-images", type=int, default=None)
    parser.add_argument("--max-new-prompts", type=int, default=None)
    parser.add_argument("--max-qa-per-image", type=int, default=DEFAULT_MAX_QA_PER_IMAGE)
    parser.add_argument(
        "--qa-generation-mode",
        choices=SUPPORTED_QA_GENERATION_MODES,
        default="source_qa",
    )
    parser.add_argument("--teacher-backend", default=None)
    parser.add_argument("--source-split-allowlist", nargs="*", default=None)
    parser.add_argument("--source-split-blocklist", nargs="*", default=None)
    parser.add_argument("--no-hash-images", action="store_true")
    parser.add_argument("--allow-missing-target-spec", action="store_true")
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prepare-api-batch", action="store_true")
    parser.add_argument("--submit-api-batch", action="store_true")
    parser.add_argument("--poll-api-batch", action="store_true")
    parser.add_argument("--parse-api-batch", action="store_true")
    parser.add_argument("--finalize-teacher-outputs", action="store_true")
    parser.add_argument("--run-api-direct", action="store_true")
    parser.add_argument("--archive-teacher-outputs", action="store_true")
    parser.add_argument(
        "--run-api-until-target",
        action="store_true",
        help="Run rolling API batches until accepted QA prompt target is reached.",
    )
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--limit-requests", type=int, default=None)
    parser.add_argument("--batch-chunk-requests", type=int, default=200)
    parser.add_argument("--max-api-rounds", type=int, default=1)
    parser.add_argument("--max-request-file-mb", type=int, default=90)
    parser.add_argument("--submit-workers", type=int, default=1)
    parser.add_argument("--direct-workers", type=int, default=4)
    parser.add_argument("--poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--poll-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--api-target-accepted-prompts", type=int, default=None)
    parser.add_argument("--archive-name", default=None)
    parser.add_argument("--archive-reason", default="smoke_excluded_from_formal")
    parser.add_argument("--archive-last-submitted-count", type=int, default=None)
    parser.add_argument("--archive-custom-id", action="append", default=[])
    parser.add_argument(
        "--no-stratified-requests",
        action="store_true",
        help="Prepare API requests in teacher_requests.jsonl order instead of source-mix order.",
    )
    parser.add_argument(
        "--include-existing-teacher-outputs",
        action="store_true",
        help="Allow API batch preparation to include custom_ids already present in teacher_outputs.jsonl.",
    )
    parser.add_argument(
        "--skip-submitted-requests",
        action="store_true",
        help="Skip custom_ids recorded in api_submitted_requests.jsonl when preparing API batches.",
    )
    parser.add_argument("--completion-window", default="24h")
    parser.add_argument(
        "--experiment-ledger-path",
        default="docs/EXPERIMENT_LEDGER.md",
        help="Ledger updated on formal --execute.",
    )
    parser.add_argument("--skip-experiment-ledger", action="store_true")
    return parser


def _main_stage3_rl(argv: list[str]) -> int:
    args = build_stage3_rl_parser().parse_args(argv)
    if args.plan and args.write_plan:
        raise ValueError("--write-plan creates a new plan and cannot be combined with --plan")
    if args.plan:
        plan = args.plan
    else:
        options = Stage3RLBuildOptions(
            project_root=str(Path.cwd()),
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            target_accepted_prompts=args.target_accepted_prompts,
            seed=args.seed,
            source_config_path=args.source_config,
            exclude_manifests=tuple(args.exclude_manifest),
            source_split_allowlist=tuple(args.source_split_allowlist or DEFAULT_SPLIT_ALLOWLIST),
            source_split_blocklist=tuple(args.source_split_blocklist or DEFAULT_SPLIT_BLOCKLIST),
            max_new_images=args.max_new_images,
            max_new_prompts=args.max_new_prompts,
            max_qa_per_image=args.max_qa_per_image,
            extend_from=args.extend_from,
            qa_generation_mode=args.qa_generation_mode,
            teacher_backend=args.teacher_backend,
            hash_images=not args.no_hash_images,
            require_target_spec=not args.allow_missing_target_spec,
            run_id=args.run_id,
        )
        built_plan = build_stage3_rl_plan(options)
        if args.write_plan:
            print_json(write_stage3_rl_plan(built_plan["output_root"], plan=built_plan))
            return 0
        plan = built_plan
    if args.preflight_only:
        report = preflight_stage3_rl_plan(plan, write_report=True)
        print_json(report)
        return 0 if report["plan_valid"] else 2
    if args.dry_run:
        print_json(execute_stage3_rl_plan(plan, dry_run=True, limit_images=args.limit_images))
        return 0
    if args.execute:
        ledger_path = None if args.skip_experiment_ledger else args.experiment_ledger_path
        print_json(
            execute_stage3_rl_plan(
                plan,
                dry_run=False,
                limit_images=args.limit_images,
                experiment_ledger_path=ledger_path,
            )
        )
        return 0
    output_root = None
    if isinstance(plan, dict):
        output_root = plan["output_root"]
    elif args.plan:
        import json

        output_root = json.loads(Path(args.plan).read_text(encoding="utf-8"))["output_root"]
    if args.prepare_api_batch:
        print_json(
            prepare_teacher_batch_files(
                output_root=output_root,
                limit_requests=args.limit_requests,
                max_request_file_mb=args.max_request_file_mb,
                stratified_requests=not args.no_stratified_requests,
                skip_existing_outputs=not args.include_existing_teacher_outputs,
                skip_submitted_requests=args.skip_submitted_requests,
            )
        )
        return 0
    if args.run_api_until_target:
        print_json(_run_api_batches_until_target(plan, args=args, output_root=Path(output_root)))
        return 0
    if args.run_api_direct:
        print_json(
            run_teacher_requests_direct(
                output_root=output_root,
                limit_requests=args.limit_requests,
                max_workers=args.direct_workers,
                stratified_requests=not args.no_stratified_requests,
                skip_existing_outputs=not args.include_existing_teacher_outputs,
                skip_submitted_requests=args.skip_submitted_requests,
            )
        )
        return 0
    if args.archive_teacher_outputs:
        print_json(
            archive_teacher_outputs(
                output_root=output_root,
                custom_ids=args.archive_custom_id,
                last_submitted_count=args.archive_last_submitted_count,
                archive_name=args.archive_name,
                reason=args.archive_reason,
            )
        )
        return 0
    if args.submit_api_batch:
        print_json(
            submit_teacher_batches(
                output_root=output_root,
                completion_window=args.completion_window,
                max_workers=args.submit_workers,
            )
        )
        return 0
    if args.poll_api_batch:
        print_json(poll_teacher_batches(output_root=output_root))
        return 0
    if args.parse_api_batch:
        print_json(parse_teacher_batch_outputs(output_root=output_root))
        return 0
    if args.finalize_teacher_outputs:
        ledger_path = None if args.skip_experiment_ledger else args.experiment_ledger_path
        print_json(finalize_teacher_outputs_stage3_rl_plan(plan, experiment_ledger_path=ledger_path))
        return 0
    return exit_not_implemented(
        "use stage3-rl with --write-plan, --preflight-only, --dry-run, --execute, "
        "--prepare-api-batch, --submit-api-batch, --poll-api-batch, --parse-api-batch, "
        "--finalize-teacher-outputs, --run-api-direct, --archive-teacher-outputs, "
        "or --run-api-until-target"
    )


def _run_api_batches_until_target(
    plan_or_path: dict | str,
    *,
    args: argparse.Namespace,
    output_root: Path,
) -> dict:
    if args.batch_chunk_requests <= 0:
        raise ValueError("--batch-chunk-requests must be positive")
    if args.max_api_rounds <= 0:
        raise ValueError("--max-api-rounds must be positive")
    plan = _load_json_plan(plan_or_path)
    target = int(
        plan["target_accepted_prompts"]
        if args.api_target_accepted_prompts is None
        else args.api_target_accepted_prompts
    )
    source_weights = _plan_source_weights(plan)
    target_source_quotas = _source_quotas(target, source_weights)
    report = {
        "schema_version": "stage3_rl_api_rolling_report_v0",
        "created_at": _utc_now_text(),
        "output_root": str(output_root),
        "target_accepted_prompts": target,
        "target_source_quotas": target_source_quotas,
        "batch_chunk_requests": int(args.batch_chunk_requests),
        "max_api_rounds": int(args.max_api_rounds),
        "submit_workers": int(args.submit_workers),
        "poll_interval_seconds": float(args.poll_interval_seconds),
        "poll_timeout_seconds": float(args.poll_timeout_seconds),
        "rounds": [],
        "initial_accepted_prompts": _accepted_prompt_count(output_root),
        "initial_source_distribution": _accepted_source_distribution(output_root),
        "final_accepted_prompts": _accepted_prompt_count(output_root),
        "final_source_distribution": _accepted_source_distribution(output_root),
        "status": "not_started",
    }
    if _accepted_source_targets_met(output_root, target=target, source_weights=source_weights):
        report["status"] = "target_already_met"
        _write_rolling_report(output_root, report)
        return report

    for round_index in range(int(args.max_api_rounds)):
        before = _accepted_prompt_count(output_root)
        before_distribution = _accepted_source_distribution(output_root)
        if _accepted_source_targets_met(output_root, target=target, source_weights=source_weights):
            report["status"] = "target_met"
            break
        round_source_weights = _source_deficit_weights(
            target=target,
            source_weights=source_weights,
            source_distribution=before_distribution,
        )
        prepare_report = prepare_teacher_batch_files(
            output_root=output_root,
            limit_requests=int(args.batch_chunk_requests),
            max_request_file_mb=int(args.max_request_file_mb),
            stratified_requests=not args.no_stratified_requests,
            skip_existing_outputs=True,
            skip_submitted_requests=True,
            source_weights_override=round_source_weights,
        )
        if int(prepare_report.get("request_count") or 0) <= 0:
            report["status"] = "no_pending_requests"
            report["rounds"].append(
                {
                    "round_index": round_index,
                    "accepted_before": before,
                    "source_distribution_before": before_distribution,
                    "round_source_weights": round_source_weights,
                    "prepare_report": prepare_report,
                    "accepted_after": before,
                    "source_distribution_after": before_distribution,
                    "status": "no_pending_requests",
                }
            )
            break
        submit_state = submit_teacher_batches(
            output_root=output_root,
            completion_window=args.completion_window,
            max_workers=int(args.submit_workers),
        )
        poll_state = _wait_for_api_batches(
            output_root=output_root,
            poll_interval_seconds=float(args.poll_interval_seconds),
            poll_timeout_seconds=float(args.poll_timeout_seconds),
        )
        statuses = _batch_statuses(poll_state)
        parse_report = None
        finalize_report = None
        if any(batch.get("downloaded_output_path") for batch in poll_state.get("batches") or []):
            parse_report = parse_teacher_batch_outputs(output_root=output_root)
            if _jsonl_count(output_root / "teacher_outputs.jsonl") > 0:
                finalize_report = finalize_teacher_outputs_stage3_rl_plan(
                    plan_or_path,
                    experiment_ledger_path=None,
                )
        after = _accepted_prompt_count(output_root)
        after_distribution = _accepted_source_distribution(output_root)
        terminal = _all_batch_statuses_terminal(statuses)
        round_status = "completed" if terminal else "poll_timeout"
        if _source_targets_met(after_distribution, target=target, source_weights=source_weights):
            round_status = "target_met"
        report["rounds"].append(
            {
                "round_index": round_index,
                "accepted_before": before,
                "accepted_after": after,
                "source_distribution_before": before_distribution,
                "source_distribution_after": after_distribution,
                "round_source_weights": round_source_weights,
                "prepare_report": prepare_report,
                "submit_batch_ids": [
                    batch.get("batch_id") for batch in submit_state.get("batches") or []
                ],
                "batch_statuses": statuses,
                "parse_report": parse_report,
                "finalize_summary": None
                if finalize_report is None
                else finalize_report.get("summary"),
                "status": round_status,
            }
        )
        report["final_accepted_prompts"] = after
        report["final_source_distribution"] = after_distribution
        if _source_targets_met(after_distribution, target=target, source_weights=source_weights):
            report["status"] = "target_met"
            break
        if not terminal:
            report["status"] = "poll_timeout"
            break
    else:
        report["status"] = "round_limit_reached"

    report["final_accepted_prompts"] = _accepted_prompt_count(output_root)
    report["final_source_distribution"] = _accepted_source_distribution(output_root)
    _write_rolling_report(output_root, report)
    return report


def _wait_for_api_batches(
    *,
    output_root: Path,
    poll_interval_seconds: float,
    poll_timeout_seconds: float,
) -> dict:
    deadline = time.monotonic() + max(0.0, poll_timeout_seconds)
    state = poll_teacher_batches(output_root=output_root)
    while not _all_batch_statuses_terminal(_batch_statuses(state)):
        if time.monotonic() >= deadline:
            return state
        time.sleep(max(0.0, poll_interval_seconds))
        state = poll_teacher_batches(output_root=output_root)
    return state


def _batch_statuses(state: dict) -> list[str]:
    return [str(batch.get("status") or "unknown") for batch in state.get("batches") or []]


def _all_batch_statuses_terminal(statuses: list[str]) -> bool:
    terminal = {"completed", "failed", "expired", "cancelled", "canceled"}
    return bool(statuses) and all(status.lower() in terminal for status in statuses)


def _accepted_prompt_count(output_root: Path) -> int:
    summary_path = output_root / "manifest_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return int(summary.get("actual_accepted_prompts") or summary.get("accepted_qa") or 0)
    return _jsonl_count(output_root / "accepted_rl_prompts.jsonl")


def _accepted_source_distribution(output_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in _read_jsonl(output_root / "accepted_rl_prompts.jsonl"):
        source = str(row.get("source_dataset") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def _accepted_source_targets_met(
    output_root: Path,
    *,
    target: int,
    source_weights: dict[str, float],
) -> bool:
    return _source_targets_met(
        _accepted_source_distribution(output_root),
        target=target,
        source_weights=source_weights,
    )


def _source_targets_met(
    source_distribution: dict[str, int],
    *,
    target: int,
    source_weights: dict[str, float],
) -> bool:
    quotas = _source_quotas(target, source_weights)
    total = sum(source_distribution.values())
    return total >= target and all(
        int(source_distribution.get(source, 0)) >= quota for source, quota in quotas.items()
    )


def _source_deficit_weights(
    *,
    target: int,
    source_weights: dict[str, float],
    source_distribution: dict[str, int],
) -> dict[str, float]:
    quotas = _source_quotas(target, source_weights)
    deficits = {
        source: max(0, quota - int(source_distribution.get(source, 0)))
        for source, quota in quotas.items()
    }
    total_deficit = sum(deficits.values())
    if total_deficit <= 0:
        return source_weights
    return {
        source: deficit / total_deficit
        for source, deficit in deficits.items()
        if deficit > 0
    }


def _source_quotas(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        return {source: 0 for source in weights}
    raw = {source: total * float(weight) for source, weight in weights.items()}
    quotas = {source: int(value) for source, value in raw.items()}
    remainder = total - sum(quotas.values())
    for source, _value in sorted(
        raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True
    )[:remainder]:
        quotas[source] += 1
    return quotas


def _plan_source_weights(plan: dict) -> dict[str, float]:
    weights = (plan.get("balance_config") or {}).get("source_weights") or {}
    return {str(key): float(value) for key, value in weights.items()}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _load_json_plan(plan_or_path: dict | str) -> dict:
    if isinstance(plan_or_path, dict):
        return plan_or_path
    return json.loads(Path(plan_or_path).read_text(encoding="utf-8"))


def _write_rolling_report(output_root: Path, report: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "api_rolling_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _utc_now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_float_mapping(values: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        key, raw = _split_key_value(value)
        parsed[key] = float(raw)
    return parsed


def _parse_string_mapping(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, raw = _split_key_value(value)
        parsed[key] = raw
    return parsed


def _split_key_value(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"expected key=value, got {value!r}")
    key, raw = value.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"empty key in {value!r}")
    return key, raw.strip()


def _git_identity() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None, None
    return commit or None, bool(status.strip())


if __name__ == "__main__":
    raise SystemExit(main())
