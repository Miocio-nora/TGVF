import json

import pytest
import revisit_vlm_clean.training.executor as training_executor
from revisit_vlm_clean.cli.benchmark import main as benchmark_main
from revisit_vlm_clean.cli.generate_data import main as generate_data_main
from revisit_vlm_clean.cli.manifest import main as manifest_main
from revisit_vlm_clean.cli.train_stage1 import main as stage1_main
from revisit_vlm_clean.cli.train_stage2 import main as stage2_main
from revisit_vlm_clean.cli.valkit import main as valkit_main
from revisit_vlm_clean.training.stage1_executor import main as stage1_executor_main
from revisit_vlm_clean.training.stage2_executor import main as stage2_executor_main


def _write_minimal_stage1_checkpoint(
    path,
    *,
    protocol: str = "protocol_c_tool_observation",
) -> None:
    import torch

    torch.save(
        {
            "tgvf_module": {"dummy.weight": torch.zeros(2, 3)},
            "config": {
                "stage": "tgvf_v3_stage1",
                "tgvf_protocol": protocol,
                "model_id": "Qwen/Qwen3-VL-8B-Thinking",
                "processor_id": "processor-unit",
                "tgvf": {"variant": "tgvf_v2_bidirectional"},
            },
            "global_step": 2000,
            "protocol_c_token_rows": {
                "protocol": protocol,
                "tokens": ["<|focus_start|>", "<|focus_end|>"],
                "token_ids": {"<|focus_start|>": 1, "<|focus_end|>": 2},
                "input_embeddings": torch.zeros(2, 4),
                "output_embeddings": torch.zeros(2, 4),
            },
        },
        path,
    )


def test_manifest_list_cli(capsys) -> None:
    assert manifest_main(["--list"]) == 0
    captured = capsys.readouterr()
    assert "core_balanced_dev_2511_seed20260625" in captured.out


def test_benchmark_dry_run_cli(capsys) -> None:
    assert (
        benchmark_main(
            [
                "--run-id",
                "dry",
                "--checkpoint-path",
                "outputs/checkpoint.pt",
                "--mode",
                "tgvf_force",
                "--post-tgvf-forward-mode",
                "kv_cache",
                "--tgvf-protocol",
                "protocol_c_tool_observation_qwen2_no_think",
                "--scoring-backend",
                "project",
                "--subset-id",
                "core_smoke_256_seed20260625",
                "--dry-run",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert '"run_id": "dry"' in captured.out
    assert '"benchmark_root": "/home/dredvpn009/Flash_Storage/datasets/benchmarks"' in captured.out
    assert '"fallback_allowed": false' in captured.out
    assert '"eval_family": "project_native_external"' in captured.out
    assert '"scoring_backend": "project"' in captured.out
    assert '"tgvf_protocol": "protocol_c_tool_observation_qwen2_no_think"' in captured.out


def test_benchmark_dry_run_cli_accepts_explicit_internal_diagnostic_family(capsys) -> None:
    assert (
        benchmark_main(
            [
                "--run-id",
                "diagnostic",
                "--checkpoint-path",
                "outputs/checkpoint.pt",
                "--eval-family",
                "internal_diagnostic",
                "--mode",
                "tgvf_force",
                "--post-tgvf-forward-mode",
                "kv_cache",
                "--subset-id",
                "diagnostic_vstar_first_1_20260626",
                "--dry-run",
            ]
        )
        == 0
    )
    assert '"eval_family": "internal_diagnostic"' in capsys.readouterr().out


def test_benchmark_cli_rejects_valkit_eval_family() -> None:
    with pytest.raises(SystemExit):
        benchmark_main(
            [
                "--run-id",
                "valkit_wrong_runner",
                "--checkpoint-path",
                "outputs/checkpoint.pt",
                "--eval-family",
                "valkit",
                "--mode",
                "tgvf_force",
                "--post-tgvf-forward-mode",
                "kv_cache",
                "--subset-id",
                "core_smoke_256_seed20260625",
                "--dry-run",
            ]
        )


def test_valkit_preflight_write_plan_cli(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint\n")
    output_dir = tmp_path / "valkit_plan"

    assert (
        valkit_main(
            [
                "--run-id",
                "valkit_unit",
                "--checkpoint-path",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--benchmark",
                "vstar",
                "--benchmark",
                "blink",
                "--write-plan",
            ]
        )
        == 0
    )

    plan = json.loads((output_dir / "valkit_plan.json").read_text())
    assert plan["valkit_plan_schema_version"] == "clean_valkit_plan_v1"
    assert plan["eval_family"] == "valkit"
    assert plan["run"]["benchmarks"] == ["vstar", "blink"]
    assert plan["runner"]["executable"] is False
    assert plan["runner"]["legacy_shell_wrapper_allowed"] is False
    report = json.loads((output_dir / "valkit_preflight_report.json").read_text())
    assert report["plan_valid"] is True
    assert report["will_launch_valkit"] is False


def test_valkit_prepare_execution_cli(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint\n")
    output_dir = tmp_path / "valkit_plan"

    assert (
        valkit_main(
            [
                "--run-id",
                "valkit_prepare",
                "--checkpoint-path",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--benchmark",
                "vstar",
                "--benchmark",
                "blink",
                "--prepare-execution",
            ]
        )
        == 0
    )

    command_path = output_dir / "valkit_prepare_execution_command.sh"
    assert command_path.exists()
    command_text = command_path.read_text()
    assert "--prepare-execution" in command_text
    assert "--benchmark vstar" in command_text
    execution_dir = output_dir / "clean_valkit_execution"
    bundle = json.loads((execution_dir / "valkit_execution_bundle.json").read_text())
    assert (
        bundle["valkit_execution_bundle_schema_version"]
        == "clean_valkit_execution_bundle_v1"
    )
    assert bundle["eval_family"] == "valkit"
    assert bundle["run"]["benchmarks"] == ["vstar", "blink"]
    assert bundle["runner"]["will_launch_valkit"] is False
    assert bundle["runner"]["valkit_runtime_ported"] is False
    assert bundle["runner"]["legacy_shell_wrapper_allowed"] is False
    plan_artifacts = bundle["runner"]["plan_artifact_identities"]
    assert plan_artifacts["valkit_plan"]["exists"] is True
    assert plan_artifacts["valkit_plan_txt"]["exists"] is True
    assert plan_artifacts["valkit_preflight_report"]["exists"] is True
    assert plan_artifacts["valkit_prepare_execution_command"]["exists"] is True
    status = json.loads((execution_dir / "valkit_execution_status.json").read_text())
    assert status["runner_status"] == "handoff_supported_valkit_runner_not_ported"
    assert status["will_launch_valkit"] is False
    assert status["plan_artifact_identities"] == plan_artifacts


def test_valkit_execute_cli_uses_clean_run_py_not_legacy_wrapper(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint\n")
    valkit_root = tmp_path / "VLMEvalKit"
    valkit_root.mkdir()
    fake_run = valkit_root / "run.py"
    fake_run.write_text(
        "import json\n"
        "import pathlib\n"
        "import sys\n"
        "work_dir = pathlib.Path(sys.argv[sys.argv.index('--work-dir') + 1])\n"
        "work_dir.mkdir(parents=True, exist_ok=True)\n"
        "(work_dir / 'argv.json').write_text(json.dumps(sys.argv), encoding='utf-8')\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "valkit_execute"
    work_dir = tmp_path / "valkit_work"

    assert (
        valkit_main(
            [
                "--run-id",
                "valkit_execute",
                "--checkpoint-path",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--benchmark",
                "vstar",
                "--benchmark",
                "blink",
                "--valkit-root",
                str(valkit_root),
                "--valkit-model-name",
                "clean_tgvf_qwen3",
                "--valkit-run-mode",
                "infer",
                "--work-dir",
                str(work_dir),
                "--execute",
            ]
        )
        == 0
    )

    execution_dir = output_dir / "clean_valkit_execution"
    plan = json.loads((output_dir / "valkit_plan.json").read_text())
    bundle = json.loads((execution_dir / "valkit_execution_bundle.json").read_text())
    status = json.loads((execution_dir / "valkit_execution_status.json").read_text())
    result = json.loads((execution_dir / "valkit_execution_result.json").read_text())
    argv = json.loads((work_dir / "argv.json").read_text())
    launch_script = (execution_dir / "valkit_launch_command.sh").read_text()
    assert plan["runner"]["executable"] is True
    assert plan["runner"]["legacy_shell_wrapper_allowed"] is False
    assert bundle["runner"]["status"] == "clean_valkit_execution_completed"
    assert bundle["runner"]["will_launch_valkit"] is True
    assert bundle["runner"]["valkit_runtime_ported"] is True
    assert bundle["runner"]["returncode"] == 0
    assert status["runner_status"] == "clean_valkit_execution_completed"
    assert status["will_launch_valkit"] is True
    assert status["returncode"] == 0
    assert result["returncode"] == 0
    assert result["plan_sha256"] == bundle["plan_sha256"]
    assert result["execution_identity"]["checkpoint_identity"]["exists"] is True
    assert result["execution_identity"]["benchmarks"] == ["vstar", "blink"]
    assert result["execution_identity"]["run_py_identity"]["exists"] is True
    assert result["execution_identity"]["work_dir_identity"]["kind"] == "directory"
    plan_artifacts = result["execution_identity"]["plan_artifact_identities"]
    assert plan_artifacts["valkit_plan"]["exists"] is True
    assert plan_artifacts["valkit_plan_txt"]["exists"] is True
    assert plan_artifacts["valkit_preflight_report"]["exists"] is True
    assert plan_artifacts["valkit_prepare_execution_command"]["exists"] is True
    assert result["execution_identity"]["launch_command_identity"]["exists"] is True
    assert result["execution_identity"]["stdout_identity"]["exists"] is True
    assert result["execution_identity"]["stderr_identity"]["exists"] is True
    assert status["execution_identity"]["plan_sha256"] == bundle["plan_sha256"]
    assert status["plan_artifact_identities"] == plan_artifacts
    assert status["result_identity"]["exists"] is True
    assert status["stdout_identity"]["exists"] is True
    assert status["stderr_identity"]["exists"] is True
    assert status["launch_command_identity"]["exists"] is True
    assert "--data" in argv
    assert "vstar" in argv
    assert "blink" in argv
    assert "--model" in argv
    assert "clean_tgvf_qwen3" in argv
    assert "--mode" in argv
    assert "infer" in argv
    assert "scripts/run_vlmevalkit_tgvf.sh" not in launch_script


def test_valkit_tgvf_mode_requires_stage2_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint\n")
    with pytest.raises(ValueError, match="stage2-checkpoint"):
        valkit_main(
            [
                "--run-id",
                "valkit_tgvf",
                "--checkpoint-path",
                str(checkpoint),
                "--output-dir",
                str(tmp_path / "out"),
                "--benchmark",
                "vstar",
                "--mode",
                "tgvf_force",
                "--dry-run",
            ]
        )


def test_benchmark_write_empty_output_cli(tmp_path) -> None:
    output_dir = tmp_path / "run"
    assert (
        benchmark_main(
            [
                "--run-id",
                "empty",
                "--checkpoint-path",
                "outputs/checkpoint.pt",
                "--mode",
                "tgvf_force",
                "--post-tgvf-forward-mode",
                "kv_cache",
                "--subset-id",
                "core_smoke_256_seed20260625",
                "--manifest-path",
                "benchmark_manifests/core_smoke_256_seed20260625.json",
                "--manifest-hash",
                "abc123",
                "--output-dir",
                str(output_dir),
                "--write-empty-output",
            ]
        )
        == 0
    )
    assert (output_dir / "run_config.json").exists()
    assert (output_dir / "run_config.txt").exists()
    assert "benchmark_root:" in (output_dir / "run_config.txt").read_text()
    assert (output_dir / "rows.jsonl").read_text() == ""
    assert "schema smoke output" in (output_dir / "summary.json").read_text()


def test_training_default_clis(capsys) -> None:
    assert stage1_main(["--print-defaults"]) == 0
    stage1_defaults = capsys.readouterr().out
    assert "matrix_ce" in stage1_defaults
    assert '"focus_action_im_end": true' in stage1_defaults
    assert '"lr_scheduler": "cosine"' in stage1_defaults
    assert '"warmup_steps": 100' in stage1_defaults
    assert '"max_grad_norm": 1.0' in stage1_defaults
    assert stage2_main(["--print-defaults"]) == 0
    stage2_defaults = capsys.readouterr().out
    assert "through_answer" in stage2_defaults
    assert '"target_modules": [' in stage2_defaults
    assert '"q_proj"' in stage2_defaults
    assert '"warmup_steps": 100' in stage2_defaults
    assert '"adam_betas": [' in stage2_defaults


def test_stage1_training_write_plan_cli(tmp_path) -> None:
    train_file = tmp_path / "stage1.train.jsonl"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "target": "mark", '
        '"evidence_description": "The mark is blue."}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "stage1_plan"

    assert (
        stage1_main(
            [
                "--run-id",
                "stage1_unit",
                "--train-file",
                str(train_file),
                "--output-dir",
                str(output_dir),
                "--global-batch",
                "32",
                "--world-size",
                "4",
                "--micro-batch-size",
                "4",
                "--write-plan",
            ]
        )
        == 0
    )

    plan = json.loads((output_dir / "training_plan.json").read_text())
    assert plan["training_plan_schema_version"] == "clean_training_plan_v1"
    assert plan["stage"] == "stage1"
    assert plan["dataset"]["train_file"]["line_count"] == 1
    assert plan["batch"] == {
        "global_batch_size": 32,
        "gradient_accumulation_steps": 2,
        "micro_batch_size": 4,
        "world_size": 4,
    }
    assert plan["training"]["token_row_mode"] == "row_only"
    assert plan["training"]["focus_action_im_end"] is True
    assert plan["optimizer"]["lr_scheduler"] == "cosine"
    assert plan["optimizer"]["warmup_steps"] == 100
    assert plan["optimizer"]["min_lr_ratio"] == 0.1
    assert plan["optimizer"]["max_grad_norm"] == 1.0
    assert plan["readout_context"] == {
        "attention_mask": {
            "blocking": "weak_strict_original_image_key_blocking_after_tgvf_append",
            "mask_original_image_after_tgvf": True,
            "scope": "stage1_readout_after_tgvf_append",
        },
        "d_append_path": "native_qwen_visual_span",
        "d_token_count": "dynamic_source_image_visual_token_count",
        "fvt_position_mode": "native_source_grid",
        "original_image_placeholder_embeddings": "replace_with_qwen_v_merge",
        "position_ids": "real_qwen3_mrope_full_trajectory",
        "visual_merger_path": "frozen_finalize_path",
    }
    assert plan["module_policy"]["trainable"] == [
        "tgvf_module",
        "protocol_c_token_rows_row_only",
    ]
    assert "qwen_visual_merger" in plan["module_policy"]["frozen"]
    assert plan["module_policy"]["visual_merger"]["trainable"] is False
    assert plan["module_policy"]["training_runtime"]["use_cache"] is False
    assert plan["clean_native_training"]["executable"] is True
    assert plan["clean_native_training"]["required_for_final_clean_project"] is True
    assert plan["clean_native_training"]["prepare_execution_supported"] is True
    assert (
        plan["clean_native_training"]["status"]
        == "clean_native_distributed_launch_supported"
    )
    assert plan["clean_native_training"]["runtime"] == "distributed_torchrun"
    assert plan["clean_native_training"]["blocking_items"] == []
    assert plan["clean_prepare_execution_command"]["final_clean_native"] is True
    assert plan["clean_prepare_execution_command"]["executable"] is True
    assert plan["clean_prepare_execution_command"]["status"] == "prepare_execution_supported"
    assert plan["clean_prepare_execution_command"]["will_launch_training"] is False
    assert plan["clean_training_command"]["final_clean_native"] is True
    assert plan["clean_training_command"]["executable"] is True
    assert plan["clean_training_command"]["will_launch_training"] is True
    assert plan["clean_training_command"]["runtime"] == "distributed_torchrun"
    assert (
        plan["clean_training_command"]["planned_entrypoint"]
        == "revisit_vlm_clean.training.stage1_executor"
    )
    assert plan["legacy_reference_command"]["final_clean_native"] is False
    assert plan["legacy_reference_command"]["executable"] is False
    assert "audit reference only" in plan["legacy_reference_command"]["unavailable_reason"]
    native_status = json.loads((output_dir / "clean_native_training_status.json").read_text())
    assert native_status["status"] == "clean_native_distributed_launch_supported"
    prepare_command_path = output_dir / "clean_prepare_execution_command.sh"
    assert prepare_command_path.stat().st_mode & 0o111
    prepare_command = prepare_command_path.read_text()
    assert prepare_command.startswith("python -m revisit_vlm_clean.training.stage1_executor")
    assert "--prepare-execution" in prepare_command
    assert "not executable" not in prepare_command
    clean_command = (output_dir / "clean_training_command.sh").read_text()
    assert "not executable" not in clean_command
    assert clean_command.startswith("torchrun --nproc-per-node 4")
    assert "revisit_vlm_clean.training.stage1_executor" in clean_command
    assert "--launch-training" in clean_command
    command = (output_dir / "legacy_reference_command.sh").read_text()
    assert command.startswith("# not executable:")
    assert "torchrun --nproc-per-node 4" in command
    assert "--focus-action-im-end" in command
    assert "--no-focus-action-im-end" not in command
    assert "--lr-scheduler cosine" in command
    assert "--warmup-steps 100" in command
    assert "--min-lr-ratio 0.1" in command
    assert "--max-grad-norm 1.0" in command
    assert "--gradient-accumulation-steps 2" in command
    assert "--protocol-token-row-mode row_only" in command


def test_stage1_training_executor_preflight_cli(tmp_path, capsys) -> None:
    train_file = tmp_path / "stage1.train.jsonl"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "target": "mark", '
        '"evidence_description": "The mark is blue."}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "stage1_plan"
    assert (
        stage1_main(
            [
                "--run-id",
                "stage1_unit",
                "--train-file",
                str(train_file),
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )

    assert (
        stage1_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--preflight-only",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"plan_valid": true' in payload
    assert '"will_launch_training": false' in payload
    assert '"stage": "stage1"' in payload
    report = json.loads((output_dir / "stage1_training_preflight_report.json").read_text())
    assert report["plan_valid"] is True
    assert report["will_launch_training"] is False
    assert report["preflight_report"].endswith("stage1_training_preflight_report.json")
    assert stage1_executor_main(["--plan", str(output_dir / "training_plan.json")]) == 2


def test_stage1_distributed_launch_requires_torchrun_env(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    train_file = tmp_path / "stage1.train.jsonl"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "target": "mark", '
        '"evidence_description": "The mark is blue."}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "stage1_plan"
    assert (
        stage1_main(
            [
                "--run-id",
                "stage1_distributed_guard",
                "--train-file",
                str(train_file),
                "--output-dir",
                str(output_dir),
                "--global-batch",
                "4",
                "--world-size",
                "2",
                "--micro-batch-size",
                "1",
                "--write-plan",
            ]
        )
        == 0
    )
    with pytest.raises(ValueError, match="distributed clean launch requires torchrun"):
        stage1_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--launch-training",
            ]
        )


def test_stage1_training_executor_prepare_execution_cli(tmp_path, capsys) -> None:
    train_file = tmp_path / "stage1.train.jsonl"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "target": "mark"}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "stage1_plan"
    execution_dir = tmp_path / "stage1_execution"
    assert (
        stage1_main(
            [
                "--run-id",
                "stage1_prepare",
                "--train-file",
                str(train_file),
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )

    assert (
        stage1_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
                "--execution-dir",
                str(execution_dir),
                "--audit-runtime",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"will_launch_training": false' in payload
    assert '"runtime_audit"' in payload
    bundle = json.loads((execution_dir / "clean_training_execution_bundle.json").read_text())
    status = json.loads((execution_dir / "clean_training_execution_status.json").read_text())
    assert bundle["training_execution_bundle_schema_version"] == (
        "clean_training_execution_bundle_v1"
    )
    assert bundle["stage"] == "stage1"
    assert bundle["clean_executor"]["status"] == "ready_for_explicit_single_process_launch"
    assert bundle["clean_executor"]["owns_execution_bundle"] is True
    assert bundle["safety"]["legacy_reference_allowed"] is False
    assert bundle["safety"]["requires_explicit_launch_training_flag"] is True
    assert bundle["readout_context"]["position_ids"] == "real_qwen3_mrope_full_trajectory"
    contract = bundle["trainer_runtime_contract"]
    assert contract["contract_schema_version"] == "clean_trainer_runtime_contract_v1"
    assert contract["status"] == "single_process_launch_supported"
    assert contract["launch_permitted"] is True
    assert contract["runtime"] == "single_process"
    assert "emit_trainable_parameter_audit" in contract["required_launch_gates"]
    assert "stage1_readout_context_uses_qwen_v_merge" in contract["required_launch_gates"]
    assert "dataset_runtime_identity.json" in contract["required_runtime_artifacts"]
    checkpoint_contract = json.loads((execution_dir / "checkpoint_contract.json").read_text())
    assert checkpoint_contract["status"] == "no_input_checkpoint_required"
    assert checkpoint_contract["input_checkpoint_required"] is False
    dataset_runtime = json.loads((execution_dir / "dataset_runtime_identity.json").read_text())
    first_batch = json.loads((execution_dir / "first_batch_identity.json").read_text())
    optimizer_groups = json.loads((execution_dir / "optimizer_groups.json").read_text())
    assert dataset_runtime["stage"] == "stage1"
    assert dataset_runtime["train_file"]["line_count"] == 1
    assert dataset_runtime["train_file"]["missing_required_counts"]["target"] == 0
    assert first_batch["materialized_batch_size"] == 1
    assert first_batch["rows"][0]["row_sha256"]
    assert optimizer_groups["status"] == "validated"
    assert optimizer_groups["group_names"] == ["tgvf_module", "protocol_c_token_rows"]
    assert optimizer_groups["groups"][0]["lr"] == 1e-4
    assert optimizer_groups["groups"][0]["weight_decay"] == 0.01
    assert bundle["plan_identity"]["sha256"]
    plan_artifacts = bundle["plan_artifact_identities"]
    assert plan_artifacts["training_plan"]["exists"] is True
    assert plan_artifacts["training_plan_txt"]["exists"] is True
    assert plan_artifacts["dataset_identity"]["exists"] is True
    assert plan_artifacts["clean_native_training_status"]["exists"] is True
    assert plan_artifacts["clean_prepare_execution_command"]["exists"] is True
    assert plan_artifacts["clean_training_command"]["exists"] is True
    assert plan_artifacts["legacy_reference_command"]["exists"] is True
    assert status["runner_status"] == "ready_for_explicit_single_process_launch"
    assert status["plan_artifact_identities"] == plan_artifacts
    assert status["trainer_runtime_contract_status"] == "single_process_launch_supported"
    assert status["will_launch_training"] is False
    assert (execution_dir / "clean_training_execution_bundle.txt").exists()
    runtime_audit = json.loads((execution_dir / "clean_training_runtime_audit.json").read_text())
    audit_status = json.loads(
        (execution_dir / "clean_training_runtime_audit_status.json").read_text()
    )
    trainable_parameters = json.loads((execution_dir / "trainable_parameters.json").read_text())
    assert runtime_audit["status"] == "blocked_before_training_loop"
    assert runtime_audit["will_launch_training"] is False
    assert runtime_audit["launch_gates"]["checkpoint_contract_status"] == (
        "no_input_checkpoint_required"
    )
    assert runtime_audit["launch_gates"]["optimizer_groups_status"] == "validated"
    assert runtime_audit["launch_gates"]["optimizer_runtime_status"] == "not_requested"
    assert runtime_audit["launch_gates"]["trainable_parameters_status"] == (
        "pending_model_load_not_actual_parameter_audit"
    )
    gate_status = {
        gate["name"]: gate["status"] for gate in runtime_audit["launch_gates"]["gates"]
    }
    assert gate_status["construct_optimizer_and_scheduler_from_plan"] == (
        "pending_real_trainer_loop"
    )
    assert gate_status["run_backward_optimizer_scheduler_step_from_plan"] == (
        "pending_real_trainer_loop"
    )
    assert audit_status["status"] == "blocked_before_training_loop"
    assert trainable_parameters["actual_model_parameters_loaded"] is False
    assert trainable_parameters["must_be_replaced_before_first_optimizer_step"] is True
    assert trainable_parameters["expected_trainable_policy"] == [
        "tgvf_module",
        "protocol_c_token_rows_row_only",
    ]


def test_training_executor_rejects_executable_legacy_reference(tmp_path) -> None:
    train_file = tmp_path / "stage1.train.jsonl"
    train_file.write_text('{"image": "/tmp/image.jpg", "question": "q"}\n', encoding="utf-8")
    output_dir = tmp_path / "stage1_plan"
    assert (
        stage1_main(
            [
                "--run-id",
                "stage1_bad_legacy",
                "--train-file",
                str(train_file),
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )
    plan_path = output_dir / "training_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["legacy_reference_command"]["executable"] = True
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy_reference_command must be non-executable"):
        stage1_executor_main(["--plan", str(plan_path), "--preflight-only"])


def test_stage1_training_executor_runtime_audit_can_write_cadence_probe(
    tmp_path,
    capsys,
) -> None:
    train_file = tmp_path / "stage1.train.jsonl"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "target": "mark"}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "stage1_plan"
    assert (
        stage1_main(
            [
                "--run-id",
                "stage1_cadence_audit",
                "--train-file",
                str(train_file),
                "--output-dir",
                str(output_dir),
                "--max-steps",
                "5",
                "--save-every",
                "2",
                "--write-plan",
            ]
        )
        == 0
    )

    assert (
        stage1_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
                "--audit-runtime",
                "--audit-cadence",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"training_cadence_runtime"' in payload
    execution_dir = output_dir / "clean_training_execution"
    cadence = json.loads((execution_dir / "training_cadence_runtime.json").read_text())
    runtime_audit = json.loads((execution_dir / "clean_training_runtime_audit.json").read_text())
    assert cadence["status"] == "actual_training_cadence_audit"
    assert cadence["actual_training_cadence_validated"] is True
    assert cadence["training_run_launched"] is False
    assert cadence["max_steps"] == 5
    assert cadence["save_every"] == 2
    assert cadence["checkpoint_save_steps"] == [2, 4, 5]
    assert cadence["final_checkpoint_saved"] is True
    assert cadence["eval_enabled"] is False
    assert cadence["eval_disabled_reason"] == "stage_has_no_eval_cadence"
    gate_status = {
        gate["name"]: gate["status"] for gate in runtime_audit["launch_gates"]["gates"]
    }
    assert runtime_audit["launch_gates"]["training_cadence_runtime_status"] == (
        "actual_training_cadence_audit"
    )
    assert gate_status["validate_training_cadence_from_plan"] == "identity_validated"


def test_stage2_training_executor_runtime_audit_can_write_cadence_probe(
    tmp_path,
    capsys,
) -> None:
    train_file = tmp_path / "stage2.train.jsonl"
    val_file = tmp_path / "stage2.val.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    row = (
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n'
    )
    train_file.write_text(row, encoding="utf-8")
    val_file.write_text(row, encoding="utf-8")
    _write_minimal_stage1_checkpoint(checkpoint)
    output_dir = tmp_path / "stage2_plan"
    assert (
        stage2_main(
            [
                "--run-id",
                "stage2_cadence_audit",
                "--train-file",
                str(train_file),
                "--val-file",
                str(val_file),
                "--stage1-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--max-steps",
                "5",
                "--save-every",
                "2",
                "--eval-every",
                "3",
                "--write-plan",
            ]
        )
        == 0
    )

    assert (
        stage2_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
                "--audit-runtime",
                "--audit-cadence",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"training_cadence_runtime"' in payload
    execution_dir = output_dir / "clean_training_execution"
    cadence = json.loads((execution_dir / "training_cadence_runtime.json").read_text())
    runtime_audit = json.loads((execution_dir / "clean_training_runtime_audit.json").read_text())
    assert cadence["status"] == "actual_training_cadence_audit"
    assert cadence["max_steps"] == 5
    assert cadence["checkpoint_save_steps"] == [2, 4, 5]
    assert cadence["eval_enabled"] is True
    assert cadence["eval_steps"] == [3, 5]
    assert cadence["final_eval_scheduled"] is True
    assert cadence["val_file_available"] is True
    gate_status = {
        gate["name"]: gate["status"] for gate in runtime_audit["launch_gates"]["gates"]
    }
    assert gate_status["validate_training_cadence_from_plan"] == "identity_validated"


def test_training_executor_rejects_bad_prepare_command(tmp_path) -> None:
    train_file = tmp_path / "stage1.train.jsonl"
    train_file.write_text('{"image": "/tmp/image.jpg", "question": "q"}\n', encoding="utf-8")
    output_dir = tmp_path / "stage1_plan"
    assert (
        stage1_main(
            [
                "--run-id",
                "stage1_bad_prepare",
                "--train-file",
                str(train_file),
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )
    plan_path = output_dir / "training_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["clean_prepare_execution_command"]["argv"].remove("--prepare-execution")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="clean_prepare_execution_command"):
        stage1_executor_main(["--plan", str(plan_path), "--preflight-only"])


def test_stage1_executor_rejects_bad_readout_context(tmp_path) -> None:
    train_file = tmp_path / "stage1.train.jsonl"
    train_file.write_text('{"image": "/tmp/image.jpg", "question": "q"}\n', encoding="utf-8")
    output_dir = tmp_path / "stage1_plan"
    assert (
        stage1_main(
            [
                "--run-id",
                "stage1_bad_readout",
                "--train-file",
                str(train_file),
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )
    plan_path = output_dir / "training_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["readout_context"]["position_ids"] = "legacy_flat_positions"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="readout_context.position_ids"):
        stage1_executor_main(["--plan", str(plan_path), "--preflight-only"])


def test_stage2_training_write_plan_cli(tmp_path) -> None:
    train_file = tmp_path / "stage2.train.jsonl"
    val_file = tmp_path / "stage2.test.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n',
        encoding="utf-8",
    )
    val_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n',
        encoding="utf-8",
    )
    _write_minimal_stage1_checkpoint(checkpoint)
    output_dir = tmp_path / "stage2_plan"

    assert (
        stage2_main(
            [
                "--run-id",
                "stage2_unit",
                "--train-file",
                str(train_file),
                "--val-file",
                str(val_file),
                "--stage1-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--global-batch",
                "128",
                "--world-size",
                "4",
                "--micro-batch-size",
                "4",
                "--write-plan",
            ]
        )
        == 0
    )

    plan = json.loads((output_dir / "training_plan.json").read_text())
    assert plan["training_plan_schema_version"] == "clean_training_plan_v1"
    assert plan["stage"] == "stage2"
    assert plan["dataset"]["train_file"]["line_count"] == 1
    assert plan["dataset"]["val_file"]["line_count"] == 1
    assert plan["dataset"]["stage1_checkpoint"]["exists"] is True
    assert plan["batch"]["gradient_accumulation_steps"] == 8
    assert plan["mask_policy"]["mask_original_image_after_tgvf_scope"] == "through_answer"
    assert plan["loss"]["weighted_span_loss"]["focus_target"] == 1.5
    assert plan["lora"] == {
        "alpha": 256,
        "bias": "none",
        "dropout": 0.05,
        "rank": 64,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }
    assert plan["optimizer"]["lr_scheduler"] == "cosine"
    assert plan["optimizer"]["warmup_steps"] == 100
    assert plan["optimizer"]["min_lr_ratio"] == 0.1
    assert plan["optimizer"]["betas"] == [0.9, 0.95]
    assert plan["optimizer"]["eps"] == 1e-8
    assert plan["optimizer"]["weight_decay"] == 0.01
    assert plan["optimizer"]["max_grad_norm"] == 1.0
    assert "qwen_lora_adapters" in plan["module_policy"]["trainable"]
    assert "qwen_visual_merger" in plan["module_policy"]["frozen"]
    assert plan["module_policy"]["visual_merger"]["trainable"] is False
    assert plan["module_policy"]["training_runtime"]["use_cache"] is False
    assert plan["module_policy"]["training_runtime"]["gradient_checkpointing"] is True
    assert (
        plan["module_policy"]["training_runtime"]["print_trainable_parameter_names_before_launch"]
        is True
    )
    assert plan["clean_native_training"]["executable"] is True
    assert plan["clean_native_training"]["legacy_reference_is_final"] is False
    assert plan["clean_native_training"]["prepare_execution_supported"] is True
    assert (
        plan["clean_native_training"]["status"]
        == "clean_native_distributed_launch_supported"
    )
    assert plan["clean_native_training"]["runtime"] == "distributed_torchrun"
    assert plan["clean_native_training"]["blocking_items"] == []
    assert plan["clean_prepare_execution_command"]["executable"] is True
    assert plan["clean_prepare_execution_command"]["status"] == "prepare_execution_supported"
    assert plan["clean_prepare_execution_command"]["will_launch_training"] is False
    assert plan["clean_training_command"]["final_clean_native"] is True
    assert plan["clean_training_command"]["executable"] is True
    assert plan["clean_training_command"]["will_launch_training"] is True
    assert plan["clean_training_command"]["runtime"] == "distributed_torchrun"
    assert (
        plan["clean_training_command"]["planned_entrypoint"]
        == "revisit_vlm_clean.training.stage2_executor"
    )
    assert plan["legacy_reference_command"]["final_clean_native"] is False
    assert plan["legacy_reference_command"]["executable"] is False
    assert "audit reference only" in plan["legacy_reference_command"]["unavailable_reason"]
    assert (output_dir / "clean_native_training_status.json").exists()
    prepare_command_path = output_dir / "clean_prepare_execution_command.sh"
    assert prepare_command_path.stat().st_mode & 0o111
    prepare_command = prepare_command_path.read_text()
    assert prepare_command.startswith("python -m revisit_vlm_clean.training.stage2_executor")
    assert "--prepare-execution" in prepare_command
    assert "not executable" not in prepare_command
    clean_command = (output_dir / "clean_training_command.sh").read_text()
    assert "not executable" not in clean_command
    assert clean_command.startswith("torchrun --nproc-per-node 4")
    assert "revisit_vlm_clean.training.stage2_executor" in clean_command
    assert "--launch-training" in clean_command
    command = (output_dir / "legacy_reference_command.sh").read_text()
    assert command.startswith("# not executable:")
    assert "--mask-original-image-after-tgvf-scope through_answer" in command
    assert "--lora-rank 64" in command
    assert "--lora-alpha 256" in command
    assert "--lora-dropout 0.05" in command
    assert "--lora-bias none" in command
    assert (
        "--lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        in command
    )
    assert "--warmup-steps 100" in command
    assert "--adam-beta1 0.9" in command
    assert "--adam-beta2 0.95" in command
    assert "--adam-eps 1e-08" in command
    assert "--weight-decay 0.01" in command
    assert "--max-grad-norm 1.0" in command
    assert "--loss-focus-target 1.5" in command


def test_stage2_training_executor_preflight_cli(tmp_path, capsys) -> None:
    train_file = tmp_path / "stage2.train.jsonl"
    val_file = tmp_path / "stage2.test.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n',
        encoding="utf-8",
    )
    val_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n',
        encoding="utf-8",
    )
    _write_minimal_stage1_checkpoint(checkpoint)
    output_dir = tmp_path / "stage2_plan"
    assert (
        stage2_main(
            [
                "--run-id",
                "stage2_unit",
                "--train-file",
                str(train_file),
                "--val-file",
                str(val_file),
                "--stage1-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )

    assert (
        stage2_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--preflight-only",
                "--preflight-report",
                str(output_dir / "reports" / "stage2_preflight.json"),
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"plan_valid": true' in payload
    assert '"will_launch_training": false' in payload
    assert '"stage": "stage2"' in payload
    report = json.loads((output_dir / "reports" / "stage2_preflight.json").read_text())
    assert report["plan_valid"] is True
    assert report["stage"] == "stage2"


def test_stage2_training_executor_prepare_execution_cli(tmp_path, capsys) -> None:
    train_file = tmp_path / "stage2.train.jsonl"
    val_file = tmp_path / "stage2.test.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n',
        encoding="utf-8",
    )
    val_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n',
        encoding="utf-8",
    )
    _write_minimal_stage1_checkpoint(checkpoint)
    output_dir = tmp_path / "stage2_plan"
    assert (
        stage2_main(
            [
                "--run-id",
                "stage2_prepare",
                "--train-file",
                str(train_file),
                "--val-file",
                str(val_file),
                "--stage1-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )

    assert (
        stage2_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
                "--audit-runtime",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"training_runtime_ported": true' in payload
    assert '"runtime_audit"' in payload
    execution_dir = output_dir / "clean_training_execution"
    bundle = json.loads((execution_dir / "clean_training_execution_bundle.json").read_text())
    assert bundle["stage"] == "stage2"
    assert bundle["mask_policy"]["mask_original_image_after_tgvf_scope"] == "through_answer"
    assert bundle["deepstack"]["enabled"] is False
    assert bundle["lora"]["rank"] == 64
    assert bundle["lora"]["target_modules"][:2] == ["q_proj", "k_proj"]
    contract = bundle["trainer_runtime_contract"]
    assert contract["stage"] == "stage2"
    assert contract["launch_function"] == "launch_training"
    assert "attach_lora_modules_from_plan" in contract["required_launch_gates"]
    assert "apply_weighted_span_losses_from_plan" in contract["required_launch_gates"]
    assert "trainable_parameters.json" in contract["required_runtime_artifacts"]
    dataset_runtime = json.loads((execution_dir / "dataset_runtime_identity.json").read_text())
    first_batch = json.loads((execution_dir / "first_batch_identity.json").read_text())
    optimizer_groups = json.loads((execution_dir / "optimizer_groups.json").read_text())
    assert dataset_runtime["stage"] == "stage2"
    assert dataset_runtime["train_file"]["need_focus"] == 1
    assert dataset_runtime["val_file"]["line_count"] == 1
    assert first_batch["requested_global_batch_size"] == 128
    assert first_batch["materialized_batch_size"] == 1
    assert first_batch["rows"][0]["need_focus"] is True
    checkpoint_contract = json.loads((execution_dir / "checkpoint_contract.json").read_text())
    assert checkpoint_contract["status"] == "validated"
    assert checkpoint_contract["global_step"] == 2000
    assert checkpoint_contract["tgvf_module"]["num_tensors"] == 1
    assert checkpoint_contract["protocol_c_token_rows"]["protocol"] == (
        "protocol_c_tool_observation"
    )
    assert optimizer_groups["status"] == "validated"
    assert optimizer_groups["group_names"] == ["llm_lora", "tgvf_refiner", "fvt_calibration"]
    assert [group["lr"] for group in optimizer_groups["groups"]] == [2e-5, 5e-6, 1e-5]
    assert optimizer_groups["optimizer"]["weight_decay"] == 0.01
    assert bundle["safety"]["legacy_reference_allowed"] is False
    plan_artifacts = bundle["plan_artifact_identities"]
    assert plan_artifacts["training_plan"]["exists"] is True
    assert plan_artifacts["training_plan_txt"]["exists"] is True
    assert plan_artifacts["dataset_identity"]["exists"] is True
    assert plan_artifacts["clean_native_training_status"]["exists"] is True
    assert plan_artifacts["clean_prepare_execution_command"]["exists"] is True
    assert plan_artifacts["clean_training_command"]["exists"] is True
    assert plan_artifacts["legacy_reference_command"]["exists"] is True
    status = json.loads((execution_dir / "clean_training_execution_status.json").read_text())
    assert status["bundle_valid"] is True
    assert status["plan_artifact_identities"] == plan_artifacts
    assert status["trainer_runtime_contract_status"] == "single_process_launch_supported"
    assert status["checkpoint_contract_status"] == "validated"
    assert status["optimizer_groups_status"] == "validated"
    assert status["will_launch_training"] is False
    runtime_audit = json.loads((execution_dir / "clean_training_runtime_audit.json").read_text())
    trainable_parameters = json.loads((execution_dir / "trainable_parameters.json").read_text())
    assert runtime_audit["stage"] == "stage2"
    assert runtime_audit["status"] == "blocked_before_training_loop"
    assert runtime_audit["launch_gates"]["checkpoint_contract_status"] == "validated"
    assert runtime_audit["launch_gates"]["optimizer_groups_status"] == "validated"
    assert runtime_audit["launch_gates"]["optimizer_runtime_status"] == "not_requested"
    assert runtime_audit["launch_gates"]["pending_real_trainer_loop"] > 0
    gate_status = {
        gate["name"]: gate["status"] for gate in runtime_audit["launch_gates"]["gates"]
    }
    assert gate_status["construct_optimizer_and_scheduler_from_plan"] == (
        "pending_real_trainer_loop"
    )
    assert gate_status["run_backward_optimizer_scheduler_step_from_plan"] == (
        "pending_real_trainer_loop"
    )
    assert trainable_parameters["expected_trainable_policy"] == [
        "qwen_lora_adapters",
        "tgvf_module_continued_from_stage1",
        "protocol_c_token_rows_restored_from_stage1",
    ]
    assert trainable_parameters["status"] == "pending_model_load_not_actual_parameter_audit"


def test_stage2_training_executor_runtime_audit_can_write_actual_parameter_audit(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import torch

    train_file = tmp_path / "stage2.train.jsonl"
    val_file = tmp_path / "stage2.val.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n',
        encoding="utf-8",
    )
    val_file.write_text(
        '{"image": "/tmp/image-val.jpg", "question": "vq", "answer": "va", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n',
        encoding="utf-8",
    )
    _write_minimal_stage1_checkpoint(checkpoint)
    output_dir = tmp_path / "stage2_plan"
    qwen = torch.nn.Linear(2, 2)
    qwen.bias.requires_grad_(False)
    tgvf = torch.nn.Sequential(torch.nn.Linear(2, 1))

    def fake_loader(bundle, *, expected_stage):
        assert expected_stage.value == "stage2"
        assert bundle["stage"] == "stage2"
        return {
            "modules": {"qwen_lora": qwen, "tgvf": tgvf},
            "loader": {"backend": "fake_parameter_audit_loader"},
        }

    monkeypatch.setattr(training_executor, "_load_training_parameter_audit_modules", fake_loader)
    assert (
        stage2_main(
            [
                "--run-id",
                "stage2_actual_parameter_audit",
                "--train-file",
                str(train_file),
                "--stage1-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )

    assert (
        stage2_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
                "--audit-runtime",
                "--audit-model-parameters",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"trainable_parameters"' in payload
    execution_dir = output_dir / "clean_training_execution"
    trainable_parameters = json.loads((execution_dir / "trainable_parameters.json").read_text())
    runtime_audit = json.loads((execution_dir / "clean_training_runtime_audit.json").read_text())
    assert trainable_parameters["status"] == "actual_model_parameter_audit"
    assert trainable_parameters["actual_model_parameters_loaded"] is True
    assert trainable_parameters["must_be_replaced_before_first_optimizer_step"] is False
    assert trainable_parameters["loader"]["backend"] == "fake_parameter_audit_loader"
    assert "qwen_lora.weight" in trainable_parameters["trainable_parameter_names"]
    assert "qwen_lora.bias" in trainable_parameters["frozen_parameter_names_sample"]
    assert trainable_parameters["module_summaries"]["qwen_lora"]["trainable_tensor_count"] == 1
    assert trainable_parameters["module_summaries"]["tgvf"]["trainable_tensor_count"] == 2
    gate_status = {
        gate["name"]: gate["status"] for gate in runtime_audit["launch_gates"]["gates"]
    }
    assert gate_status["load_model_and_processor"] == "identity_validated"
    assert gate_status["construct_optimizer_and_scheduler_from_plan"] == (
        "pending_real_trainer_loop"
    )
    assert gate_status["emit_trainable_parameter_audit"] == "identity_validated"
    assert gate_status["save_checkpoint_with_clean_contract"] == "pending_real_trainer_loop"
    assert runtime_audit["launch_gates"]["optimizer_runtime_status"] == "not_requested"


def test_stage2_training_executor_runtime_audit_can_write_actual_optimizer_audit(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import torch

    train_file = tmp_path / "stage2.train.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n',
        encoding="utf-8",
    )
    _write_minimal_stage1_checkpoint(checkpoint)
    output_dir = tmp_path / "stage2_plan"
    qwen = torch.nn.Linear(2, 2)
    qwen.bias.requires_grad_(False)
    tgvf = torch.nn.Sequential(torch.nn.Linear(2, 1))

    def fake_loader(bundle, *, expected_stage):
        assert expected_stage.value == "stage2"
        assert bundle["stage"] == "stage2"
        return {
            "modules": {"qwen_lora": qwen, "tgvf": tgvf},
            "loader": {"backend": "fake_parameter_optimizer_audit_loader"},
        }

    monkeypatch.setattr(training_executor, "_load_training_parameter_audit_modules", fake_loader)
    assert (
        stage2_main(
            [
                "--run-id",
                "stage2_actual_optimizer_audit",
                "--train-file",
                str(train_file),
                "--stage1-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )

    assert (
        stage2_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
                "--audit-runtime",
                "--audit-optimizer",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"optimizer_runtime"' in payload
    execution_dir = output_dir / "clean_training_execution"
    trainable_parameters = json.loads((execution_dir / "trainable_parameters.json").read_text())
    optimizer_runtime = json.loads((execution_dir / "optimizer_runtime.json").read_text())
    runtime_audit = json.loads((execution_dir / "clean_training_runtime_audit.json").read_text())
    assert trainable_parameters["status"] == "actual_model_parameter_audit"
    assert trainable_parameters["actual_model_parameters_loaded"] is True
    assert optimizer_runtime["status"] == "actual_optimizer_scheduler_audit"
    assert optimizer_runtime["actual_optimizer_constructed"] is True
    assert optimizer_runtime["actual_scheduler_constructed"] is True
    assert optimizer_runtime["planned_group_names"] == [
        "llm_lora",
        "tgvf_refiner",
        "fvt_calibration",
    ]
    assert optimizer_runtime["constructed_group_names"] == ["llm_lora", "tgvf_refiner"]
    assert optimizer_runtime["empty_planned_group_names"] == ["fvt_calibration"]
    assert [group["lr"] for group in optimizer_runtime["constructed_groups"]] == [2e-5, 5e-6]
    assert optimizer_runtime["optimizer"]["param_group_count"] == 2
    assert optimizer_runtime["scheduler"]["name"] == "cosine"
    gate_status = {
        gate["name"]: gate["status"] for gate in runtime_audit["launch_gates"]["gates"]
    }
    assert runtime_audit["launch_gates"]["optimizer_runtime_status"] == (
        "actual_optimizer_scheduler_audit"
    )
    assert gate_status["construct_optimizer_and_scheduler_from_plan"] == "identity_validated"
    assert gate_status["run_backward_optimizer_scheduler_step_from_plan"] == (
        "pending_real_trainer_loop"
    )
    assert gate_status["save_checkpoint_with_clean_contract"] == "pending_real_trainer_loop"


def test_stage2_training_executor_runtime_audit_can_write_optimizer_step_probe(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import torch

    train_file = tmp_path / "stage2.train.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n',
        encoding="utf-8",
    )
    _write_minimal_stage1_checkpoint(checkpoint)
    output_dir = tmp_path / "stage2_plan"
    qwen = torch.nn.Linear(2, 2)
    qwen.bias.requires_grad_(False)
    tgvf = torch.nn.Sequential(torch.nn.Linear(2, 1))

    def fake_loader(bundle, *, expected_stage):
        assert expected_stage.value == "stage2"
        assert bundle["stage"] == "stage2"
        return {
            "modules": {"qwen_lora": qwen, "tgvf": tgvf},
            "loader": {"backend": "fake_optimizer_step_audit_loader"},
        }

    def fake_step_probe(*, bundle, artifacts, loaded_modules):
        assert bundle["stage"] == "stage2"
        assert artifacts["optimizer_groups"]["status"] == "validated"
        parameters = [
            parameter
            for module in (qwen, tgvf)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        loss_tensor = sum(parameter.square().sum() for parameter in parameters)
        return {
            "forward_completed": True,
            "sample_count": 1,
            "loss_total": float(loss_tensor.detach()),
            "loss_tensor": loss_tensor,
            "loss_focus": 1.0,
            "loss_no_focus": 0.0,
            "loss_visual_token_manifold": 0.0,
            "mask_original_image_after_tgvf": True,
            "debug": {
                "fast_batched_stage2": True,
                "focus_count": 1,
                "no_focus_count": 0,
                "focus_loss_token_weight": 3.5,
                "no_focus_loss_token_weight": 0.0,
                "mask_original_image_after_tgvf_prob": 1.0,
                "mask_original_image_after_tgvf_scope": "through_answer",
                "focus_sample_mask_active_rate": 1.0,
                "no_focus_mask_active_rate": 0.0,
            },
        }

    monkeypatch.setattr(training_executor, "_load_training_parameter_audit_modules", fake_loader)
    monkeypatch.setattr(training_executor, "_run_stage2_training_step_probe", fake_step_probe)
    assert (
        stage2_main(
            [
                "--run-id",
                "stage2_optimizer_step_audit",
                "--train-file",
                str(train_file),
                "--stage1-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )

    assert (
        stage2_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
                "--audit-runtime",
                "--audit-optimizer-step",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"optimizer_step_runtime"' in payload
    execution_dir = output_dir / "clean_training_execution"
    optimizer_runtime = json.loads((execution_dir / "optimizer_runtime.json").read_text())
    training_step_runtime = json.loads((execution_dir / "training_step_runtime.json").read_text())
    optimizer_step_runtime = json.loads(
        (execution_dir / "optimizer_step_runtime.json").read_text()
    )
    runtime_audit = json.loads((execution_dir / "clean_training_runtime_audit.json").read_text())
    assert optimizer_runtime["status"] == "actual_optimizer_scheduler_audit"
    assert optimizer_runtime["optimizer"]["max_grad_norm"] == 1.0
    assert training_step_runtime["actual_training_step_forward"] is True
    assert training_step_runtime["backward_called"] is False
    assert optimizer_step_runtime["status"] == "actual_backward_optimizer_scheduler_step_audit"
    assert optimizer_step_runtime["backward_called"] is True
    assert optimizer_step_runtime["optimizer_step_called"] is True
    assert optimizer_step_runtime["scheduler_step_called"] is True
    assert optimizer_step_runtime["checkpoint_published"] is False
    assert optimizer_step_runtime["training_loop_launched"] is False
    assert optimizer_step_runtime["max_grad_norm"] == 1.0
    assert optimizer_step_runtime["grad_before_clip"]["tensors_with_grad"] > 0
    assert optimizer_step_runtime["grad_after_zero"]["tensors_with_grad"] == 0
    gate_status = {
        gate["name"]: gate["status"] for gate in runtime_audit["launch_gates"]["gates"]
    }
    assert runtime_audit["launch_gates"]["optimizer_step_runtime_status"] == (
        "actual_backward_optimizer_scheduler_step_audit"
    )
    assert gate_status["construct_optimizer_and_scheduler_from_plan"] == "identity_validated"
    assert gate_status["run_backward_optimizer_scheduler_step_from_plan"] == (
        "identity_validated"
    )
    assert gate_status["save_checkpoint_with_clean_contract"] == "pending_real_trainer_loop"


def test_stage2_training_executor_runtime_audit_can_write_trainer_loop_probe(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import torch

    train_file = tmp_path / "stage2.train.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n',
        encoding="utf-8",
    )
    _write_minimal_stage1_checkpoint(checkpoint)
    output_dir = tmp_path / "stage2_plan"
    qwen = torch.nn.Linear(2, 2)
    qwen.bias.requires_grad_(False)
    tgvf = torch.nn.Sequential(torch.nn.Linear(2, 1))
    step_calls = 0

    def fake_loader(bundle, *, expected_stage):
        assert expected_stage.value == "stage2"
        assert bundle["stage"] == "stage2"
        return {
            "modules": {"qwen_lora": qwen, "tgvf": tgvf},
            "loader": {"backend": "fake_trainer_loop_audit_loader"},
        }

    def fake_step_probe(*, bundle, artifacts, loaded_modules):
        nonlocal step_calls
        step_calls += 1
        assert bundle["stage"] == "stage2"
        assert artifacts["dataset_runtime_identity"]["stage"] == "stage2"
        parameters = [
            parameter
            for module in (qwen, tgvf)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        loss_tensor = torch.stack([parameter.square().sum() for parameter in parameters]).sum()
        return {
            "forward_completed": True,
            "sample_count": 1,
            "loss_total": float(loss_tensor.detach()),
            "loss_tensor": loss_tensor,
            "loss_focus": 1.0,
            "loss_no_focus": 0.0,
            "loss_visual_token_manifold": 0.0,
            "mask_original_image_after_tgvf": True,
            "debug": {
                "fast_batched_stage2": True,
                "focus_count": 1,
                "no_focus_count": 0,
                "focus_loss_token_weight": 3.5,
                "no_focus_loss_token_weight": 0.0,
                "mask_original_image_after_tgvf_prob": 1.0,
                "mask_original_image_after_tgvf_scope": "through_answer",
                "focus_sample_mask_active_rate": 1.0,
                "no_focus_mask_active_rate": 0.0,
            },
        }

    monkeypatch.setattr(training_executor, "_load_training_parameter_audit_modules", fake_loader)
    monkeypatch.setattr(training_executor, "_run_stage2_training_step_probe", fake_step_probe)
    assert (
        stage2_main(
            [
                "--run-id",
                "stage2_trainer_loop_audit",
                "--train-file",
                str(train_file),
                "--stage1-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--global-batch",
                "2",
                "--micro-batch-size",
                "1",
                "--write-plan",
            ]
        )
        == 0
    )

    assert (
        stage2_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
                "--audit-runtime",
                "--audit-trainer-loop",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"trainer_loop_runtime"' in payload
    execution_dir = output_dir / "clean_training_execution"
    trainer_loop_runtime = json.loads((execution_dir / "trainer_loop_runtime.json").read_text())
    runtime_audit = json.loads((execution_dir / "clean_training_runtime_audit.json").read_text())
    assert trainer_loop_runtime["status"] == "actual_gradient_accumulation_trainer_loop_audit"
    assert trainer_loop_runtime["actual_trainer_loop_probe"] is True
    assert trainer_loop_runtime["gradient_accumulation_steps"] == 2
    assert trainer_loop_runtime["micro_steps_run"] == 2
    assert trainer_loop_runtime["backward_micro_steps"] == 2
    assert trainer_loop_runtime["optimizer_steps_completed"] == 1
    assert trainer_loop_runtime["scheduler_steps_completed"] == 1
    assert trainer_loop_runtime["checkpoint_published"] is False
    assert trainer_loop_runtime["training_run_launched"] is False
    assert trainer_loop_runtime["grad_after_accumulation"]["tensors_with_grad"] > 0
    assert trainer_loop_runtime["grad_after_zero"]["tensors_with_grad"] == 0
    assert step_calls == 3
    gate_status = {
        gate["name"]: gate["status"] for gate in runtime_audit["launch_gates"]["gates"]
    }
    assert runtime_audit["launch_gates"]["trainer_loop_runtime_status"] == (
        "actual_gradient_accumulation_trainer_loop_audit"
    )
    assert gate_status["run_backward_optimizer_scheduler_step_from_plan"] == (
        "identity_validated"
    )
    assert gate_status["run_gradient_accumulation_loop_from_plan"] == "identity_validated"
    assert gate_status["save_checkpoint_with_clean_contract"] == "pending_real_trainer_loop"


def test_stage2_training_executor_runtime_audit_can_publish_post_loop_checkpoint(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import torch

    train_file = tmp_path / "stage2.train.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n',
        encoding="utf-8",
    )
    _write_minimal_stage1_checkpoint(checkpoint)
    output_dir = tmp_path / "stage2_plan"
    qwen = torch.nn.Linear(2, 2)
    qwen.bias.requires_grad_(False)
    tgvf = torch.nn.Sequential(torch.nn.Linear(2, 1))
    step_calls = 0

    def fake_loader(bundle, *, expected_stage):
        assert expected_stage.value == "stage2"
        assert bundle["stage"] == "stage2"
        return {
            "modules": {"qwen_lora": qwen, "tgvf": tgvf},
            "loader": {"backend": "fake_checkpoint_publish_audit_loader"},
        }

    def fake_step_probe(*, bundle, artifacts, loaded_modules):
        nonlocal step_calls
        step_calls += 1
        assert bundle["stage"] == "stage2"
        parameters = [
            parameter
            for module in (qwen, tgvf)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        loss_tensor = torch.stack([parameter.square().sum() for parameter in parameters]).sum()
        return {
            "forward_completed": True,
            "sample_count": 1,
            "loss_total": float(loss_tensor.detach()),
            "loss_tensor": loss_tensor,
            "loss_focus": 1.0,
            "loss_no_focus": 0.0,
            "loss_visual_token_manifold": 0.0,
            "mask_original_image_after_tgvf": True,
            "debug": {
                "fast_batched_stage2": True,
                "focus_count": 1,
                "no_focus_count": 0,
                "focus_loss_token_weight": 3.5,
                "no_focus_loss_token_weight": 0.0,
                "mask_original_image_after_tgvf_prob": 1.0,
                "mask_original_image_after_tgvf_scope": "through_answer",
                "focus_sample_mask_active_rate": 1.0,
                "no_focus_mask_active_rate": 0.0,
            },
        }

    monkeypatch.setattr(training_executor, "_load_training_parameter_audit_modules", fake_loader)
    monkeypatch.setattr(training_executor, "_run_stage2_training_step_probe", fake_step_probe)
    assert (
        stage2_main(
            [
                "--run-id",
                "stage2_checkpoint_publish_audit",
                "--train-file",
                str(train_file),
                "--stage1-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--global-batch",
                "2",
                "--micro-batch-size",
                "1",
                "--write-plan",
            ]
        )
        == 0
    )

    assert (
        stage2_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
                "--audit-runtime",
                "--audit-checkpoint-publish",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"training_checkpoint_publish_runtime"' in payload
    execution_dir = output_dir / "clean_training_execution"
    trainer_loop_runtime = json.loads((execution_dir / "trainer_loop_runtime.json").read_text())
    publish_runtime = json.loads(
        (execution_dir / "training_checkpoint_publish_runtime.json").read_text()
    )
    runtime_audit = json.loads((execution_dir / "clean_training_runtime_audit.json").read_text())
    assert not (execution_dir / "checkpoint_runtime.json").exists()
    assert trainer_loop_runtime["actual_trainer_loop_probe"] is True
    assert publish_runtime["status"] == "actual_training_checkpoint_publish_audit"
    assert publish_runtime["actual_checkpoint_publish_probe_saved"] is True
    assert publish_runtime["actual_checkpoint_publish_probe_loaded"] is True
    assert publish_runtime["global_step"] == 1
    assert publish_runtime["micro_step"] == 2
    assert publish_runtime["missing_required_keys"] == []
    assert publish_runtime["state_checks_ok"] is True
    assert publish_runtime["step_checks_ok"] is True
    assert publish_runtime["step_checks"]["global_step"]["ok"] is True
    assert publish_runtime["step_checks"]["micro_step"]["ok"] is True
    assert publish_runtime["state_checks"]["tgvf_module"]["ok"] is True
    assert publish_runtime["state_checks"]["qwen_lora"]["ok"] is True
    assert (execution_dir / "training_checkpoint_publish_probe_step_1.pt").exists()
    assert step_calls == 3
    gate_status = {
        gate["name"]: gate["status"] for gate in runtime_audit["launch_gates"]["gates"]
    }
    assert runtime_audit["launch_gates"]["training_checkpoint_publish_runtime_status"] == (
        "actual_training_checkpoint_publish_audit"
    )
    assert gate_status["run_gradient_accumulation_loop_from_plan"] == "identity_validated"
    assert gate_status["save_checkpoint_with_clean_contract"] == "identity_validated"
    assert gate_status["publish_training_checkpoint_after_trainer_loop"] == (
        "identity_validated"
    )


def test_stage2_training_executor_runtime_audit_can_resume_published_checkpoint(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import torch

    train_file = tmp_path / "stage2.train.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n',
        encoding="utf-8",
    )
    _write_minimal_stage1_checkpoint(checkpoint)
    output_dir = tmp_path / "stage2_plan"
    loader_modules = []
    step_calls = 0

    def make_modules():
        qwen = torch.nn.Linear(2, 2)
        qwen.bias.requires_grad_(False)
        tgvf = torch.nn.Sequential(torch.nn.Linear(2, 1))
        loader_modules.append((qwen, tgvf))
        return qwen, tgvf

    def fake_loader(bundle, *, expected_stage):
        assert expected_stage.value == "stage2"
        assert bundle["stage"] == "stage2"
        qwen, tgvf = make_modules()
        return {
            "modules": {"qwen_lora": qwen, "tgvf": tgvf},
            "loader": {
                "backend": "fake_checkpoint_resume_audit_loader",
                "loader_call": len(loader_modules),
            },
        }

    def fake_step_probe(*, bundle, artifacts, loaded_modules):
        nonlocal step_calls
        step_calls += 1
        assert bundle["stage"] == "stage2"
        modules = loaded_modules["modules"]
        parameters = [
            parameter
            for module in (modules["qwen_lora"], modules["tgvf"])
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        loss_tensor = torch.stack([parameter.square().sum() for parameter in parameters]).sum()
        return {
            "forward_completed": True,
            "sample_count": 1,
            "loss_total": float(loss_tensor.detach()),
            "loss_tensor": loss_tensor,
            "loss_focus": 1.0,
            "loss_no_focus": 0.0,
            "loss_visual_token_manifold": 0.0,
            "mask_original_image_after_tgvf": True,
            "debug": {
                "fast_batched_stage2": True,
                "focus_count": 1,
                "no_focus_count": 0,
                "focus_loss_token_weight": 3.5,
                "no_focus_loss_token_weight": 0.0,
                "mask_original_image_after_tgvf_prob": 1.0,
                "mask_original_image_after_tgvf_scope": "through_answer",
                "focus_sample_mask_active_rate": 1.0,
                "no_focus_mask_active_rate": 0.0,
            },
        }

    monkeypatch.setattr(training_executor, "_load_training_parameter_audit_modules", fake_loader)
    monkeypatch.setattr(training_executor, "_run_stage2_training_step_probe", fake_step_probe)
    assert (
        stage2_main(
            [
                "--run-id",
                "stage2_checkpoint_resume_audit",
                "--train-file",
                str(train_file),
                "--stage1-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--global-batch",
                "2",
                "--micro-batch-size",
                "1",
                "--write-plan",
            ]
        )
        == 0
    )

    assert (
        stage2_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
                "--audit-runtime",
                "--audit-launch-readiness",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"training_checkpoint_resume_runtime"' in payload
    assert '"training_launch_readiness"' in payload
    execution_dir = output_dir / "clean_training_execution"
    resume_runtime = json.loads(
        (execution_dir / "training_checkpoint_resume_runtime.json").read_text()
    )
    cadence_runtime = json.loads((execution_dir / "training_cadence_runtime.json").read_text())
    launch_readiness = json.loads(
        (execution_dir / "training_launch_readiness.json").read_text()
    )
    runtime_audit = json.loads((execution_dir / "clean_training_runtime_audit.json").read_text())
    runtime_status = json.loads(
        (execution_dir / "clean_training_runtime_audit_status.json").read_text()
    )
    assert len(loader_modules) == 2
    assert step_calls == 3
    assert resume_runtime["status"] == "actual_training_checkpoint_resume_audit"
    assert resume_runtime["actual_checkpoint_resume_probe_loaded"] is True
    assert resume_runtime["model_state_loaded"] is True
    assert resume_runtime["optimizer_state_loaded"] is True
    assert resume_runtime["scheduler_state_loaded"] is True
    assert resume_runtime["state_checks_ok"] is True
    assert resume_runtime["step_checks_ok"] is True
    assert resume_runtime["protocol_rows_ok"] is True
    assert resume_runtime["global_step"] == 1
    assert resume_runtime["micro_step"] == 2
    assert resume_runtime["fresh_loader"]["loader_call"] == 2
    assert resume_runtime["state_checks"]["tgvf_module"]["ok"] is True
    assert resume_runtime["state_checks"]["qwen_lora"]["ok"] is True
    assert resume_runtime["optimizer"]["state_entry_count"] > 0
    assert cadence_runtime["status"] == "actual_training_cadence_audit"
    assert cadence_runtime["actual_training_cadence_validated"] is True
    gate_status = {
        gate["name"]: gate["status"] for gate in runtime_audit["launch_gates"]["gates"]
    }
    assert runtime_audit["launch_gates"]["training_checkpoint_resume_runtime_status"] == (
        "actual_training_checkpoint_resume_audit"
    )
    assert runtime_audit["launch_gates"]["training_cadence_runtime_status"] == (
        "actual_training_cadence_audit"
    )
    assert gate_status["publish_training_checkpoint_after_trainer_loop"] == (
        "identity_validated"
    )
    assert gate_status["resume_training_from_clean_checkpoint"] == "identity_validated"
    assert gate_status["validate_training_cadence_from_plan"] == "identity_validated"
    assert gate_status["apply_deepstack_training_scope_when_enabled"] == (
        "identity_validated"
    )
    assert launch_readiness["status"] == "launch_contract_ready_explicit_launch_required"
    assert launch_readiness["all_required_gates_identity_validated"] is True
    assert launch_readiness["contract_ready_for_trainer_loop"] is True
    assert launch_readiness["launch_permitted"] is False
    assert (
        launch_readiness["launch_disabled_reason"]
        == "readiness_audit_does_not_launch_training"
    )
    assert launch_readiness["pending_gates"] == []
    assert launch_readiness["unknown_gates"] == []
    assert launch_readiness["unexpected_blockers"] == []
    assert launch_readiness["remaining_blockers"] == []
    assert launch_readiness["deepstack"]["enabled"] is False
    assert launch_readiness["deepstack"]["training_scope_gate_status"] == (
        "identity_validated"
    )
    assert runtime_audit["launch_readiness"]["status"] == launch_readiness["status"]
    assert runtime_status["training_launch_readiness_status"] == launch_readiness["status"]
    assert runtime_status["launch_contract_ready_for_trainer_loop"] is True
    assert runtime_status["launch_permitted"] is False


def test_stage2_training_executor_can_launch_single_process_training_loop(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import torch

    train_file = tmp_path / "stage2.train.jsonl"
    val_file = tmp_path / "stage2.val.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image-0.jpg", "question": "q0", "answer": "a0", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence", '
        '"target": "t0", "evidence_description": "d0"}\n'
        '{"image": "/tmp/image-1.jpg", "question": "q1", "answer": "a1", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence", '
        '"target": "t1", "evidence_description": "d1"}\n'
        '{"image": "/tmp/image-2.jpg", "question": "q2", "answer": "a2", '
        '"need_focus": false, "evidence_state": "sufficient_visual_evidence"}\n',
        encoding="utf-8",
    )
    val_file.write_text(
        '{"image": "/tmp/image-val-0.jpg", "question": "vq0", "answer": "va0", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence", '
        '"target": "vt0", "evidence_description": "vd0"}\n'
        '{"image": "/tmp/image-val-1.jpg", "question": "vq1", "answer": "va1", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence", '
        '"target": "vt1", "evidence_description": "vd1"}\n',
        encoding="utf-8",
    )
    _write_minimal_stage1_checkpoint(checkpoint)
    output_dir = tmp_path / "stage2_plan"
    qwen = torch.nn.Linear(2, 2)
    qwen.bias.requires_grad_(False)
    tgvf = torch.nn.Sequential(torch.nn.Linear(2, 1))
    loader_calls = 0
    train_step_calls = 0
    validation_step_calls = 0

    def fake_loader(bundle, *, expected_stage):
        nonlocal loader_calls
        loader_calls += 1
        assert expected_stage.value == "stage2"
        assert bundle["stage"] == "stage2"
        return {
            "modules": {"qwen_lora": qwen, "tgvf": tgvf},
            "loader": {"backend": "fake_single_process_launch_loader"},
        }

    def fake_step_probe(*, bundle, artifacts, loaded_modules, samples=None):
        nonlocal train_step_calls, validation_step_calls
        assert bundle["stage"] == "stage2"
        assert samples
        sample_questions = [sample.question for sample in samples]
        if sample_questions[0].startswith("vq"):
            validation_step_calls += 1
        else:
            train_step_calls += 1
            assert sample_questions[0].startswith("q")
        modules = loaded_modules["modules"]
        parameters = [
            parameter
            for module in (modules["qwen_lora"], modules["tgvf"])
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        loss_tensor = torch.stack([parameter.square().sum() for parameter in parameters]).sum()
        return {
            "forward_completed": True,
            "sample_count": 1,
            "loss_total": float(loss_tensor.detach()),
            "loss_tensor": loss_tensor,
            "loss_focus": 1.0,
            "loss_no_focus": 0.0,
            "loss_visual_token_manifold": 0.0,
            "mask_original_image_after_tgvf": True,
            "debug": {
                "fast_batched_stage2": True,
                "focus_count": 1,
                "no_focus_count": 0,
                "focus_loss_token_weight": 3.5,
                "no_focus_loss_token_weight": 0.0,
                "mask_original_image_after_tgvf_prob": 1.0,
                "mask_original_image_after_tgvf_scope": "through_answer",
                "focus_sample_mask_active_rate": 1.0,
                "no_focus_mask_active_rate": 0.0,
            },
        }

    monkeypatch.setattr(training_executor, "_load_training_parameter_audit_modules", fake_loader)
    monkeypatch.setattr(training_executor, "_run_stage2_training_step_probe", fake_step_probe)
    assert (
        stage2_main(
            [
                "--run-id",
                "stage2_single_process_launch",
                "--train-file",
                str(train_file),
                "--val-file",
                str(val_file),
                "--stage1-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--global-batch",
                "2",
                "--micro-batch-size",
                "1",
                "--max-steps",
                "2",
                "--save-every",
                "1",
                "--eval-every",
                "1",
                "--write-plan",
            ]
        )
        == 0
    )
    plan = json.loads((output_dir / "training_plan.json").read_text())
    assert plan["clean_native_training"]["executable"] is True
    assert (
        plan["clean_native_training"]["status"]
        == "clean_native_single_process_launch_supported"
    )
    assert plan["clean_training_command"]["executable"] is True
    assert plan["clean_training_command"]["will_launch_training"] is True
    assert plan["dataset"]["val_file"]["line_count"] == 2
    clean_command_path = output_dir / "clean_training_command.sh"
    assert clean_command_path.stat().st_mode & 0o111
    clean_command = clean_command_path.read_text()
    assert clean_command.startswith("python -m revisit_vlm_clean.training.stage2_executor")
    assert "--launch-training" in clean_command

    assert (
        stage2_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--launch-training",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"training_launch_result"' in payload
    assert '"will_launch_training": true' in payload
    execution_dir = output_dir / "clean_training_execution"
    bundle = json.loads((execution_dir / "clean_training_execution_bundle.json").read_text())
    launch_result = json.loads(
        (execution_dir / "clean_training_launch_result.json").read_text()
    )
    launch_status = json.loads(
        (execution_dir / "clean_training_launch_status.json").read_text()
    )
    runtime = json.loads((execution_dir / "single_process_training_runtime.json").read_text())
    assert loader_calls == 1
    assert train_step_calls == 4
    assert validation_step_calls == 2
    assert launch_result["status"] == "clean_single_process_training_completed"
    assert launch_result["training_runtime_ported"] is True
    assert launch_result["optimizer_steps_completed"] == 2
    assert launch_result["in_training_validation_enabled"] is True
    assert launch_result["validation_steps"] == [1, 2]
    assert launch_result["validation_record_count"] == 2
    assert launch_result["plan_identity"]["sha256"] == bundle["plan_identity"]["sha256"]
    assert launch_result["plan_artifact_identities"] == bundle["plan_artifact_identities"]
    assert launch_result["bundle_identity"]["exists"] is True
    assert launch_result["dataset_identity"]["train_file"]["line_count"] == 3
    assert launch_result["dataset_identity"]["val_file"]["line_count"] == 2
    assert launch_result["dataset_identity"]["first_batch"]["materialized_batch_size"] == 2
    assert launch_result["input_checkpoint_contract"]["status"] == "validated"
    assert launch_result["input_checkpoint_contract"]["checkpoint_identity"]["exists"] is True
    assert launch_result["runtime_artifact_identities"]["training_runtime"]["exists"] is True
    assert launch_result["runtime_artifact_identities"]["checkpoint_contract"]["exists"] is True
    assert runtime["optimizer_steps_completed"] == 2
    assert runtime["micro_steps_completed"] == 4
    assert runtime["checkpoint_save_steps"] == [1, 2]
    assert runtime["in_training_validation_enabled"] is True
    assert runtime["validation_steps"] == [1, 2]
    assert len(runtime["validation_records"]) == 2
    assert runtime["validation_records"][0]["dataset_path"] == str(val_file)
    train_indices = [
        micro_step["sample_trace"][0]["sample_index"]
        for step in runtime["step_records"]
        for micro_step in step["micro_steps"]
    ]
    assert train_indices == [0, 1, 2, 0]
    validation_indices = [
        record["sample_trace"][0]["sample_index"]
        for record in runtime["validation_records"]
    ]
    assert validation_indices == [0, 1]
    assert runtime["train_cursor"]["mode"] == "target_focus_ratio_cycle"
    assert runtime["train_cursor"]["target_focus_ratio"] == 0.8
    assert runtime["validation_cursor"]["mode"] == "sequential_cycle"
    assert runtime["validation_records"][0]["backward_called"] is False
    assert runtime["validation_records"][0]["optimizer_step_called"] is False
    assert len(runtime["checkpoint_records"]) == 2
    assert (execution_dir / "checkpoint_step_1.pt").exists()
    assert (execution_dir / "checkpoint_step_2.pt").exists()
    assert launch_status["final_checkpoint"] == str(execution_dir / "checkpoint_step_2.pt")
    assert launch_status["plan_identity"]["sha256"] == launch_result["plan_identity"]["sha256"]
    assert launch_status["plan_artifact_identities"] == (
        launch_result["plan_artifact_identities"]
    )
    assert launch_status["dataset_identity"]["train_file"]["line_count"] == 3
    assert launch_status["input_checkpoint_contract"]["status"] == "validated"
    assert launch_status["final_checkpoint_identity"]["sha256"] == (
        launch_result["final_checkpoint_identity"]["sha256"]
    )
    assert launch_status["runtime_artifact_identity_count"] >= 8
    assert launch_status["in_training_validation_enabled"] is True
    assert launch_status["validation_record_count"] == 2
    assert launch_status["unsupported_runtime_features"] == []


def test_stage2_training_executor_runtime_audit_can_write_checkpoint_audit(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import torch

    train_file = tmp_path / "stage2.train.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n',
        encoding="utf-8",
    )
    _write_minimal_stage1_checkpoint(checkpoint)
    output_dir = tmp_path / "stage2_plan"
    qwen = torch.nn.Linear(2, 2)
    qwen.bias.requires_grad_(False)
    tgvf = torch.nn.Sequential(torch.nn.Linear(2, 1))

    def fake_loader(bundle, *, expected_stage):
        assert expected_stage.value == "stage2"
        assert bundle["stage"] == "stage2"
        return {
            "modules": {"qwen_lora": qwen, "tgvf": tgvf},
            "loader": {"backend": "fake_checkpoint_audit_loader"},
        }

    monkeypatch.setattr(training_executor, "_load_training_parameter_audit_modules", fake_loader)
    assert (
        stage2_main(
            [
                "--run-id",
                "stage2_checkpoint_audit",
                "--train-file",
                str(train_file),
                "--stage1-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )

    assert (
        stage2_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
                "--audit-runtime",
                "--audit-checkpoint",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"checkpoint_runtime"' in payload
    execution_dir = output_dir / "clean_training_execution"
    optimizer_runtime = json.loads((execution_dir / "optimizer_runtime.json").read_text())
    checkpoint_runtime = json.loads((execution_dir / "checkpoint_runtime.json").read_text())
    runtime_audit = json.loads((execution_dir / "clean_training_runtime_audit.json").read_text())
    assert optimizer_runtime["status"] == "actual_optimizer_scheduler_audit"
    assert checkpoint_runtime["status"] == "actual_checkpoint_save_load_audit"
    assert checkpoint_runtime["actual_checkpoint_saved"] is True
    assert checkpoint_runtime["actual_checkpoint_loaded"] is True
    assert checkpoint_runtime["missing_required_keys"] == []
    assert checkpoint_runtime["state_checks_ok"] is True
    assert checkpoint_runtime["state_checks"]["tgvf_module"]["ok"] is True
    assert checkpoint_runtime["state_checks"]["qwen_lora"]["ok"] is True
    assert (execution_dir / "checkpoint_runtime_probe.pt").exists()
    gate_status = {
        gate["name"]: gate["status"] for gate in runtime_audit["launch_gates"]["gates"]
    }
    assert runtime_audit["launch_gates"]["checkpoint_runtime_status"] == (
        "actual_checkpoint_save_load_audit"
    )
    assert gate_status["construct_optimizer_and_scheduler_from_plan"] == "identity_validated"
    assert gate_status["save_checkpoint_with_clean_contract"] == "identity_validated"


def test_stage1_training_executor_runtime_audit_checkpoint_includes_protocol_rows(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import torch

    train_file = tmp_path / "stage1.train.jsonl"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "target": "mark"}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "stage1_plan"
    qwen = torch.nn.Linear(2, 2)
    tgvf = torch.nn.Sequential(torch.nn.Linear(2, 1))
    protocol_rows = {
        "protocol": "protocol_c_tool_observation",
        "tokens": ["<|focus_start|>", "<|focus_end|>"],
        "token_ids": {"<|focus_start|>": 1, "<|focus_end|>": 2},
        "input_embeddings": torch.zeros(2, 2),
        "output_embeddings": torch.ones(2, 2),
    }

    def fake_loader(bundle, *, expected_stage):
        assert expected_stage.value == "stage1"
        assert bundle["stage"] == "stage1"
        return {
            "modules": {"qwen": qwen, "tgvf": tgvf},
            "checkpoint_extras": {"protocol_c_token_rows": protocol_rows},
            "loader": {"backend": "fake_stage1_checkpoint_audit_loader"},
        }

    monkeypatch.setattr(training_executor, "_load_training_parameter_audit_modules", fake_loader)
    assert (
        stage1_main(
            [
                "--run-id",
                "stage1_checkpoint_audit",
                "--train-file",
                str(train_file),
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )

    assert (
        stage1_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
                "--audit-runtime",
                "--audit-checkpoint",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"checkpoint_runtime"' in payload
    execution_dir = output_dir / "clean_training_execution"
    checkpoint_runtime = json.loads((execution_dir / "checkpoint_runtime.json").read_text())
    runtime_audit = json.loads((execution_dir / "clean_training_runtime_audit.json").read_text())
    assert checkpoint_runtime["status"] == "actual_checkpoint_save_load_audit"
    assert checkpoint_runtime["missing_required_keys"] == []
    assert checkpoint_runtime["state_checks_ok"] is True
    assert checkpoint_runtime["protocol_c_token_rows"] is not None
    assert checkpoint_runtime["protocol_c_token_rows"]["protocol"] == (
        "protocol_c_tool_observation"
    )
    gate_status = {
        gate["name"]: gate["status"] for gate in runtime_audit["launch_gates"]["gates"]
    }
    assert gate_status["save_checkpoint_with_clean_contract"] == "identity_validated"


def test_stage2_training_executor_runtime_audit_can_write_training_step_probe(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import torch

    train_file = tmp_path / "stage2.train.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n',
        encoding="utf-8",
    )
    _write_minimal_stage1_checkpoint(checkpoint)
    output_dir = tmp_path / "stage2_plan"
    qwen = torch.nn.Linear(2, 2)
    qwen.bias.requires_grad_(False)
    tgvf = torch.nn.Sequential(torch.nn.Linear(2, 1))

    def fake_loader(bundle, *, expected_stage):
        assert expected_stage.value == "stage2"
        assert bundle["stage"] == "stage2"
        return {
            "modules": {"qwen_lora": qwen, "tgvf": tgvf},
            "loader": {"backend": "fake_training_step_audit_loader"},
        }

    def fake_step_probe(*, bundle, artifacts, loaded_modules):
        assert bundle["stage"] == "stage2"
        assert artifacts["dataset_runtime_identity"]["stage"] == "stage2"
        assert loaded_modules["modules"]["qwen_lora"] is qwen
        return {
            "forward_completed": True,
            "sample_count": 1,
            "loss_total": 1.25,
            "loss_focus": 1.0,
            "loss_no_focus": 0.0,
            "loss_visual_token_manifold": 0.0,
            "mask_original_image_after_tgvf": True,
            "debug": {
                "fast_batched_stage2": True,
                "focus_count": 1,
                "no_focus_count": 0,
                "focus_loss_token_weight": 3.5,
                "no_focus_loss_token_weight": 0.0,
                "mask_original_image_after_tgvf_prob": 1.0,
                "mask_original_image_after_tgvf_scope": "through_answer",
                "focus_sample_mask_active_rate": 1.0,
                "no_focus_mask_active_rate": 0.0,
            },
        }

    monkeypatch.setattr(training_executor, "_load_training_parameter_audit_modules", fake_loader)
    monkeypatch.setattr(training_executor, "_run_stage2_training_step_probe", fake_step_probe)
    assert (
        stage2_main(
            [
                "--run-id",
                "stage2_training_step_audit",
                "--train-file",
                str(train_file),
                "--stage1-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )

    assert (
        stage2_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
                "--audit-runtime",
                "--audit-training-step",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"training_step_runtime"' in payload
    execution_dir = output_dir / "clean_training_execution"
    step_runtime = json.loads((execution_dir / "training_step_runtime.json").read_text())
    runtime_audit = json.loads((execution_dir / "clean_training_runtime_audit.json").read_text())
    assert step_runtime["status"] == "actual_stage2_training_step_forward_audit"
    assert step_runtime["actual_training_step_forward"] is True
    assert step_runtime["backward_called"] is False
    assert step_runtime["optimizer_step_called"] is False
    assert step_runtime["fast_batched_stage2_used"] is True
    assert step_runtime["weighted_span_loss_applied"] is True
    assert step_runtime["mask_scope_applied"] is True
    assert step_runtime["loss_total_finite"] is True
    assert step_runtime["expected_weighted_span_loss"]["focus_target"] == 1.5
    gate_status = {
        gate["name"]: gate["status"] for gate in runtime_audit["launch_gates"]["gates"]
    }
    assert runtime_audit["launch_gates"]["training_step_runtime_status"] == (
        "actual_stage2_training_step_forward_audit"
    )
    assert gate_status["use_fast_batched_stage2_path"] == "identity_validated"
    assert gate_status["apply_weighted_span_losses_from_plan"] == "identity_validated"
    assert gate_status["apply_original_image_mask_scope_from_plan"] == "identity_validated"
    assert gate_status["save_checkpoint_with_clean_contract"] == "pending_real_trainer_loop"


def test_stage1_training_executor_runtime_audit_can_write_training_step_probe(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import torch

    train_file = tmp_path / "stage1.train.jsonl"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "target": "mark", '
        '"evidence_description": "The mark is visible."}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "stage1_plan"
    qwen = torch.nn.Linear(2, 2)
    tgvf = torch.nn.Sequential(torch.nn.Linear(2, 1))

    def fake_loader(bundle, *, expected_stage):
        assert expected_stage.value == "stage1"
        assert bundle["stage"] == "stage1"
        return {
            "modules": {"qwen": qwen, "tgvf": tgvf},
            "loader": {"backend": "fake_stage1_training_step_audit_loader"},
        }

    def fake_step_probe(*, bundle, artifacts, loaded_modules):
        assert bundle["stage"] == "stage1"
        assert artifacts["dataset_runtime_identity"]["stage"] == "stage1"
        assert loaded_modules["modules"]["qwen"] is qwen
        return {
            "forward_completed": True,
            "sample_count": 1,
            "loss_total": 2.0,
            "loss_gen": 1.0,
            "loss_visual_token_manifold": 0.5,
            "loss_same_image_negative": 0.25,
            "debug": {
                "stage": "tgvf_v3_stage1",
                "loss_weights": {
                    "gen": 1.0,
                    "visual_token_manifold": 0.1,
                    "same_image_negative": 1.0,
                    "contrastive_alignment": 0.0,
                },
                "same_image_negative_mode": "matrix_ce",
                "readout_append_mode": "qwen3_visual_special_tokens_embedding_replace",
                "position_ids_source": "qwen3_native_source_grid_full_trajectory",
                "attention_mask_mode": "weak_strict_after_tgvf",
                "image_keys_blocked_for_tgvf_evidence_answer": True,
                "visual_token_manifold_active": True,
            },
        }

    monkeypatch.setattr(training_executor, "_load_training_parameter_audit_modules", fake_loader)
    monkeypatch.setattr(training_executor, "_run_stage1_training_step_probe", fake_step_probe)
    assert (
        stage1_main(
            [
                "--run-id",
                "stage1_training_step_audit",
                "--train-file",
                str(train_file),
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )

    assert (
        stage1_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
                "--audit-runtime",
                "--audit-training-step",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"training_step_runtime"' in payload
    execution_dir = output_dir / "clean_training_execution"
    step_runtime = json.loads((execution_dir / "training_step_runtime.json").read_text())
    runtime_audit = json.loads((execution_dir / "clean_training_runtime_audit.json").read_text())
    assert step_runtime["status"] == "actual_stage1_training_step_forward_audit"
    assert step_runtime["actual_training_step_forward"] is True
    assert step_runtime["backward_called"] is False
    assert step_runtime["stage1_readout_context_applied"] is True
    assert step_runtime["stage1_position_ids_applied"] is True
    assert step_runtime["stage1_matrix_ce_and_manifold_losses_applied"] is True
    assert step_runtime["observed_same_image_negative_mode"] == "matrix_ce"
    gate_status = {
        gate["name"]: gate["status"] for gate in runtime_audit["launch_gates"]["gates"]
    }
    assert runtime_audit["launch_gates"]["training_step_runtime_status"] == (
        "actual_stage1_training_step_forward_audit"
    )
    assert gate_status["stage1_readout_context_uses_qwen_v_merge"] == "identity_validated"
    assert gate_status["stage1_position_ids_use_real_qwen3_mrope"] == "identity_validated"
    assert gate_status["stage1_matrix_ce_and_manifold_losses_match_plan"] == (
        "identity_validated"
    )


def test_stage1_training_executor_runtime_audit_can_write_optimizer_step_probe(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    import torch

    train_file = tmp_path / "stage1.train.jsonl"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "target": "mark", '
        '"evidence_description": "The mark is visible."}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "stage1_plan"
    qwen = torch.nn.Linear(2, 2)
    tgvf = torch.nn.Sequential(torch.nn.Linear(2, 1))

    def fake_loader(bundle, *, expected_stage):
        assert expected_stage.value == "stage1"
        assert bundle["stage"] == "stage1"
        return {
            "modules": {"qwen": qwen, "tgvf": tgvf},
            "loader": {"backend": "fake_stage1_optimizer_step_audit_loader"},
        }

    def fake_step_probe(*, bundle, artifacts, loaded_modules):
        assert bundle["stage"] == "stage1"
        assert artifacts["optimizer_groups"]["status"] == "validated"
        parameters = [
            parameter
            for module in (qwen, tgvf)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        loss_tensor = sum(parameter.square().sum() for parameter in parameters)
        return {
            "forward_completed": True,
            "sample_count": 1,
            "loss_total": float(loss_tensor.detach()),
            "loss_tensor": loss_tensor,
            "loss_gen": 1.0,
            "loss_visual_token_manifold": 0.5,
            "loss_same_image_negative": 0.25,
            "debug": {
                "stage": "tgvf_v3_stage1",
                "loss_weights": {
                    "gen": 1.0,
                    "visual_token_manifold": 0.1,
                    "same_image_negative": 1.0,
                    "contrastive_alignment": 0.0,
                },
                "same_image_negative_mode": "matrix_ce",
                "readout_append_mode": "qwen3_visual_special_tokens_embedding_replace",
                "position_ids_source": "qwen3_native_source_grid_full_trajectory",
                "attention_mask_mode": "weak_strict_after_tgvf",
                "image_keys_blocked_for_tgvf_evidence_answer": True,
                "visual_token_manifold_active": True,
            },
        }

    monkeypatch.setattr(training_executor, "_load_training_parameter_audit_modules", fake_loader)
    monkeypatch.setattr(training_executor, "_run_stage1_training_step_probe", fake_step_probe)
    assert (
        stage1_main(
            [
                "--run-id",
                "stage1_optimizer_step_audit",
                "--train-file",
                str(train_file),
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )

    assert (
        stage1_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
                "--audit-runtime",
                "--audit-optimizer-step",
            ]
        )
        == 0
    )
    payload = capsys.readouterr().out
    assert '"optimizer_step_runtime"' in payload
    execution_dir = output_dir / "clean_training_execution"
    optimizer_step_runtime = json.loads(
        (execution_dir / "optimizer_step_runtime.json").read_text()
    )
    runtime_audit = json.loads((execution_dir / "clean_training_runtime_audit.json").read_text())
    assert optimizer_step_runtime["status"] == "actual_backward_optimizer_scheduler_step_audit"
    assert optimizer_step_runtime["backward_called"] is True
    assert optimizer_step_runtime["optimizer_step_called"] is True
    assert optimizer_step_runtime["scheduler_step_called"] is True
    assert optimizer_step_runtime["grad_before_clip"]["tensors_with_grad"] > 0
    assert optimizer_step_runtime["grad_after_zero"]["tensors_with_grad"] == 0
    gate_status = {
        gate["name"]: gate["status"] for gate in runtime_audit["launch_gates"]["gates"]
    }
    assert gate_status["stage1_readout_context_uses_qwen_v_merge"] == "identity_validated"
    assert gate_status["stage1_position_ids_use_real_qwen3_mrope"] == "identity_validated"
    assert gate_status["stage1_matrix_ce_and_manifold_losses_match_plan"] == (
        "identity_validated"
    )
    assert gate_status["run_backward_optimizer_scheduler_step_from_plan"] == (
        "identity_validated"
    )


def test_stage2_prepare_execution_rejects_missing_required_dataset_fields(tmp_path) -> None:
    train_file = tmp_path / "stage2.train.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a"}\n',
        encoding="utf-8",
    )
    checkpoint.write_bytes(b"checkpoint\n")
    output_dir = tmp_path / "stage2_plan"
    assert (
        stage2_main(
            [
                "--run-id",
                "stage2_bad_dataset",
                "--train-file",
                str(train_file),
                "--stage1-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )

    with pytest.raises(ValueError, match="missing required fields"):
        stage2_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
            ]
        )


def test_stage2_prepare_execution_rejects_unloadable_stage1_checkpoint(tmp_path) -> None:
    train_file = tmp_path / "stage2.train.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n',
        encoding="utf-8",
    )
    checkpoint.write_bytes(b"not a torch checkpoint\n")
    output_dir = tmp_path / "stage2_plan"
    assert (
        stage2_main(
            [
                "--run-id",
                "stage2_bad_checkpoint",
                "--train-file",
                str(train_file),
                "--stage1-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )

    with pytest.raises(ValueError, match="not loadable by torch"):
        stage2_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
            ]
        )


def test_stage2_deepstack_plan_disables_legacy_command(tmp_path, capsys) -> None:
    train_file = tmp_path / "stage2.train.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a"}\n',
        encoding="utf-8",
    )
    checkpoint.write_bytes(b"checkpoint\n")

    assert (
        stage2_main(
            [
                "--run-id",
                "stage2_deepstack",
                "--train-file",
                str(train_file),
                "--stage1-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(tmp_path / "stage2_deepstack_plan"),
                "--mask-original-image-after-tgvf-scope",
                "evidence_only",
                "--deepstack-enabled",
                "--dry-run",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    deepstack_plan = payload["deepstack_training_plan"]
    assert deepstack_plan["enabled"] is True
    assert deepstack_plan["original_image_scope"] == "evidence_only"
    assert deepstack_plan["execution_supported"] is True
    assert deepstack_plan["runtime_hooks"]["all_required_hooks_implemented"] is True
    assert deepstack_plan["original_image_deepstack"]["restore_for_answer"] is True
    assert deepstack_plan["current_training_path"]["qwen3_deepstack_features_injected"] is True
    assert deepstack_plan["current_training_path"]["post_d_deepstack_scope_mask_applied"] is True
    assert deepstack_plan["current_training_path"]["answer_stage_restore_supported"] is True
    assert payload["clean_native_training"]["executable"] is True
    assert payload["clean_native_training"]["launch_training_supported"] is True
    assert payload["legacy_reference_command"]["executable"] is False
    assert (
        payload["legacy_reference_command"]["unavailable_reason"]
        == "historical Stage2 script has no DeepStack training controls"
    )


def test_stage2_deepstack_prepare_writes_training_plan(tmp_path) -> None:
    train_file = tmp_path / "stage2.train.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a", '
        '"need_focus": true, "evidence_state": "need_local_visual_evidence"}\n',
        encoding="utf-8",
    )
    _write_minimal_stage1_checkpoint(checkpoint)
    output_dir = tmp_path / "stage2_deepstack_plan"
    assert (
        stage2_main(
            [
                "--run-id",
                "stage2_deepstack_prepare",
                "--train-file",
                str(train_file),
                "--stage1-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--mask-original-image-after-tgvf-scope",
                "through_answer",
                "--deepstack-enabled",
                "--write-plan",
            ]
        )
        == 0
    )
    assert (
        stage2_executor_main(
            [
                "--plan",
                str(output_dir / "training_plan.json"),
                "--prepare-execution",
            ]
        )
        == 0
    )
    execution_dir = output_dir / "clean_training_execution"
    deepstack_plan = json.loads((execution_dir / "deepstack_training_plan.json").read_text())
    bundle = json.loads((execution_dir / "clean_training_execution_bundle.json").read_text())
    assert deepstack_plan["schema_version"] == "clean_deepstack_training_plan_v1"
    assert deepstack_plan["enabled"] is True
    assert deepstack_plan["execution_supported"] is True
    assert deepstack_plan["runtime_hooks"]["all_required_hooks_implemented"] is True
    assert deepstack_plan["runtime_hooks"]["hooks"][
        "capture_original_image_deepstack_features"
    ]["status"] == "ported"
    assert deepstack_plan["runtime_hooks"]["hooks"][
        "carry_original_image_deepstack_through_post_tgvf_append"
    ]["status"] == "ported"
    assert deepstack_plan["runtime_hooks"]["hooks"][
        "apply_post_tgvf_deepstack_scope_mask"
    ]["required"] is True
    assert deepstack_plan["runtime_hooks"]["hooks"][
        "apply_post_tgvf_deepstack_scope_mask"
    ]["status"] == "ported"
    assert deepstack_plan["runtime_hooks"]["hooks"][
        "restore_deepstack_for_answer_when_scope_requires"
    ]["required"] is False
    assert deepstack_plan["runtime_hooks"]["hooks"][
        "restore_deepstack_for_answer_when_scope_requires"
    ]["status"] == "not_required"
    assert deepstack_plan["blocking_items"] == []
    assert deepstack_plan["scope_contract"]["runtime_hooks"] == deepstack_plan["runtime_hooks"]
    assert deepstack_plan["original_image_deepstack"]["block_after_tgvf_append"] is True
    assert deepstack_plan["original_image_deepstack"]["restore_for_answer"] is False
    assert deepstack_plan["d_deepstack_features"]["required_for_current_mainline"] is False
    assert deepstack_plan["fvt_visual_token_path"] == "v_merge_level_visual_tokens"
    assert deepstack_plan["scope_contract"]["surface"] == "stage2_training"
    assert deepstack_plan["scope_contract"]["original_image_deepstack"] == (
        deepstack_plan["original_image_deepstack"]
    )
    assert deepstack_plan["scope_contract"]["original_image_deepstack"][
        "block_query_end"
    ] is None
    assert bundle["runtime_artifacts"]["deepstack_training_plan"] == str(
        execution_dir / "deepstack_training_plan.json"
    )
    assert "deepstack_training_plan.json" in bundle["trainer_runtime_contract"][
        "required_runtime_artifacts"
    ]
    assert bundle["deepstack"]["enabled"] is True
    assert bundle["deepstack_training_plan"]["execution_supported"] is True


def test_generate_data_dry_run_cli(tmp_path, capsys) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    source = input_root / "stage2.train.jsonl"
    source.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"manifest_hash": "toyhash"}\n', encoding="utf-8")

    assert (
        generate_data_main(
            [
                "--run-id",
                "data_plan",
                "--stage",
                "stage2_protocol_c",
                "--input-root",
                str(input_root),
                "--input-files",
                "stage2.train.jsonl",
                "--output-dir",
                str(tmp_path / "out"),
                "--source-manifest-path",
                str(manifest),
                "--source-manifest-hash",
                "toyhash",
                "--transform",
                "choice_to_open_answer",
                "--field-weight",
                "focus_target=1.5",
                "--mask-policy",
                "scope=through_answer",
                "--dry-run",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert '"run_id": "data_plan"' in captured.out
    assert '"output_schema_version": "clean_data_generation_v1"' in captured.out
    assert '"stage": "stage2_protocol_c"' in captured.out
    assert '"line_count": 2' in captured.out
    assert '"focus_target": 1.5' in captured.out
    assert '"scope": "through_answer"' in captured.out
    assert '"generated_data_written": false' in captured.out


def test_generate_data_write_plan_cli(tmp_path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    source = input_root / "stage1.train.jsonl"
    source.write_text('{"id": "a"}\n', encoding="utf-8")
    output_dir = tmp_path / "plan"

    assert (
        generate_data_main(
            [
                "--run-id",
                "stage1_plan",
                "--stage",
                "stage1_protocol_c_focus",
                "--input-root",
                str(input_root),
                "--input-files",
                "stage1.train.jsonl",
                "--output-dir",
                str(output_dir),
                "--write-plan",
            ]
        )
        == 0
    )

    assert (output_dir / "data_generation_config.json").exists()
    assert (output_dir / "data_generation_config.txt").exists()
    assert (output_dir / "input_files.json").exists()
    assert (output_dir / "data_generation_report.json").exists()
    config = json.loads((output_dir / "data_generation_config.json").read_text())
    assert config["output_schema_version"] == "clean_data_generation_v1"
    assert "identity_only: true" in (output_dir / "data_generation_config.txt").read_text()
    assert (
        "generated_data_written: false" in (output_dir / "data_generation_config.txt").read_text()
    )


def test_generate_data_execute_choice_to_open_answer(tmp_path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    source = input_root / "stage2.train.jsonl"
    source.write_text(
        '{"question": "Pick one\\n(A) red\\n(B) blue\\nAnswer only with the option letter.", '
        '"choices": ["red", "blue"], "answer": "B", "answer_format": "multiple_choice", '
        '"value_span_text": "B. blue", "focus_steps": [{"value_span_text": "B. blue"}]}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "converted"

    assert (
        generate_data_main(
            [
                "--run-id",
                "choice_open",
                "--stage",
                "choice_to_open_answer",
                "--input-root",
                str(input_root),
                "--input-files",
                "stage2.train.jsonl",
                "--output-dir",
                str(output_dir),
                "--transform",
                "choice_to_open_answer",
                "--execute",
            ]
        )
        == 0
    )

    row = json.loads((output_dir / "stage2.train.jsonl").read_text().strip())
    assert row["question"] == "Pick one"
    assert row["choices"] == []
    assert row["answer"] == "blue"
    assert row["answer_format"] == "short_text"
    assert row["focus_steps"][0]["value_span_text"] == "blue"
    report = (output_dir / "data_generation_report.json").read_text()
    assert '"generated_data_written": true' in report
    assert '"converted_choice": 1' in report
    report_json = json.loads(report)
    assert report_json["summary"]["total_generated_lines"] == 1
    assert report_json["output_files"][0]["line_count"] == 1
    assert len(report_json["split_hashes"]["stage2.train.jsonl"]) == 64
    config_txt = (output_dir / "data_generation_config.txt").read_text()
    assert "identity_only: false" in config_txt
    assert "generated_data_written: true" in config_txt
    assert "split_hash[stage2.train.jsonl]:" in config_txt


def test_generate_data_execute_clean_imend(tmp_path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    source = input_root / "stage1.train.jsonl"
    source.write_text(
        '{"question": "clean", "answer": "ok"}\n'
        '{"question": "bad metadata: polluted", "answer": "no"}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "cleaned"

    assert (
        generate_data_main(
            [
                "--run-id",
                "clean_imend",
                "--stage",
                "clean_splits",
                "--input-root",
                str(input_root),
                "--input-files",
                "stage1.train.jsonl",
                "--output-dir",
                str(output_dir),
                "--transform",
                "clean_imend",
                "--execute",
            ]
        )
        == 0
    )

    rows = (output_dir / "stage1.train.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert '"question": "clean"' in rows[0]
    report = (output_dir / "transform_report.json").read_text()
    assert '"input": 2' in report
    assert '"kept": 1' in report
    assert '"dropped": 1' in report


def test_generate_data_execute_v4_to_protocol_c(tmp_path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    source = input_root / "teacher.accepted.jsonl"
    records = [
        {
            "item_type": "single_refocus",
            "teacher_prompt_version": "tgvf_v4_teacher",
            "image": "/tmp/image.jpg",
            "stable_image_uid": "toy:image",
            "source_dataset": "toy",
            "source_profile": "unit",
            "question": "What color is the mark?",
            "choices": [{"label": "A", "text": "red"}, {"label": "B", "text": "blue"}],
            "answer": "B. blue",
            "answer_text": "blue",
            "answer_format": "multiple_choice",
            "evidence_types": ["color_attribute"],
            "confidence": 0.9,
            "uid": "toy:0",
            "question_type": "color_attribute",
            "focus_category": "color_surface",
            "trace": [
                {"type": "think", "text": "Need a closer look."},
                {
                    "type": "focus",
                    "focus_text": "the small mark",
                    "focused_evidence": "The mark is blue.",
                    "metadata": {
                        "focus_descriptor_cues": ["location", "color"],
                        "target_leakage_risk": "low",
                        "evidence_type": "color_attribute",
                    },
                },
                {"type": "think", "text": "It is blue."},
                {"type": "answer", "text": "B. blue"},
            ],
        },
        {
            "item_type": "no_refocus_answer",
            "image": "/tmp/image.jpg",
            "question": "Is this visible?",
            "answer": "yes",
            "answer_text": "yes",
            "trace": [{"type": "think", "text": "The image is sufficient."}],
        },
    ]
    source.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    output_dir = tmp_path / "protocol"

    assert (
        generate_data_main(
            [
                "--run-id",
                "v4_protocol",
                "--stage",
                "stage2_protocol_c",
                "--input-root",
                str(input_root),
                "--input-files",
                "teacher.accepted.jsonl",
                "--output-dir",
                str(output_dir),
                "--transform",
                "v4_to_protocol_c",
                "--execute",
            ]
        )
        == 0
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "teacher.accepted.jsonl").read_text().splitlines()
    ]
    assert rows[0]["schema_version"] == "tgvf_teacher_schema_v4_stage2_compat"
    assert rows[0]["trajectory_type"] == "single_focus"
    assert rows[0]["need_focus"] is True
    assert rows[0]["target"] == "the small mark"
    assert rows[0]["evidence_description"] == "The mark is blue."
    assert rows[0]["pre_focus_think"] == "Need a closer look."
    assert rows[0]["post_focus_think"] == "It is blue."
    assert rows[1]["trajectory_type"] == "direct_answer"
    assert rows[1]["need_focus"] is False
    report = (output_dir / "transform_report.json").read_text()
    assert '"written_single_focus": 1' in report
    assert '"written_direct_answer": 1' in report


def test_generate_data_execute_v4_to_stage1_protocol_c_focus(tmp_path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    source = input_root / "teacher.accepted.jsonl"
    records = [
        {
            "item_type": "multi_refocus",
            "teacher_prompt_version": "tgvf_v4_teacher",
            "schema_version": "tgvf_teacher_schema_v4",
            "image": "/tmp/chart.png",
            "stable_image_uid": "toy:chart",
            "source_dataset": "toy",
            "source_profile": "chart",
            "question": "Which bar is closest to 100?",
            "choices": [{"label": "A", "text": "Capital"}, {"label": "B", "text": "Goods"}],
            "answer": "A. Capital",
            "answer_text": "Capital",
            "answer_format": "multiple_choice",
            "evidence_types": ["chart_value"],
            "confidence": 0.8,
            "uid": "toy:multi:0",
            "question_type": "math_reasoning",
            "focus_category": "chart_table_region",
            "trace": [
                {"type": "think", "text": "Need to find bars near 100."},
                {
                    "type": "focus",
                    "focus_text": "bars near the 100 mark",
                    "focused_evidence": "Capital at 93.47 and Goods at 49.58 are nearest.",
                    "metadata": {
                        "confidence": 0.95,
                        "focus_descriptor_cues": ["chart_anchor", "number_like"],
                        "target_leakage_risk": "low",
                        "evidence_type": "chart_value",
                    },
                },
                {"type": "think", "text": "Need the exact label for 93.47."},
                {
                    "type": "focus",
                    "focus_text": "the label under the 93.47 bar",
                    "focused_evidence": "The 93.47 bar is labeled Capital.",
                    "metadata": {
                        "confidence": 0.96,
                        "focus_descriptor_cues": ["chart_anchor", "text_like"],
                        "target_leakage_risk": "low",
                        "evidence_type": "chart_value",
                    },
                },
                {"type": "answer", "text": "A. Capital"},
            ],
        },
        {
            "item_type": "no_refocus_answer",
            "image": "/tmp/chart.png",
            "question": "Is the title visible?",
            "answer": "yes",
            "trace": [{"type": "think", "text": "The title is clear."}],
        },
    ]
    source.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    output_dir = tmp_path / "stage1"

    assert (
        generate_data_main(
            [
                "--run-id",
                "stage1_focus",
                "--stage",
                "stage1_protocol_c_focus",
                "--input-root",
                str(input_root),
                "--input-files",
                "teacher.accepted.jsonl",
                "--output-dir",
                str(output_dir),
                "--transform",
                "v4_to_stage1_protocol_c_focus",
                "--execute",
            ]
        )
        == 0
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "teacher.accepted.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["schema_version"] == "tgvf_teacher_schema_v4_stage1_compat"
    assert rows[0]["source_schema_version"] == "tgvf_teacher_schema_v4"
    assert rows[0]["uid"] == "toy:multi:0::focus1"
    assert rows[0]["source_uid"] == "toy:multi:0"
    assert rows[0]["trajectory_type"] == "single_focus"
    assert rows[0]["focus_step_index"] == 1
    assert rows[0]["target"] == "bars near the 100 mark"
    assert rows[0]["evidence_description"] == "Capital at 93.47 and Goods at 49.58 are nearest."
    assert rows[0]["confidence"] == 0.95
    assert rows[0]["short_answer"] == "Capital"
    assert rows[0]["answer_format"] == "multiple_choice"
    assert rows[1]["uid"] == "toy:multi:0::focus2"
    assert rows[1]["target_cues"] == ["chart_anchor", "text_like"]
    report = (output_dir / "transform_report.json").read_text()
    assert '"source_rows": 2' in report
    assert '"converted_focus_rows": 2' in report
    assert '"skipped_no_refocus_answer": 1' in report
