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
    status = json.loads((execution_dir / "valkit_execution_status.json").read_text())
    assert status["runner_status"] == "handoff_supported_valkit_runner_not_ported"
    assert status["will_launch_valkit"] is False


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
    assert plan["clean_native_training"]["executable"] is False
    assert plan["clean_native_training"]["required_for_final_clean_project"] is True
    assert plan["clean_native_training"]["prepare_execution_supported"] is True
    assert (
        plan["clean_native_training"]["status"]
        == "handoff_supported_trainer_loop_not_ported"
    )
    assert plan["clean_prepare_execution_command"]["final_clean_native"] is True
    assert plan["clean_prepare_execution_command"]["executable"] is True
    assert plan["clean_prepare_execution_command"]["status"] == "prepare_execution_supported"
    assert plan["clean_prepare_execution_command"]["will_launch_training"] is False
    assert plan["clean_training_command"]["final_clean_native"] is True
    assert plan["clean_training_command"]["executable"] is False
    assert (
        plan["clean_training_command"]["planned_entrypoint"]
        == "revisit_vlm_clean.training.stage1_executor"
    )
    assert plan["legacy_reference_command"]["final_clean_native"] is False
    assert plan["legacy_reference_command"]["executable"] is False
    assert "audit reference only" in plan["legacy_reference_command"]["unavailable_reason"]
    native_status = json.loads((output_dir / "clean_native_training_status.json").read_text())
    assert native_status["status"] == "handoff_supported_trainer_loop_not_ported"
    prepare_command_path = output_dir / "clean_prepare_execution_command.sh"
    assert prepare_command_path.stat().st_mode & 0o111
    prepare_command = prepare_command_path.read_text()
    assert prepare_command.startswith("python -m revisit_vlm_clean.training.stage1_executor")
    assert "--prepare-execution" in prepare_command
    assert "not executable" not in prepare_command
    clean_command = (output_dir / "clean_training_command.sh").read_text()
    assert "not executable" in clean_command
    assert "revisit_vlm_clean.training.stage1_executor" in clean_command
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
    assert bundle["clean_executor"]["status"] == "trainer_loop_not_ported"
    assert bundle["clean_executor"]["owns_execution_bundle"] is True
    assert bundle["safety"]["legacy_reference_allowed"] is False
    assert bundle["readout_context"]["position_ids"] == "real_qwen3_mrope_full_trajectory"
    contract = bundle["trainer_runtime_contract"]
    assert contract["contract_schema_version"] == "clean_trainer_runtime_contract_v1"
    assert contract["status"] == "not_ported"
    assert contract["launch_permitted"] is False
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
    assert status["runner_status"] == "trainer_loop_not_ported"
    assert status["trainer_runtime_contract_status"] == "not_ported"
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
    assert plan["clean_native_training"]["executable"] is False
    assert plan["clean_native_training"]["legacy_reference_is_final"] is False
    assert plan["clean_native_training"]["prepare_execution_supported"] is True
    assert (
        plan["clean_native_training"]["status"]
        == "handoff_supported_trainer_loop_not_ported"
    )
    assert plan["clean_prepare_execution_command"]["executable"] is True
    assert plan["clean_prepare_execution_command"]["status"] == "prepare_execution_supported"
    assert plan["clean_prepare_execution_command"]["will_launch_training"] is False
    assert plan["clean_training_command"]["final_clean_native"] is True
    assert plan["clean_training_command"]["executable"] is False
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
    assert "not executable" in clean_command
    assert "revisit_vlm_clean.training.stage2_executor" in clean_command
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
    assert '"training_runtime_ported": false' in payload
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
    status = json.loads((execution_dir / "clean_training_execution_status.json").read_text())
    assert status["bundle_valid"] is True
    assert status["trainer_runtime_contract_status"] == "not_ported"
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
    assert gate_status["save_checkpoint_with_clean_contract"] == "pending_real_trainer_loop"


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

    payload = capsys.readouterr().out
    assert '"enabled": true' in payload
    assert '"original_image_scope": "evidence_only"' in payload
    assert '"clean_native_training"' in payload
    assert '"executable": false' in payload
    assert "DeepStack original-image injection/masking" in payload
    assert "historical Stage2 script has no DeepStack training controls" in payload


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
