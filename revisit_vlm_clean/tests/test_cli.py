import json

import pytest
from revisit_vlm_clean.cli.benchmark import main as benchmark_main
from revisit_vlm_clean.cli.generate_data import main as generate_data_main
from revisit_vlm_clean.cli.manifest import main as manifest_main
from revisit_vlm_clean.cli.train_stage1 import main as stage1_main
from revisit_vlm_clean.cli.train_stage2 import main as stage2_main
from revisit_vlm_clean.cli.valkit import main as valkit_main
from revisit_vlm_clean.training.stage1_executor import main as stage1_executor_main
from revisit_vlm_clean.training.stage2_executor import main as stage2_executor_main


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
    assert "matrix_ce" in capsys.readouterr().out
    assert stage2_main(["--print-defaults"]) == 0
    assert "through_answer" in capsys.readouterr().out


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
    assert plan["module_policy"]["trainable"] == [
        "tgvf_module",
        "protocol_c_token_rows_row_only",
    ]
    assert "qwen_visual_merger" in plan["module_policy"]["frozen"]
    assert plan["module_policy"]["visual_merger"]["trainable"] is False
    assert plan["module_policy"]["training_runtime"]["use_cache"] is False
    assert plan["clean_native_training"]["executable"] is False
    assert plan["clean_native_training"]["required_for_final_clean_project"] is True
    assert plan["clean_training_command"]["final_clean_native"] is True
    assert plan["clean_training_command"]["executable"] is False
    assert (
        plan["clean_training_command"]["planned_entrypoint"]
        == "revisit_vlm_clean.training.stage1_executor"
    )
    assert plan["legacy_reference_command"]["final_clean_native"] is False
    native_status = json.loads((output_dir / "clean_native_training_status.json").read_text())
    assert native_status["status"] == "not_implemented"
    clean_command = (output_dir / "clean_training_command.sh").read_text()
    assert "not executable" in clean_command
    assert "revisit_vlm_clean.training.stage1_executor" in clean_command
    command = (output_dir / "legacy_reference_command.sh").read_text()
    assert "torchrun --nproc-per-node 4" in command
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


def test_stage2_training_write_plan_cli(tmp_path) -> None:
    train_file = tmp_path / "stage2.train.jsonl"
    val_file = tmp_path / "stage2.test.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a"}\n',
        encoding="utf-8",
    )
    val_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a"}\n',
        encoding="utf-8",
    )
    checkpoint.write_bytes(b"checkpoint\n")
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
    assert plan["clean_training_command"]["final_clean_native"] is True
    assert plan["clean_training_command"]["executable"] is False
    assert (
        plan["clean_training_command"]["planned_entrypoint"]
        == "revisit_vlm_clean.training.stage2_executor"
    )
    assert plan["legacy_reference_command"]["final_clean_native"] is False
    assert (output_dir / "clean_native_training_status.json").exists()
    clean_command = (output_dir / "clean_training_command.sh").read_text()
    assert "not executable" in clean_command
    assert "revisit_vlm_clean.training.stage2_executor" in clean_command
    command = (output_dir / "legacy_reference_command.sh").read_text()
    assert "--mask-original-image-after-tgvf-scope through_answer" in command
    assert "--loss-focus-target 1.5" in command


def test_stage2_training_executor_preflight_cli(tmp_path, capsys) -> None:
    train_file = tmp_path / "stage2.train.jsonl"
    val_file = tmp_path / "stage2.test.jsonl"
    checkpoint = tmp_path / "stage1.pt"
    train_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a"}\n',
        encoding="utf-8",
    )
    val_file.write_text(
        '{"image": "/tmp/image.jpg", "question": "q", "answer": "a"}\n',
        encoding="utf-8",
    )
    checkpoint.write_bytes(b"checkpoint\n")
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
