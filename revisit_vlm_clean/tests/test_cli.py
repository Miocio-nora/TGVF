import json

from revisit_vlm_clean.cli.benchmark import main as benchmark_main
from revisit_vlm_clean.cli.generate_data import main as generate_data_main
from revisit_vlm_clean.cli.manifest import main as manifest_main
from revisit_vlm_clean.cli.train_stage1 import main as stage1_main
from revisit_vlm_clean.cli.train_stage2 import main as stage2_main


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
    assert '"scoring_backend": "project"' in captured.out
    assert '"tgvf_protocol": "protocol_c_tool_observation_qwen2_no_think"' in captured.out


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
    assert "identity_only: true" in (output_dir / "data_generation_config.txt").read_text()
    assert "generated_data_written: false" in (output_dir / "data_generation_config.txt").read_text()


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
