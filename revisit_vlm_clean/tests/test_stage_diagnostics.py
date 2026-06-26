from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from revisit_vlm_clean.cli.stage_diagnostics import main as diagnostics_main
from revisit_vlm_clean import stage_diagnostics


def _write_inputs(tmp_path):
    checkpoint = tmp_path / "checkpoint_step_2000.pt"
    eval_jsonl = tmp_path / "stage1_focus.test.jsonl"
    checkpoint.write_bytes(b"placeholder")
    eval_jsonl.write_text("{}\n", encoding="utf-8")
    return checkpoint, eval_jsonl


def test_stage_diagnostics_dry_run_expands_all_tasks(tmp_path, capsys) -> None:
    checkpoint, eval_jsonl = _write_inputs(tmp_path)
    output_dir = tmp_path / "diagnostics"
    assert (
        diagnostics_main(
            [
                "--run-id",
                "stage1_diag",
                "--stage",
                "stage1",
                "--checkpoint",
                str(checkpoint),
                "--eval-jsonl",
                str(eval_jsonl),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["eval_family"] == "internal_diagnostic"
    assert payload["diagnostic_kind"] == "stage1_style_fvt_readout_regression"
    assert payload["tasks"] == ["readout", "query", "distribution"]
    assert payload["environment"]["WANDB_LOG_EVAL"] == "0"
    assert payload["expected_reports"]["readout"].endswith("readout_eval_report.json")


def test_stage_diagnostics_write_plan_records_command_and_env(tmp_path) -> None:
    checkpoint, eval_jsonl = _write_inputs(tmp_path)
    output_dir = tmp_path / "diagnostics"
    assert (
        diagnostics_main(
            [
                "--run-id",
                "stage2_diag",
                "--stage",
                "stage2",
                "--checkpoint",
                str(checkpoint),
                "--eval-jsonl",
                str(eval_jsonl),
                "--output-dir",
                str(output_dir),
                "--tasks",
                "readout,distribution",
                "--processor-id",
                "processor_step_1200",
                "--write-plan",
            ]
        )
        == 0
    )
    plan = json.loads((output_dir / "clean_stage_diagnostic_plan.json").read_text())
    script = (output_dir / "clean_stage_diagnostic_command.sh").read_text()
    assert plan["stage"] == "stage2"
    assert plan["tasks"] == ["readout", "distribution"]
    assert plan["environment"]["PROCESSOR_ID"] == "processor_step_1200"
    assert plan["command"] == [
        "bash",
        "eval/run_tgvf_v3_eval_suite.sh",
        str(checkpoint),
        "readout,distribution",
    ]
    assert "export EVAL_JSONL=" in script
    assert "eval/run_tgvf_v3_eval_suite.sh" in script


def test_stage_diagnostics_execute_invokes_legacy_eval_suite_with_clean_env(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint, eval_jsonl = _write_inputs(tmp_path)
    output_dir = tmp_path / "diagnostics"
    calls = []

    def fake_run(command, *, cwd, env, check):
        calls.append({"command": command, "cwd": cwd, "env": dict(env), "check": check})
        report_path = output_dir / "readout" / "readout_eval_report.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text('{"metrics": {"mean_nll_correct_D": 1.0}}\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(stage_diagnostics.subprocess, "run", fake_run)
    assert (
        diagnostics_main(
            [
                "--run-id",
                "stage1_diag_execute",
                "--stage",
                "stage1",
                "--checkpoint",
                str(checkpoint),
                "--eval-jsonl",
                str(eval_jsonl),
                "--output-dir",
                str(output_dir),
                "--tasks",
                "readout",
                "--execute",
            ]
        )
        == 0
    )
    assert calls
    assert calls[0]["command"] == [
        "bash",
        "eval/run_tgvf_v3_eval_suite.sh",
        str(checkpoint),
        "readout",
    ]
    assert calls[0]["env"]["OUT_ROOT"] == str(output_dir)
    assert calls[0]["env"]["EVAL_JSONL"] == str(eval_jsonl)
    status = json.loads((output_dir / "clean_stage_diagnostic_status.json").read_text())
    assert status["status"] == "completed"
    assert status["reports"]["readout"]["exists"] is True


def test_stage_diagnostics_wandb_upload_requires_explicit_project(tmp_path) -> None:
    checkpoint, eval_jsonl = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="wandb-project"):
        diagnostics_main(
            [
                "--run-id",
                "bad_wandb",
                "--stage",
                "stage1",
                "--checkpoint",
                str(checkpoint),
                "--eval-jsonl",
                str(eval_jsonl),
                "--output-dir",
                str(tmp_path / "diagnostics"),
                "--wandb-log-eval",
                "--dry-run",
            ]
        )
