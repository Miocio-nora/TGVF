from __future__ import annotations

import json

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
    assert payload["d_conditions"] == [
        "correct_D",
        "no_D",
        "random_D",
        "wrong_same_image_D",
        "wrong_diff_image_D",
    ]
    assert payload["execution_backend"]["name"] == "clean_native_stage_diagnostics"
    assert payload["execution_backend"]["legacy_bridge"] is False
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
    assert plan["processor_id"] == "processor_step_1200"
    assert plan["execution_backend"]["legacy_bridge"] is False
    assert plan["command"][:3] == ["python", "-m", "revisit_vlm_clean.cli.stage_diagnostics"]
    assert "eval/run_tgvf_v3_eval_suite.sh" not in script
    assert "--execute" in plan["command"]
    assert "--processor-id" in plan["command"]


def test_stage_diagnostics_can_disable_stage2_lora_for_legacy_readout(tmp_path, capsys) -> None:
    checkpoint, eval_jsonl = _write_inputs(tmp_path)
    output_dir = tmp_path / "diagnostics"
    assert (
        diagnostics_main(
            [
                "--run-id",
                "stage2_diag_base_qwen",
                "--stage",
                "stage2",
                "--checkpoint",
                str(checkpoint),
                "--eval-jsonl",
                str(eval_jsonl),
                "--output-dir",
                str(output_dir),
                "--no-stage2-load-lora",
                "--dry-run",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["config"]["stage2_load_lora"] is False
    assert "--no-stage2-load-lora" in plan["command"]
    assert plan["execution_backend"]["forward_semantics"] == (
        "stage2_tgvf_module_with_base_qwen_readout; qwen_lora ignored by request"
    )


def test_stage_diagnostics_execute_invokes_clean_native_executor(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint, eval_jsonl = _write_inputs(tmp_path)
    output_dir = tmp_path / "diagnostics"
    calls = []

    def fake_execute(plan):
        calls.append(plan)
        report_path = output_dir / "readout" / "readout_eval_report.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text('{"metrics": {"mean_nll_correct_D": 1.0}}\n')
        return {"readout": {"report": str(report_path), "num_rows": 1}}

    monkeypatch.setattr(stage_diagnostics, "execute_stage_diagnostic_tasks", fake_execute)
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
    assert calls[0]["execution_backend"]["name"] == "clean_native_stage_diagnostics"
    assert calls[0]["execution_backend"]["legacy_bridge"] is False
    assert calls[0]["checkpoint"] == str(checkpoint)
    assert calls[0]["eval_jsonl"] == str(eval_jsonl)
    status = json.loads((output_dir / "clean_stage_diagnostic_status.json").read_text())
    assert status["status"] == "completed"
    assert status["reports"]["readout"]["exists"] is True


def test_stage_diagnostics_resolves_d_deepstack_from_checkpoint_config() -> None:
    config = stage_diagnostics.StageDiagnosticConfig(
        run_id="stage1_ddeep_diag",
        stage="stage1",
        checkpoint="checkpoint.pt",
        eval_jsonl="eval.jsonl",
        output_dir="out",
    )
    resolved = stage_diagnostics._resolved_tgvf_module_config(
        checkpoint={
            "config": {
                "tgvf": {
                    "variant": "tgvf_v2_bidirectional",
                    "spatial_merge_size": 2,
                    "d_deepstack_enabled": True,
                    "d_deepstack_branch_layers": [8, 16, 24],
                }
            }
        },
        config=config,
        dims={"spatial_merge_size": 2},
    )
    assert resolved["d_deepstack_enabled"] is True
    assert resolved["d_deepstack_branch_layers"] == (8, 16, 24)


def test_stage_diagnostics_json_safe_serializes_sets_stably() -> None:
    payload = stage_diagnostics._json_safe({"tokens": {"b", "a"}, "ids": frozenset({2, 1})})
    assert payload == {"tokens": ["a", "b"], "ids": [1, 2]}
    json.dumps(payload, sort_keys=True)


def test_stage_diagnostics_wandb_upload_is_not_implemented(tmp_path) -> None:
    checkpoint, eval_jsonl = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="W&B upload is not implemented"):
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
