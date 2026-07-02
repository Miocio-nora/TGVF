import json

import pytest
from revisit_vlm_clean.cli.benchmark import main as benchmark_main
from revisit_vlm_clean.schema import ForwardMode
from revisit_vlm_clean.stage2_runtime import Stage2RuntimeConfig, eval_jsonl_identity


def test_stage2_runtime_config_rejects_missing_files(tmp_path) -> None:
    config = Stage2RuntimeConfig(
        stage2_checkpoint=str(tmp_path / "missing.pt"),
        eval_jsonl=str(tmp_path / "missing.jsonl"),
        append_forward_mode=ForwardMode.KV_CACHE,
    )

    with pytest.raises(FileNotFoundError):
        config.validate()


def test_stage2_runtime_config_rejects_nonpositive_max_tokens(tmp_path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    jsonl = tmp_path / "eval.jsonl"
    ckpt.write_bytes(b"placeholder")
    jsonl.write_text('{"image": "a.jpg", "need_focus": true}\n')
    config = Stage2RuntimeConfig(
        stage2_checkpoint=str(ckpt),
        eval_jsonl=str(jsonl),
        append_forward_mode=ForwardMode.KV_CACHE,
        max_tokens=0,
    )

    with pytest.raises(ValueError, match="max_tokens"):
        config.validate()


def test_eval_jsonl_identity_counts_focus_and_no_focus(tmp_path) -> None:
    path = tmp_path / "stage2.jsonl"
    path.write_text(
        json.dumps(
            {
                "image": "a.jpg",
                "need_focus": True,
                "trajectory_type": "single_focus",
                "target": "text",
            }
        )
        + "\n"
        + json.dumps({"image": "b.jpg", "need_focus": False, "trajectory_type": "direct_answer"})
        + "\n"
    )

    identity = eval_jsonl_identity(path)

    assert identity["n_rows"] == 2
    assert identity["need_focus"] == 1
    assert identity["no_focus"] == 1


def test_validate_stage2_runtime_requires_identity_paths(capsys) -> None:
    with pytest.raises(ValueError, match="--stage2-checkpoint"):
        benchmark_main(
            [
                "--run-id",
                "stage2",
                "--checkpoint-path",
                "outputs/checkpoint.pt",
                "--mode",
                "tgvf_force",
                "--post-tgvf-forward-mode",
                "kv_cache",
                "--subset-id",
                "core_smoke_256_seed20260625",
                "--validate-stage2-runtime",
            ]
        )
    capsys.readouterr()
