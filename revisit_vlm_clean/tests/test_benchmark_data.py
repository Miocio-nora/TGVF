import json

from revisit_vlm_clean.benchmark_data import materialize_samples_from_manifest_path
from revisit_vlm_clean.cli.benchmark import main as benchmark_main


def _write_toy_vstar(root):
    snapshot = root / "vstar_bench" / "snapshot"
    image_dir = snapshot / "direct_attributes"
    image_dir.mkdir(parents=True)
    (image_dir / "toy.jpg").write_bytes(b"not-a-real-image")
    source = snapshot / "test_questions.jsonl"
    source.write_text(
        json.dumps(
            {
                "image": "direct_attributes/toy.jpg",
                "text": "What color is the toy?\n(A) red\n(B) blue",
                "category": "direct_attributes",
                "question_id": "toy-0",
                "label": "B",
            }
        )
        + "\n"
    )
    return source


def _write_toy_manifest(path):
    payload = {
        "manifest_id": "toy_manifest",
        "manifest_hash": "toyhash",
        "samples": [
            {
                "sample_id": "vstar_test_questions_191/toy/toy_000000",
                "benchmark": "vstar_bench",
                "population_id": "vstar_test_questions_191",
                "source_file": "vstar_bench/snapshot/test_questions.jsonl",
                "metadata": {
                    "row_index": 0,
                    "raw_id": "toy-0",
                    "question_id": "toy-0",
                    "category": "direct_attributes",
                    "label": "B",
                },
            }
        ],
    }
    path.write_text(json.dumps(payload) + "\n")
    return path


def _write_toy_ocrbench(root):
    snapshot = root / "ocrbench_v2" / "snapshot"
    snapshot.mkdir(parents=True)
    source = snapshot / "toy.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "ocr-0",
                "question": "Read the word.",
                "type": "text recognition en",
                "answers": ["blue", "BLUE"],
                "eval": "case sensitive",
                "dataset_name": "toy_ocr",
                "precision": 0,
            }
        )
        + "\n"
    )
    return source


def _write_toy_ocrbench_manifest(path):
    payload = {
        "manifest_id": "ocrbench_toy",
        "manifest_hash": "ocrhash",
        "samples": [
            {
                "sample_id": "ocrbench_v2/toy/0",
                "benchmark": "ocrbench_v2",
                "population_id": "ocrbench_v2_data_test_10000",
                "source_file": "ocrbench_v2/snapshot/toy.jsonl",
                "metadata": {"row_index": 0, "raw_id": "ocr-0"},
            }
        ],
    }
    path.write_text(json.dumps(payload) + "\n")
    return path


def _write_fake_ocrbench_official(root):
    eval_path = root / "ocrbench_v2" / "official_code" / "OCRBench_v2" / "eval_scripts" / "eval.py"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text(
        """
import json


def process_predictions(input_path, output_path):
    with open(input_path, encoding="utf-8") as handle:
        rows = json.load(handle)
    for row in rows:
        answers = row.get("answers") or []
        if not isinstance(answers, list):
            answers = [answers]
        row["score"] = 1.0 if str(row.get("predict") or "") in {str(item) for item in answers} else 0.0
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle)
""".strip()
        + "\n"
    )
    return eval_path


def test_materialize_samples_from_manifest_path(tmp_path) -> None:
    root = tmp_path / "benchmarks"
    _write_toy_vstar(root)
    manifest_path = _write_toy_manifest(tmp_path / "manifest.json")

    samples = materialize_samples_from_manifest_path(manifest_path, benchmark_root=root)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.sample_id == "vstar_test_questions_191/toy/toy_000000"
    assert sample.choices == ("red", "blue")
    assert sample.gold_answer == "B"
    assert sample.primary_media["exists"] is True


def test_materialize_samples_preserves_official_scorer_metadata(tmp_path) -> None:
    root = tmp_path / "benchmarks"
    _write_toy_ocrbench(root)
    manifest_path = _write_toy_ocrbench_manifest(tmp_path / "ocr_manifest.json")

    samples = materialize_samples_from_manifest_path(manifest_path, benchmark_root=root)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.gold_answer == "blue"
    assert sample.metadata["answers"] == ["blue", "BLUE"]
    assert sample.metadata["type"] == "text recognition en"
    assert sample.metadata["eval"] == "case sensitive"
    assert sample.metadata["precision"] == 0


def test_benchmark_materialize_samples_cli(tmp_path) -> None:
    root = tmp_path / "benchmarks"
    _write_toy_vstar(root)
    manifest_path = _write_toy_manifest(tmp_path / "manifest.json")
    output_dir = tmp_path / "out"

    assert (
        benchmark_main(
            [
                "--run-id",
                "materialize",
                "--checkpoint-path",
                "outputs/checkpoint.pt",
                "--mode",
                "tgvf_force",
                "--post-tgvf-forward-mode",
                "kv_cache",
                "--population-id",
                "vstar_test_questions_191",
                "--manifest-path",
                str(manifest_path),
                "--manifest-hash",
                "toyhash",
                "--benchmark-root",
                str(root),
                "--output-dir",
                str(output_dir),
                "--materialize-samples",
            ]
        )
        == 0
    )
    rows = (output_dir / "materialized_rows.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["gold_answer"] == "B"
    assert "sample materialization smoke" in (output_dir / "summary.json").read_text()


def test_benchmark_render_inputs_cli(tmp_path) -> None:
    root = tmp_path / "benchmarks"
    _write_toy_vstar(root)
    manifest_path = _write_toy_manifest(tmp_path / "manifest.json")
    output_dir = tmp_path / "render"

    assert (
        benchmark_main(
            [
                "--run-id",
                "render",
                "--checkpoint-path",
                "outputs/checkpoint.pt",
                "--mode",
                "tgvf_softforce",
                "--softforce-prompt-text",
                "Use focus tool.",
                "--post-tgvf-forward-mode",
                "kv_cache",
                "--population-id",
                "vstar_test_questions_191",
                "--manifest-path",
                str(manifest_path),
                "--manifest-hash",
                "toyhash",
                "--benchmark-root",
                str(root),
                "--output-dir",
                str(output_dir),
                "--render-inputs",
            ]
        )
        == 0
    )
    rows = (output_dir / "rendered_inputs.jsonl").read_text().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["user_prompt"].endswith("\n\nUse focus tool.")
    assert row["requires_tgvf_controller"] is True
    assert "rendered input smoke" in (output_dir / "summary.json").read_text()


def test_benchmark_execute_dry_run_cli(tmp_path) -> None:
    root = tmp_path / "benchmarks"
    _write_toy_vstar(root)
    manifest_path = _write_toy_manifest(tmp_path / "manifest.json")
    output_dir = tmp_path / "exec"

    assert (
        benchmark_main(
            [
                "--run-id",
                "exec",
                "--checkpoint-path",
                "outputs/checkpoint.pt",
                "--mode",
                "original",
                "--post-tgvf-forward-mode",
                "kv_cache",
                "--population-id",
                "vstar_test_questions_191",
                "--manifest-path",
                str(manifest_path),
                "--manifest-hash",
                "toyhash",
                "--benchmark-root",
                str(root),
                "--output-dir",
                str(output_dir),
                "--execute",
                "--runner-backend",
                "dry_run",
            ]
        )
        == 0
    )
    rows = (output_dir / "rows.jsonl").read_text().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["raw_output"] == "B"
    assert row["score"] == 1.0
    assert row["scorer_name"] == "project_choice_exact_match"
    assert row["official_tool_used"] is False
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["accuracy"] == 1.0
    assert summary["runner_backend"]["backend"] == "dry_run"


def test_benchmark_execute_dry_run_auto_uses_blink_official_choice(tmp_path) -> None:
    manifest = {
        "manifest_id": "blink_toy",
        "manifest_hash": "blinkhash",
        "samples": [
            {
                "sample_id": "blink/sample/0",
                "benchmark": "blink",
                "population_id": "blink_val_all_subtasks_1901",
                "source_file": "blink/snapshot/Counting/val-00000-of-00001.parquet",
                "metadata": {"row_index": 0, "raw_id": "0"},
            }
        ],
    }
    manifest_path = tmp_path / "blink_manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")

    from revisit_vlm_clean.benchmark_data import BenchmarkSample
    from revisit_vlm_clean.rendering import render_benchmark_inputs
    from revisit_vlm_clean.runner import BackendConfig, run_benchmark_rows
    from revisit_vlm_clean.schema import EvalMode, ForwardMode, RunConfig

    sample = BenchmarkSample(
        sample_id="blink/sample/0",
        benchmark="blink",
        population_id="blink_val_all_subtasks_1901",
        source_file="blink/snapshot/Counting/val-00000-of-00001.parquet",
        question="Count the circles.\n(A) one\n(B) two",
        media=({"kind": "path", "path": "/tmp/nonexistent.jpg", "exists": False},),
        choices=("one", "two"),
        gold_answer="B",
        metadata={"row_index": 0},
    )
    config = RunConfig(
        run_id="blink",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.ORIGINAL,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        population_id="blink_val_all_subtasks_1901",
        manifest_path=str(manifest_path),
        manifest_hash="blinkhash",
    )
    rows, summary = run_benchmark_rows(
        [sample],
        render_benchmark_inputs([sample], config),
        config=config,
        backend_config=BackendConfig(backend="dry_run"),
    )

    assert rows[0]["benchmark"] == "blink"
    assert rows[0]["scorer_name"] == "official_blink_exact_match"
    assert rows[0]["official_tool_used"] is True
    assert summary.accuracy == 1.0


def test_benchmark_execute_dry_run_uses_ocrbench_official_batch_scorer(tmp_path) -> None:
    root = tmp_path / "benchmarks"
    eval_path = _write_fake_ocrbench_official(root)

    from revisit_vlm_clean.benchmark_data import BenchmarkSample
    from revisit_vlm_clean.rendering import render_benchmark_inputs
    from revisit_vlm_clean.runner import BackendConfig, run_benchmark_rows
    from revisit_vlm_clean.schema import (
        EvalMode,
        ForwardMode,
        ParserScorerIdentity,
        RunConfig,
        ScoringBackend,
    )

    sample = BenchmarkSample(
        sample_id="ocrbench_v2/sample/0",
        benchmark="ocrbench_v2",
        population_id="ocrbench_v2_data_test_10000",
        source_file="ocrbench_v2/snapshot/toy.jsonl",
        question="Read the word.",
        choices=(),
        gold_answer="blue",
        metadata={"type": "text recognition en", "answers": ["blue"]},
    )
    config = RunConfig(
        run_id="ocrbench",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.ORIGINAL,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        population_id="ocrbench_v2_data_test_10000",
        benchmark_root=str(root),
        parser_scorer=ParserScorerIdentity(
            scoring_backend=ScoringBackend.OFFICIAL,
            fallback_allowed=False,
        ),
    )
    rows, summary = run_benchmark_rows(
        [sample],
        render_benchmark_inputs([sample], config),
        config=config,
        backend_config=BackendConfig(backend="dry_run"),
    )

    assert rows[0]["raw_output"] == "blue"
    assert rows[0]["score"] == 1.0
    assert rows[0]["scorer_name"] == "official_ocrbench_v2"
    assert rows[0]["official_tool_used"] is True
    assert rows[0]["official_tool_path"] == str(eval_path)
    assert summary.accuracy == 1.0
