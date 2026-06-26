import json

import pytest
from revisit_vlm_clean.benchmark_data import materialize_samples_from_manifest_path
from revisit_vlm_clean.cli.benchmark import main as benchmark_main
from revisit_vlm_clean.cli.merge_benchmark import main as merge_main
from revisit_vlm_clean.schema import RunConfig


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


def _write_two_toy_vstar(root):
    snapshot = root / "vstar_bench" / "snapshot"
    image_dir = snapshot / "direct_attributes"
    image_dir.mkdir(parents=True)
    (image_dir / "toy0.jpg").write_bytes(b"not-a-real-image")
    (image_dir / "toy1.jpg").write_bytes(b"not-a-real-image")
    source = snapshot / "test_questions.jsonl"
    rows = [
        {
            "image": "direct_attributes/toy0.jpg",
            "text": "What color is toy 0?\n(A) red\n(B) blue",
            "category": "direct_attributes",
            "question_id": "toy-0",
            "label": "A",
        },
        {
            "image": "direct_attributes/toy1.jpg",
            "text": "What color is toy 1?\n(A) red\n(B) blue",
            "category": "direct_attributes",
            "question_id": "toy-1",
            "label": "B",
        },
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return source


def _write_two_toy_manifest(path):
    samples = []
    for row_index, label in enumerate(("A", "B")):
        samples.append(
            {
                "sample_id": f"vstar_test_questions_191/toy/toy_{row_index:06d}",
                "benchmark": "vstar_bench",
                "population_id": "vstar_test_questions_191",
                "source_file": "vstar_bench/snapshot/test_questions.jsonl",
                "metadata": {
                    "row_index": row_index,
                    "raw_id": f"toy-{row_index}",
                    "question_id": f"toy-{row_index}",
                    "category": "direct_attributes",
                    "label": label,
                },
            }
        )
    payload = {
        "manifest_id": "two_toy_manifest",
        "manifest_hash": "twohash",
        "samples": samples,
        "source_population_ids": ["vstar_test_questions_191"],
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
        row["score"] = (
            1.0
            if str(row.get("predict") or "") in {str(item) for item in answers}
            else 0.0
        )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle)
""".strip()
        + "\n"
    )
    return eval_path


def _write_fake_mmmu_pro_official(root):
    eval_path = root / "mmmu_pro" / "official_code" / "mmmu-pro" / "evaluate.py"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text(
        """
def get_multi_choice_info(options):
    letters = []
    index2ans = {}
    for index, option in enumerate(options):
        letter = chr(ord("A") + index)
        letters.append(letter)
        index2ans[letter] = option
    return index2ans, letters


def parse_multi_choice_response(response, all_choices, index2ans):
    for choice in all_choices:
        if f"({choice})" in response or response.strip() == choice:
            return choice
    for choice, answer in index2ans.items():
        if str(answer).lower() in str(response).lower():
            return choice
    return all_choices[0]


def eval_multi_choice(gold_i, pred_i):
    return gold_i == pred_i
""".strip()
        + "\n"
    )
    return eval_path


def _write_fake_mathvista_official(root):
    eval_path = root / "mathvista" / "official_code" / "evaluation" / "calculate_score.py"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text("# fake MathVista official scorer path for clean tests\n")
    return eval_path


def _write_fake_mathverse_official(root):
    eval_path = root / "mathverse" / "official_code" / "evaluation" / "score_answer_s2.py"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text("# fake MathVerse official scorer path for clean tests\n")
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


def test_benchmark_manifest_subset_id_mismatch_fails(tmp_path) -> None:
    root = tmp_path / "benchmarks"
    _write_toy_vstar(root)
    manifest_path = _write_toy_manifest(tmp_path / "manifest.json")

    with pytest.raises(ValueError, match="manifest id mismatch"):
        benchmark_main(
            [
                "--run-id",
                "bad_subset",
                "--checkpoint-path",
                "outputs/checkpoint.pt",
                "--mode",
                "tgvf_force",
                "--post-tgvf-forward-mode",
                "kv_cache",
                "--subset-id",
                "core_smoke_256_seed20260625",
                "--manifest-path",
                str(manifest_path),
                "--manifest-hash",
                "toyhash",
                "--benchmark-root",
                str(root),
                "--output-dir",
                str(tmp_path / "out"),
                "--materialize-samples",
            ]
        )


def test_benchmark_manifest_population_id_mismatch_fails(tmp_path) -> None:
    root = tmp_path / "benchmarks"
    _write_toy_vstar(root)
    manifest_path = _write_toy_manifest(tmp_path / "manifest.json")

    with pytest.raises(ValueError, match="manifest population mismatch"):
        benchmark_main(
            [
                "--run-id",
                "bad_population",
                "--checkpoint-path",
                "outputs/checkpoint.pt",
                "--mode",
                "tgvf_force",
                "--post-tgvf-forward-mode",
                "kv_cache",
                "--population-id",
                "hr_bench_4k_800",
                "--manifest-path",
                str(manifest_path),
                "--manifest-hash",
                "toyhash",
                "--benchmark-root",
                str(root),
                "--output-dir",
                str(tmp_path / "out"),
                "--materialize-samples",
            ]
        )


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
    run_config = json.loads((output_dir / "run_config.json").read_text())
    assert run_config["output_schema_version"] == "clean_benchmark_run_v1"
    assert run_config["started_at"].endswith("+00:00")
    assert run_config["benchmark_source_manifest"]["artifact_path"] == "benchmark_sources.json"
    assert run_config["benchmark_source_manifest"]["source_file_count"] == 1
    assert run_config["benchmark_source_manifest"]["sample_count"] == 1
    assert run_config["benchmark_source_manifest"]["all_files_exist"] is True
    assert (
        run_config["parser_scorer"]["model_output_parser"]
        == "revisit_vlm_clean.scoring.parse_and_score:v3_external"
    )
    assert (
        run_config["parser_scorer"]["choice_parser"]
        == "revisit_vlm_clean.scoring.extract_choice_strict"
    )
    assert "started_at:" in (output_dir / "run_config.txt").read_text()
    row = json.loads(rows[0])
    assert row["raw_output"] == "B"
    assert row["final_output"] == "B"
    assert row["score"] == 1.0
    assert row["scorer_name"] == "project_choice_exact_match"
    assert row["official_tool_used"] is False
    assert row["eval_family"] == "project_native_external"
    assert row["num_shards"] == 1
    assert row["shard_index"] == 0
    assert row["tgvf_protocol"] == "protocol_c_tool_observation"
    assert row["post_tgvf_continuation"] == "natural_continue"
    assert row["post_tgvf_forward_mode"] == "kv_cache"
    assert row["deepstack"] == {
        "d_features_enabled": False,
        "enabled": False,
        "original_image_scope": "off",
    }
    assert row["deepstack_execution"] == {
        "backend": "dry_run",
        "deepstack_caution": None,
        "execution_supported_for_requested_state": True,
        "fvt_append_path": None,
        "fvt_append_uses_deepstack": None,
        "fvt_position_mode": None,
        "notes": [],
        "requested": {
            "d_features_enabled": False,
            "enabled": False,
            "original_image_scope": "off",
        },
        "requested_enabled": False,
        "requested_scope": "off",
        "resolved_backend": "dry_run",
        "schema_version": "clean_deepstack_execution_row_v1",
    }
    assert row["parser_scorer"]["scoring_backend"] == "auto"
    assert (
        row["parser_scorer"]["model_output_parser"]
        == "revisit_vlm_clean.scoring.parse_and_score:v3_external"
    )
    assert (
        row["parser_scorer"]["choice_parser"]
        == "revisit_vlm_clean.scoring.extract_choice_strict"
    )
    assert row["d_condition"] is None
    assert row["trigger_policy"] == {
        "allows_no_focus": True,
        "mode": "original",
        "policy": "none_original",
        "requires_focus": False,
        "schema_version": "clean_benchmark_trigger_policy_v1",
        "softforce_prompt_text": "",
    }
    assert row["d_shape"] is None
    assert row["continuation_metadata"]["schema_version"] == (
        "clean_benchmark_continuation_metadata_v1"
    )
    assert row["continuation_metadata"]["post_tgvf_continuation"] == "natural_continue"
    assert row["continuation_metadata"]["post_tgvf_forward_mode"] == "kv_cache"
    assert row["continuation_metadata"]["final_output_source"] == "direct_or_capture_output"
    assert row["continuation_metadata"]["append_success"] is None
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["accuracy"] == 1.0
    assert summary["runner_backend"]["backend"] == "dry_run"
    assert summary["deepstack_execution"] == {
        "all_reported_fvt_append_uses_deepstack": None,
        "any_fvt_append_uses_deepstack": False,
        "deepstack_caution_rows": 0,
        "fvt_append_reported_rows": 0,
        "fvt_append_uses_deepstack_rows": 0,
        "n_rows": 1,
        "requested_enabled_rows": 0,
        "schema_version": "clean_deepstack_execution_summary_v1",
        "unsupported_requested_rows": 0,
    }
    assert (
        summary["parser_scorer"]["model_output_parser"]
        == "revisit_vlm_clean.scoring.parse_and_score:v3_external"
    )
    verification = summary["manifest_verification"]
    assert verification["schema_version"] == "clean_benchmark_manifest_verification_v1"
    assert verification["manifest_hash"] == "toyhash"
    assert verification["sample_manifest_sample_count"] == 1
    assert verification["rows_count"] == 1
    assert verification["row_count_matches_sample_manifest"] is True
    assert verification["row_sample_id_order_matches_manifest"] is True
    assert verification["source_manifest_matches_sample_manifest"] is True
    assert verification["all_source_files_exist"] is True
    comparability = summary["comparability"]
    assert comparability["schema_version"] == "clean_benchmark_comparability_v1"
    assert comparability["comparable"] is True
    assert comparability["clean_core"] is True
    assert comparability["clean_core_population_ids"] == ["vstar_test_questions_191"]
    assert comparability["subset_run"] is False
    assert comparability["population_run"] is True
    assert comparability["population_expected_n"] == 191
    assert comparability["population_expected_n_matches"] is False
    assert comparability["side_result"] is False
    assert comparability["invalid_for_baseline"] is False
    assert comparability["comparison_scope"] == "clean_core_population_subset"
    breakdowns = summary["result_breakdowns"]
    assert breakdowns["schema_version"] == "clean_benchmark_result_breakdowns_v1"
    assert breakdowns["by_benchmark"]["vstar_bench"]["accuracy"] == 1.0
    assert breakdowns["by_population_id"]["vstar_test_questions_191"]["n_rows"] == 1
    assert breakdowns["by_method"]["original"]["n_scored"] == 1
    assert breakdowns["by_d_condition"]["none"]["n_rows"] == 1
    assert breakdowns["choice_counts"]["prediction_counts"] == {"B": 1}
    assert breakdowns["choice_counts"]["gold_counts"] == {"B": 1}
    assert breakdowns["official_scoring"]["scorer_names"] == {"project_choice_exact_match": 1}
    assert breakdowns["official_scoring"]["official_tool_used_rows"] == 0
    benchmark_sources = json.loads((output_dir / "benchmark_sources.json").read_text())
    assert benchmark_sources["schema_version"] == "clean_benchmark_source_manifest_v1"
    assert benchmark_sources["manifest_hash"] == "toyhash"
    assert benchmark_sources["source_file_count"] == 1
    assert benchmark_sources["sample_count"] == 1
    assert benchmark_sources["all_files_exist"] is True
    source_file = benchmark_sources["source_files"][0]
    assert source_file["source_file"] == "vstar_bench/snapshot/test_questions.jsonl"
    assert source_file["sample_count"] == 1
    assert source_file["row_indices"] == [0]
    assert len(source_file["sha256"]) == 64


def test_benchmark_execute_dry_run_shards_manifest_deterministically(tmp_path) -> None:
    root = tmp_path / "benchmarks"
    _write_two_toy_vstar(root)
    manifest_path = _write_two_toy_manifest(tmp_path / "manifest.json")
    output_dir = tmp_path / "exec_shard"

    assert (
        benchmark_main(
            [
                "--run-id",
                "exec_shard",
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
                "twohash",
                "--num-shards",
                "2",
                "--shard-index",
                "1",
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

    rows = [json.loads(line) for line in (output_dir / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "vstar_test_questions_191/toy/toy_000001"
    assert rows[0]["gold_answer"] == "B"
    assert rows[0]["num_shards"] == 2
    assert rows[0]["shard_index"] == 1

    sample_manifest = json.loads((output_dir / "sample_manifest.json").read_text())
    assert sample_manifest["manifest_id"] == "two_toy_manifest__shard_1_of_2"
    assert sample_manifest["source_manifest_hash"] == "twohash"
    assert sample_manifest["num_shards"] == 2
    assert sample_manifest["shard_index"] == 1
    assert len(sample_manifest["samples"]) == 1
    run_config = json.loads((output_dir / "run_config.json").read_text())
    assert run_config["manifest_hash"] == sample_manifest["manifest_hash"]
    assert run_config["num_shards"] == 2
    assert run_config["shard_index"] == 1
    benchmark_sources = json.loads((output_dir / "benchmark_sources.json").read_text())
    assert benchmark_sources["manifest_hash"] == sample_manifest["manifest_hash"]
    assert benchmark_sources["source_manifest_hash"] == "twohash"
    assert benchmark_sources["sample_count"] == 1
    assert benchmark_sources["source_files"][0]["row_indices"] == [1]
    assert run_config["benchmark_source_manifest"]["source_manifest_hash"] == "twohash"
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["manifest_verification"]["source_manifest_hash"] == "twohash"
    assert summary["manifest_verification"]["sample_manifest_sample_count"] == 1
    assert summary["manifest_verification"]["row_sample_id_order_matches_manifest"] is True


def test_merge_benchmark_shards_restores_source_manifest_order(tmp_path) -> None:
    root = tmp_path / "benchmarks"
    _write_two_toy_vstar(root)
    manifest_path = _write_two_toy_manifest(tmp_path / "manifest.json")
    shard_dirs = [tmp_path / "shard0", tmp_path / "shard1"]
    for shard_index, output_dir in enumerate(shard_dirs):
        assert (
            benchmark_main(
                [
                    "--run-id",
                    f"exec_shard_{shard_index}",
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
                    "twohash",
                    "--num-shards",
                    "2",
                    "--shard-index",
                    str(shard_index),
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

    merged = tmp_path / "merged"
    assert (
        merge_main(
            [
                "--output-dir",
                str(merged),
                "--run-id",
                "merged_two_toy",
                "--expected-num-shards",
                "2",
                "--expected-source-manifest-hash",
                "twohash",
                str(shard_dirs[1]),
                str(shard_dirs[0]),
            ]
        )
        == 0
    )

    rows = [json.loads(line) for line in (merged / "rows.jsonl").read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == [
        "vstar_test_questions_191/toy/toy_000000",
        "vstar_test_questions_191/toy/toy_000001",
    ]
    assert [row["gold_answer"] for row in rows] == ["A", "B"]
    summary = json.loads((merged / "summary.json").read_text())
    assert summary["n_rows"] == 2
    assert summary["accuracy"] == 1.0
    assert summary["manifest_hash"] == "twohash"
    assert summary["deepstack_execution"]["n_rows"] == 2
    assert summary["deepstack_execution"]["fvt_append_reported_rows"] == 0
    assert summary["deepstack_execution"]["any_fvt_append_uses_deepstack"] is False
    assert summary["merge_metadata"]["shard_indices"] == [0, 1]
    assert summary["merge_metadata"]["merge_order"] == "source_manifest_order_modulo"
    assert summary["benchmark_source_manifest"]["source_manifest_hash"] == "twohash"
    assert summary["benchmark_source_manifest"]["sample_count"] == 2
    assert (
        summary["parser_scorer"]["model_output_parser"]
        == "revisit_vlm_clean.scoring.parse_and_score:v3_external"
    )
    assert summary["deepstack"] == {
        "d_features_enabled": False,
        "enabled": False,
        "original_image_scope": "off",
    }
    assert summary["post_tgvf_forward_mode"] == "kv_cache"
    assert summary["post_tgvf_continuation"] == "natural_continue"
    assert summary["eval_family"] == "project_native_external"
    assert summary["tgvf_protocol"] == "protocol_c_tool_observation"
    assert summary["runner_backend"] == {
        "alias_targets": [],
        "backend_counts": {"dry_run": 2},
        "deprecated_alias_rows": 0,
        "resolved_backend_counts": {"dry_run": 2},
        "schema_version": "clean_merged_runner_backend_summary_v1",
        "stage2_generic_alias_rows": 0,
    }
    verification = summary["manifest_verification"]
    assert verification["schema_version"] == "clean_benchmark_manifest_verification_v1"
    assert verification["manifest_hash"] == "twohash"
    assert verification["sample_manifest_sample_count"] == 2
    assert verification["rows_count"] == 2
    assert verification["row_count_matches_sample_manifest"] is True
    assert verification["row_sample_id_order_matches_manifest"] is True
    assert verification["source_manifest_matches_sample_manifest"] is True
    assert verification["merged_from_shards"] is True
    assert verification["merge_metadata"]["merge_order"] == "source_manifest_order_modulo"
    comparability = summary["comparability"]
    assert comparability["clean_core"] is True
    assert comparability["population_run"] is True
    assert comparability["population_expected_n_matches"] is False
    assert comparability["merged_from_shards"] is True
    assert comparability["comparison_scope"] == "clean_core_population_subset"
    breakdowns = summary["result_breakdowns"]
    assert breakdowns["by_benchmark"]["vstar_bench"]["n_rows"] == 2
    assert breakdowns["by_method"]["original"]["accuracy"] == 1.0
    assert breakdowns["choice_counts"]["prediction_counts"] == {"A": 1, "B": 1}
    assert breakdowns["choice_counts"]["gold_counts"] == {"A": 1, "B": 1}
    sample_manifest = json.loads((merged / "sample_manifest.json").read_text())
    assert sample_manifest["manifest_hash"] == "twohash"
    assert sample_manifest["merged_from_shards"]["merged_row_count"] == 2
    run_config = RunConfig.from_json((merged / "run_config.json").read_text())
    assert run_config.run_id == "merged_two_toy"
    assert run_config.manifest_hash == "twohash"
    assert run_config.num_shards == 2
    assert run_config.shard_index == 0
    assert run_config.benchmark_source_manifest["source_manifest_hash"] == "twohash"
    assert run_config.benchmark_source_manifest["sample_count"] == 2
    merge_metadata = json.loads((merged / "merge_metadata.json").read_text())
    assert merge_metadata["source_manifest_hash"] == "twohash"
    benchmark_sources = json.loads((merged / "benchmark_sources.json").read_text())
    assert benchmark_sources["manifest_hash"] == "twohash"
    assert benchmark_sources["source_manifest_hash"] == "twohash"
    assert benchmark_sources["sample_count"] == 2
    assert benchmark_sources["source_files"][0]["row_indices"] == [0, 1]


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
    assert rows[0]["runner_backend"] == "dry_run"
    assert rows[0]["resolved_runner_backend"] == "dry_run"
    assert rows[0]["runner_backend_deprecated_alias"] is False
    assert rows[0]["runner_backend_stage2_generic_alias"] is False
    assert rows[0]["runner_backend_alias_target"] is None
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


def test_benchmark_execute_dry_run_uses_mmmu_pro_official_batch_scorer(tmp_path) -> None:
    root = tmp_path / "benchmarks"
    eval_path = _write_fake_mmmu_pro_official(root)

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
        sample_id="mmmu_pro/sample/0",
        benchmark="mmmu_pro",
        population_id="mmmu_pro_standard_10_test_1730",
        source_file="mmmu_pro/snapshot/standard (10 options)/toy.parquet",
        question="Which option is correct?",
        choices=("red", "blue"),
        gold_answer="B",
        metadata={"subject": "Art"},
    )
    config = RunConfig(
        run_id="mmmu",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.ORIGINAL,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        population_id="mmmu_pro_standard_10_test_1730",
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

    assert rows[0]["raw_output"] == "B"
    assert rows[0]["parsed_answer"] == "B"
    assert rows[0]["score"] == 1.0
    assert rows[0]["scorer_name"] == "official_mmmu_pro"
    assert rows[0]["official_tool_used"] is True
    assert rows[0]["official_tool_path"] == str(eval_path)
    assert summary.accuracy == 1.0


def test_benchmark_execute_dry_run_uses_mathvista_official_scorer(tmp_path) -> None:
    root = tmp_path / "benchmarks"
    eval_path = _write_fake_mathvista_official(root)

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
        sample_id="mathvista/sample/0",
        benchmark="mathvista",
        population_id="mathvista_testmini_1000",
        source_file="mathvista/snapshot/data/toy.parquet",
        question="What is the value?",
        choices=(),
        gold_answer="2",
        metadata={"question_type": "free_form", "answer_type": "integer"},
    )
    config = RunConfig(
        run_id="mathvista",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.ORIGINAL,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        population_id="mathvista_testmini_1000",
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

    assert rows[0]["raw_output"] == "2"
    assert rows[0]["parsed_answer"] == "2"
    assert rows[0]["prediction"] == "2"
    assert rows[0]["score"] == 1.0
    assert rows[0]["scorer_name"] == "official_mathvista"
    assert rows[0]["official_tool_path"] == str(eval_path)
    assert rows[0]["llm_judge_used"] is False
    assert summary.accuracy == 1.0


def test_benchmark_execute_dry_run_uses_mathverse_official_scorer(tmp_path) -> None:
    root = tmp_path / "benchmarks"
    eval_path = _write_fake_mathverse_official(root)

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
        sample_id="mathverse/sample/0",
        benchmark="mathverse",
        population_id="mathverse_testmini_3940",
        source_file="mathverse/snapshot/testmini.json",
        question="Select the answer.\nA:40\nB:60\nC:120\nD:140",
        choices=("40", "60", "120", "140"),
        gold_answer="D",
        metadata={"problem_version": "Text Dominant", "question_type": "multi-choice"},
    )
    config = RunConfig(
        run_id="mathverse",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.ORIGINAL,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        population_id="mathverse_testmini_3940",
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

    assert rows[0]["raw_output"] == "D"
    assert rows[0]["parsed_answer"] == "D"
    assert rows[0]["score"] == 1.0
    assert rows[0]["scorer_name"] == "official_mathverse"
    assert rows[0]["official_tool_path"] == str(eval_path)
    assert rows[0]["llm_judge_used"] is False
    assert summary.accuracy == 1.0
