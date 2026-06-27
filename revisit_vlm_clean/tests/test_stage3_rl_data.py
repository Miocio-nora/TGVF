import json
from pathlib import Path

from revisit_vlm_clean.cli.generate_data import main as generate_data_main
from revisit_vlm_clean.stage3_rl_data.api_runner import archive_teacher_outputs, prepare_teacher_batch_files
from revisit_vlm_clean.stage3_rl_data.pipeline import load_exclusion_manifests
from revisit_vlm_clean.stage3_rl_data.schemas import (
    TargetSpec,
    infer_answer_type,
    infer_eval_metric,
    normalize_answer,
    stable_image_uid,
    stable_qa_id,
    stable_sample_id,
    target_is_generic,
    validate_target_spec,
)
from revisit_vlm_clean.stage3_rl_data.sources import _qa_item


def test_stage3_stable_ids_are_deterministic() -> None:
    uid = stable_image_uid("fixture", "img_1", "/tmp/img_1.png")
    qa_id = stable_qa_id("fixture", "row_1", "What color is the sign?", "blue")
    sample_id = stable_sample_id("bundle_1", qa_id)
    assert uid == stable_image_uid("fixture", "img_1", "/elsewhere/img_1.png")
    assert qa_id == stable_qa_id("fixture", "row_1", "What color is the sign?", "blue")
    assert sample_id == stable_sample_id("bundle_1", qa_id)


def test_stage3_target_validators() -> None:
    leaked = TargetSpec(
        target_text="the blue sign above the shop entrance",
        focus_type="attribute",
    )
    assert "target_leakage" in validate_target_spec(leaked, answer="blue")
    assert target_is_generic("the object")
    allowed = TargetSpec(
        target_text="the sign above the shop entrance near the door",
        focus_type="attribute",
    )
    assert validate_target_spec(allowed, answer="blue") == []
    page_count_field = TargetSpec(
        target_text="the subject line following the Re label below the page count",
        focus_type="document_field",
    )
    assert validate_target_spec(page_count_field, answer="Search information") == []
    task_like = TargetSpec(
        target_text="count the people standing near the shop entrance",
        focus_type="counting",
    )
    assert "target_contains_task_verb" in validate_target_spec(task_like, answer="2")
    instruction_like = TargetSpec(
        target_text="the region to read the printed label near the shelf",
        focus_type="ocr",
    )
    assert "target_contains_task_verb" in validate_target_spec(instruction_like, answer="sale")


def test_stage3_answer_metric_sanity() -> None:
    assert normalize_answer("Two.") == "2"
    assert normalize_answer("2") == "2"
    assert infer_answer_type("Yes") == "boolean_state"
    assert infer_answer_type("12", question="How many windows are visible?") == "count"
    assert infer_answer_type("1/8/93") == "date"
    assert infer_answer_type("B", choices=["A", "B"]) == "multiple_choice"
    assert infer_eval_metric("multiple_choice", ["A", "B"]) == "mcq"
    assert infer_eval_metric("date") == "date"


def test_stage3_source_choices_are_preserved_but_prompt_is_open_answer() -> None:
    row = _qa_item(
        source_dataset="fixture",
        source_profile="natural_image",
        source_record_id="mc1",
        question="Which option is higher?",
        answer="B. high",
        aliases=["B. high"],
        choices=["low", "high"],
        evidence_type="attribute",
    )

    assert row["choices"] == []
    assert row["original_choices"] == ["low", "high"]
    assert row["gold_answer"] == "high"
    assert row["answer_format"] == "short_text"
    assert row["answer_type"] == "short_text"
    assert row["eval_metric"] == "normalized_exact_match"
    assert row["metadata"]["choice_to_open_answer"]["original_answer"] == "B. high"


def test_stage3_exclusion_manifest_reads_common_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "exclude.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "stable_image_uid": "fixture:img_a",
                "image": str(tmp_path / "img_a.png"),
                "source": {"source_id": "source-row-1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    exclusion = load_exclusion_manifests([manifest])
    assert "fixture:img_a" in exclusion.stable_image_uids
    assert "source-row-1" in exclusion.source_ids
    assert exclusion.rows == 1


def test_stage3_fixture_cli_plan_execute_extend(tmp_path: Path) -> None:
    dataset_root, source_config, exclude_manifest = _write_fixture_dataset(tmp_path)
    output_root = tmp_path / "stage3_rl" / "v0"
    ledger_path = tmp_path / "EXPERIMENT_LEDGER.md"

    assert (
        generate_data_main(
            [
                "stage3-rl",
                "--run-id",
                "fixture_v0",
                "--dataset-root",
                str(dataset_root),
                "--source-config",
                str(source_config),
                "--exclude-manifest",
                str(exclude_manifest),
                "--output-root",
                str(output_root),
                "--target-accepted-prompts",
                "4",
                "--max-new-images",
                "20",
                "--seed",
                "7",
                "--write-plan",
            ]
        )
        == 0
    )
    plan = output_root / "stage3_rl_data_plan.json"
    assert plan.exists()
    assert generate_data_main(["stage3-rl", "--plan", str(plan), "--preflight-only"]) == 0
    assert generate_data_main(["stage3-rl", "--plan", str(plan), "--dry-run", "--limit-images", "3"]) == 0
    assert (output_root / "dry_run" / "accepted_rl_prompts.jsonl").exists()

    assert (
        generate_data_main(
            [
                "stage3-rl",
                "--plan",
                str(plan),
                "--execute",
                "--experiment-ledger-path",
                str(ledger_path),
            ]
        )
        == 0
    )

    accepted = _read_jsonl(output_root / "accepted_rl_prompts.jsonl")
    rejected = _read_jsonl(output_root / "rejected_rl_prompts.jsonl")
    summary = json.loads((output_root / "manifest_summary.json").read_text())
    assert len(accepted) == 4
    assert summary["accepted_qa"] == 4
    assert all(row["target_spec"] for row in accepted)
    rejected_reasons = {reason for row in rejected for reason in row["rejection_reasons"]}
    assert {"duplicate_question", "target_leakage", "generic_target", "missing_image", "blocked_split"} <= rejected_reasons
    assert not any(row["stable_image_uid"] == "fixture_a:img_excluded" for row in accepted)
    assert "Stage3 RL source-QA pool" in ledger_path.read_text(encoding="utf-8")

    output_root_v1 = tmp_path / "stage3_rl" / "v1"
    assert (
        generate_data_main(
            [
                "stage3-rl",
                "--run-id",
                "fixture_v1",
                "--dataset-root",
                str(dataset_root),
                "--source-config",
                str(source_config),
                "--exclude-manifest",
                str(exclude_manifest),
                "--output-root",
                str(output_root_v1),
                "--extend-from",
                str(output_root),
                "--target-accepted-prompts",
                "6",
                "--max-new-images",
                "20",
                "--seed",
                "8",
                "--write-plan",
            ]
        )
        == 0
    )
    plan_v1 = output_root_v1 / "stage3_rl_data_plan.json"
    assert generate_data_main(["stage3-rl", "--plan", str(plan_v1), "--execute", "--skip-experiment-ledger"]) == 0
    accepted_v1 = _read_jsonl(output_root_v1 / "accepted_rl_prompts.jsonl")
    assert [row["sample_id"] for row in accepted_v1[: len(accepted)]] == [
        row["sample_id"] for row in accepted
    ]
    assert len(accepted_v1) >= len(accepted)


def test_stage3_teacher_triage_dry_run_writes_requests(tmp_path: Path) -> None:
    dataset_root, source_config, exclude_manifest = _write_fixture_dataset(tmp_path)
    output_root = tmp_path / "stage3_rl" / "teacher_triage"

    assert (
        generate_data_main(
            [
                "stage3-rl",
                "--run-id",
                "fixture_teacher_triage",
                "--dataset-root",
                str(dataset_root),
                "--source-config",
                str(source_config),
                "--exclude-manifest",
                str(exclude_manifest),
                "--output-root",
                str(output_root),
                "--target-accepted-prompts",
                "4",
                "--max-new-images",
                "20",
                "--qa-generation-mode",
                "teacher_triage",
                "--teacher-backend",
                "gpt-5.4",
                "--write-plan",
            ]
        )
        == 0
    )
    plan = output_root / "stage3_rl_data_plan.json"
    assert generate_data_main(["stage3-rl", "--plan", str(plan), "--dry-run", "--limit-images", "3"]) == 0

    dry_root = output_root / "dry_run"
    requests = _read_jsonl(dry_root / "teacher_requests.jsonl")
    summary = json.loads((dry_root / "teacher_request_summary.json").read_text(encoding="utf-8"))
    manifest_summary = json.loads((dry_root / "manifest_summary.json").read_text(encoding="utf-8"))

    assert requests
    assert summary["requests"] == len(requests)
    assert summary["source_qa_candidates"] >= len(requests)
    assert manifest_summary["actual_accepted_prompts"] == 0
    assert manifest_summary["teacher_request_summary"]["requests"] == len(requests)
    first = requests[0]
    assert first["schema_version"] == "stage3_rl_teacher_request_v0"
    assert first["teacher_version"] == "stage3_rl_gpt54_triage_v1"
    assert first["model"] == "gpt-5.4"
    assert first["source_qas"]
    assert "source_qa_kept" in first["prompt"]["user"]
    assert "Do not output more than 4 items" in first["prompt"]["user"]
    assert first["response_schema"]["properties"]["items"]["maxItems"] == 4
    assert first["api_payload_template"]["input"][1]["content"][1]["image_url"] == (
        "<FILLED_BY_TEACHER_RUNNER>"
    )


def test_stage3_prepare_api_batch_uses_source_mix_and_cleans_stale_parts(tmp_path: Path) -> None:
    output_root = tmp_path / "stage3_rl" / "api_mix"
    output_root.mkdir(parents=True)
    image_path = output_root / "image.png"
    image_path.write_bytes(_png_bytes())
    (output_root / "stage3_rl_data_plan.json").write_text(
        json.dumps(
            {
                "random_seed": 11,
                "balance_config": {
                    "source_weights": {
                        "visual_genome": 0.40,
                        "textvqa": 0.30,
                        "docvqa": 0.20,
                        "chartqa": 0.10,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for source in ("visual_genome", "textvqa", "docvqa", "chartqa"):
        for index in range(10):
            rows.append(
                {
                    "custom_id": f"{source}_{index}",
                    "request_id": f"req_{source}_{index}",
                    "stable_image_uid": f"{source}:img_{index}",
                    "image_path": str(image_path),
                    "source_dataset": source,
                    "source_profile": "natural_image",
                    "api_payload_template": _api_payload_template(),
                }
            )
    _write_jsonl(output_root / "teacher_requests.jsonl", rows)
    _write_jsonl(output_root / "teacher_outputs.jsonl", [{"custom_id": "visual_genome_0"}])
    archive_report = archive_teacher_outputs(
        output_root=output_root,
        custom_ids=["textvqa_0"],
        archive_name="unit_smoke",
        reason="unit_test_archive",
    )
    assert archive_report["custom_id_count"] == 1
    request_dir = output_root / "api_requests"
    request_dir.mkdir()
    stale_path = request_dir / "batch_requests_part_999.jsonl"
    stale_path.write_text("{}\n", encoding="utf-8")

    report = prepare_teacher_batch_files(output_root=output_root, limit_requests=20)
    index = _read_jsonl(request_dir / "request_index.jsonl")

    assert not stale_path.exists()
    assert report["request_count"] == 20
    assert report["request_file_count"] == 1
    assert report["skipped_archived_requests"] == 1
    assert report["skipped_existing_outputs"] == 1
    assert report["selection_mode"] == "source_mix_stratified_v0"
    assert report["source_distribution"] == {
        "visual_genome": 8,
        "textvqa": 6,
        "docvqa": 4,
        "chartqa": 2,
    }
    assert {row["custom_id"] for row in index}.isdisjoint({"visual_genome_0", "textvqa_0"})
    assert len(_read_jsonl(Path(report["request_files"][0]))) == 20


def _write_fixture_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_root = tmp_path / "datasets"
    image_root = dataset_root / "images"
    image_root.mkdir(parents=True)
    for name in (
        "img_a.png",
        "img_b.png",
        "img_c.png",
        "img_d.png",
        "img_e.png",
        "img_f.png",
        "img_excluded.png",
    ):
        (image_root / name).write_bytes(_png_bytes())

    source_a = dataset_root / "source_a.jsonl"
    source_b = dataset_root / "source_b.jsonl"
    _write_jsonl(
        source_a,
        [
            {
                "id": "a1",
                "image_id": "img_a",
                "image_path": "images/img_a.png",
                "question": "What color is the shop sign?",
                "answer": "blue",
                "target": "the sign above the shop entrance near the door",
                "focus_type": "attribute",
                "split": "train",
            },
            {
                "id": "a2",
                "image_id": "img_a",
                "image_path": "images/img_a.png",
                "question": "What color is the shop sign?",
                "answer": "blue",
                "target": "the sign above the shop entrance near the door",
                "focus_type": "attribute",
                "split": "train",
            },
            {
                "id": "a3",
                "image_id": "img_a",
                "image_path": "images/img_a.png",
                "question": "What color is the awning?",
                "answer": "red",
                "target": "the red awning above the shop entrance",
                "focus_type": "attribute",
                "split": "train",
            },
            {
                "id": "a4",
                "image_id": "img_missing",
                "image_path": "images/missing.png",
                "question": "What word is printed on the tag?",
                "answer": "sale",
                "target": "the printed tag near the lower shelf",
                "focus_type": "ocr",
                "split": "train",
            },
            {
                "id": "a5",
                "image_id": "img_b",
                "image_path": "images/img_b.png",
                "question": "What number is on the door?",
                "answer": "12",
                "target": "the number plaque beside the front door",
                "focus_type": "ocr",
                "split": "val",
            },
            {
                "id": "a6",
                "image_id": "img_b",
                "image_path": "images/img_b.png",
                "question": "What material is the table?",
                "answer": "wood",
                "target": "the object",
                "focus_type": "attribute",
                "split": "train",
            },
            {
                "id": "a7",
                "image_id": "img_b",
                "image_path": "images/img_b.png",
                "question": "How many cups are on the table?",
                "answer": "2",
                "split": "train",
            },
            {
                "id": "a8",
                "image_id": "img_excluded",
                "image_path": "images/img_excluded.png",
                "question": "What color is the excluded sign?",
                "answer": "green",
                "target": "the excluded shop sign above the doorway",
                "focus_type": "attribute",
                "split": "train",
            },
        ],
    )
    _write_jsonl(
        source_b,
        [
            {
                "id": "b1",
                "image_id": "img_c",
                "image_path": "images/img_c.png",
                "question": "What value is marked near 2019?",
                "answer": "42",
                "target": "the line chart point near the 2019 tick",
                "focus_type": "chart_value",
                "split": "train",
            },
            {
                "id": "b2",
                "image_id": "img_c",
                "image_path": "images/img_c.png",
                "question": "Which label is at the top of the legend?",
                "answer": "sales",
                "target": "the legend label at the top of the chart",
                "focus_type": "chart_value",
                "split": "train",
            },
            {
                "id": "b3",
                "image_id": "img_d",
                "image_path": "images/img_d.png",
                "question": "What value is shown for the leftmost bar?",
                "answer": "18",
                "target": "the leftmost bar and its value label",
                "focus_type": "chart_value",
                "split": "train",
            },
            {
                "id": "b4",
                "image_id": "img_e",
                "image_path": "images/img_e.png",
                "question": "What label appears under the second column?",
                "answer": "profit",
                "target": "the second column and its lower axis label",
                "focus_type": "chart_value",
                "split": "train",
            },
            {
                "id": "b5",
                "image_id": "img_f",
                "image_path": "images/img_f.png",
                "question": "What number is printed in the total cell?",
                "answer": "64",
                "target": "the total cell near the bottom of the table",
                "focus_type": "table_cell",
                "split": "train",
            },
        ],
    )

    source_config = dataset_root / "sources.json"
    source_config.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "fixture_a",
                        "adapter": "generic_jsonl",
                        "path": str(source_a),
                        "source_profile": "natural_image",
                        "metadata": {
                            "image_path_field": "image_path",
                            "image_id_field": "image_id",
                            "question_field": "question",
                            "answer_field": "answer",
                            "source_record_id_field": "id",
                            "target_field": "target",
                        },
                    },
                    {
                        "name": "fixture_b",
                        "adapter": "generic_jsonl",
                        "path": str(source_b),
                        "source_profile": "chart",
                        "metadata": {
                            "image_path_field": "image_path",
                            "image_id_field": "image_id",
                            "question_field": "question",
                            "answer_field": "answer",
                            "source_record_id_field": "id",
                            "target_field": "target",
                            "evidence_type": "chart_value",
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    exclude_manifest = dataset_root / "exclude.jsonl"
    exclude_manifest.write_text(
        json.dumps({"stable_image_uid": "fixture_a:img_excluded"}) + "\n",
        encoding="utf-8",
    )
    return dataset_root, source_config, exclude_manifest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x01\x01\x01\x00"
        b"\x18\xdd\x8d\xb0"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _api_payload_template() -> dict:
    return {
        "model": "gpt-5.4",
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": "system"}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "user"},
                    {"type": "input_image", "image_url": "<FILLED_BY_TEACHER_RUNNER>"},
                ],
            },
        ],
    }
