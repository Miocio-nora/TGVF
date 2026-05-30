from __future__ import annotations

import json
from pathlib import Path

import pytest

from tgvf_eval.adapters import BENCHMARK_NAMES, BenchmarkRegistry, _extract_choices, _extract_media
from tgvf_eval.config import image_budget_kwargs, method_config_from_name, validate_fvt_append_mode, validate_video_foveation_mode
from tgvf_eval.force_ablation import (
    _base_row,
    ablation_spec,
    answer_distribution_report,
    summarize_ablation_rows,
    target_quality_flags,
    validate_force_ablation_mode,
)
from tgvf_eval.parsing import parse_multiple_choice
from tgvf_eval.prompts import build_force_prompt, build_free_prompt
from tgvf_eval.results import ResultWriter, make_run_paths
from tgvf_eval.sampling import deterministic_sample
from tgvf_eval.video_foveation import VideoFoveationController


def test_benchmark_registry_lists_all_local_benchmark_names() -> None:
    assert set(BenchmarkRegistry.names()) == {
        "vstar_bench",
        "hr_bench_4k",
        "ocrbench_v2",
        "blink",
        "mmmu_pro",
        "mathvista",
        "mathverse",
        "ovo_bench",
    }
    assert BENCHMARK_NAMES == BenchmarkRegistry.names()


def test_tier_sampling_returns_deterministic_sample_ids() -> None:
    class Item:
        def __init__(self, sample_id: str, category: str) -> None:
            self.sample_id = sample_id
            self.metadata = {"category": category}

    samples = [Item(f"{index:03d}", "a" if index % 2 == 0 else "b") for index in range(100)]
    first = deterministic_sample(samples, benchmark="vstar_bench", tier="light", limit=10, seed=7)
    second = deterministic_sample(samples, benchmark="vstar_bench", tier="light", limit=10, seed=7)
    assert [item.sample_id for item in first] == [item.sample_id for item in second]
    assert len(first) == 10


def test_prompt_only_force_prompt_contains_required_stop_span() -> None:
    prompt = build_force_prompt("What is shown?")
    assert "<|foveate|>" in prompt
    assert "<|/foveate|>" in prompt
    assert "object or region phrase" in prompt
    assert "Do not answer before foveating" in prompt


def test_prompt_only_free_prompt_allows_direct_answer_or_foveation_span() -> None:
    prompt = build_free_prompt("What is shown?")
    assert "<|foveate|>target<|/foveate|>" in prompt
    assert "answer directly" in prompt


def test_cot_default_is_false_and_enabled_changes_config() -> None:
    default = method_config_from_name("direct_qwen")
    enabled = method_config_from_name("direct_qwen", cot_enabled=True)
    assert default.cot_enabled is False
    assert default.cot_prompt_source == "none"
    assert default.nextframe_reuse_ttl is None
    assert enabled.cot_enabled is True
    assert enabled.cot_prompt_source == "minimal"


def test_image_budget_presets_map_to_qwen_pixel_limits() -> None:
    assert image_budget_kwargs("low") == {"max_pixels": 1280 * 28 * 28}
    assert image_budget_kwargs("mid") == {"max_pixels": 4096 * 28 * 28}
    assert image_budget_kwargs("high") == {"max_pixels": 8192 * 28 * 28}
    with pytest.raises(ValueError):
        image_budget_kwargs("ultra")


def test_qwen_content_builder_strips_video_nframes_from_image_items() -> None:
    from revisit_vlm.tgvf_capture import _image_content, _looks_like_video

    content = _image_content("frame.png", {"nframes": 8, "max_pixels": 4096 * 28 * 28})
    assert _looks_like_video("clip.mp4") is True
    assert content["type"] == "image"
    assert content["image"] == "frame.png"
    assert content["max_pixels"] == 4096 * 28 * 28
    assert "nframes" not in content


def test_forced_foveation_bracket_wrapper_keeps_target_tokens_free() -> None:
    import torch
    from revisit_vlm.tgvf_training import ForcedFoveationBracketWrapper

    class Output:
        def __init__(self) -> None:
            self.logits = torch.zeros(1, 1, 10)
            self.logits[:, -1, 9] = 5.0

    class Model:
        def __call__(self, *args, **kwargs):
            return Output()

        def prepare_inputs_for_generation(self, *args, **kwargs):
            return {}

        def parameters(self):
            return iter(())

    wrapped = ForcedFoveationBracketWrapper(
        Model(),
        start_ids=[1, 2],
        end_ids=[3, 4],
        max_target_tokens=2,
        suppress_ids=[8],
    )
    chosen = []
    for _ in range(6):
        out = wrapped(input_ids=torch.ones(1, 1, dtype=torch.long))
        chosen.append(int(out.logits[:, -1, :].argmax(dim=-1).item()))
    assert chosen == [1, 2, 9, 9, 3, 4]


def test_embedded_base64_images_are_decoded_for_parquet_benchmarks(tmp_path: Path) -> None:
    tiny_png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
        "x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    media = _extract_media({"image": tiny_png}, base_dir=tmp_path, benchmark_dir=tmp_path)
    assert media
    assert not isinstance(media[0], str)
    assert getattr(media[0], "size", None) == (1, 1)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Answer: C", "C"),
        ("The answer is (B).", "B"),
        ("A", "A"),
        ("I choose option D because...", "D"),
    ],
)
def test_multiple_choice_parser_extracts_letters(raw: str, expected: str) -> None:
    assert parse_multiple_choice(raw, valid_options="ABCD") == expected


def test_field_choices_are_extracted_and_option_text_can_parse() -> None:
    from tgvf_eval.parsing import parse_prediction

    record = {"question": "What number is shown?", "A": "27B", "B": "37B", "answer": "A"}
    choices = _extract_choices(record)
    assert choices == ["27B", "37B"]
    assert parse_prediction("The number displayed is 27B.", choices=choices) == "A"


def test_result_writer_resume_skip_works(tmp_path: Path) -> None:
    paths = make_run_paths(tmp_path, run_id="resume_test")
    writer = ResultWriter(paths, benchmark="vstar_bench", method="direct_qwen")
    writer.append_row({"sample_id": "a", "benchmark": "vstar_bench", "method": "direct_qwen"})
    assert writer.completed_ids() == {"a"}


def test_result_writer_serializes_embedded_pil_media(tmp_path: Path) -> None:
    from PIL import Image

    paths = make_run_paths(tmp_path, run_id="pil_media_test")
    writer = ResultWriter(paths, benchmark="hr_bench_4k", method="tgvf_module_force")
    writer.append_row(
        {
            "sample_id": "img",
            "benchmark": "hr_bench_4k",
            "method": "tgvf_module_force",
            "media": [Image.new("RGB", (2, 3))],
        }
    )
    row = json.loads(writer.prediction_path.read_text())
    assert row["media"] == [{"type": "image", "filename": "<embedded_image>", "size": [2, 3]}]


def test_video_foveation_mode_enum_validates() -> None:
    validate_video_foveation_mode("off")
    validate_video_foveation_mode("per_frame_reencode")
    validate_video_foveation_mode("nextframe_encode_no_reencode")
    with pytest.raises(ValueError):
        validate_video_foveation_mode("bad")


def test_nextframe_no_reencode_reuses_hq_without_regenerating_next_frame() -> None:
    generated: list[int] = []

    def target_generator(index: int, frame: str) -> str:
        generated.append(index)
        return f"Hq-{index}"

    encoded: list[tuple[str, str]] = []

    def encoder(hq: str, frame: str) -> str:
        encoded.append((hq, frame))
        return f"{hq}:{frame}"

    controller = VideoFoveationController(
        mode="nextframe_encode_no_reencode",
        nextframe_reuse_ttl=1,
        target_generator=target_generator,
        fvt_encoder=encoder,
    )
    outputs, stats = controller.process_frames(["f0", "f1", "f2"])
    assert generated == [0, 2]
    assert encoded[1] == ("Hq-0", "f1")
    assert outputs == ["Hq-0:f0", "Hq-0:f1", "Hq-2:f2"]
    assert stats.num_reuse_encode_events == 1


def test_per_frame_reencode_regenerates_hq_per_frame() -> None:
    generated: list[int] = []
    controller = VideoFoveationController(
        mode="per_frame_reencode",
        target_generator=lambda index, frame: generated.append(index) or f"Hq-{index}",
        fvt_encoder=lambda hq, frame: (hq, frame),
    )
    _outputs, stats = controller.process_frames(["f0", "f1", "f2"])
    assert generated == [0, 1, 2]
    assert stats.num_reencode_events == 3
    assert stats.num_reuse_encode_events == 0


def test_second_full_forward_is_false_in_module_config() -> None:
    config = method_config_from_name("tgvf_module_free")
    assert config.second_full_forward_allowed is False


def test_official_tool_path_resolution_fails_clearly(tmp_path: Path) -> None:
    root = tmp_path / "benchmarks"
    tools = root / "_tools"
    (root / "vstar_bench").mkdir(parents=True)
    tools.mkdir()
    (tools / "benchmark_manifest.json").write_text(json.dumps([{"name": "vstar_bench"}]))
    adapter = BenchmarkRegistry.get("vstar_bench", benchmark_root=root, tools_root=tools)
    info = adapter.official_tool_info()
    assert info.official_tool_used is False
    assert "no local official scorer" in (info.note or "")


def test_force_target_mode_generated_does_not_flag_fixed_target() -> None:
    flags = target_quality_flags("small text near the top", force_target_mode="generated")
    assert flags["target_was_fixed"] is False
    assert flags["target_is_generic"] is False


def test_target_quality_detector_flags_generic_targets() -> None:
    flags = target_quality_flags("visual target description")
    assert flags["target_is_generic"] is True
    assert flags["target_is_malformed"] is False
    malformed = target_quality_flags("visual target description|What is the color?")
    assert malformed["target_is_generic"] is True
    assert malformed["target_is_malformed"] is True


def test_answer_distribution_report_counts_predicted_and_gold_letters() -> None:
    rows = [
        {"parsed_answer": "A", "gold_answer": "A", "score": 1.0, "category": "attribute"},
        {"parsed_answer": "A", "gold_answer": "B", "score": 0.0, "category": "attribute"},
        {"parsed_answer": "B", "gold_answer": "B", "score": 1.0, "category": "spatial"},
        {"parsed_answer": "", "gold_answer": "C", "score": 0.0, "category": "spatial"},
    ]
    report = answer_distribution_report(rows)
    assert report["predicted_letter_histogram"] == {"A": 2, "B": 1}
    assert report["gold_letter_histogram"] == {"A": 1, "B": 2, "C": 1}
    assert report["correct_by_predicted_letter"]["A"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert report["parse_fail_count"] == 1


def test_force_ablation_config_validates_required_modes() -> None:
    assert validate_force_ablation_mode("force_correct_D_repeat_options") == "force_correct_D_repeat_options"
    assert validate_force_ablation_mode("native_append_compare") == "native_append_compare"
    with pytest.raises(ValueError):
        validate_force_ablation_mode("bad_mode")
    assert ablation_spec("force_random_D_repeat_options").fvt_mode == "random"
    native = ablation_spec("force_correct_D_native_pseudo_image")
    assert native.fvt_append_mode == "qwen_native_pseudo_image"
    legacy = ablation_spec("force_correct_D_legacy_text_positions")
    assert legacy.fvt_append_mode == "legacy_text_positions"


def test_fvt_append_mode_config_validates() -> None:
    validate_fvt_append_mode("qwen_native_pseudo_image")
    validate_fvt_append_mode("legacy_text_positions")
    config = method_config_from_name("tgvf_module_force", fvt_append_mode="legacy_text_positions")
    assert config.fvt_append_mode == "legacy_text_positions"
    with pytest.raises(ValueError):
        method_config_from_name("tgvf_module_force", fvt_append_mode="bad")


def test_second_full_forward_false_in_mocked_force_ablation_summary() -> None:
    rows = [
        {
            "score": 1.0,
            "raw_output": "A<|im_end|>",
            "parsed_answer": "A",
            "immediate_im_end": False,
            "triggered": True,
            "second_full_forward_used": False,
            "cache_preserved": True,
            "used_logits_source": "after_append",
            "debug_metadata": {"append_mode": "new_user_turn"},
            "wall_time_sec": 0.1,
        }
    ]
    summary = summarize_ablation_rows(
        rows,
        run_id="mock",
        ablation_name="force_correct_D_repeat_options",
        benchmark="vstar_bench",
    )
    assert summary["second_full_forward_used_count"] == 0
    assert summary["used_logits_source_counts"] == {"after_append": 1}


def test_per_sample_ablation_record_includes_target_and_repeat_options() -> None:
    from tgvf_eval.adapters import BenchmarkSample

    sample = BenchmarkSample(
        benchmark="vstar_bench",
        sample_id="s1",
        question="What color?\n(A) red\n(B) blue",
        choices=["red", "blue"],
        gold_answer="A",
        metadata={"category": "attribute"},
    )
    row = _base_row(sample, ablation_spec("force_correct_D_repeat_options"))
    assert row["ablation"] == "force_correct_D_repeat_options"
    assert row["repeat_options_in_continuation"] is True
    assert row["options"] == {"A": "red", "B": "blue"}
