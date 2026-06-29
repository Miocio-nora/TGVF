import pytest
from revisit_vlm_clean.schema import (
    DeepStackScope,
    DeepStackState,
    EvalMode,
    ForwardMode,
    RunConfig,
)


def test_run_config_json_roundtrip() -> None:
    config = RunConfig(
        run_id="smoke",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.TGVF_FORCE,
        subset_id="core_smoke_256_seed20260625",
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
    )
    restored = RunConfig.from_json(config.to_json())
    assert restored == config
    assert restored.benchmark_root == "/home/dredvpn009/Flash_Storage/datasets/benchmarks"
    assert restored.output_schema_version == "clean_benchmark_run_v1"
    assert restored.started_at is None
    assert restored.num_shards == 1
    assert restored.shard_index == 0
    assert restored.execution_backend is None
    assert (
        restored.parser_scorer.model_output_parser
        == "revisit_vlm_clean.scoring.parse_and_score:v3_external"
    )
    assert (
        restored.parser_scorer.choice_parser
        == "revisit_vlm_clean.scoring.extract_choice_strict"
    )


def test_free_mode_rejects_prompt_suffix() -> None:
    config = RunConfig(
        run_id="bad",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.TGVF_FREE,
        subset_id="core_smoke_256_seed20260625",
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        prompt_suffix="use focus tool",
    )
    with pytest.raises(ValueError, match="must not include prompt"):
        config.validate()


def test_run_config_rejects_invalid_shard_identity() -> None:
    config = RunConfig(
        run_id="bad_shard",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.ORIGINAL,
        subset_id="core_smoke_256_seed20260625",
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        num_shards=2,
        shard_index=2,
    )
    with pytest.raises(ValueError, match="shard_index"):
        config.validate()


def test_deepstack_enabled_requires_scope() -> None:
    with pytest.raises(ValueError, match="requires a non-off"):
        DeepStackState(enabled=True, original_image_scope=DeepStackScope.OFF).validate()


def test_deepstack_enabled_accepts_no_block_scope() -> None:
    DeepStackState(enabled=True, original_image_scope=DeepStackScope.NO_BLOCK).validate()


def test_deepstack_disabled_rejects_scope() -> None:
    with pytest.raises(ValueError, match="disabled DeepStack"):
        DeepStackState(enabled=False, original_image_scope=DeepStackScope.THROUGH_ANSWER).validate()
    with pytest.raises(ValueError, match="disabled DeepStack"):
        DeepStackState(enabled=False, original_image_scope=DeepStackScope.NO_BLOCK).validate()
