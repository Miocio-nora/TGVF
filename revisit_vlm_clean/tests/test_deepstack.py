import pytest
import torch
from revisit_vlm_clean.deepstack import (
    build_cached_chunk_original_image_key_block_attention_mask,
    build_original_image_key_block_attention_mask,
    build_qwen3_deepstack_payload,
    build_qwen3_original_image_deepstack_payload,
    build_single_query_original_image_key_block_attention_mask,
)


def test_qwen3_original_image_deepstack_payload_masks_only_original_tokens() -> None:
    features = [torch.ones(2, 4), torch.full((2, 4), 2.0)]
    payload = build_qwen3_original_image_deepstack_payload(
        sequence_length=7,
        original_image_token_indices=torch.tensor([1, 3]),
        deepstack_features=features,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert payload.visual_pos_masks.tolist() == [[False, True, False, True, False, False, False]]
    assert [list(item.shape) for item in payload.deepstack_visual_embeds] == [[2, 4], [2, 4]]
    assert payload.to_debug_dict()["visual_pos_mask_policy"] == "original_image_tokens_only"
    assert payload.to_debug_dict()["d_deepstack_features_enabled"] is False


def test_qwen3_original_image_deepstack_payload_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="token count mismatch"):
        build_qwen3_original_image_deepstack_payload(
            sequence_length=7,
            original_image_token_indices=torch.tensor([1, 3]),
            deepstack_features=[torch.ones(3, 4)],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_qwen3_deepstack_payload_combines_original_and_d_in_position_order() -> None:
    original = [torch.tensor([[10.0], [30.0]])]
    d_features = [torch.tensor([[20.0], [50.0]])]

    payload = build_qwen3_deepstack_payload(
        sequence_length=7,
        original_image_token_indices=torch.tensor([1, 3]),
        original_deepstack_features=original,
        d_token_indices=torch.tensor([2, 5]),
        d_deepstack_features=d_features,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert payload.visual_pos_masks.tolist() == [[False, True, True, True, False, True, False]]
    assert payload.deepstack_visual_embeds[0].view(-1).tolist() == [10.0, 20.0, 30.0, 50.0]
    debug = payload.to_debug_dict()
    assert debug["d_deepstack_features_enabled"] is True
    assert debug["d_token_count"] == 2
    assert debug["combined_token_count"] == 4


def test_original_image_key_block_attention_mask_blocks_query_span() -> None:
    mask = build_original_image_key_block_attention_mask(
        attention_mask_2d=torch.ones(1, 6, dtype=torch.long),
        original_image_token_indices=torch.tensor([1, 2]),
        block_query_start=4,
        dtype=torch.float32,
    )

    assert list(mask.shape) == [1, 1, 6, 6]
    blocked = torch.finfo(torch.float32).min
    assert mask[0, 0, 4, 1].item() == blocked
    assert mask[0, 0, 5, 2].item() == blocked
    assert mask[0, 0, 3, 1].item() == 0.0
    assert mask[0, 0, 0, 1].item() == blocked


def test_single_query_original_image_key_block_attention_mask_blocks_original_keys() -> None:
    mask = build_single_query_original_image_key_block_attention_mask(
        attention_mask_2d=torch.ones(1, 5, dtype=torch.long),
        original_image_token_indices=torch.tensor([0, 2]),
        dtype=torch.float32,
    )

    assert list(mask.shape) == [1, 1, 1, 5]
    blocked = torch.finfo(torch.float32).min
    assert mask[0, 0, 0, 0].item() == blocked
    assert mask[0, 0, 0, 2].item() == blocked
    assert mask[0, 0, 0, 4].item() == 0.0


def test_cached_chunk_original_image_key_block_attention_mask_is_causal() -> None:
    mask = build_cached_chunk_original_image_key_block_attention_mask(
        attention_mask_2d=torch.ones(1, 6, dtype=torch.long),
        original_image_token_indices=torch.tensor([1]),
        query_start=3,
        query_length=3,
        dtype=torch.float32,
    )

    assert list(mask.shape) == [1, 1, 3, 6]
    blocked = torch.finfo(torch.float32).min
    assert mask[0, 0, 0, 1].item() == blocked
    assert mask[0, 0, 1, 1].item() == blocked
    assert mask[0, 0, 2, 1].item() == blocked
    assert mask[0, 0, 0, 4].item() == blocked
    assert mask[0, 0, 1, 4].item() == 0.0
    assert mask[0, 0, 2, 5].item() == 0.0


def test_cached_chunk_original_image_key_block_can_skip_tail_queries() -> None:
    mask = build_cached_chunk_original_image_key_block_attention_mask(
        attention_mask_2d=torch.ones(1, 6, dtype=torch.long),
        original_image_token_indices=torch.tensor([1]),
        query_start=3,
        query_length=3,
        block_query_offset=1,
        dtype=torch.float32,
    )

    blocked = torch.finfo(torch.float32).min
    assert mask[0, 0, 0, 1].item() == 0.0
    assert mask[0, 0, 1, 1].item() == blocked
    assert mask[0, 0, 2, 1].item() == blocked
    assert mask[0, 0, 0, 4].item() == blocked
    assert mask[0, 0, 1, 4].item() == 0.0
