import torch

from revisit_vlm.tgvf_v3_stage1 import build_weak_strict_attention_mask
from revisit_vlm.tgvf_v3_stage2 import original_image_mask_block_query_end, sample_original_image_mask_active


def test_stage2_evidence_only_mask_reopens_original_image_keys_at_answer() -> None:
    attention_2d = torch.ones((1, 8), dtype=torch.long)
    original_image_keys = torch.tensor([1, 2], dtype=torch.long)

    mask = build_weak_strict_attention_mask(
        attention_mask_2d=attention_2d,
        original_image_token_indices=original_image_keys,
        block_query_start=4,
        block_query_end=6,
        dtype=torch.float32,
    )

    assert mask[0, 0, 3, 1].item() == 0.0
    assert mask[0, 0, 4, 1].item() < 0.0
    assert mask[0, 0, 5, 2].item() < 0.0
    assert mask[0, 0, 6, 1].item() == 0.0


def test_stage2_original_image_mask_probability_extremes() -> None:
    assert sample_original_image_mask_active(enabled=True, probability=1.0, device="cpu")
    assert not sample_original_image_mask_active(enabled=True, probability=0.0, device="cpu")
    assert not sample_original_image_mask_active(enabled=False, probability=1.0, device="cpu")


def test_stage2_through_answer_scope_keeps_old_mask_behavior() -> None:
    assert original_image_mask_block_query_end(answer_query_start=6, scope="through_answer") is None
    assert original_image_mask_block_query_end(answer_query_start=6, scope="evidence_only") == 6
