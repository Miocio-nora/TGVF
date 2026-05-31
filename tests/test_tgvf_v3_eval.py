from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.eval_v3_fvt_distribution import distribution_row
from eval.v3_common import (
    V3EvalFeatureCacheItem,
    can_score_fvt_for_item,
    group_indices_by_v3_image,
    same_image_wrong_index_v3,
    v3_group_id,
)
from revisit_vlm.qwen3_vl_tgvf import Qwen3FocusCapture, Qwen3SourceVisualGeometry
from revisit_vlm.tgvf_v3_stage1 import TGVFv3Stage1Sample


def _sample(image_id: str, target: str) -> TGVFv3Stage1Sample:
    return TGVFv3Stage1Sample(
        image="/tmp/image.jpg",
        image_id=image_id,
        question="What is the local detail?",
        target=target,
        evidence_description="The local detail reads ABC.",
        evidence_type="ocr_text",
        source_profile="scene_text",
        target_style="visual_cue",
        target_cues=["location", "text_like"],
        metadata={"answer_type": "text_string", "visual_difficulty": "medium"},
    )


def _item(sample: TGVFv3Stage1Sample, *, source_count: int, d_dim: int, v_dim: int) -> V3EvalFeatureCacheItem:
    geometry = Qwen3SourceVisualGeometry(
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        video_grid_thw=None,
        source_visual_position_ids=torch.zeros((3, source_count), dtype=torch.long),
        source_visual_token_indices=torch.arange(source_count),
        source_visual_token_count=source_count,
        image_token_id=1,
        position_ids_shape=[3, 1, source_count],
        mm_token_type_ids_present=True,
        extraction_mode="test",
        errors=[],
    )
    capture = Qwen3FocusCapture(
        target_text=sample.target,
        target_token_ids=[10, 11],
        target_hidden_states=torch.randn(2, d_dim),
        generated_ids=[10, 11],
        generated_text=f"<FOCUS>{sample.target}</FOCUS>",
        generated_hidden_states=torch.randn(2, d_dim),
        past_key_values=None,
        attention_mask=torch.ones(1, 8, dtype=torch.long),
        cache_position=None,
        input_ids=torch.ones(1, 8, dtype=torch.long),
        last_logits=None,
        model_kwargs={},
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        video_grid_thw=None,
        source_visual_geometry=geometry,
        capture_found=True,
        second_full_forward_used=False,
    )
    return V3EvalFeatureCacheItem(
        sample=sample,
        uid=f"{sample.image_id}:{sample.target}",
        target_hidden_states=torch.randn(2, d_dim),
        pre_merge_visual_tokens=torch.randn(source_count * 4, v_dim),
        merged_visual_tokens=torch.randn(source_count, v_dim),
        foveated_visual_tokens=torch.randn(source_count, d_dim),
        capture=capture,
        shapes={"H_q": [2, d_dim], "V_pre": [source_count * 4, v_dim], "V_merge": [source_count, v_dim], "D": [source_count, d_dim]},
        readout_metadata={"position_mode": "native_source_grid"},
    )


def test_v3_eval_groups_same_image_targets() -> None:
    items = [
        _item(_sample("img-a", "the small text near the top"), source_count=4, d_dim=8, v_dim=8),
        _item(_sample("img-a", "the label near the bottom"), source_count=4, d_dim=8, v_dim=8),
        _item(_sample("img-b", "the number inside the circle"), source_count=4, d_dim=8, v_dim=8),
    ]
    groups = group_indices_by_v3_image(items)
    assert v3_group_id(items[0].sample) == "img-a"
    assert groups["img-a"] == [0, 1]
    assert same_image_wrong_index_v3(groups, items[0], 0) == 1
    assert same_image_wrong_index_v3(groups, items[2], 2) is None


def test_v3_eval_shape_compatibility_uses_source_visual_count() -> None:
    item = _item(_sample("img-a", "the small text near the top"), source_count=4, d_dim=8, v_dim=8)
    assert can_score_fvt_for_item(item, torch.randn(4, 8))
    assert not can_score_fvt_for_item(item, torch.randn(5, 8))


def test_v3_distribution_row_handles_qwen3_dim_mismatch() -> None:
    item = _item(_sample("img-a", "the small text near the top"), source_count=4, d_dim=8, v_dim=6)
    row = distribution_row(item)
    assert row["manifold_active"] is False
    assert row["manifold_loss"] is None
    assert row["mean_mse"] is None
    assert row["norm_ratio_D_to_Vmerge"] is not None
