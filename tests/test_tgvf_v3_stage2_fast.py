from __future__ import annotations

import torch

from revisit_vlm.qwen3_vl_tgvf import NEED_LOCAL_EVIDENCE
from revisit_vlm.tgvf_v3_stage2 import TGVFv3Stage2Sample
from revisit_vlm.tgvf_v3_stage2_fast import (
    _BaseItem,
    _FocusPrepared,
    _batched_focus_deepstack_inputs,
)


def test_batched_focus_deepstack_inputs_merge_original_and_d_by_position() -> None:
    sample = TGVFv3Stage2Sample(
        image="/tmp/image.jpg",
        question="What is shown?",
        answer="A",
        need_focus=True,
        evidence_state=NEED_LOCAL_EVIDENCE,
        trajectory_type="single_focus",
        target="mark",
        evidence_description="The mark is visible.",
    )
    base = _BaseItem(
        sample=sample,
        input_ids=torch.zeros((1, 6), dtype=torch.long),
        attention_mask=torch.ones((1, 6), dtype=torch.long),
        model_inputs={
            "_deepstack_visual_embeds": [
                torch.tensor([[10.0], [20.0]]),
                torch.tensor([[100.0], [200.0]]),
            ]
        },
        image_grid_thw=torch.tensor([[1, 1, 2]], dtype=torch.long),
        image_token_indices=torch.tensor([1, 4], dtype=torch.long),
        visual_pre_count=8,
        visual_merge_count=2,
    )
    prepared = _FocusPrepared(
        sample=sample,
        item=base,
        action_ids=torch.zeros((1, 1), dtype=torch.long),
        action_weights=torch.ones((1, 1), dtype=torch.float32),
        target_start=0,
        target_end=1,
        target_hidden_states=torch.zeros((1, 2)),
        pre_merge_visual_tokens=torch.zeros((8, 1)),
        merged_visual_tokens=torch.zeros((2, 1)),
        foveated_visual_tokens=torch.zeros((1, 1)),
        value_span_matched=False,
        final_input_ids=torch.zeros((1, 6), dtype=torch.long),
        final_labels=torch.zeros((1, 6), dtype=torch.long),
        final_weights=torch.ones((1, 6), dtype=torch.float32),
        final_inputs_embeds=torch.zeros((1, 6, 1)),
        final_attention_mask_2d=torch.ones((1, 6), dtype=torch.long),
        final_attention_mask=torch.zeros((1, 1, 6, 6)),
        final_position_ids=torch.zeros((3, 1, 6), dtype=torch.long),
        final_mm_token_type_ids=torch.zeros((1, 6), dtype=torch.long),
        d_token_indices=torch.tensor([3], dtype=torch.long),
        d_deepstack_visual_embeds=[
            torch.tensor([[30.0]]),
            torch.tensor([[300.0]]),
        ],
        masked_image_key_count=2,
        image_key_mask_active=True,
        mask_mode="weak_strict_original_image_keys_4d",
        fvt_shape=[1, 1],
        target_hidden_shape=[1, 2],
    )

    payload = _batched_focus_deepstack_inputs(
        items=[prepared],
        max_len=6,
        device="cpu",
        dtype=torch.float32,
    )

    assert payload["visual_pos_masks"].tolist() == [[False, True, False, True, True, False]]
    assert payload["deepstack_visual_embeds"][0].view(-1).tolist() == [10.0, 30.0, 20.0]
    assert payload["deepstack_visual_embeds"][1].view(-1).tolist() == [100.0, 300.0, 200.0]
