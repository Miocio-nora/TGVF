from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from revisit_vlm.qwen3_vl_tgvf import Qwen3FocusCapture, Qwen3SourceVisualGeometry
from revisit_vlm.tgvf_v3_stage1 import (
    TGVFv3Stage1Dataset,
    build_weak_strict_attention_mask,
    compute_v3_stage1_lm_loss,
    freeze_qwen_backbone,
    prepare_v3_stage1_readout_inputs,
)


class TinyQwen3Tokenizer:
    def __init__(self) -> None:
        self.vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<|vision_start|>": 101,
            "<|image_pad|>": 102,
            "<|vision_end|>": 103,
        }
        self.unk_token_id = 1
        self.eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        tokens = text.replace("\n", " \n ").split()
        ids = []
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 200
            ids.append(self.vocab[token])
        return ids

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)


class TinyQwen3(nn.Module):
    def __init__(self, vocab_size: int = 512, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vision_start_token_id=101,
            image_token_id=102,
            vision_end_token_id=103,
            hidden_size=hidden_size,
        )
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        self.model = SimpleNamespace(compute_3d_position_ids=self.compute_3d_position_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def compute_3d_position_ids(
        self,
        *,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        past_key_values: object | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del inputs_embeds, image_grid_thw, video_grid_thw, past_key_values, mm_token_type_ids
        if attention_mask is None:
            base = torch.arange(input_ids.shape[-1], device=input_ids.device).view(1, -1)
        else:
            base = attention_mask.long().cumsum(-1) - 1
        position_ids = base.view(1, input_ids.shape[0], input_ids.shape[-1]).repeat(3, 1, 1)
        image_mask = input_ids == self.config.image_token_id
        image_positions = torch.nonzero(image_mask[0], as_tuple=False).view(-1)
        for offset, token_index in enumerate(image_positions.tolist()):
            position_ids[:, 0, token_index] = torch.tensor(
                [base[0, token_index], base[0, token_index], base[0, token_index] + offset],
                device=input_ids.device,
            )
        return position_ids

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool = True,
        **_kwargs: object,
    ) -> SimpleNamespace:
        del input_ids, attention_mask, position_ids, image_grid_thw, mm_token_type_ids, labels, return_dict
        hidden = torch.cumsum(inputs_embeds, dim=1)
        return SimpleNamespace(logits=self.head(hidden))


def _capture() -> Qwen3FocusCapture:
    input_ids = torch.tensor([[11, 102, 102, 12, 13, 14]])
    attention_mask = torch.ones_like(input_ids)
    source_positions = torch.tensor([[1, 1], [1, 1], [1, 2]])
    return Qwen3FocusCapture(
        target_text="the small text",
        target_token_ids=[13, 14],
        target_hidden_states=torch.randn(2, 8),
        generated_ids=[13, 14],
        generated_text="<FOCUS>the small text</FOCUS>",
        generated_hidden_states=torch.randn(2, 8),
        past_key_values=None,
        attention_mask=attention_mask,
        cache_position=None,
        input_ids=input_ids,
        last_logits=None,
        model_kwargs={},
        image_grid_thw=torch.tensor([[1, 2, 4]]),
        source_visual_geometry=Qwen3SourceVisualGeometry(
            image_grid_thw=torch.tensor([[1, 2, 4]]),
            video_grid_thw=None,
            source_visual_position_ids=source_positions,
            source_visual_token_indices=torch.tensor([1, 2]),
            source_visual_token_count=2,
            image_token_id=102,
            position_ids_shape=[3, 1, 6],
            mm_token_type_ids_present=True,
            extraction_mode="test",
            errors=[],
        ),
        target_token_start=0,
        target_token_end=2,
        capture_found=True,
    )


def test_v3_stage1_dataset_keeps_focus_and_skips_direct_rows(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"image":"a.jpg","question":"q","target":"the small text","evidence_description":"It reads EXP.","need_focus":true,"trajectory_type":"single_focus","evidence_state":"need_local_visual_evidence"}',
                '{"image":"b.jpg","question":"q","target":"","evidence_description":"A dog is visible.","need_focus":false,"trajectory_type":"direct_answer","evidence_state":"sufficient_visual_evidence"}',
            ]
        )
        + "\n"
    )

    dataset = TGVFv3Stage1Dataset(path)

    assert len(dataset) == 1
    assert dataset[0].target_style == "unknown"
    assert dataset[0].target_cues == []
    assert len(dataset.skipped_rows) == 1


def test_weak_strict_mask_blocks_only_post_tgvf_queries() -> None:
    attention = torch.ones((1, 8), dtype=torch.long)
    mask = build_weak_strict_attention_mask(
        attention_mask_2d=attention,
        original_image_token_indices=torch.tensor([1, 2]),
        block_query_start=5,
        dtype=torch.float32,
    )

    assert mask.shape == (1, 1, 8, 8)
    assert mask[0, 0, 4, 1].item() == 0.0
    assert mask[0, 0, 5, 1].item() < 0.0
    assert mask[0, 0, 6, 0].item() == 0.0


def test_v3_stage1_readout_loss_backprops_to_d_not_frozen_qwen() -> None:
    tokenizer = TinyQwen3Tokenizer()
    model = TinyQwen3()
    freeze_qwen_backbone(model)
    d = torch.randn(2, 8, requires_grad=True)
    merged_visual_tokens = torch.randn(2, 8)

    inputs = prepare_v3_stage1_readout_inputs(
        model=model,
        tokenizer_or_processor=tokenizer,
        capture=_capture(),
        evidence_description="It reads EXP.",
        foveated_visual_tokens=d,
        merged_visual_tokens=merged_visual_tokens,
        mask_original_image_after_tgvf=True,
    )
    loss, _ = compute_v3_stage1_lm_loss(model=model, readout_inputs=inputs)
    loss.backward()

    assert torch.isfinite(loss)
    assert inputs["mask_mode"] == "weak_strict_original_image_keys_4d"
    assert inputs["blocked_original_image_keys_for_post_tgvf"] is True
    assert inputs["pre_tgvf_queries_keep_original_image_keys"] is True
    assert inputs["original_image_embeds_replaced"] is True
    assert torch.allclose(inputs["inputs_embeds"][0, torch.tensor([1, 2])], merged_visual_tokens)
    assert d.grad is not None
    assert torch.isfinite(d.grad).all()
    assert any(d.grad.abs().flatten() > 0)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not parameter.requires_grad for parameter in model.parameters())
