from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from revisit_vlm.tgvf_foveal import (
    FovealCrossMerger,
    PooledFovealCrossAttention,
    Qwen2VLPreMergeVisualHook,
    TargetSlotFovealCrossMerger,
    TGVFv2Bidirectional,
    TGVFv2CrossAttention,
    TGVFv2VPTGating,
    TokenFovealCrossAttention,
    append_bracketed_fvt,
    append_fvt_result_and_open_answer_turn,
    append_text_instruction_to_state,
    bracketed_fvt_token_ids,
    build_fvt_answer_instruction,
    build_text_answer_instruction,
    continue_generation_from_state,
    make_fake_image_grid,
    _fvt_grid_thw,
)


class FakeTokenizer:
    unk_token_id = -1

    token_to_id = {
        "<|vision_start|>": 101,
        "<|image_pad|>": 102,
        "<|vision_end|>": 103,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_token_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [40 + index for index, token in enumerate(text.split()) if token]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return " ".join(str(token_id) for token_id in token_ids)


class FakeCache:
    def __init__(self, seq_length: int) -> None:
        self.seq_length = seq_length

    def get_seq_length(self) -> int:
        return self.seq_length


class FakeAppendModel(nn.Module):
    def __init__(self, *, vocab_size: int = 256, hidden_size: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.forward_lengths: list[int] = []
        self.seen_past: list[FakeCache | None] = []
        self.seen_position_shapes: list[tuple[int, ...] | None] = []
        self.seen_input_id_shapes: list[tuple[int, ...] | None] = []
        self.seen_image_grid_thw: list[list[list[int]] | None] = []
        self.seen_mm_token_type_counts: list[int | None] = []

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: FakeCache | None = None,
        position_ids: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        use_cache: bool = False,
        return_dict: bool = False,
        **_: Any,
    ) -> SimpleNamespace:
        assert use_cache
        assert return_dict
        seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
        self.forward_lengths.append(seq_len)
        self.seen_past.append(past_key_values)
        self.seen_position_shapes.append(
            None if position_ids is None else tuple(position_ids.shape)
        )
        self.seen_input_id_shapes.append(None if input_ids is None else tuple(input_ids.shape))
        self.seen_image_grid_thw.append(None if image_grid_thw is None else image_grid_thw.detach().cpu().tolist())
        self.seen_mm_token_type_counts.append(None if mm_token_type_ids is None else int(mm_token_type_ids.sum().detach().cpu().item()))
        old_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        logits = torch.zeros((1, seq_len, self.embedding.num_embeddings))
        logits[:, -1, 7] = 1.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=FakeCache(old_len + seq_len),
        )


def test_token_foveal_cross_attention_shapes_and_attention() -> None:
    target = torch.randn(3, 8)
    visual = torch.randn(11, 5)
    module = TokenFovealCrossAttention(d_lm=8, d_v=5)

    output = module(target_hidden_states=target, pre_merge_visual_tokens=visual)

    assert output.foveated_visual_tokens.shape == (3, 8)
    attention = output.attention_debug["attention_weights"]
    assert attention.shape == (3, 11)
    assert torch.isfinite(attention).all()
    assert output.debug_metadata["variant_name"] == "token_foveal_cross_attention"


def test_token_foveal_cross_attention_fixed_output_length() -> None:
    target = torch.randn(5, 8)
    visual = torch.randn(9, 5)
    module = TokenFovealCrossAttention(d_lm=8, d_v=5, num_output_tokens=2)

    output = module(target_hidden_states=target, pre_merge_visual_tokens=visual)

    assert output.foveated_visual_tokens.shape == (2, 8)
    assert output.attention_debug["attention_weights"].shape == (2, 9)


def test_pooled_foveal_cross_attention_shapes() -> None:
    target = torch.randn(4, 8)
    visual = torch.randn(13, 5)
    module = PooledFovealCrossAttention(d_lm=8, d_v=5, num_fvt_tokens=6)

    output = module(target_hidden_states=target, pre_merge_visual_tokens=visual)

    assert output.foveated_visual_tokens.shape == (6, 8)
    assert output.attention_debug["attention_weights"].shape == (6, 13)
    assert torch.isfinite(output.foveated_visual_tokens).all()


def test_foveal_cross_merger_shapes_and_subslot_attention() -> None:
    target = torch.randn(4, 8)
    visual = torch.randn(17, 5)
    module = FovealCrossMerger(d_lm=8, d_v=5, num_fvt_tokens=3, spatial_merge_size=2)

    output = module(target_hidden_states=target, pre_merge_visual_tokens=visual)

    assert output.foveated_visual_tokens.shape == (3, 8)
    assert output.attention_debug["attention_weights"].shape == (12, 17)
    assert output.attention_debug["sub_slot_attention_weights"].shape == (3, 4, 17)
    assert output.debug_metadata["merger_initialization"] == "random"


def test_tgvf_v2_vpt_gating_conditions_visual_tokens_in_grid_order() -> None:
    target = torch.randn(4, 8)
    visual = torch.randn(20, 5)
    module = TGVFv2VPTGating(d_lm=8, d_v=5, spatial_merge_size=2)

    output = module(target_hidden_states=target, pre_merge_visual_tokens=visual)

    assert output.foveated_visual_tokens.shape == (5, 8)
    assert torch.isfinite(output.foveated_visual_tokens).all()
    assert output.attention_debug["attention_weights"].shape == (1, 20)
    assert output.attention_debug["condition_scores"].shape == (20,)
    assert output.debug_metadata["variant_name"] == "tgvf_v2_vpt_gating"
    assert output.debug_metadata["tgvf_version"] == "v2"
    assert output.debug_metadata["visual_residual"] is True
    assert output.debug_metadata["padding_visual_tokens"] == 0


def test_tgvf_v2_cross_attention_conditions_each_visual_token() -> None:
    target = torch.randn(4, 8)
    visual = torch.randn(20, 5)
    module = TGVFv2CrossAttention(d_lm=8, d_v=5, spatial_merge_size=2, attn_dim=7)

    output = module(target_hidden_states=target, pre_merge_visual_tokens=visual)

    assert output.foveated_visual_tokens.shape == (5, 8)
    assert torch.isfinite(output.foveated_visual_tokens).all()
    assert output.attention_debug["attention_weights"].shape == (1, 20)
    assert output.attention_debug["visual_to_target_attention"].shape == (20, 4)
    assert output.debug_metadata["variant_name"] == "tgvf_v2_cross_attention"
    assert output.debug_metadata["tgvf_version"] == "v2"
    assert output.debug_metadata["visual_residual"] is True


def test_tgvf_v2_bidirectional_conditions_visual_tokens_after_target_read() -> None:
    target = torch.randn(4, 8)
    visual = torch.randn(20, 5)
    module = TGVFv2Bidirectional(d_lm=8, d_v=5, spatial_merge_size=2, attn_dim=7)

    output = module(target_hidden_states=target, pre_merge_visual_tokens=visual)

    assert output.foveated_visual_tokens.shape == (5, 8)
    assert torch.isfinite(output.foveated_visual_tokens).all()
    assert output.attention_debug["attention_weights"].shape == (1, 20)
    assert output.attention_debug["target_to_visual_attention"].shape == (4, 20)
    assert output.attention_debug["visual_to_target_attention"].shape == (20, 4)
    assert output.debug_metadata["variant_name"] == "tgvf_v2_bidirectional"
    assert output.debug_metadata["tgvf_version"] == "v2"
    assert output.debug_metadata["visual_residual"] is True


def test_tgvf_v2_group_merger_pads_to_spatial_merge_group() -> None:
    target = torch.randn(4, 8)
    visual = torch.randn(18, 5)
    module = TGVFv2CrossAttention(d_lm=8, d_v=5, spatial_merge_size=2)

    output = module(target_hidden_states=target, pre_merge_visual_tokens=visual)

    assert output.foveated_visual_tokens.shape == (5, 8)
    assert output.debug_metadata["padding_visual_tokens"] == 2


def test_tgvf_v2_gradients_flow_to_conditioner_and_inputs() -> None:
    target = torch.randn(4, 8, requires_grad=True)
    visual = torch.randn(20, 5, requires_grad=True)
    modules = (
        TGVFv2VPTGating(d_lm=8, d_v=5, spatial_merge_size=2),
        TGVFv2CrossAttention(d_lm=8, d_v=5, spatial_merge_size=2, attn_dim=7),
        TGVFv2Bidirectional(d_lm=8, d_v=5, spatial_merge_size=2, attn_dim=7),
    )

    for module in modules:
        if target.grad is not None:
            target.grad.zero_()
        if visual.grad is not None:
            visual.grad.zero_()
        output = module(target_hidden_states=target, pre_merge_visual_tokens=visual)
        output.foveated_visual_tokens.mean().backward()

        assert target.grad is not None and torch.isfinite(target.grad).all()
        assert visual.grad is not None and torch.isfinite(visual.grad).all()
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()
            if parameter.requires_grad
        )


def test_bracketed_fvt_token_ids_uses_visual_brackets_and_placeholders() -> None:
    token_ids = bracketed_fvt_token_ids(
        tokenizer_or_processor=FakeTokenizer(),
        model=None,
        num_fvt_tokens=3,
    )

    assert token_ids.tolist() == [101, 102, 102, 102, 103]


def test_bracketed_append_replaces_only_placeholder_embeddings_and_extends_cache() -> None:
    model = FakeAppendModel(hidden_size=8)
    tokenizer = FakeTokenizer()
    old_cache = FakeCache(seq_length=5)
    state = SimpleNamespace(
        past_key_values=old_cache,
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        model_kwargs={"rope_deltas": torch.zeros((1, 1), dtype=torch.long)},
    )
    fvt = torch.randn(3, 8)

    result = append_bracketed_fvt(
        model=model,
        tokenizer_or_processor=tokenizer,
        generation_state=state,
        foveated_visual_tokens=fvt,
    )

    assert model.forward_lengths == [5]
    assert old_cache.get_seq_length() == 5
    assert result.past_key_values.get_seq_length() == 10
    assert result.appended_token_ids.tolist() == [101, 102, 102, 102, 103]
    assert result.attention_mask.shape == (1, 10)
    assert result.input_ids.shape == (1, 10)
    assert model.seen_past[0] is old_cache
    assert model.seen_position_shapes[0] == (3, 1, 5)
    assert torch.allclose(result.appended_inputs_embeds[0, 1:4], fvt, atol=1e-6)
    assert result.debug_metadata["bracketed_append_used"] is True
    assert result.debug_metadata["second_full_forward_used"] is False


def test_fvt_answer_instruction_restores_multiple_choice_format() -> None:
    instruction = build_fvt_answer_instruction(
        target_text="red dustpan",
        benchmark_answer_format="multiple_choice",
        option_letters=["A", "B", "C", "D"],
    )

    assert "red dustpan" in instruction
    assert "Output exactly one letter: A, B, C, D." in instruction
    assert "Do not explain." in instruction


def test_repeat_options_continuation_contains_question_and_options() -> None:
    instruction = build_fvt_answer_instruction(
        target_text="red dustpan",
        benchmark_answer_format="multiple_choice",
        option_letters=["A", "B", "C", "D"],
        original_question="What color is the dustpan?",
        option_texts=["red", "white", "blue", "green"],
        repeat_options_in_continuation=True,
    )

    assert "Question:\nWhat color is the dustpan?" in instruction
    assert "A. red" in instruction
    assert "D. green" in instruction
    assert "Use the original image and the focused visual evidence." in instruction


def test_minimal_continuation_does_not_include_options() -> None:
    instruction = build_text_answer_instruction(
        benchmark_answer_format="multiple_choice",
        option_letters=["A", "B", "C", "D"],
        original_question="What color is the dustpan?",
        option_texts=["red", "white", "blue", "green"],
        repeat_options_in_continuation=False,
    )

    assert "What color is the dustpan?" not in instruction
    assert "A. red" not in instruction
    assert "Output exactly one letter: A, B, C, D." in instruction


def test_fvt_grid_thw_uses_near_square_fake_grid_for_merged_tokens() -> None:
    grid = _fvt_grid_thw(num_fvt_tokens=16, spatial_merge_size=2, device=None)

    assert grid.tolist() == [[1, 8, 8]]
    assert make_fake_image_grid(num_fvt_tokens=16, spatial_merge_size=2, device=None).tolist() == [[1, 8, 8]]


def test_append_fvt_result_opens_new_answer_turn_and_uses_after_append_logits() -> None:
    model = FakeAppendModel(hidden_size=8)
    old_cache = FakeCache(seq_length=5)
    state = SimpleNamespace(
        past_key_values=old_cache,
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        last_logits=torch.zeros((1, 1, 256)),
        model_kwargs={"rope_deltas": torch.zeros((1, 1), dtype=torch.long)},
    )

    result = append_fvt_result_and_open_answer_turn(
        model=model,
        tokenizer_or_processor=FakeTokenizer(),
        generation_state=state,
        foveated_visual_tokens=torch.randn(2, 8),
        target_text="red dustpan",
        benchmark_answer_format="multiple_choice",
        option_letters=["A", "B", "C", "D"],
    )

    assert model.forward_lengths == [result.appended_token_ids.numel()]
    assert old_cache.get_seq_length() == 5
    assert result.past_key_values.get_seq_length() == 5 + result.appended_token_ids.numel()
    assert result.debug_metadata["append_mode"] == "qwen_native_pseudo_image"
    assert result.debug_metadata["turn_append_mode"] == "new_user_turn"
    assert result.debug_metadata["fvt_position_mode"] == "manual_fake_grid_mrope"
    assert result.debug_metadata["position_ids_source"] == "manual_fake_grid_mrope"
    assert result.debug_metadata["fake_image_grid_thw"] == [[1, 2, 4]]
    assert result.debug_metadata["source_image_grid_thw"] is None
    assert result.debug_metadata["fvt_grid_thw"] == [[1, 2, 4]]
    assert result.debug_metadata["fvt_grid_source"] == "near_square_fake_grid"
    assert result.debug_metadata["spatial_merge_size"] == 2
    assert result.debug_metadata["expected_llm_image_tokens"] == 2
    assert result.debug_metadata["actual_image_pad_token_count"] == 2
    assert result.debug_metadata["mm_token_type_ids_present"] is True
    assert result.debug_metadata["image_pad_mm_type_is_image"] is True
    assert result.debug_metadata["mm_token_type_ids_fvt_count"] == 2
    assert result.debug_metadata["image_position_ids_are_3d"] is True
    assert result.debug_metadata["text_position_ids_are_1d"] is True
    assert result.debug_metadata["visual_tower_called_for_fvt"] is False
    assert model.seen_input_id_shapes[0] == (1, result.appended_token_ids.numel())
    assert model.seen_image_grid_thw[0] == [[1, 2, 4]]
    assert model.seen_mm_token_type_counts[0] == 2
    assert result.debug_metadata["used_logits_source"] == "after_append"
    assert result.debug_metadata["final_assistant_prefix_appended"] is True
    assert result.debug_metadata["stop_checked_on_appended_context"] is False
    assert result.debug_metadata["cache_preserved"] is True
    assert result.debug_metadata["second_full_forward_used"] is False
    assert "<|im_start|>assistant" in result.debug_metadata["appended_context_text_without_FVT"]
    assert result.last_logits is not state.last_logits


def test_append_fvt_result_can_use_source_image_grid_positions() -> None:
    model = FakeAppendModel(hidden_size=8)
    state = SimpleNamespace(
        past_key_values=FakeCache(seq_length=5),
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        last_logits=torch.zeros((1, 1, 256)),
        model_kwargs={"rope_deltas": torch.zeros((1, 1), dtype=torch.long)},
    )

    result = append_fvt_result_and_open_answer_turn(
        model=model,
        tokenizer_or_processor=FakeTokenizer(),
        generation_state=state,
        foveated_visual_tokens=torch.randn(2, 8),
        target_text="red dustpan",
        benchmark_answer_format="multiple_choice",
        option_letters=["A", "B", "C", "D"],
        image_grid_thw=torch.tensor([[1, 4, 2]]),
    )

    assert result.debug_metadata["position_ids_source"] == "source_image_grid_mrope"
    assert result.debug_metadata["fake_image_grid_thw"] is None
    assert result.debug_metadata["source_image_grid_thw"] == [[1, 4, 2]]
    assert result.debug_metadata["fvt_grid_thw"] == [[1, 4, 2]]
    assert result.debug_metadata["fvt_grid_source"] == "source_image_grid_thw"
    assert model.seen_image_grid_thw[0] == [[1, 4, 2]]


def test_append_fvt_result_legacy_text_positions_keeps_old_soft_prompt_path() -> None:
    model = FakeAppendModel(hidden_size=8)
    old_cache = FakeCache(seq_length=5)
    state = SimpleNamespace(
        past_key_values=old_cache,
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        last_logits=torch.zeros((1, 1, 256)),
        model_kwargs={"rope_deltas": torch.zeros((1, 1), dtype=torch.long)},
    )

    result = append_fvt_result_and_open_answer_turn(
        model=model,
        tokenizer_or_processor=FakeTokenizer(),
        generation_state=state,
        foveated_visual_tokens=torch.randn(2, 8),
        target_text="red dustpan",
        benchmark_answer_format="multiple_choice",
        option_letters=["A", "B", "C", "D"],
        fvt_append_mode="legacy_text_positions",
    )

    assert result.debug_metadata["append_mode"] == "legacy_text_positions"
    assert result.debug_metadata["position_ids_source"] == "text_style_chunk"
    assert result.debug_metadata["fake_image_grid_thw"] is None
    assert result.debug_metadata["mm_token_type_ids_present"] is False
    assert result.debug_metadata["image_pad_mm_type_is_image"] is False
    assert result.debug_metadata["image_position_ids_are_3d"] is False
    assert result.debug_metadata["text_position_ids_are_1d"] is True
    assert result.debug_metadata["visual_tower_called_for_fvt"] is False
    assert model.seen_input_id_shapes[0] is None
    assert model.seen_image_grid_thw[0] is None
    assert model.seen_mm_token_type_counts[0] is None
    assert result.debug_metadata["used_logits_source"] == "after_append"
    assert result.debug_metadata["second_full_forward_used"] is False


def test_continue_generation_after_bracketed_append_uses_single_token_steps() -> None:
    model = FakeAppendModel(hidden_size=8)
    state = SimpleNamespace(
        past_key_values=FakeCache(seq_length=5),
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        model_kwargs={"rope_deltas": torch.zeros((1, 1), dtype=torch.long)},
    )
    appended = append_bracketed_fvt(
        model=model,
        tokenizer_or_processor=FakeTokenizer(),
        generation_state=state,
        foveated_visual_tokens=torch.randn(2, 8),
    )

    continuation = continue_generation_from_state(
        model=model,
        tokenizer_or_processor=FakeTokenizer(),
        generation_state=appended,
        max_new_tokens=2,
    )

    assert continuation.generated_ids == [7, 7]
    assert model.forward_lengths == [4, 1, 1]
    assert continuation.past_key_values.get_seq_length() == 11


def test_text_instruction_append_preserves_cache_after_fvt_append() -> None:
    model = FakeAppendModel(hidden_size=8)
    old_cache = FakeCache(seq_length=5)
    state = SimpleNamespace(
        past_key_values=old_cache,
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        model_kwargs={"rope_deltas": torch.zeros((1, 1), dtype=torch.long)},
    )
    appended = append_bracketed_fvt(
        model=model,
        tokenizer_or_processor=FakeTokenizer(),
        generation_state=state,
        foveated_visual_tokens=torch.randn(2, 8),
    )

    instructed = append_text_instruction_to_state(
        model=model,
        tokenizer_or_processor=FakeTokenizer(),
        generation_state=appended,
        instruction="Use focused evidence. Do not mention foveation.",
    )

    assert model.forward_lengths == [4, 7]
    assert model.seen_past[0] is old_cache
    assert model.seen_past[1] is appended.past_key_values
    assert instructed.past_key_values.get_seq_length() == 16
    assert instructed.attention_mask.shape == (1, 16)
    assert instructed.debug_metadata["continuation_instruction_appended"] is True
    assert instructed.debug_metadata["instruction_token_count"] == 7
    assert instructed.debug_metadata["second_full_forward_used"] is False


def test_qwen2vl_pre_merge_visual_hook_captures_merger_input() -> None:
    class FakeMerger(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=0, keepdim=True)

    class FakeVisual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.merger = FakeMerger()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.merger(x)

    model = SimpleNamespace(visual=FakeVisual())
    tokens = torch.randn(7, 5)

    with Qwen2VLPreMergeVisualHook(model) as hook:
        model.visual(tokens)

    assert hook.pre_merge_visual_tokens is not None
    assert torch.allclose(hook.pre_merge_visual_tokens, tokens)


def test_target_slot_foveal_cross_merger_unbatched_attention_and_metadata() -> None:
    target = torch.randn(5, 8)
    visual = torch.randn(13, 6)
    module = TargetSlotFovealCrossMerger(
        d_lm=8,
        d_v=6,
        num_fvt_tokens=3,
        spatial_merge_size=2,
        num_heads=2,
        return_attention=True,
    )

    output = module(target_hidden_states=target, pre_merge_visual_tokens=visual)

    assert output.foveated_visual_tokens.shape == (3, 8)
    assert output.attention_debug["target_attn_weights"].shape[-2:] == (12, 5)
    assert output.attention_debug["visual_attn_weights"].shape[-2:] == (12, 13)
    assert output.attention_debug["sub_slot_attention_weights"].shape == (3, 4, 13)
    assert output.debug_metadata["variant_name"] == "target_slot_foveal_cross_merger"
    assert output.debug_metadata["uses_single_q_global"] is False
    assert output.debug_metadata["direct_query_residual"] is False
    assert output.debug_metadata["sub_slots_per_fvt"] == 4
    assert torch.isfinite(output.foveated_visual_tokens).all()


def test_target_slot_foveal_cross_merger_batched_forward() -> None:
    target = torch.randn(2, 4, 8)
    visual = torch.randn(2, 9, 6)
    module = TargetSlotFovealCrossMerger(
        d_lm=8,
        d_v=6,
        num_fvt_tokens=2,
        spatial_merge_size=2,
        num_heads=2,
    )

    output = module(target_hidden_states=target, pre_merge_visual_tokens=visual)

    assert output.foveated_visual_tokens.shape == (2, 2, 8)
    assert output.attention_debug["attention_weights"].shape == (2, 8, 9)


def test_target_slot_foveal_cross_merger_masks_and_calibration_are_finite() -> None:
    target = torch.randn(2, 5, 8)
    visual = torch.randn(2, 7, 6)
    target_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    visual_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0]])
    module = TargetSlotFovealCrossMerger(
        d_lm=8,
        d_v=6,
        num_fvt_tokens=2,
        spatial_merge_size=2,
        num_heads=2,
        use_fvt_calibration=True,
        return_attention=True,
    )

    output = module(
        target_hidden_states=target,
        pre_merge_visual_tokens=visual,
        target_attention_mask=target_mask,
        visual_attention_mask=visual_mask,
    )

    assert output.foveated_visual_tokens.shape == (2, 2, 8)
    assert torch.isfinite(output.foveated_visual_tokens).all()
    assert torch.isfinite(output.attention_debug["target_attn_entropy"]).all()
    assert torch.isfinite(output.attention_debug["visual_topk_mass"]).all()


def test_target_slot_foveal_cross_merger_gradients_flow() -> None:
    target = torch.randn(4, 8)
    visual = torch.randn(10, 6)
    module = TargetSlotFovealCrossMerger(
        d_lm=8,
        d_v=6,
        num_fvt_tokens=2,
        spatial_merge_size=2,
        num_heads=2,
        use_fvt_calibration=True,
    )

    output = module(target_hidden_states=target, pre_merge_visual_tokens=visual)
    output.foveated_visual_tokens.mean().backward()

    assert module.sub_slot_embed.grad is not None
    assert module.target_proj.weight.grad is not None
    assert module.visual_key_proj.weight.grad is not None
    assert module.visual_value_proj.weight.grad is not None
    assert module.group_merger[1].weight.grad is not None
    assert module.fvt_scale is not None and module.fvt_scale.grad is not None
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def test_target_slot_foveal_cross_merger_query_residual_default_off() -> None:
    module = TargetSlotFovealCrossMerger(d_lm=8, d_v=6, num_fvt_tokens=2, num_heads=2)
    assert module.use_query_residual is False
    assert module.query_residual_gate is None

    residual_module = TargetSlotFovealCrossMerger(
        d_lm=8,
        d_v=6,
        num_fvt_tokens=2,
        num_heads=2,
        use_query_residual=True,
        query_residual_init=0.0,
    )
    assert residual_module.query_residual_gate is not None
    assert residual_module.query_residual_gate.item() == 0.0
