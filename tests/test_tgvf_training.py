from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from revisit_vlm.tgvf_foveal import (
    TargetSlotFovealCrossMerger,
    TGVFv2Bidirectional,
    TGVFv2CrossAttention,
    TGVFv2VPTGating,
)
from revisit_vlm.tgvf_training import (
    IGNORE_INDEX,
    LossWeights,
    attention_diagnostics,
    TeacherGuideDataset,
    TeacherGuideSample,
    TGVFTrainingConfig,
    build_tgvf_module,
    contrastive_alignment_loss,
    freeze_qwen2vl,
    fvt_norm_diagnostics,
    load_tgvf_module_checkpoint,
    prepare_readout_inputs,
    readout_prompt_parts,
    same_image_negative_groups,
    same_image_negative_loss,
    same_image_negative_matrix_ce_loss,
    same_image_negative_matrix_ce_score_gradients,
    same_image_negative_pairs,
    save_tgvf_checkpoint,
    summarize_diagnostics,
    visual_token_manifold_loss,
)


class TinyTokenizer:
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

    def __call__(
        self, texts: list[str], padding: bool = True, return_tensors: str = "pt"
    ) -> dict[str, torch.Tensor]:
        del return_tensors
        encoded = [self.encode(text, add_special_tokens=False) for text in texts]
        max_len = max(len(ids) for ids in encoded) if padding else None
        rows = []
        masks = []
        for ids in encoded:
            pad = (max_len or len(ids)) - len(ids)
            rows.append(ids + [0] * pad)
            masks.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class TinyLM(nn.Module):
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
        self.last_forward_kwargs: dict[str, object] = {}

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        position_ids: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        self.last_forward_kwargs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }
        del attention_mask, return_dict
        token_embeds = inputs_embeds if inputs_embeds is not None else self.embed(input_ids)
        hidden = torch.cumsum(token_embeds, dim=1)
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=IGNORE_INDEX,
            )
        hidden_states = (hidden,) if output_hidden_states else None
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=hidden_states)


def test_teacher_guide_dataset_loads_jsonl_and_warns_on_leakage(tmp_path: Path) -> None:
    path = tmp_path / "teacher.jsonl"
    record = {
        "image": "sample.jpg",
        "image_id": "img-1",
        "question": "What is printed below the barcode?",
        "target": "the EXP text below the barcode",
        "evidence_description": "The small text below the barcode reads EXP 08/2026.",
        "short_answer": "EXP 08/2026",
    }
    path.write_text(json.dumps(record) + "\n")

    with pytest.warns(UserWarning, match="target may leak"):
        dataset = TeacherGuideDataset(path)

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.image == str((tmp_path / "sample.jpg").resolve())
    assert sample.question == record["question"]
    assert sample.target == record["target"]


def test_default_loss_weights_are_conservative() -> None:
    weights = LossWeights()
    assert weights.gen == 1.0
    assert weights.visual_token_manifold == pytest.approx(0.1)
    assert weights.same_image_negative == 0.0
    assert weights.contrastive_alignment == 0.0


def test_training_config_defaults_readout_prompt_target_dropout() -> None:
    config = TGVFTrainingConfig()

    assert config.readout_prompt_target_dropout == pytest.approx(0.3)


def test_token_direct_can_use_per_target_token_output_length() -> None:
    module = build_tgvf_module(
        variant="token_direct",
        d_lm=8,
        d_v=5,
        num_foveated_tokens=None,
    )
    output = module(
        target_hidden_states=torch.randn(7, 8),
        pre_merge_visual_tokens=torch.randn(11, 5),
    )

    assert output.foveated_visual_tokens.shape == (7, 8)
    assert output.debug_metadata["output_mode"] == "per_target_token"


def test_non_token_direct_rejects_none_num_foveated_tokens() -> None:
    with pytest.raises(ValueError, match="supported only for"):
        build_tgvf_module(
            variant="pooled",
            d_lm=8,
            d_v=5,
            num_foveated_tokens=None,
        )


def test_visual_token_manifold_loss_returns_finite_scalar() -> None:
    loss = visual_token_manifold_loss(torch.randn(4, 8), torch.randn(12, 8))
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_attention_diagnostics_report_slot_entropy_mass_and_coverage() -> None:
    diagnostics = attention_diagnostics(
        {
            "attention_weights": torch.tensor(
                [
                    [0.7, 0.2, 0.1],
                    [0.0, 0.5, 0.5],
                ]
            )
        },
        topk=2,
    )

    assert diagnostics["top1_attn_mass_values"] == pytest.approx([0.7, 0.5])
    assert diagnostics["top2_attn_mass_values"] == pytest.approx([0.9, 1.0])
    assert diagnostics["top2_visual_token_coverage"] == pytest.approx(1.0)
    assert len(diagnostics["attn_entropy_values"]) == 2


def test_attention_diagnostics_average_sub_slots_to_fvt_slots() -> None:
    diagnostics = attention_diagnostics(
        {
            "sub_slot_attention_weights": torch.tensor(
                [
                    [[0.8, 0.2], [0.6, 0.4]],
                    [[0.1, 0.9], [0.3, 0.7]],
                ]
            )
        },
        topk=1,
    )

    assert diagnostics["top1_attn_mass_values"] == pytest.approx([0.7, 0.8])
    assert diagnostics["top1_visual_token_coverage"] == pytest.approx(1.0)


def test_fvt_norm_diagnostics_and_summary_values() -> None:
    norm = fvt_norm_diagnostics(
        foveated_visual_tokens=torch.tensor([[3.0, 4.0], [0.0, 2.0]]),
        merged_visual_tokens=torch.tensor([[0.0, 2.0], [0.0, 4.0]]),
    )
    summary = summarize_diagnostics([norm])

    assert norm["d_norm_values"] == pytest.approx([5.0, 2.0])
    assert norm["v_merge_norm_values"] == pytest.approx([2.0, 4.0])
    assert norm["norm_ratio_values"] == pytest.approx([5.0 / 3.0, 2.0 / 3.0])
    assert summary["d_norm_mean"] == pytest.approx(3.5)
    assert summary["norm_ratio_max"] == pytest.approx(5.0 / 3.0)


def test_disabled_optional_losses_skip_cleanly() -> None:
    contrastive = contrastive_alignment_loss([torch.randn(2, 8)], torch.randn(1, 8))
    same = same_image_negative_loss(
        positive_log_likelihoods=torch.empty(0),
        negative_log_likelihoods=torch.empty(0),
    )
    pairs = same_image_negative_pairs(
        [
            TeacherGuideSample("a.jpg", "q", "target a", "desc a", image_id="a"),
            TeacherGuideSample("b.jpg", "q", "target b", "desc b", image_id="b"),
        ]
    )
    assert contrastive.item() == 0.0
    assert same.item() == 0.0
    assert pairs == []


def test_same_image_negative_groups_fallback_to_image_path() -> None:
    samples = [
        TeacherGuideSample("same.jpg", "q", "target a", "desc a"),
        TeacherGuideSample("same.jpg", "q", "target b", "desc b"),
        TeacherGuideSample("other.jpg", "q", "target c", "desc c"),
    ]

    assert same_image_negative_groups(samples) == [[0, 1]]
    assert same_image_negative_pairs(samples) == [(0, 1), (1, 0)]


def test_same_image_negative_matrix_ce_prefers_diagonal_scores() -> None:
    good_scores = torch.tensor(
        [
            [5.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
            [1.0, 0.0, 3.0],
        ]
    )
    bad_scores = torch.tensor(
        [
            [0.0, 5.0, 1.0],
            [4.0, 0.0, 1.0],
            [1.0, 3.0, 0.0],
        ]
    )

    good_loss = same_image_negative_matrix_ce_loss([good_scores])
    bad_loss = same_image_negative_matrix_ce_loss([bad_scores])

    assert good_loss < bad_loss
    assert good_loss.item() < 0.1


def test_same_image_negative_matrix_ce_score_gradients_match_autograd() -> None:
    scores = torch.tensor(
        [
            [2.0, 0.5, -1.0],
            [0.0, 1.5, 0.25],
            [-0.5, 0.75, 1.0],
        ],
        requires_grad=True,
    )

    expected_loss = same_image_negative_matrix_ce_loss([scores])
    expected_loss.backward()
    loss, gradients = same_image_negative_matrix_ce_score_gradients([scores.detach()])

    assert torch.allclose(loss, expected_loss.detach())
    assert len(gradients) == 1
    assert torch.allclose(gradients[0], scores.grad)


def test_readout_inputs_mask_prompt_and_replace_only_fvt_positions() -> None:
    tokenizer = TinyTokenizer()
    model = TinyLM()
    d = torch.randn(2, 8)
    inputs = prepare_readout_inputs(
        model=model,
        tokenizer_or_processor=tokenizer,
        target="small date",
        evidence_description="EXP 08 2026",
        foveated_visual_tokens=d,
    )

    before, after = readout_prompt_parts("small date")
    before_len = len(tokenizer.encode(before, add_special_tokens=False))
    after_len = len(tokenizer.encode(after, add_special_tokens=False))
    assert inputs["input_ids"][0, before_len].item() == 101
    assert inputs["input_ids"][0, before_len + 1].item() == 102
    assert inputs["input_ids"][0, before_len + 2].item() == 102
    assert inputs["input_ids"][0, before_len + 3].item() == 103
    assert torch.allclose(inputs["inputs_embeds"][0, before_len + 1 : before_len + 3], d)
    assert inputs["readout_append_mode"] == "qwen_native_pseudo_image"
    assert inputs["fake_image_grid_thw"] == [[1, 2, 4]]
    assert inputs["source_image_grid_thw"] is None
    assert inputs["fvt_grid_thw"] == [[1, 2, 4]]
    assert inputs["fvt_grid_source"] == "near_square_fake_grid"
    assert inputs["spatial_merge_size"] == 2
    assert inputs["expected_llm_image_tokens"] == 2
    assert inputs["actual_image_pad_token_count"] == 2
    assert inputs["mm_token_type_ids_present"] is True
    assert inputs["image_pad_mm_type_is_image"] is True
    assert inputs["position_ids_source"] == "manual_fake_grid_mrope"
    assert inputs["image_position_ids_are_3d"] is True
    assert inputs["text_position_ids_are_1d"] is True
    assert inputs["visual_tower_called_for_fvt"] is False
    assert inputs["mm_token_type_ids"][0, before_len].item() == 0
    assert inputs["mm_token_type_ids"][0, before_len + 1].item() == 1
    assert inputs["mm_token_type_ids"][0, before_len + 2].item() == 1
    assert inputs["mm_token_type_ids"][0, before_len + 3].item() == 0
    assert inputs["position_ids"].shape == (3, 1, inputs["input_ids"].shape[-1])

    answer_start = before_len + 4 + after_len
    assert inputs["answer_start"] == answer_start
    assert (inputs["labels"][0, :answer_start] == IGNORE_INDEX).all()
    assert (inputs["labels"][0, answer_start:] != IGNORE_INDEX).all()


def test_readout_inputs_can_use_source_image_grid_positions() -> None:
    tokenizer = TinyTokenizer()
    model = TinyLM()
    d = torch.randn(2, 8)
    source_grid = torch.tensor([[1, 4, 2]])

    inputs = prepare_readout_inputs(
        model=model,
        tokenizer_or_processor=tokenizer,
        target="small date",
        evidence_description="EXP 08 2026",
        foveated_visual_tokens=d,
        image_grid_thw=source_grid,
    )

    assert inputs["fake_image_grid_thw"] is None
    assert inputs["source_image_grid_thw"] == [[1, 4, 2]]
    assert inputs["fvt_grid_thw"] == [[1, 4, 2]]
    assert inputs["fvt_grid_source"] == "source_image_grid_thw"
    assert inputs["position_ids_source"] == "source_image_grid_mrope"
    assert inputs["expected_llm_image_tokens"] == 2


def test_readout_inputs_reject_source_grid_token_count_mismatch() -> None:
    tokenizer = TinyTokenizer()
    model = TinyLM()
    d = torch.randn(2, 8)

    with pytest.raises(AssertionError, match="expected_llm_image_tokens=4"):
        prepare_readout_inputs(
            model=model,
            tokenizer_or_processor=tokenizer,
            target="small date",
            evidence_description="EXP 08 2026",
            foveated_visual_tokens=d,
            image_grid_thw=torch.tensor([[1, 4, 4]]),
        )


def test_readout_prompt_can_omit_target() -> None:
    before, after = readout_prompt_parts(None)

    assert before == "<|im_start|>user\n"
    assert "Target:" not in before
    assert "Target:" not in after
    assert "Describe only what is visible in this focused evidence." in after
    assert after.endswith("<|im_end|>\n<|im_start|>assistant\n")


def test_readout_inputs_without_target_still_replace_fvt_positions() -> None:
    tokenizer = TinyTokenizer()
    model = TinyLM()
    d = torch.randn(2, 8)
    inputs = prepare_readout_inputs(
        model=model,
        tokenizer_or_processor=tokenizer,
        target=None,
        evidence_description="EXP 08 2026",
        foveated_visual_tokens=d,
    )

    before, _ = readout_prompt_parts(None)
    before_len = len(tokenizer.encode(before, add_special_tokens=False))
    assert inputs["input_ids"][0, before_len].item() == 101
    assert torch.allclose(inputs["inputs_embeds"][0, before_len + 1 : before_len + 3], d)


def test_readout_loss_backprops_to_fvt_not_frozen_lm() -> None:
    tokenizer = TinyTokenizer()
    model = TinyLM()
    freeze_qwen2vl(model)
    d = torch.randn(2, 8, requires_grad=True)
    inputs = prepare_readout_inputs(
        model=model,
        tokenizer_or_processor=tokenizer,
        target="small date",
        evidence_description="EXP 08 2026",
        foveated_visual_tokens=d,
    )
    outputs = model(
        input_ids=inputs["input_ids"],
        inputs_embeds=inputs["inputs_embeds"],
        attention_mask=inputs["attention_mask"],
        position_ids=inputs["position_ids"],
        image_grid_thw=inputs["image_grid_thw"],
        mm_token_type_ids=inputs["mm_token_type_ids"],
        labels=inputs["labels"],
        return_dict=True,
    )
    outputs.loss.backward()

    assert model.last_forward_kwargs["image_grid_thw"].tolist() == [[1, 2, 4]]
    assert int(model.last_forward_kwargs["mm_token_type_ids"].sum().item()) == 2
    assert tuple(model.last_forward_kwargs["position_ids"].shape) == (3, 1, inputs["input_ids"].shape[-1])
    assert d.grad is not None
    assert torch.isfinite(d.grad).all()
    assert any(d.grad.abs().flatten() > 0)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_tgvf_module_params_remain_trainable_when_qwen_is_frozen() -> None:
    model = TinyLM()
    freeze_qwen2vl(model)
    module = build_tgvf_module(
        variant="foveal_cross_merger",
        d_lm=8,
        d_v=6,
        num_foveated_tokens=3,
        spatial_merge_size=2,
    )
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.requires_grad for parameter in module.parameters())


def test_tgvf_checkpoint_saves_module_without_qwen_weights(tmp_path: Path) -> None:
    module = build_tgvf_module(
        variant="pooled",
        d_lm=8,
        d_v=6,
        num_foveated_tokens=2,
    )
    path = tmp_path / "tgvf.pt"
    save_tgvf_checkpoint(
        path=path,
        foveal_module=module,
        config={"variant": "pooled"},
        global_step=7,
    )

    fresh = build_tgvf_module(
        variant="pooled",
        d_lm=8,
        d_v=6,
        num_foveated_tokens=2,
    )
    checkpoint = load_tgvf_module_checkpoint(fresh, path)
    assert checkpoint["global_step"] == 7
    assert "tgvf_module" in checkpoint
    assert "qwen" not in checkpoint


def test_build_tgvf_module_supports_target_slot_variant() -> None:
    module = build_tgvf_module(
        variant="target_slot_foveal_cross_merger",
        d_lm=8,
        d_v=6,
        num_foveated_tokens=2,
        spatial_merge_size=2,
    )

    assert isinstance(module, TargetSlotFovealCrossMerger)
    assert module.num_fvt_tokens == 2


def test_build_tgvf_module_supports_v2_dynamic_token_variants() -> None:
    variants = {
        "tgvf_v2_vpt_gating": TGVFv2VPTGating,
        "tgvf_v2_cross_attention": TGVFv2CrossAttention,
        "tgvf_v2_bidirectional": TGVFv2Bidirectional,
    }
    for variant, expected_type in variants.items():
        module = build_tgvf_module(
            variant=variant,
            d_lm=8,
            d_v=5,
            num_foveated_tokens=None,
            spatial_merge_size=2,
            attn_dim=7,
        )
        output = module(
            target_hidden_states=torch.randn(4, 8),
            pre_merge_visual_tokens=torch.randn(20, 5),
        )

        assert isinstance(module, expected_type)
        assert output.foveated_visual_tokens.shape == (20, 5)
        assert output.debug_metadata["final_fvt_requires_qwen_visual_merger"] is True
        assert output.debug_metadata["conditioned_visual_tokens_shape"] == [20, 5]
        assert output.debug_metadata["tgvf_version"] == "v2"
