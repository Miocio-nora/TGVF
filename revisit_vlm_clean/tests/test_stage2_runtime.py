import json
from types import SimpleNamespace

import pytest
from revisit_vlm_clean.cli.benchmark import main as benchmark_main
from revisit_vlm_clean.schema import ForwardMode
from revisit_vlm_clean.stage2_runtime import Stage2RuntimeConfig, eval_jsonl_identity


def test_stage2_matrix_ce_cursor_keeps_complete_single_focus_same_image_groups() -> None:
    from revisit_vlm.tgvf_v3_stage2 import Stage2SameImageFocusBatchCursor

    samples = [
        SimpleNamespace(
            image_id="shared",
            image="/tmp/shared.jpg",
            need_focus=True,
            trajectory_type="single_focus",
        )
        for _ in range(5)
    ]
    samples.extend(
        [
            SimpleNamespace(
                image_id="shared",
                image="/tmp/shared.jpg",
                need_focus=True,
                trajectory_type="multi_focus",
            ),
            SimpleNamespace(
                image_id="shared",
                image="/tmp/shared.jpg",
                need_focus=False,
                trajectory_type="direct_answer",
            ),
        ]
    )
    cursor = Stage2SameImageFocusBatchCursor(samples, group_size=4, seed=7)

    batch = cursor.next_batch()

    assert len(batch["samples"]) == 4
    assert len(set(batch["sample_indices"])) == 4
    assert all(sample.trajectory_type == "single_focus" for sample in batch["samples"])
    assert cursor.summary()["eligible_sample_count"] == 5
    assert cursor.summary()["drop_incomplete_group_remainders"] is True


def test_stage2_matrix_ce_cross_items_swap_only_candidate_d() -> None:
    import torch

    from revisit_vlm.tgvf_v3_stage2_fast import (
        _BaseItem,
        _FocusPrepared,
        _stage2_matrix_ce_cross_items,
    )

    sample = SimpleNamespace(target="target")
    base = _BaseItem(
        sample=sample,
        input_ids=torch.zeros((1, 4), dtype=torch.long),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        model_inputs={"_deepstack_visual_embeds": [torch.zeros((1, 2))]},
        image_grid_thw=torch.ones((1, 3), dtype=torch.long),
        image_token_indices=torch.tensor([0]),
        visual_pre_count=1,
        visual_merge_count=1,
    )

    def prepared(value: float) -> _FocusPrepared:
        d = torch.full((2, 2), value)
        return _FocusPrepared(
            sample=sample,
            item=base,
            action_ids=torch.zeros((1, 1), dtype=torch.long),
            action_weights=torch.ones((1, 1)),
            target_start=0,
            target_end=1,
            target_hidden_states=torch.zeros((1, 2)),
            pre_merge_visual_tokens=torch.zeros((2, 2)),
            merged_visual_tokens=torch.zeros((2, 2)),
            foveated_visual_tokens=d,
            value_span_matched=False,
            final_input_ids=torch.zeros((1, 4), dtype=torch.long),
            final_labels=torch.full((1, 4), -100, dtype=torch.long),
            matrix_ce_labels=torch.tensor([[-100, -100, 3, -100]]),
            final_weights=torch.zeros((1, 4)),
            final_inputs_embeds=torch.zeros((1, 4, 2)),
            final_attention_mask_2d=torch.ones((1, 4), dtype=torch.long),
            final_attention_mask=torch.zeros((1, 1, 4, 4)),
            final_position_ids=torch.zeros((3, 1, 4), dtype=torch.long),
            final_mm_token_type_ids=torch.zeros((1, 4), dtype=torch.long),
            fvt_token_indices=torch.tensor([1, 2]),
            d_token_indices=torch.tensor([1, 2]),
            d_deepstack_visual_embeds=[d.clone()],
            masked_image_key_count=1,
            image_key_mask_active=True,
            mask_mode="unit",
            fvt_shape=[2, 2],
            target_hidden_shape=[1, 2],
        )

    cross = _stage2_matrix_ce_cross_items([prepared(1.0), prepared(2.0)])

    assert len(cross) == 4
    assert torch.equal(cross[1].final_inputs_embeds[0, 1:3], torch.full((2, 2), 2.0))
    assert torch.equal(cross[1].final_labels, cross[0].final_labels)
    assert torch.equal(cross[1].d_deepstack_visual_embeds[0], torch.full((2, 2), 2.0))


def test_stage2_matrix_ce_reuses_one_same_image_vision_encode(monkeypatch) -> None:
    import torch

    import revisit_vlm.tgvf_v3_stage2_fast as stage2_fast

    items = [
        stage2_fast._BaseItem(
            sample=SimpleNamespace(),
            input_ids=torch.zeros((1, 2), dtype=torch.long),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            model_inputs={},
            image_grid_thw=torch.ones((1, 3), dtype=torch.long),
            image_token_indices=torch.tensor([0]),
            visual_pre_count=2,
            visual_merge_count=1,
        )
        for _ in range(4)
    ]
    calls = []

    def fake_attach(*, qwen_model, base_items, device):
        calls.append((qwen_model, len(base_items), device))
        base_items[0].model_inputs.update(
            {
                "_v_pre": torch.ones((2, 2)),
                "_v_merge": torch.ones((1, 2)),
                "_deepstack_visual_embeds": [torch.ones((1, 2))],
                "_deepstack_pre_merge_visual_embeds": [torch.ones((2, 2))],
            }
        )

    monkeypatch.setattr(stage2_fast, "_attach_batched_vision_features", fake_attach)

    stage2_fast._attach_same_image_vision_features(
        qwen_model="model",
        base_items=items,
        device="cpu",
    )

    assert calls == [("model", 1, "cpu")]
    assert all(item.model_inputs["_v_pre"] is items[0].model_inputs["_v_pre"] for item in items)


def test_stage2_runtime_config_rejects_missing_files(tmp_path) -> None:
    config = Stage2RuntimeConfig(
        stage2_checkpoint=str(tmp_path / "missing.pt"),
        eval_jsonl=str(tmp_path / "missing.jsonl"),
        append_forward_mode=ForwardMode.KV_CACHE,
    )

    with pytest.raises(FileNotFoundError):
        config.validate()


def test_stage2_runtime_config_rejects_nonpositive_max_tokens(tmp_path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    jsonl = tmp_path / "eval.jsonl"
    ckpt.write_bytes(b"placeholder")
    jsonl.write_text('{"image": "a.jpg", "need_focus": true}\n')
    config = Stage2RuntimeConfig(
        stage2_checkpoint=str(ckpt),
        eval_jsonl=str(jsonl),
        append_forward_mode=ForwardMode.KV_CACHE,
        max_tokens=0,
    )

    with pytest.raises(ValueError, match="max_tokens"):
        config.validate()


def test_eval_jsonl_identity_counts_focus_and_no_focus(tmp_path) -> None:
    path = tmp_path / "stage2.jsonl"
    path.write_text(
        json.dumps(
            {
                "image": "a.jpg",
                "need_focus": True,
                "trajectory_type": "single_focus",
                "target": "text",
            }
        )
        + "\n"
        + json.dumps({"image": "b.jpg", "need_focus": False, "trajectory_type": "direct_answer"})
        + "\n"
    )

    identity = eval_jsonl_identity(path)

    assert identity["n_rows"] == 2
    assert identity["need_focus"] == 1
    assert identity["no_focus"] == 1


def test_validate_stage2_runtime_requires_identity_paths(capsys) -> None:
    with pytest.raises(ValueError, match="--stage2-checkpoint"):
        benchmark_main(
            [
                "--run-id",
                "stage2",
                "--checkpoint-path",
                "outputs/checkpoint.pt",
                "--mode",
                "tgvf_force",
                "--post-tgvf-forward-mode",
                "kv_cache",
                "--subset-id",
                "core_smoke_256_seed20260625",
                "--validate-stage2-runtime",
            ]
        )
    capsys.readouterr()
