from __future__ import annotations

from types import SimpleNamespace

import pytest

import eval.eval_v3_stage2_protocol as stage2_protocol
from eval.eval_v3_stage2_protocol import Stage2ProtocolEvaluator, _validate_peft_load_result
from scripts.chat_tgvf_v3 import (
    DEFAULT_TOOLOBS_CKPT,
    DEFAULT_TOOLOBS_PROCESSOR,
    INVALID_FOCUS_ACTION_MESSAGE,
    apply_chat_profile_defaults,
    collapse_repetitive_answer_text,
    invalid_or_direct_capture_row,
    sync_chat_args_from_evaluator,
)
from revisit_vlm.qwen3_vl_tgvf import parse_v3_action
from revisit_vlm.tgvf_v3_stage2 import Stage2LossWeights


def _stage2_args(tmp_path, **overrides: object) -> SimpleNamespace:
    args = SimpleNamespace(
        stage2_checkpoint="/tmp/checkpoint.pt",
        eval_jsonl="/tmp/eval.jsonl",
        output_dir=str(tmp_path),
        device="cpu",
        tgvf_protocol="legacy_v3_tags",
        processor_id=None,
        min_confidence=None,
        max_focus=None,
        max_no_focus=None,
        num_shards=1,
        shard_index=0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _patch_minimal_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "config": {
                "tgvf_protocol": "protocol_c_tool_observation",
                "processor_id": "/tmp/processor-from-checkpoint",
            },
            "qwen_lora": {},
            "tgvf_module": {},
        }

    class FakeDataset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.samples = [
                SimpleNamespace(need_focus=True, image_id="img-1", image="/tmp/image.jpg"),
            ]

    monkeypatch.setattr(stage2_protocol.torch, "load", fake_load)
    monkeypatch.setattr(stage2_protocol, "TGVFv3Stage2Dataset", FakeDataset)


def _chat_args(**overrides: object) -> SimpleNamespace:
    args = SimpleNamespace(
        chat_profile="current",
        stage2_checkpoint=None,
        processor_id=None,
        tgvf_protocol=None,
        post_tgvf_continuation=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_chat_profile_defaults_fill_current_checkpoint_processor_protocol() -> None:
    args = _chat_args()

    apply_chat_profile_defaults(args)

    assert args.chat_profile == "toolobs"
    assert args.stage2_checkpoint == DEFAULT_TOOLOBS_CKPT
    assert args.processor_id == DEFAULT_TOOLOBS_PROCESSOR
    assert args.tgvf_protocol == "protocol_c_tool_observation"


def test_chat_answer_cleanup_collapses_repeated_option_tail() -> None:
    assert collapse_repetitive_answer_text("C. C. 1. 1. 1. 1") == "C"
    assert collapse_repetitive_answer_text("A. white") == "A. white"


def test_chat_profile_custom_checkpoint_defers_processor_and_protocol_to_checkpoint() -> None:
    args = _chat_args(stage2_checkpoint="/tmp/custom-checkpoint.pt")

    apply_chat_profile_defaults(args)

    assert args.stage2_checkpoint == "/tmp/custom-checkpoint.pt"
    assert args.processor_id is None
    assert args.tgvf_protocol == "legacy_v3_tags"


def test_chat_args_sync_to_checkpoint_resolved_evaluator_args() -> None:
    args = _chat_args(stage2_checkpoint="/tmp/custom-checkpoint.pt")
    apply_chat_profile_defaults(args)
    evaluator_args = SimpleNamespace(
        tgvf_protocol="protocol_c_tool_observation",
        processor_id="/tmp/processor-from-checkpoint",
    )

    sync_chat_args_from_evaluator(args, evaluator_args)

    assert args.tgvf_protocol == "protocol_c_tool_observation"
    assert args.processor_id == "/tmp/processor-from-checkpoint"


def test_stage2_evaluator_uses_checkpoint_processor_and_protocol(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_minimal_checkpoint(monkeypatch)
    args = _stage2_args(tmp_path)

    Stage2ProtocolEvaluator(args)

    assert args.tgvf_protocol == "protocol_c_tool_observation"
    assert args.processor_id == "/tmp/processor-from-checkpoint"


def test_stage2_evaluator_rejects_protocol_checkpoint_mismatch(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_minimal_checkpoint(monkeypatch)
    args = _stage2_args(tmp_path, tgvf_protocol="protocol_e_action_evidence_special")

    with pytest.raises(ValueError, match="Stage2 checkpoint protocol mismatch"):
        Stage2ProtocolEvaluator(args)


def test_stage2_free_capture_uses_training_direct_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = object.__new__(Stage2ProtocolEvaluator)
    evaluator.args = SimpleNamespace(
        tgvf_protocol="protocol_c_tool_observation",
        max_action_tokens=12,
    )
    evaluator.model = object()
    evaluator.processor = SimpleNamespace()
    evaluator.device = "cpu"
    evaluator._image = lambda sample: {"path": sample.image}
    sample = SimpleNamespace(image="/tmp/image.jpg", prompt_question="What is shown?")
    expected_messages = [{"role": "user", "content": "direct"}]

    def fake_build_direct_messages(image: object, question: str) -> list[dict[str, object]]:
        assert image == {"path": "/tmp/image.jpg"}
        assert question == "What is shown?"
        return expected_messages

    def fake_capture_focus_single_pass_qwen3(*_args: object, **kwargs: object) -> SimpleNamespace:
        assert kwargs["messages"] is expected_messages
        assert kwargs["force_action_prefix"] is False
        assert kwargs["protocol"] == "protocol_c_tool_observation"
        return SimpleNamespace(generated_text="")

    monkeypatch.setattr(stage2_protocol, "build_direct_messages", fake_build_direct_messages)
    monkeypatch.setattr(stage2_protocol, "capture_focus_single_pass_qwen3", fake_capture_focus_single_pass_qwen3)

    capture = evaluator._capture_free_router(sample)

    assert capture.generated_text == ""


def test_malformed_protocol_c_free_capture_is_not_displayed_as_answer() -> None:
    raw = "<think>\n<|focus_start|><|focus_end|><|focus_start|>bad<|focus_end|>"
    parsed = parse_v3_action(raw, protocol="protocol_c_tool_observation")
    capture = SimpleNamespace(
        generated_text=raw,
        generated_ids=[1, 2, 3],
        stop_reason="max_new_tokens",
        malformed=False,
        errors=[],
        second_full_forward_used=False,
    )

    row = invalid_or_direct_capture_row(
        image="/tmp/image.jpg",
        question="What is shown?",
        mode="free",
        capture=capture,
        parsed=parsed,
    )

    assert row["invalid_action"] is True
    assert row["answer"] == INVALID_FOCUS_ACTION_MESSAGE
    assert row["raw_output"] == raw


def test_stage2_default_loss_weights_match_legacy_value1() -> None:
    weights = Stage2LossWeights()

    assert weights.evidence_state == pytest.approx(0.2)
    assert weights.focus_target == pytest.approx(1.5)
    assert weights.evidence == pytest.approx(1.0)
    assert weights.value_span == pytest.approx(1.0)
    assert weights.answer == pytest.approx(1.0)
    assert weights.no_focus_evidence_state == pytest.approx(0.2)
    assert weights.no_focus_answer == pytest.approx(1.0)


def test_peft_load_validation_ignores_base_model_missing_keys() -> None:
    load_result = SimpleNamespace(
        missing_keys=["base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"],
        unexpected_keys=[],
    )

    _validate_peft_load_result(load_result)


@pytest.mark.parametrize(
    "load_result",
    [
        SimpleNamespace(missing_keys=["base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"], unexpected_keys=[]),
        SimpleNamespace(missing_keys=["base_model.model.lm_head.modules_to_save.default.weight"], unexpected_keys=[]),
        SimpleNamespace(missing_keys=[], unexpected_keys=["base_model.model.extra.weight"]),
    ],
)
def test_peft_load_validation_rejects_incomplete_adapter_load(load_result: SimpleNamespace) -> None:
    with pytest.raises(RuntimeError, match="Incomplete Stage2 LoRA load"):
        _validate_peft_load_result(load_result)
