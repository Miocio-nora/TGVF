from __future__ import annotations

import json
from pathlib import Path

from revisit_vlm_clean.cli.train_stage3_grpo import main as plan_main
from revisit_vlm_clean.cli.stage3_grpo_schedule import main as schedule_main
from revisit_vlm_clean.cli.stage3_grpo_stepwise import (
    _apply_checkpoint_retention,
    _stepwise_wandb_metrics,
)
from revisit_vlm_clean.cli.stage3_grpo_judge import main as judge_main
from revisit_vlm_clean.cli.stage3_grpo_prepare_judge_models import main as prepare_judge_main
from revisit_vlm_clean.stage3_grpo.data import BalancedPromptSampler, load_stage3_samples
from revisit_vlm_clean.stage3_grpo.grpo import group_advantages, grpo_loss_from_tensors
from revisit_vlm_clean.stage3_grpo.model_prepare import (
    JudgeModelPrepareConfig,
    run_judge_model_prepare,
)
from revisit_vlm_clean.stage3_grpo.native_replay import (
    replay_inputs_with_generated_text,
    selected_token_logprobs_from_logits,
)
from revisit_vlm_clean.stage3_grpo.policy_reference import (
    capture_frozen_policy_reference,
    swapped_policy_reference,
)
from revisit_vlm_clean.stage3_grpo.probe import ProbeCache
from revisit_vlm_clean.stage3_grpo.reward import answer_is_correct, score_rollout_reward
from revisit_vlm_clean.stage3_grpo.rollout import FakeRolloutEngine
from revisit_vlm_clean.stage3_grpo.rollout import _reference_model_context, build_rollout_engine
from revisit_vlm_clean.stage3_grpo.schemas import (
    JudgeConfig,
    RewardConfig,
    RolloutRecord,
    RolloutConfig,
    STAGE3_GRPO_PLAN_SCHEMA_VERSION,
    Stage3Sample,
    Stage3GRPOConfig,
    TrainConfig,
)
from revisit_vlm_clean.stage3_grpo.trainer import (
    Stage3GRPOTrainer,
    _stage3_configure_native_trainables,
    _stage3_distributed_average_gradients,
    _stage3_qwen_lora_contract_summary,
    _stage3_qwen_lora_state_for_checkpoint,
    _stage3_rollout_replay_token_count,
    _stage3_manual_sgd_step,
    _stage3_native_adamw,
    _stage3_clip_grad_norm,
    _stage3_zero_grad,
    native_grpo_readiness_report,
)
from revisit_vlm_clean.stage3_grpo.judge import JudgeBundle
from revisit_vlm_clean.stage3_grpo.judge_runner import (
    OfflineJudgeConfig,
    apply_judge_chat_template,
    build_judge_prompt_text,
    judge_no_thinking_bad_words_ids,
    parse_judge_json,
    parse_judge_text_fallback,
    run_offline_judge,
    strip_thinking_generation_prompt,
)
from revisit_vlm_clean.training.stage3_grpo_executor import (
    main as executor_main,
    _native_stage2_checkpoint_preflight,
    preflight_stage3_grpo_plan,
)
from revisit_vlm_clean.schema import DeepStackScope, DeepStackState


def test_stage3_samples_and_balanced_sampler(tmp_path: Path) -> None:
    data = _write_rl_fixture(tmp_path)
    samples = load_stage3_samples(data)
    assert [sample.sample_id for sample in samples] == ["s1", "s2", "s3"]
    assert samples[1].target_text == "the small blue label on the box"

    sampler = BalancedPromptSampler(samples, seed=7)
    batch = sampler.next_batch(3)
    assert len(batch) == 3
    assert {sample.sample_id for sample in batch} <= {"s1", "s2", "s3"}
    summary = sampler.summary()
    assert summary["sample_count"] == 3
    assert sum(summary["bucket_counts"].values()) == 0


def test_stage3_reward_uses_answer_and_tool_hint(tmp_path: Path) -> None:
    record = {
        "sample_id": "s_tool",
        "image_path": "/tmp/image.png",
        "question": "What word is printed on the small blue label?",
        "gold_answer": "Open",
        "answer_aliases": ["OPEN"],
        "source_dataset": "textvqa",
        "answer_type": "ocr_text",
        "eval_metric": "normalized_exact_match",
        "tool_need_hint": "likely_required",
        "reference_target": "the small blue label on the box",
        "target_spec": {"target_text": "the small blue label on the box"},
    }
    sample = Stage3Sample.from_record(record)
    engine = FakeRolloutEngine(RolloutConfig(group_size=4, max_tool_calls=1))
    rollout = engine.free_rollout(sample, rollout_id=0)
    assert answer_is_correct("open", sample)

    reward = score_rollout_reward(
        sample=sample,
        rollout=rollout,
        reward_config=RewardConfig(w_focus=0.0, w_ground=0.0),
        probe_cache=ProbeCache(),
        judge_bundle=JudgeBundle(JudgeConfig(enabled=False)),
        tau=0.25,
        missing_probe_policy="teacher_hint",
        hint_label_weight=0.5,
        protocol="protocol_c_tool_observation",
        max_tool_calls=1,
    )
    assert reward.answer_correct is True
    assert reward.reward_answer == 1.0
    assert reward.tool_label == "tool_needed"
    assert reward.reward_tool > 0
    assert reward.reward_protocol == 0.0


def test_stage3_reward_gates_answer_when_required_tool_is_not_used() -> None:
    sample = Stage3Sample.from_record(
        {
            "sample_id": "s_tool_no_trigger",
            "image_path": "/tmp/image.png",
            "question": "What word is printed on the small blue label?",
            "gold_answer": "Open",
            "answer_aliases": ["OPEN"],
            "source_dataset": "textvqa",
            "answer_type": "ocr_text",
            "eval_metric": "normalized_exact_match",
            "tool_need_hint": "likely_required",
            "reference_target": "the small blue label on the box",
            "target_spec": {"target_text": "the small blue label on the box"},
        }
    )
    rollout = RolloutRecord(
        sample_id=sample.sample_id,
        rollout_id=0,
        rollout_type="free",
        raw_output="Open",
        final_answer="Open",
        used_tool=False,
        num_tool_calls=0,
        targets=(),
    )

    reward = score_rollout_reward(
        sample=sample,
        rollout=rollout,
        reward_config=RewardConfig(w_focus=0.0, w_ground=0.0),
        probe_cache=ProbeCache(),
        judge_bundle=JudgeBundle(JudgeConfig(enabled=False)),
        tau=0.25,
        missing_probe_policy="teacher_hint",
        hint_label_weight=1.0,
        protocol="protocol_c_tool_observation",
        max_tool_calls=1,
    )

    assert reward.answer_correct is True
    assert reward.tool_label == "tool_needed"
    assert reward.reward_answer == 0.0
    assert reward.reward_tool == -2.0
    assert reward.reward_total == -2.0
    assert reward.metadata["tool_required"] is True
    assert reward.metadata["answer_reward_gated"] is True
    assert reward.metadata["answer_reward_gate_reason"] == "required_tool_not_used"


def test_stage3_reward_softly_encourages_optional_tool_without_answer_gate() -> None:
    sample = Stage3Sample.from_record(
        {
            "sample_id": "s_optional_tool",
            "image_path": "/tmp/image.png",
            "question": "What word is printed on the sign?",
            "gold_answer": "Open",
            "answer_aliases": ["OPEN"],
            "source_dataset": "textvqa",
            "answer_type": "ocr_text",
            "eval_metric": "normalized_exact_match",
            "tool_need_hint": "optional_tool",
            "reference_target": "the sign text",
            "target_spec": {"target_text": "the sign text"},
        }
    )
    engine = FakeRolloutEngine(RolloutConfig(group_size=4, max_tool_calls=1))
    used_tool_rollout = engine.forced_on_clean(sample, rollout_id=0)
    no_tool_rollout = RolloutRecord(
        sample_id=sample.sample_id,
        rollout_id=1,
        rollout_type="free",
        raw_output="Open",
        final_answer="Open",
        used_tool=False,
        num_tool_calls=0,
        targets=(),
    )

    kwargs = {
        "sample": sample,
        "reward_config": RewardConfig(w_focus=0.0, w_ground=0.0),
        "probe_cache": ProbeCache(),
        "judge_bundle": JudgeBundle(JudgeConfig(enabled=False)),
        "tau": 0.25,
        "missing_probe_policy": "teacher_hint",
        "hint_label_weight": 0.5,
        "protocol": "protocol_c_tool_observation",
        "max_tool_calls": 1,
    }
    used_tool_reward = score_rollout_reward(rollout=used_tool_rollout, **kwargs)
    no_tool_reward = score_rollout_reward(rollout=no_tool_rollout, **kwargs)

    assert used_tool_reward.tool_label == "tool_optional"
    assert used_tool_reward.reward_tool > 0.0
    assert used_tool_reward.metadata["tool_required"] is False
    assert used_tool_reward.metadata["answer_reward_gated"] is False
    assert no_tool_reward.tool_label == "tool_optional"
    assert no_tool_reward.reward_tool == 0.0
    assert no_tool_reward.reward_answer == 1.0
    assert no_tool_reward.metadata["answer_reward_gated"] is False


def test_stage3_reward_uses_configured_judge_cache_miss_reward(tmp_path: Path) -> None:
    record = {
        "sample_id": "s_tool",
        "image_path": "/tmp/image.png",
        "question": "What word is printed on the small blue label?",
        "gold_answer": "Open",
        "answer_aliases": [],
        "source_dataset": "textvqa",
        "answer_type": "ocr_text",
        "eval_metric": "normalized_exact_match",
        "tool_need_hint": "likely_required",
        "reference_target": "the small blue label on the box",
        "target_spec": {"target_text": "the small blue label on the box"},
    }
    sample = Stage3Sample.from_record(record)
    rollout = FakeRolloutEngine(RolloutConfig(group_size=2, max_tool_calls=1)).free_rollout(
        sample,
        rollout_id=0,
    )
    reward = score_rollout_reward(
        sample=sample,
        rollout=rollout,
        reward_config=RewardConfig(w_focus=1.0, w_ground=1.0, grounding_zero_reward=-1.0),
        probe_cache=ProbeCache(),
        judge_bundle=JudgeBundle(
            JudgeConfig(enabled=True, mode="cache_only", cache_miss_reward=-0.25),
            output_dir=tmp_path,
        ),
        tau=0.25,
        missing_probe_policy="teacher_hint",
        hint_label_weight=0.5,
        protocol="protocol_c_tool_observation",
        max_tool_calls=1,
    )
    assert rollout.used_tool is True
    assert reward.focus_judge is None
    assert reward.grounding_judge is None
    assert reward.reward_focus == -0.25
    assert reward.reward_ground == -0.25
    assert reward.metadata["focus_judge_hit"] is False
    assert reward.metadata["grounding_judge_hit"] is False


def test_stage3_grpo_math_smoke() -> None:
    advantages = group_advantages([1.0, 2.0, 3.0], eps=1e-6)
    assert advantages[0] < 0
    assert abs(sum(advantages)) < 1e-5

    loss, stats = grpo_loss_from_tensors(
        new_logprobs=[[-1.0, -1.1], [-0.9, -1.2]],
        old_logprobs=[[-1.0, -1.0], [-1.0, -1.0]],
        advantages=[1.0, -1.0],
        loss_mask=[[1, 1], [1, 0]],
        ref_logprobs=[[-1.0, -1.0], [-1.0, -1.0]],
        clip_range=0.2,
        kl_coef=0.01,
    )
    assert float(loss.detach().cpu()) == stats["loss"]
    assert stats["token_count"] == 3.0


def test_stage3_grpo_loss_preserves_new_logprob_gradients() -> None:
    import torch

    new_logprobs = torch.tensor([[-1.0, -1.1]], requires_grad=True)
    old_logprobs = new_logprobs.detach().clone()
    loss, _stats = grpo_loss_from_tensors(
        new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        advantages=torch.tensor([1.0]),
        loss_mask=torch.ones_like(new_logprobs),
        ref_logprobs=old_logprobs,
        clip_range=0.2,
        kl_coef=0.0,
    )

    loss.backward()

    assert new_logprobs.grad is not None
    assert torch.all(new_logprobs.grad < 0)


def test_stage3_clip_grad_norm_zero_disables_norm_and_clip() -> None:
    import torch

    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.tensor([3.0, 4.0])

    grad_norm = _stage3_clip_grad_norm([param], 0.0)

    assert grad_norm == 0.0
    assert param.grad.tolist() == [3.0, 4.0]


def test_stage3_distributed_average_gradients_allreduces_missing_grads(monkeypatch) -> None:
    import torch
    import torch.distributed as dist

    calls: list[tuple[tuple[int, ...], float]] = []

    def fake_all_reduce(tensor, op=None):
        calls.append((tuple(tensor.shape), float(tensor.detach().sum().cpu())))
        tensor.add_(2.0)

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    with_grad = torch.nn.Parameter(torch.ones(2))
    with_grad.grad = torch.tensor([1.0, 3.0])
    missing_grad = torch.nn.Parameter(torch.ones(3))
    missing_grad.grad = None

    _stage3_distributed_average_gradients(
        [with_grad, missing_grad],
        {"distributed": True, "world_size": 2},
    )

    assert calls == [((2,), 4.0), ((3,), 0.0)]
    assert torch.allclose(with_grad.grad, torch.tensor([1.5, 2.5]))
    assert missing_grad.grad is not None
    assert torch.allclose(missing_grad.grad, torch.ones(3))


def test_stage3_native_adamw_uses_non_foreach_non_fused_path() -> None:
    import torch

    param = torch.nn.Parameter(torch.ones(2))
    optimizer = _stage3_native_adamw([param], lr=1e-6)

    assert optimizer.param_groups[0]["foreach"] is False
    assert optimizer.param_groups[0]["fused"] is False


def test_stage3_manual_sgd_step_updates_and_zeroes_grads() -> None:
    import torch

    param = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    param.grad = torch.tensor([0.25, -0.5])

    _stage3_manual_sgd_step([param], lr=0.1)
    _stage3_zero_grad([param])

    assert torch.allclose(param.detach(), torch.tensor([0.975, -0.95]))
    assert param.grad is None


def test_stage3_train_config_accepts_manual_sgd() -> None:
    TrainConfig(optimizer="manual_sgd").validate()


def test_stage3_train_config_accepts_reference_policy_modes() -> None:
    TrainConfig(reference_policy="frozen_stage2").validate()
    TrainConfig(reference_policy="on_policy_detached").validate()


def test_stage3_config_round_trips_deepstack_state(tmp_path: Path) -> None:
    config = Stage3GRPOConfig(
        run_id="deepstack_roundtrip",
        rl_data_path=str(tmp_path / "rl.jsonl"),
        output_dir=str(tmp_path / "out"),
        policy_checkpoint=str(tmp_path / "stage2.pt"),
        deepstack=DeepStackState(
            enabled=True,
            original_image_scope=DeepStackScope.EVIDENCE_ONLY,
        ),
    )

    restored = Stage3GRPOConfig.from_dict(config.to_dict())

    assert restored.deepstack.enabled is True
    assert restored.deepstack.original_image_scope == DeepStackScope.EVIDENCE_ONLY


def test_stage3_configure_native_trainables_keeps_policy_adapters_only() -> None:
    import torch

    class TinyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = torch.nn.Linear(2, 2)
            self.register_parameter("lora_adapter", torch.nn.Parameter(torch.ones(2)))

    policy = TinyPolicy()
    foveal = torch.nn.Linear(2, 2)

    summary = _stage3_configure_native_trainables(policy, foveal)

    assert policy.base.weight.requires_grad is False
    assert policy.lora_adapter.requires_grad is True
    assert all(param.requires_grad is False for param in foveal.parameters())
    assert summary["policy"]["trainable_parameters"] == 2
    assert summary["foveal_module"]["trainable_parameters"] == 0


def test_stage3_checkpoint_state_preserves_source_protocol_token_payload(monkeypatch) -> None:
    import torch
    import peft

    source_qwen = {
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(
            1, 1
        ),
        "base_model.model.model.language_model.embed_tokens.weight": torch.full((2, 2), 2.0),
        "base_model.model.lm_head.weight": torch.full((2, 2), 3.0),
    }

    monkeypatch.setattr(
        peft,
        "get_peft_model_state_dict",
        lambda _model: {
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(
                1, 1
            )
        },
    )

    state = _stage3_qwen_lora_state_for_checkpoint(
        object(),
        source_checkpoint={"qwen_lora": source_qwen},
    )
    contract = _stage3_qwen_lora_contract_summary(state)

    assert torch.equal(
        state["base_model.model.model.language_model.embed_tokens.weight"],
        source_qwen["base_model.model.model.language_model.embed_tokens.weight"],
    )
    assert torch.equal(
        state["base_model.model.lm_head.weight"],
        source_qwen["base_model.model.lm_head.weight"],
    )
    assert contract["has_full_protocol_token_payload"] is True
    assert contract["qwen_lora_key_count"] == 3


def test_stage3_preflight_rejects_checkpoint_missing_protocol_token_payload(tmp_path: Path) -> None:
    import torch

    broken = tmp_path / "broken_stage3.pt"
    torch.save(
        {
            "config": {"tgvf": {}, "training": {}, "tgvf_protocol": "protocol_c_tool_observation"},
            "global_step": 1,
            "qwen_lora": {
                "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(
                    1, 1
                )
            },
            "tgvf_module": {"dummy": torch.zeros(1)},
        },
        broken,
    )

    report = _native_stage2_checkpoint_preflight(broken)

    assert report["status"] == "failed"
    assert "stage2_checkpoint_missing_protocol_token_payload" in report["errors"]


def test_stage3_preflight_accepts_checkpoint_with_embed_and_lm_head_payload(
    tmp_path: Path,
) -> None:
    import torch

    valid = tmp_path / "valid_stage3.pt"
    torch.save(
        {
            "config": {"tgvf": {}, "training": {}, "tgvf_protocol": "protocol_c_tool_observation"},
            "global_step": 1,
            "qwen_lora": {
                "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(
                    1, 1
                ),
                "base_model.model.model.language_model.embed_tokens.weight": torch.zeros(2, 2),
                "base_model.model.lm_head.weight": torch.zeros(2, 2),
            },
            "tgvf_module": {"dummy": torch.zeros(1)},
        },
        valid,
    )

    report = _native_stage2_checkpoint_preflight(valid)

    assert report["status"] == "passed"
    assert report["qwen_lora_contract"]["has_full_protocol_token_payload"] is True


def test_stage3_native_replay_gathers_next_token_logprobs() -> None:
    import torch

    logits = torch.full((1, 5, 4), -10.0)
    logits[0, 1, 2] = 10.0
    logits[0, 2, 3] = 10.0
    values = selected_token_logprobs_from_logits(
        logits,
        generated_token_ids=[2, 3],
        prompt_len=2,
    )
    assert values.shape == (2,)
    assert torch.all(values > -1e-3)


def test_stage3_native_replay_extends_mm_token_type_ids() -> None:
    import torch

    replay_inputs = replay_inputs_with_generated_text(
        {
            "input_ids": torch.ones((1, 5), dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 1, 1, 0, 0]], dtype=torch.long),
            "rope_deltas": torch.zeros((1, 1), dtype=torch.long),
        },
        torch.tensor([[9, 10]], dtype=torch.long),
    )

    assert replay_inputs["input_ids"].shape[-1] == 7
    assert replay_inputs["attention_mask"].shape[-1] == 7
    assert replay_inputs["mm_token_type_ids"].shape[-1] == 7
    assert replay_inputs["mm_token_type_ids"][0, -2:].tolist() == [0, 0]
    assert "rope_deltas" not in replay_inputs


def test_stage3_native_readiness_requires_segmented_replay_inputs() -> None:
    rollout = RolloutRecord(
        sample_id="s1",
        rollout_id=0,
        rollout_type="free",
        raw_output="raw",
        final_answer="red",
        used_tool=True,
        num_tool_calls=1,
        targets=("the cup",),
        token_ids=(1, 2, 3),
        old_logprobs=(-1.0, -1.1, -1.2),
        loss_mask=(1, 1, 1),
        protocol={
            "focus_generated_ids": [1],
            "focus_generated_logprobs": [-1.0],
            "continuation_generated_ids": [2, 3],
            "continuation_generated_logprobs": [-1.1, -1.2],
        },
        runtime={"backend": "native_single_focus"},
    )
    report = native_grpo_readiness_report(
        [rollout],
        [{"sample_id": "s1", "rollout_id": 0, "reward_total": 1.0}],
        {("s1", 0): 0.0},
    )
    assert report["status"] == "ready"
    assert report["all_rollouts_have_segmented_replay_inputs"] is True
    assert report["unique_blockers"] == []


def test_stage3_native_readiness_treats_recorded_old_logprobs_as_diagnostic() -> None:
    rollout = RolloutRecord(
        sample_id="s1",
        rollout_id=0,
        rollout_type="free",
        raw_output="raw",
        final_answer="red",
        used_tool=True,
        num_tool_calls=1,
        targets=("the cup",),
        token_ids=(1, 2, 3),
        old_logprobs=(),
        loss_mask=(1, 1, 1),
        protocol={
            "focus_generated_ids": [1],
            "focus_generated_logprobs": [],
            "continuation_generated_ids": [2, 3],
            "continuation_generated_logprobs": [],
        },
        runtime={"backend": "native_single_focus"},
    )

    report = native_grpo_readiness_report(
        [rollout],
        [{"sample_id": "s1", "rollout_id": 0, "reward_total": 1.0}],
        {("s1", 0): 0.0},
    )

    assert report["status"] == "ready"
    assert report["recorded_old_logprobs_are_diagnostic_only"] is True
    assert report["rollouts_with_aligned_recorded_old_logprobs"] == 0
    assert _stage3_rollout_replay_token_count(rollout) == 3


def test_stage3_reference_context_does_not_disable_stage2_adapter() -> None:
    class Policy:
        def __init__(self) -> None:
            self.disable_called = False

        def disable_adapter(self):
            self.disable_called = True
            raise AssertionError("Stage3 reference must not disable the Stage2 adapter")

    policy = Policy()
    with _reference_model_context(policy):
        pass

    assert policy.disable_called is False


def test_stage3_policy_reference_snapshot_swaps_and_restores_adapter() -> None:
    import torch

    class Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = torch.nn.Linear(1, 1)
            self.register_parameter("lora_adapter", torch.nn.Parameter(torch.tensor([1.0])))

    policy = Policy()
    _stage3_configure_native_trainables(policy, None)
    snapshot = capture_frozen_policy_reference(policy)
    with torch.no_grad():
        policy.lora_adapter.fill_(5.0)

    with swapped_policy_reference(policy, snapshot):
        assert torch.allclose(policy.lora_adapter.detach(), torch.tensor([1.0]))

    assert torch.allclose(policy.lora_adapter.detach(), torch.tensor([5.0]))


def test_stage3_native_update_uses_frozen_stage2_reference_snapshot(tmp_path: Path) -> None:
    import torch

    class Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_parameter("lora_adapter", torch.nn.Parameter(torch.tensor([0.25])))

    class Runtime:
        def __init__(self, model) -> None:
            self.model = model
            self.foveal_module = None

    class Engine:
        def __init__(self, model) -> None:
            self._engine = Runtime(model)
            self.reference_calls = 0
            self.call_order = []

        def replay_rollout_logprobs(
            self,
            sample,
            rollout,
            *,
            reference=False,
            reference_policy_snapshot=None,
        ):
            if reference:
                self.call_order.append("reference")
                self.reference_calls += 1
                assert reference_policy_snapshot is not None
                assert reference_policy_snapshot.source == "stage2_policy_before_stage3_updates"
                return torch.full((2,), -1.5)
            self.call_order.append("policy")
            value = self._engine.model.lora_adapter.view(())
            return torch.stack([value, value + 0.1])

    data = _write_rl_fixture(tmp_path)
    config = Stage3GRPOConfig(
        run_id="native_frozen_ref",
        rl_data_path=str(data),
        output_dir=str(tmp_path / "out"),
        policy_checkpoint=str(tmp_path / "stage2.pt"),
        train=TrainConfig(
            optimizer="manual_sgd",
            reference_policy="frozen_stage2",
            learning_rate=0.01,
            kl_coef=0.1,
            max_grad_norm=0.0,
        ),
    )
    samples = load_stage3_samples(data)
    model = Policy()
    engine = Engine(model)
    trainer = Stage3GRPOTrainer.__new__(Stage3GRPOTrainer)
    trainer.config = config
    trainer.engine = engine
    trainer.samples = samples
    trainer.progress_path = tmp_path / "progress.jsonl"
    trainer.distributed = {
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "distributed": False,
    }
    rollouts = [
        RolloutRecord(
            sample_id="s1",
            rollout_id=0,
            rollout_type="free",
            raw_output="a",
            final_answer="a",
            used_tool=False,
            num_tool_calls=0,
            protocol={"focus_generated_ids": [1, 2], "continuation_generated_ids": []},
        ),
        RolloutRecord(
            sample_id="s1",
            rollout_id=1,
            rollout_type="free",
            raw_output="b",
            final_answer="b",
            used_tool=False,
            num_tool_calls=0,
            protocol={"focus_generated_ids": [3, 4], "continuation_generated_ids": []},
        ),
    ]
    rewards = [
        {"sample_id": "s1", "rollout_id": 0, "reward_total": 0.0},
        {"sample_id": "s1", "rollout_id": 1, "reward_total": 1.0},
    ]

    update = trainer.native_grpo_update(rollouts, rewards, global_step=1)

    assert engine.reference_calls == 2
    assert engine.call_order == ["reference", "reference", "policy", "policy"]
    assert update["reference_policy"] == "frozen_stage2"
    assert update["reference_logprobs_source"] == "teacher_forced_frozen_stage2_snapshot"
    assert update["reference_policy_snapshot"]["tensor_count"] == 1


def test_stage3_judge_json_parser() -> None:
    parsed = parse_judge_json(
        'The score is:\n```json\n{"focus_score": 2, "reason": "clear target"}\n```',
        kind="focus",
    )
    assert parsed["focus_score"] == 2
    assert parsed["score"] == 2
    parsed_ground = parse_judge_json(
        'prefix {"grounding_score": 1, "reason": "mostly grounded"} suffix',
        kind="grounding",
    )
    assert parsed_ground["grounding_score"] == 1


def test_stage3_judge_text_fallback_parser() -> None:
    parsed = parse_judge_text_fallback(
        "The target is relevant, specific, and executable visual content for the question.",
        kind="focus",
    )
    assert parsed["focus_score"] == 2
    assert parsed["reason"].startswith("parse_fallback:")
    parsed_ground = parse_judge_text_fallback(
        "The reasoning contradicts the image and cannot support the final answer.",
        kind="grounding",
    )
    assert parsed_ground["grounding_score"] == 0


def test_stage3_judge_chat_template_disables_thinking() -> None:
    class CaptureProcessor:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def apply_chat_template(self, messages: list[dict[str, object]], **kwargs: object) -> str:
            self.kwargs = dict(kwargs)
            return "templated"

    processor = CaptureProcessor()

    text = apply_judge_chat_template(processor, [{"role": "user", "content": []}])

    assert text == "templated"
    assert processor.kwargs is not None
    assert processor.kwargs["enable_thinking"] is False
    assert processor.kwargs["tokenize"] is False
    assert processor.kwargs["add_generation_prompt"] is True


def test_stage3_judge_chat_template_strips_hardcoded_thinking_prompt() -> None:
    text = (
        "<|im_start|>user\nReturn JSON only.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )

    stripped = strip_thinking_generation_prompt(text)

    assert stripped == "<|im_start|>user\nReturn JSON only.<|im_end|>\n<|im_start|>assistant\n"
    assert "<think>" not in stripped


def test_stage3_judge_prompt_text_prefills_json_after_plain_assistant() -> None:
    class Processor:
        def apply_chat_template(self, messages: list[dict[str, object]], **kwargs: object) -> str:
            return "<|im_start|>user\nReturn JSON.<|im_end|>\n<|im_start|>assistant\n<think>\n"

    text = build_judge_prompt_text(
        Processor(),
        [{"role": "user", "content": []}],
        enable_thinking=False,
        response_prefix="{",
    )

    assert text.endswith("<|im_start|>assistant\n{")
    assert "<think>" not in text


def test_stage3_judge_no_thinking_bad_words_include_think_tokens() -> None:
    class Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            mapping = {"<think>": [101], "</think>": [102]}
            return mapping[text]

    assert judge_no_thinking_bad_words_ids(Tokenizer()) == [[101], [102]]


def test_stage3_judge_chat_template_falls_back_without_thinking_kwarg() -> None:
    class LegacyProcessor:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None
            self.calls = 0

        def apply_chat_template(self, messages: list[dict[str, object]], **kwargs: object) -> str:
            self.calls += 1
            if "enable_thinking" in kwargs:
                raise TypeError("unexpected keyword argument 'enable_thinking'")
            self.kwargs = dict(kwargs)
            return "legacy"

    processor = LegacyProcessor()

    text = apply_judge_chat_template(processor, [{"role": "user", "content": []}])

    assert text == "legacy"
    assert processor.calls == 2
    assert processor.kwargs is not None
    assert "enable_thinking" not in processor.kwargs


def test_stage3_offline_judge_fake_backend_writes_caches(tmp_path: Path) -> None:
    pending = _write_judge_pending_fixture(tmp_path)
    output_dir = tmp_path / "judge_out"
    summary = run_offline_judge(
        OfflineJudgeConfig(
            pending_path=str(pending),
            output_dir=str(output_dir),
            backend="fake",
            model_id="fake-local-qwen-vl",
        )
    )
    assert summary["scored_rows"] == 2
    assert summary["enable_thinking"] is False
    assert summary["response_prefix"] == "{"
    focus_cache = output_dir / "focus_judge_cache.jsonl"
    grounding_cache = output_dir / "grounding_judge_cache.jsonl"
    assert focus_cache.exists()
    assert grounding_cache.exists()
    focus = json.loads(focus_cache.read_text(encoding="utf-8").splitlines()[0])
    ground = json.loads(grounding_cache.read_text(encoding="utf-8").splitlines()[0])
    assert focus["focus_score"] == 2
    assert ground["grounding_score"] == 2

    preflight = run_offline_judge(
        OfflineJudgeConfig(
            pending_path=str(pending),
            output_dir=str(output_dir),
            backend="fake",
            model_id="fake-local-qwen-vl",
        ),
        preflight_only=True,
    )
    assert preflight["rows_to_score"] == 0
    assert preflight["enable_thinking"] is False
    assert preflight["response_prefix"] == "{"


def test_stage3_offline_judge_local_preflight_resolves_model_root(tmp_path: Path) -> None:
    pending = _write_judge_pending_fixture(tmp_path)
    model_root = tmp_path / "models"
    model_dir = model_root / "Qwen3-VL-32B-Thinking"
    _write_minimal_qwen_vl_model_dir(model_dir)

    preflight = run_offline_judge(
        OfflineJudgeConfig(
            pending_path=str(pending),
            output_dir=str(tmp_path / "judge_local_preflight"),
            backend="local_qwen_vl",
            model_id="Qwen/Qwen3-VL-32B-Thinking",
            model_root=str(model_root),
            require_local_model=True,
        ),
        preflight_only=True,
    )

    assert preflight["status"] == "passed"
    assert preflight["model"]["status"] == "ready"
    assert preflight["model"]["resolved_model_id"] == str(model_dir)
    assert preflight["model"]["model"]["identity"]["model_type"] == "qwen3_vl"


def test_stage3_offline_judge_local_preflight_rejects_missing_model(
    tmp_path: Path,
) -> None:
    pending = _write_judge_pending_fixture(tmp_path)

    preflight = run_offline_judge(
        OfflineJudgeConfig(
            pending_path=str(pending),
            output_dir=str(tmp_path / "judge_missing_model"),
            backend="local_qwen_vl",
            model_id="Qwen/Qwen3-VL-32B-Thinking",
            model_root=str(tmp_path / "models"),
            require_local_model=True,
        ),
        preflight_only=True,
    )

    assert preflight["status"] == "failed"
    assert preflight["model"]["status"] == "missing"
    assert any("local_model_missing" in item for item in preflight["errors"])


def test_stage3_prepare_judge_models_reports_local_and_missing_targets(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "models"
    _write_minimal_qwen_vl_model_dir(model_root / "Qwen3-VL-8B-Thinking")

    report = run_judge_model_prepare(
        JudgeModelPrepareConfig(
            output_dir=str(tmp_path / "prepare"),
            model_root=str(model_root),
            targets=("qwen3_vl_8b_thinking", "qwen3_vl_32b_thinking"),
            target_sets=(),
        ),
        preflight_only=True,
    )

    assert report["status"] == "missing_models"
    assert report["ready_targets"] == ["qwen3_vl_8b_thinking"]
    assert report["missing_targets"] == ["qwen3_vl_32b_thinking"]
    assert (tmp_path / "prepare" / "stage3_grpo_judge_model_prepare_plan.json").exists()


def test_stage3_prepare_judge_models_cli_prints_presets() -> None:
    assert prepare_judge_main(["--print-presets"]) == 0


def test_stage3_cli_plan_accepts_processor_directory(tmp_path: Path) -> None:
    data = _write_rl_fixture(tmp_path)
    checkpoint = tmp_path / "stage2_checkpoint.pt"
    checkpoint.write_text("fake checkpoint", encoding="utf-8")
    processor_dir = tmp_path / "processor"
    processor_dir.mkdir()
    (processor_dir / "preprocessor_config.json").write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "stage3_grpo_plan"

    assert (
        plan_main(
            [
                "--write-plan",
                "--run-id",
                "stage3_grpo_processor_dir",
                "--rl-data-path",
                str(data),
                "--policy-checkpoint",
                str(checkpoint),
                "--processor-id",
                str(processor_dir),
                "--output-dir",
                str(output_dir),
                "--runtime-backend",
                "fake",
                "--no-judge-enabled",
            ]
        )
        == 0
    )

    plan = json.loads((output_dir / "stage3_grpo_training_plan.json").read_text(encoding="utf-8"))
    assert plan["processor_identity"]["type"] == "directory"
    assert plan["processor_identity"]["child_count"] == 1
    assert plan["config"]["reward"]["gate_answer_without_required_tool"] is True
    assert plan["config"]["reward"]["required_tool_no_call_penalty"] == 2.0
    assert plan["summary"]["reward_gate"]["gate_answer_without_required_tool"] is True
    text = (output_dir / "stage3_grpo_training_plan.txt").read_text(encoding="utf-8")
    assert "reward_gate: gate_answer_without_required_tool=True" in text


def test_stage3_cli_plan_records_wandb_config(tmp_path: Path) -> None:
    data = _write_rl_fixture(tmp_path)
    checkpoint = tmp_path / "stage2_checkpoint.pt"
    checkpoint.write_text("fake checkpoint", encoding="utf-8")
    output_dir = tmp_path / "stage3_grpo_wandb_plan"

    assert (
        plan_main(
            [
                "--write-plan",
                "--run-id",
                "stage3_grpo_wandb",
                "--rl-data-path",
                str(data),
                "--policy-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--runtime-backend",
                "fake",
                "--wandb-project",
                "tgvf-stage3",
                "--wandb-mode",
                "disabled",
                "--wandb-run-name",
                "unit-wandb",
                "--wandb-group",
                "stage3-smoke",
                "--wandb-tags",
                "stage3,grpo,unit",
                "--wandb-log-checkpoint-artifact",
            ]
        )
        == 0
    )

    plan = json.loads((output_dir / "stage3_grpo_training_plan.json").read_text(encoding="utf-8"))
    assert plan["config"]["wandb"]["project"] == "tgvf-stage3"
    assert plan["config"]["wandb"]["mode"] == "disabled"
    assert plan["config"]["wandb"]["name"] == "unit-wandb"
    assert plan["config"]["wandb"]["group"] == "stage3-smoke"
    assert plan["config"]["wandb"]["tags"] == ["stage3", "grpo", "unit"]
    assert plan["config"]["wandb"]["log_artifacts"] is True
    assert plan["config"]["wandb"]["log_checkpoint_artifact"] is True
    assert plan["summary"]["wandb_enabled"] is False
    text = (output_dir / "stage3_grpo_training_plan.txt").read_text(encoding="utf-8")
    assert "wandb_project: tgvf-stage3" in text
    assert "wandb_mode: disabled" in text


def test_stage3_sample_schedule_controls_rollout_prompts(tmp_path: Path) -> None:
    data = _write_rl_fixture(tmp_path)
    schedule_dir = tmp_path / "schedule"
    assert (
        schedule_main(
            [
                "--write-schedule",
                "--run-id",
                "stage3_schedule_unit",
                "--rl-data-path",
                str(data),
                "--output-dir",
                str(schedule_dir),
                "--steps",
                "2",
                "--world-size",
                "1",
                "--per-device-prompt-batch-size",
                "1",
                "--gradient-accumulation-steps",
                "1",
                "--seed",
                "11",
            ]
        )
        == 0
    )
    schedule_path = schedule_dir / "stage3_grpo_sample_schedule.jsonl"
    rows = [
        json.loads(line)
        for line in schedule_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert len({row["sample_id"] for row in rows}) == 2
    assert len({row["stable_image_uid"] for row in rows}) == 2

    checkpoint = tmp_path / "stage2_checkpoint.pt"
    checkpoint.write_text("fake checkpoint", encoding="utf-8")
    output_dir = tmp_path / "stage3_grpo_scheduled"
    assert (
        plan_main(
            [
                "--write-plan",
                "--run-id",
                "stage3_grpo_scheduled",
                "--rl-data-path",
                str(data),
                "--policy-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--sample-schedule-path",
                str(schedule_path),
                "--sample-schedule-start-step",
                "2",
                "--runtime-backend",
                "fake",
                "--group-size",
                "2",
                "--per-device-prompt-batch-size",
                "1",
                "--max-steps",
                "1",
                "--no-judge-enabled",
            ]
        )
        == 0
    )
    plan = json.loads((output_dir / "stage3_grpo_training_plan.json").read_text(encoding="utf-8"))
    assert plan["sample_schedule_identity"]["rows"] == 2
    assert plan["summary"]["sample_schedule_start_step"] == 2
    assert executor_main(["--plan", str(output_dir / "stage3_grpo_training_plan.json"), "--rollout-only"]) == 0
    rollout_rows = [
        json.loads(line)
        for line in (output_dir / "rollout_debug.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["sample_id"] for row in rollout_rows} == {rows[1]["sample_id"]}
    assert all(
        row["runtime"]["sample_schedule"]["global_step"] == 2
        for row in rollout_rows
    )


def test_stage3_tool_exploration_soft_prompt_modes(tmp_path: Path) -> None:
    data = _write_rl_fixture(tmp_path)
    schedule_path = tmp_path / "soft_prompt_schedule.jsonl"
    schedule_rows = [
        {
            "global_step": 1,
            "rank": 0,
            "accumulation_index": 0,
            "prompt_index": 0,
            "sample_id": "s2",
            "tool_need_hint": "likely_required",
        },
        {
            "global_step": 1,
            "rank": 0,
            "accumulation_index": 0,
            "prompt_index": 1,
            "sample_id": "s3",
            "tool_need_hint": "optional_tool",
        },
        {
            "global_step": 1,
            "rank": 0,
            "accumulation_index": 0,
            "prompt_index": 2,
            "sample_id": "s1",
            "tool_need_hint": "no_tool",
        },
    ]
    schedule_path.write_text(
        "\n".join(json.dumps(row) for row in schedule_rows) + "\n",
        encoding="utf-8",
    )
    config = Stage3GRPOConfig(
        run_id="soft_prompt_modes",
        rl_data_path=str(data),
        output_dir=str(tmp_path / "out"),
        policy_checkpoint=str(tmp_path / "stage2.pt"),
        sample_schedule_path=str(schedule_path),
        rollout=RolloutConfig(
            group_size=5,
            runtime_backend="fake",
            tool_exploration_apply_to="tool_needed_or_optional",
            tool_exploration_soft_count=2,
            tool_exploration_prompt_text="Use the focus tool if useful.",
        ),
        judge=JudgeConfig(enabled=False),
        train=TrainConfig(per_device_prompt_batch_size=3),
    )

    trainer = Stage3GRPOTrainer(config)
    rollouts = trainer.rollout_batch(global_step=1, accumulation_index=0)

    s2_rollouts = sorted(
        [rollout for rollout in rollouts if rollout.sample_id == "s2"],
        key=lambda rollout: rollout.rollout_id,
    )
    assert [rollout.rollout_type for rollout in s2_rollouts] == [
        "free",
        "free",
        "free",
        "soft_tool_prompt",
        "soft_tool_prompt",
    ]
    assert all(
        rollout.runtime["question_suffix"] == "Use the focus tool if useful."
        for rollout in s2_rollouts[-2:]
    )
    s3_rollouts = sorted(
        [rollout for rollout in rollouts if rollout.sample_id == "s3"],
        key=lambda rollout: rollout.rollout_id,
    )
    assert [rollout.rollout_type for rollout in s3_rollouts] == [
        "free",
        "free",
        "free",
        "soft_tool_prompt",
        "soft_tool_prompt",
    ]
    assert all(
        rollout.rollout_type == "free"
        for rollout in rollouts
        if rollout.sample_id == "s1"
    )
    assert all(
        rollout.runtime["sample_schedule"]["global_step"] == 1
        for rollout in rollouts
    )


def test_stage3_cli_plan_records_tool_exploration(tmp_path: Path) -> None:
    data = _write_rl_fixture(tmp_path)
    checkpoint = tmp_path / "stage2_checkpoint.pt"
    checkpoint.write_text("fake checkpoint", encoding="utf-8")
    output_dir = tmp_path / "stage3_grpo_soft_prompt_plan"

    assert (
        plan_main(
            [
                "--write-plan",
                "--run-id",
                "stage3_grpo_soft_prompt",
                "--rl-data-path",
                str(data),
                "--policy-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--runtime-backend",
                "fake",
                "--group-size",
                "20",
                "--tool-exploration-soft-count",
                "8",
                "--tool-exploration-prompt-text",
                "Use the focus tool if useful.",
            ]
        )
        == 0
    )
    plan = json.loads((output_dir / "stage3_grpo_training_plan.json").read_text(encoding="utf-8"))
    assert plan["config"]["rollout"]["tool_exploration_soft_count"] == 8
    assert plan["summary"]["tool_exploration"]["soft_count"] == 8
    text = (output_dir / "stage3_grpo_training_plan.txt").read_text(encoding="utf-8")
    assert "tool_exploration: apply_to=tool_needed soft_count=8 hard_count=0" in text


def test_stage3_stepwise_checkpoint_retention_keeps_latest_and_milestones(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "stepwise"
    for step in range(1, 8):
        step_dir = output_root / f"step_{step:06d}"
        step_dir.mkdir(parents=True)
        (step_dir / "checkpoint_step_1.pt").write_text(f"checkpoint {step}", encoding="utf-8")
    state = {
        "completed_steps": list(range(1, 8)),
        "current_checkpoint": str(output_root / "step_000007" / "checkpoint_step_1.pt"),
    }

    summary = _apply_checkpoint_retention(
        output_root=output_root,
        state=state,
        keep_last=2,
        keep_every=3,
        keep_steps={1},
    )

    kept_steps = {
        int(path.parent.name.split("_", 1)[1])
        for path in output_root.glob("step_*/checkpoint_step_1.pt")
    }
    assert kept_steps == {1, 3, 6, 7}
    assert summary["deleted_count"] == 3
    assert (output_root / "checkpoint_retention_summary.json").exists()


def test_stage3_stepwise_wandb_metrics_uses_outer_global_step(tmp_path: Path) -> None:
    step_dir = tmp_path / "step_000031"
    step_dir.mkdir()
    (step_dir / "train_metrics.jsonl").write_text(
        json.dumps(
            {
                "global_step": 1,
                "optimizer_step": 1,
                "loss": 0.25,
                "distributed_loss_mean": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rewards = [
        {
            "reward_total": 2.0,
            "reward_answer": 1.0,
            "reward_tool": 1.0,
            "reward_focus": 0.5,
            "reward_ground": 0.0,
            "reward_protocol": 0.0,
            "answer_correct": True,
            "tool_label": "tool_needed",
            "tool_label_source": "teacher_hint",
            "metadata": {"used_tool": True},
            "focus_judge": {"score": 1},
            "grounding_judge": None,
        },
        {
            "reward_total": 0.0,
            "reward_answer": 0.0,
            "reward_tool": 0.0,
            "reward_focus": 0.0,
            "reward_ground": 0.0,
            "reward_protocol": 0.0,
            "answer_correct": False,
            "tool_label": "unknown",
            "tool_label_source": "none",
            "metadata": {"used_tool": False},
            "focus_judge": None,
            "grounding_judge": None,
        },
    ]
    (step_dir / "reward_breakdown.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rewards) + "\n",
        encoding="utf-8",
    )
    rollouts = [
        {"used_tool": True, "num_tool_calls": 1, "protocol": {}},
        {"used_tool": False, "num_tool_calls": 0, "protocol": {"native_errors": ["bad"]}},
    ]
    (step_dir / "rollout_debug.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rollouts) + "\n",
        encoding="utf-8",
    )

    metrics = _stepwise_wandb_metrics(
        step_dir=step_dir,
        global_step=31,
        step_result={"pending_rows": 2},
    )

    assert metrics["trainer/global_step"] == 31
    assert metrics["train/local_global_step"] == 1.0
    assert metrics["train/loss"] == 0.25
    assert metrics["train/distributed_loss_mean"] == 0.5
    assert metrics["reward/reward_total_mean"] == 1.0
    assert metrics["reward/answer_accuracy"] == 0.5
    assert metrics["rollout/tool_trigger_rate"] == 0.5
    assert metrics["rollout/malformed_rate"] == 0.5
    assert metrics["reward/focus_judge_hit_rate"] == 1.0


def test_stage3_native_rollout_backend_constructs_without_loading(tmp_path: Path) -> None:
    data = _write_rl_fixture(tmp_path)
    config = Stage3GRPOConfig(
        run_id="native_construct",
        rl_data_path=str(data),
        output_dir=str(tmp_path / "out"),
        policy_checkpoint=str(tmp_path / "future_stage2.pt"),
        rollout=RolloutConfig(runtime_backend="native_single_focus"),
    )
    engine = build_rollout_engine(config)
    assert engine.__class__.__name__ == "NativeSingleFocusRolloutEngine"


def test_stage3_native_rollout_run_config_carries_deepstack(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import revisit_vlm_clean.stage2_native as stage2_native

    class FakeNativeStage2Engine:
        def __init__(self, *, stage2_config, backend_options=None) -> None:
            self.stage2_config = stage2_config
            self.backend_options = dict(backend_options or {})
            self.run_config = None
            self.loaded_sample = None

        def prepare(self, run_config):
            self.run_config = run_config

        def _ensure_loaded(self, sample):
            self.loaded_sample = sample

    monkeypatch.setattr(stage2_native, "NativeStage2Engine", FakeNativeStage2Engine)
    data = _write_rl_fixture(tmp_path)
    config = Stage3GRPOConfig(
        run_id="native_deepstack_config",
        rl_data_path=str(data),
        output_dir=str(tmp_path / "out"),
        policy_checkpoint=str(tmp_path / "future_stage2.pt"),
        deepstack=DeepStackState(
            enabled=True,
            original_image_scope=DeepStackScope.THROUGH_ANSWER,
        ),
        rollout=RolloutConfig(runtime_backend="native_single_focus"),
    )
    engine = build_rollout_engine(config)
    sample = load_stage3_samples(data)[0]

    native = engine._ensure_engine(sample)  # noqa: SLF001

    assert native.run_config.deepstack.enabled is True
    assert native.run_config.deepstack.original_image_scope == DeepStackScope.THROUGH_ANSWER
    assert native.run_config.execution_backend["deepstack"]["enabled"] is True


def test_stage3_native_preflight_validates_stage2_checkpoint_contract(
    tmp_path: Path,
) -> None:
    import torch

    data = _write_rl_fixture(tmp_path)
    checkpoint = tmp_path / "stage2_checkpoint.pt"
    torch.save(
        {
            "qwen_lora": {
                "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(
                    1, 1
                ),
                "base_model.model.model.language_model.embed_tokens.weight": torch.zeros(2, 2),
                "base_model.model.lm_head.weight": torch.zeros(2, 2),
            },
            "tgvf_module": {},
            "config": {
                "stage": "stage2",
                "tgvf_protocol": "protocol_c_tool_observation",
                "model_id": "Qwen/Qwen3-VL-8B-Thinking",
                "processor_id": None,
                "tgvf": {"variant": "tgvf_v2_bidirectional"},
                "training": {"max_steps": 1200, "max_image_resolution": 512},
            },
            "global_step": 12,
        },
        checkpoint,
    )
    config = Stage3GRPOConfig(
        run_id="native_preflight",
        rl_data_path=str(data),
        output_dir=str(tmp_path / "out"),
        policy_checkpoint=str(checkpoint),
        rollout=RolloutConfig(runtime_backend="native_single_focus"),
    )
    plan = {
        "schema_version": STAGE3_GRPO_PLAN_SCHEMA_VERSION,
        "config": config.to_dict(),
    }
    plan_path = tmp_path / "stage3_grpo_training_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    report = preflight_stage3_grpo_plan(plan_path, plan, config)

    assert report["status"] == "passed"
    assert report["stage2_checkpoint"]["status"] == "passed"
    assert report["stage2_checkpoint"]["global_step"] == 12
    assert report["stage2_checkpoint"]["has_training_config"] is True
    assert report["judge_cache_status"]["status"] == "cache_missing"
    assert any("focus judge reward" in item for item in report["warnings"])


def test_stage3_native_preflight_rejects_non_stage2_checkpoint(tmp_path: Path) -> None:
    data = _write_rl_fixture(tmp_path)
    checkpoint = tmp_path / "not_a_stage2_checkpoint.pt"
    checkpoint.write_text("fake checkpoint", encoding="utf-8")
    config = Stage3GRPOConfig(
        run_id="native_preflight_reject",
        rl_data_path=str(data),
        output_dir=str(tmp_path / "out"),
        policy_checkpoint=str(checkpoint),
        rollout=RolloutConfig(runtime_backend="native_single_focus"),
    )
    plan = {
        "schema_version": STAGE3_GRPO_PLAN_SCHEMA_VERSION,
        "config": config.to_dict(),
    }
    plan_path = tmp_path / "stage3_grpo_training_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    report = preflight_stage3_grpo_plan(plan_path, plan, config)

    assert report["status"] == "failed"
    assert report["stage2_checkpoint"]["status"] == "failed"
    assert any("stage2_checkpoint_load_failed" in item for item in report["errors"])


def test_stage3_judge_cli_fake_backend(tmp_path: Path) -> None:
    pending = _write_judge_pending_fixture(tmp_path)
    output_dir = tmp_path / "judge_cli"
    assert (
        judge_main(
            [
                "--pending-path",
                str(pending),
                "--output-dir",
                str(output_dir),
                "--backend",
                "fake",
                "--model-id",
                "fake-local-qwen-vl",
                "--preflight-only",
            ]
        )
        == 0
    )
    assert (
        judge_main(
            [
                "--pending-path",
                str(pending),
                "--output-dir",
                str(output_dir),
                "--backend",
                "fake",
                "--model-id",
                "fake-local-qwen-vl",
                "--execute",
            ]
        )
        == 0
    )
    assert (output_dir / "judge_run_summary.json").exists()


def test_stage3_cli_plan_preflight_rollout_and_launch(tmp_path: Path) -> None:
    data = _write_rl_fixture(tmp_path)
    checkpoint = tmp_path / "stage2_checkpoint.pt"
    checkpoint.write_text("fake checkpoint", encoding="utf-8")
    output_dir = tmp_path / "stage3_grpo_smoke"

    assert (
        plan_main(
            [
                "--write-plan",
                "--run-id",
                "stage3_grpo_test",
                "--rl-data-path",
                str(data),
                "--policy-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--runtime-backend",
                "fake",
                "--deepstack-enabled",
                "--deepstack-original-image-scope",
                "evidence_only",
                "--optimizer",
                "manual_sgd",
                "--group-size",
                "4",
                "--per-device-prompt-batch-size",
                "1",
                "--max-steps",
                "3",
                "--save-steps",
                "2",
                "--gradient-accumulation-steps",
                "2",
                "--no-judge-enabled",
            ]
        )
        == 0
    )
    plan_path = output_dir / "stage3_grpo_training_plan.json"
    assert plan_path.exists()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["summary"]["rl_sample_count"] == 3
    assert plan["config"]["model_id"] == "Qwen/Qwen3-VL-8B-Thinking"
    assert plan["config"]["train"]["optimizer"] == "manual_sgd"
    assert plan["config"]["train"]["reference_policy"] == "frozen_stage2"
    assert plan["summary"]["reference_policy"] == "frozen_stage2"
    assert plan["config"]["deepstack"] == {
        "d_features_enabled": False,
        "enabled": True,
        "original_image_scope": "evidence_only",
    }
    assert plan["summary"]["deepstack"]["enabled"] is True

    assert executor_main(["--plan", str(plan_path), "--preflight-only"]) == 0
    preflight = json.loads(
        (output_dir / "stage3_grpo_preflight_report.json").read_text(encoding="utf-8")
    )
    assert preflight["deepstack"]["original_image_scope"] == "evidence_only"
    assert preflight["reference_policy"] == "frozen_stage2"
    assert executor_main(["--plan", str(plan_path), "--prepare-execution"]) == 0
    bundle = json.loads(
        (
            output_dir
            / "stage3_grpo_execution"
            / "stage3_grpo_execution_bundle.json"
        ).read_text(encoding="utf-8")
    )
    assert bundle["deepstack"]["enabled"] is True
    assert (
        executor_main(
            [
                "--plan",
                str(plan_path),
                "--precompute-probes",
                "--limit-prompts",
                "2",
            ]
        )
        == 0
    )
    assert executor_main(["--plan", str(plan_path), "--rollout-only"]) == 0
    assert executor_main(["--plan", str(plan_path), "--launch-training"]) == 0
    assert (output_dir / "rollout_debug.jsonl").exists()
    assert (output_dir / "reward_breakdown.jsonl").exists()
    assert (output_dir / "train_metrics.json").exists()
    assert (output_dir / "train_metrics.jsonl").exists()
    assert (output_dir / "checkpoint_step_2.pt").exists()
    assert (output_dir / "checkpoint_step_3.pt").exists()
    launch_result = json.loads((output_dir / "stage3_grpo_launch_result.json").read_text(encoding="utf-8"))
    assert launch_result["status"] == "stage3_grpo_training_completed"
    assert launch_result["global_step"] == 3
    assert launch_result["aggregate"]["step_count"] == 3
    assert launch_result["aggregate"]["rollout_count"] == 24
    assert len((output_dir / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert len((output_dir / "reward_breakdown.jsonl").read_text(encoding="utf-8").splitlines()) == 24
    assert "checkpoint_step_3.pt" in (output_dir / "LATEST_CHECKPOINT.txt").read_text(encoding="utf-8")


def test_stage3_cli_deepstack_enabled_defaults_to_no_block(tmp_path: Path) -> None:
    data = _write_rl_fixture(tmp_path)
    checkpoint = tmp_path / "stage2_checkpoint.pt"
    checkpoint.write_text("fake checkpoint", encoding="utf-8")
    output_dir = tmp_path / "stage3_grpo_no_block_plan"

    assert (
        plan_main(
            [
                "--write-plan",
                "--run-id",
                "stage3_grpo_no_block",
                "--rl-data-path",
                str(data),
                "--policy-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--runtime-backend",
                "fake",
                "--deepstack-enabled",
            ]
        )
        == 0
    )

    plan = json.loads(
        (output_dir / "stage3_grpo_training_plan.json").read_text(encoding="utf-8")
    )
    assert plan["config"]["deepstack"] == {
        "d_features_enabled": False,
        "enabled": True,
        "original_image_scope": "no_block",
    }


def _write_rl_fixture(tmp_path: Path) -> Path:
    rows = [
        {
            "schema_version": "stage3_rl_qa_prompt_v0",
            "sample_id": "s1",
            "stable_image_uid": "img1",
            "image_path": str(tmp_path / "img1.png"),
            "image_sha256": "a" * 64,
            "source_dataset": "visual_genome",
            "source_split": "train",
            "question": "What color is the cup?",
            "gold_answer": "red",
            "answer_aliases": ["crimson"],
            "answer_type": "short_text",
            "eval_metric": "normalized_exact_match",
            "evidence_type": "attribute",
            "difficulty": "easy",
            "tool_need_hint": "no_tool",
            "reference_target": "the cup",
            "target_spec": {"target_text": "the cup"},
            "rl_metadata": {"is_rl_train_eligible": True},
        },
        {
            "schema_version": "stage3_rl_qa_prompt_v0",
            "sample_id": "s2",
            "stable_image_uid": "img2",
            "image_path": str(tmp_path / "img2.png"),
            "image_sha256": "b" * 64,
            "source_dataset": "textvqa",
            "source_split": "train",
            "question": "What word is printed on the small blue label?",
            "gold_answer": "open",
            "answer_aliases": [],
            "answer_type": "ocr_text",
            "eval_metric": "normalized_exact_match",
            "evidence_type": "ocr_text",
            "difficulty": "medium",
            "tool_need_hint": "likely_required",
            "reference_target": "the small blue label on the box",
            "target_spec": {"target_text": "the small blue label on the box"},
            "rl_metadata": {"is_rl_train_eligible": True},
        },
        {
            "schema_version": "stage3_rl_qa_prompt_v0",
            "sample_id": "s3",
            "stable_image_uid": "img3",
            "image_path": str(tmp_path / "img3.png"),
            "image_sha256": "c" * 64,
            "source_dataset": "chartqa",
            "source_split": "train",
            "question": "How many bars are above 10?",
            "gold_answer": "2",
            "answer_aliases": [],
            "answer_type": "number",
            "eval_metric": "numeric",
            "evidence_type": "chart_value",
            "difficulty": "hard",
            "tool_need_hint": "optional_tool",
            "reference_target": "the bars above the 10 mark",
            "target_spec": {"target_text": "the bars above the 10 mark"},
            "rl_metadata": {"is_rl_train_eligible": True},
        },
    ]
    path = tmp_path / "accepted_rl_prompts.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _write_judge_pending_fixture(tmp_path: Path) -> Path:
    image = tmp_path / "img.png"
    image.write_bytes(b"fake image placeholder")
    rows = [
        {
            "schema_version": "stage3_grpo_judge_cache_v0",
            "kind": "focus",
            "cache_key": "focus-key",
            "judge_model": "offline_cache",
            "prompt_version": "stage3_grpo_judge_v0",
            "sample_id": "s1",
            "image_path": str(image),
            "image_sha256": "a" * 64,
            "question": "What word is printed on the label?",
            "choices": [],
            "target": "the small blue label on the box",
            "post_tool_reasoning": "",
            "final_answer": "",
            "status": "pending",
        },
        {
            "schema_version": "stage3_grpo_judge_cache_v0",
            "kind": "grounding",
            "cache_key": "ground-key",
            "judge_model": "offline_cache",
            "prompt_version": "stage3_grpo_judge_v0",
            "sample_id": "s1",
            "image_path": str(image),
            "image_sha256": "a" * 64,
            "question": "What word is printed on the label?",
            "choices": [],
            "target": "the small blue label on the box",
            "post_tool_reasoning": "The focused label shows the word open.",
            "final_answer": "open",
            "status": "pending",
        },
    ]
    path = tmp_path / "judge_pending.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _write_minimal_qwen_vl_model_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "model_type": "qwen3_vl",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (path / "preprocessor_config.json").write_text("{}\n", encoding="utf-8")
