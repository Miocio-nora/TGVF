from __future__ import annotations

import json
from pathlib import Path

from revisit_vlm_clean.cli.train_stage3_grpo import main as plan_main
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
from revisit_vlm_clean.stage3_grpo.probe import ProbeCache
from revisit_vlm_clean.stage3_grpo.reward import answer_is_correct, score_rollout_reward
from revisit_vlm_clean.stage3_grpo.rollout import FakeRolloutEngine
from revisit_vlm_clean.stage3_grpo.rollout import build_rollout_engine
from revisit_vlm_clean.stage3_grpo.schemas import (
    JudgeConfig,
    RewardConfig,
    RolloutRecord,
    RolloutConfig,
    STAGE3_GRPO_PLAN_SCHEMA_VERSION,
    Stage3Sample,
    Stage3GRPOConfig,
)
from revisit_vlm_clean.stage3_grpo.trainer import native_grpo_readiness_report
from revisit_vlm_clean.stage3_grpo.judge import JudgeBundle
from revisit_vlm_clean.stage3_grpo.judge_runner import (
    OfflineJudgeConfig,
    parse_judge_json,
    run_offline_judge,
)
from revisit_vlm_clean.training.stage3_grpo_executor import (
    main as executor_main,
    preflight_stage3_grpo_plan,
)


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


def test_stage3_native_preflight_validates_stage2_checkpoint_contract(
    tmp_path: Path,
) -> None:
    import torch

    data = _write_rl_fixture(tmp_path)
    checkpoint = tmp_path / "stage2_checkpoint.pt"
    torch.save(
        {
            "qwen_lora": {},
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

    assert executor_main(["--plan", str(plan_path), "--preflight-only"]) == 0
    assert executor_main(["--plan", str(plan_path), "--prepare-execution"]) == 0
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
