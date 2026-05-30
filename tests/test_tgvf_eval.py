from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from eval.common import (
    EvalFeatureCacheItem,
    EvalSample,
    compute_readout_nll,
    different_image_index,
    group_indices_by_image,
    prepare_target_only_readout_inputs,
    random_d_like,
    same_image_wrong_index,
    shard_eval_samples,
)
from eval.merge_results import merge_query, merge_readout
from eval.metrics import (
    char_f1,
    exact_match,
    grouped_means,
    normalize_text,
    tensor_distribution_stats,
    token_f1,
)
from eval.progress_monitor import render_progress
from eval.upload_eval_wandb import collect_eval_metrics
from revisit_vlm.tgvf_training import IGNORE_INDEX
from tgvf_eval.adapters import BenchmarkSample
from tgvf_eval.config import method_config_from_name
from tgvf_eval.model_runner import QwenTGVFModelRunner
from tgvf_eval.prompts import build_prompt


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

    def decode(self, ids: list[int], **_kwargs: object) -> str:
        reverse = {value: key for key, value in self.vocab.items()}
        return " ".join(reverse.get(token_id, "<unk>") for token_id in ids)


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

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool = True,
        **_kwargs: object,
    ) -> SimpleNamespace:
        del attention_mask, return_dict
        hidden = inputs_embeds if inputs_embeds is not None else self.embed(input_ids)
        logits = self.head(torch.cumsum(hidden, dim=1))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return SimpleNamespace(loss=loss, logits=logits)


def _item(uid: str, group: str, d_offset: float = 0.0) -> EvalFeatureCacheItem:
    sample = EvalSample(
        uid=uid,
        stable_image_uid=group,
        image=f"{group}.jpg",
        question="q",
        target=f"target {uid}",
        evidence_description=f"description {uid}",
    )
    d = torch.ones(2, 4) * d_offset
    return EvalFeatureCacheItem(
        sample=sample,
        target_hidden_states=torch.randn(2, 4),
        pre_merge_visual_tokens=torch.randn(3, 5),
        merged_visual_tokens=torch.randn(3, 4),
        foveated_visual_tokens=d,
        image_grid_thw=None,
        capture_text="",
        shapes={"D": [2, 4]},
    )


def test_qwen_runner_passes_complete_eval_prompt_without_capture_rewrap() -> None:
    config = method_config_from_name("tgvf_module_force")
    sample = BenchmarkSample(
        benchmark="toy",
        sample_id="0",
        question="What letter is on the small label?",
        media=["image.jpg"],
        choices=["A", "B", "C"],
    )
    prompt = build_prompt(sample.question, config).prompt
    runner = QwenTGVFModelRunner(model_path="unused")

    messages = runner._single_user_messages(sample, prompt, config)

    text_items = [item for item in messages[0]["content"] if item.get("type") == "text"]
    assert len(text_items) == 1
    assert text_items[0]["text"] == prompt
    assert "If fine-grained visual evidence is needed" not in text_items[0]["text"]
    assert text_items[0]["text"].count("Before answering, select") == 1


def test_target_only_readout_masks_prompt_tokens() -> None:
    tokenizer = TinyTokenizer()
    model = TinyLM()
    inputs = prepare_target_only_readout_inputs(
        model=model,
        tokenizer_or_processor=tokenizer,
        target="small date",
        evidence_description="EXP 08 2026",
        device=torch.device("cpu"),
    )

    answer_start = inputs["answer_start"]
    assert (inputs["labels"][0, :answer_start] == IGNORE_INDEX).all()
    assert (inputs["labels"][0, answer_start:] != IGNORE_INDEX).all()
    assert inputs["answer_token_count"] == 3


def test_compute_readout_nll_returns_finite_for_no_d() -> None:
    tokenizer = TinyTokenizer()
    model = TinyLM()
    result = compute_readout_nll(
        model=model,
        tokenizer_or_processor=tokenizer,
        target_text="small date",
        evidence_description="EXP 08 2026",
        foveated_visual_tokens=None,
        device=torch.device("cpu"),
    )

    assert result["answer_token_count"] == 3
    assert torch.isfinite(torch.tensor(result["avg_nll"]))


def test_wrong_d_selection_avoids_same_sample() -> None:
    items = [_item("a", "img1"), _item("b", "img1"), _item("c", "img2")]
    groups = group_indices_by_image(items)

    assert same_image_wrong_index(groups, items[0], 0) == 1
    assert different_image_index(items, 0) == 2


def test_random_d_like_is_deterministic() -> None:
    d = torch.zeros(2, 3)
    reference = torch.randn(5, 3)
    first = random_d_like(d, reference=reference, seed=123)
    second = random_d_like(d, reference=reference, seed=123)

    assert torch.allclose(first, second)
    assert first.shape == d.shape


def test_distribution_metrics_return_finite_values() -> None:
    stats = tensor_distribution_stats(torch.randn(4, 8))

    assert stats["shape"] == [4, 8]
    assert stats["finite_rate"] == 1.0
    assert stats["mean_token_norm"] > 0


def test_metric_helpers_and_grouping() -> None:
    assert normalize_text("EXP-08/2026!") == "exp 08 2026"
    assert exact_match("EXP 08 2026", "exp-08/2026") == 1.0
    assert token_f1("red small label", "small label") > 0
    assert char_f1("abc", "abd") > 0
    rows = [
        {"evidence_type": "ocr", "delta": 1.0},
        {"evidence_type": "ocr", "delta": 3.0},
        {"evidence_type": "chart", "delta": 2.0},
    ]
    grouped = grouped_means(rows, group_key="evidence_type", metric_keys=["delta"])
    assert grouped["ocr"]["delta"] == 2.0
    assert grouped["chart"]["delta"] == 2.0


def test_image_sharding_keeps_same_image_together() -> None:
    samples = [
        EvalSample(
            uid="a0",
            stable_image_uid="img-a",
            image="a.jpg",
            question="q",
            target="t",
            evidence_description="d",
        ),
        EvalSample(
            uid="a1",
            stable_image_uid="img-a",
            image="a.jpg",
            question="q",
            target="t",
            evidence_description="d",
        ),
        EvalSample(
            uid="b0",
            stable_image_uid="img-b",
            image="b.jpg",
            question="q",
            target="t",
            evidence_description="d",
        ),
        EvalSample(
            uid="b1",
            stable_image_uid="img-b",
            image="b.jpg",
            question="q",
            target="t",
            evidence_description="d",
        ),
    ]
    shards = [
        shard_eval_samples(samples, num_shards=2, shard_index=index, shard_key="image")
        for index in range(2)
    ]
    locations = {}
    for shard_index, shard in enumerate(shards):
        for sample in shard:
            previous = locations.setdefault(sample.group_id, shard_index)
            assert previous == shard_index
    assert sorted(sample.uid for shard in shards for sample in shard) == ["a0", "a1", "b0", "b1"]


def test_render_progress_aggregates_shards_on_one_line() -> None:
    line = render_progress(
        "readout",
        [
            {"desc": "readout: build FVT", "current": 2, "total": 10},
            {"desc": "readout: build FVT", "current": 3, "total": 10},
            None,
        ],
        width=10,
    )

    assert "[readout]" in line
    assert "readout: build FVT 5/20" in line
    assert "starting 1w" in line
    assert "\n" not in line


def test_merge_readout_shards_writes_combined_report(tmp_path: Path) -> None:
    input_root = tmp_path / "readout"
    for shard, uid, delta in [(0, "a", 1.0), (1, "b", 3.0)]:
        shard_dir = input_root / f"shard_{shard}_of_2"
        shard_dir.mkdir(parents=True)
        row = {
            "uid": uid,
            "evidence_type": "ocr",
            "source_profile": "scene_text",
            "answer_type": "text_string",
            "visual_difficulty": "clear",
            "nlls": {
                "correct_D_plus_target": 2.0,
                "target_only_no_D": 2.0 + delta,
                "random_D_plus_target": 4.0,
                "wrong_D_same_image_plus_target": 5.0,
                "wrong_D_different_image_plus_target": 6.0,
            },
            "delta_correct_vs_target_only": delta,
            "delta_correct_vs_random": 2.0,
            "delta_correct_vs_wrong_same": 3.0,
            "delta_correct_vs_wrong_diff": 4.0,
        }
        (shard_dir / "per_sample_results.jsonl").write_text(json.dumps(row) + "\n")

    report = merge_readout(input_root, input_root)

    assert report["num_samples_evaluated"] == 2
    assert report["metrics"]["mean_delta_correct_vs_target_only"] == 2.0
    assert (input_root / "readout_eval_report.json").exists()
    assert sum(1 for _ in (input_root / "per_sample_results.jsonl").open()) == 2


def test_merge_query_shards_writes_combined_report(tmp_path: Path) -> None:
    input_root = tmp_path / "query"
    for shard in [0, 1]:
        shard_dir = input_root / f"shard_{shard}_of_2"
        shard_dir.mkdir(parents=True)
        item = {
            "uid": f"item-{shard}",
            "stable_image_uid": f"img-{shard}",
            "evidence_type": "ocr",
            "source_profile": "scene_text",
            "diagonal_rank": 1,
            "diagonal_gap": 0.5,
            "top1": 1.0,
            "top2": 1.0,
            "group_size": 3,
        }
        group = {
            "stable_image_uid": f"img-{shard}",
            "group_size": 3,
            "score_type": "nll_lower_is_better",
            "nll_matrix": [[1.0]],
            "diagonal_ranks": [1],
            "diagonal_gaps": [0.5],
        }
        (shard_dir / "per_item_results.jsonl").write_text(json.dumps(item) + "\n")
        (shard_dir / "per_group_results.jsonl").write_text(json.dumps(group) + "\n")

    report = merge_query(input_root, input_root)

    assert report["num_groups_evaluated"] == 2
    assert report["metrics"]["retrieval_top1"] == 1.0
    assert (input_root / "query_sensitivity_report.json").exists()


def test_merge_query_applies_global_cap_after_shards(tmp_path: Path) -> None:
    input_root = tmp_path / "query_cap"
    for shard in [0, 1]:
        shard_dir = input_root / f"shard_{shard}_of_2"
        shard_dir.mkdir(parents=True)
        item_rows = []
        group_rows = []
        for offset in range(2):
            group_id = f"img-{shard}-{offset}"
            item_rows.append(
                {
                    "uid": f"item-{group_id}",
                    "stable_image_uid": group_id,
                    "evidence_type": "ocr",
                    "source_profile": "scene_text",
                    "diagonal_rank": 1,
                    "diagonal_gap": 0.5,
                    "top1": 1.0,
                    "top2": 1.0,
                    "group_size": 3,
                }
            )
            group_rows.append(
                {
                    "stable_image_uid": group_id,
                    "group_size": 3,
                    "score_type": "nll_lower_is_better",
                    "nll_matrix": [[1.0]],
                    "diagonal_ranks": [1],
                    "diagonal_gaps": [0.5],
                }
            )
        (shard_dir / "per_item_results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in item_rows)
        )
        (shard_dir / "per_group_results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in group_rows)
        )

    report = merge_query(input_root, input_root, max_groups=3, require_groups=4)

    assert report["num_groups_evaluated"] == 3
    assert report["num_items_evaluated"] == 3


def test_collect_eval_metrics_flattens_merged_reports(tmp_path: Path) -> None:
    readout_dir = tmp_path / "readout"
    query_dir = tmp_path / "query_sensitivity"
    dist_dir = tmp_path / "fvt_distribution"
    end_dir = tmp_path / "end2end_forced"
    for path in (readout_dir, query_dir, dist_dir, end_dir):
        path.mkdir()

    (readout_dir / "readout_eval_report.json").write_text(
        json.dumps({"metrics": {"mean_nll_correct_D": 1.5}, "config": {"skip": 1}})
    )
    (query_dir / "query_sensitivity_report.json").write_text(
        json.dumps({"metrics": {"retrieval_top1": 0.25}, "num_groups_evaluated": 420})
    )
    (dist_dir / "fvt_distribution_report.json").write_text(
        json.dumps({"metrics": {"finite_rate": 1.0, "collapse_warning": False}})
    )
    (end_dir / "end2end_eval_report.json").write_text(
        json.dumps({"mode": "forced", "highlight": {"correct_D_token_f1": 0.4}})
    )

    metrics = collect_eval_metrics(tmp_path)

    assert metrics["eval/readout/metrics/mean_nll_correct_D"] == 1.5
    assert metrics["eval/query/metrics/retrieval_top1"] == 0.25
    assert metrics["eval/query/num_groups_evaluated"] == 420
    assert metrics["eval/distribution/metrics/finite_rate"] == 1.0
    assert metrics["eval/distribution/metrics/collapse_warning"] is False
    assert metrics["eval/end2end_forced/mode"] == "forced"
    assert metrics["eval/end2end_forced/highlight/correct_D_token_f1"] == 0.4
    assert "eval/readout/config/skip" not in metrics
