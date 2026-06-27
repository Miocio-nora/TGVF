"""Source adapters for Stage3 RL gold-QA pools."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .schemas import (
    infer_answer_type,
    infer_difficulty,
    infer_eval_metric,
    infer_evidence_type,
    normalize_answer,
    stable_hash,
    stable_bundle_id,
    stable_image_uid,
    stable_qa_id,
)

DEFAULT_DATASET_ROOT = "/home/dredvpn009/Flash_Storage/datasets"
DEFAULT_SOURCE_MIX = {
    "visual_genome": 0.40,
    "textvqa": 0.30,
    "docvqa": 0.20,
    "chartqa": 0.10,
}


@dataclass(frozen=True)
class SourceConfig:
    name: str
    adapter: str
    path: str
    split: str = "train"
    source_profile: str = "unknown"
    enabled: bool = True
    image_root: str | None = None
    image_roots: tuple[str, ...] = ()
    max_qa_per_image: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "adapter": self.adapter,
            "path": self.path,
            "split": self.split,
            "source_profile": self.source_profile,
            "enabled": self.enabled,
            "image_root": self.image_root,
            "image_roots": list(self.image_roots),
            "max_qa_per_image": self.max_qa_per_image,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceConfig":
        image_roots = payload.get("image_roots") or ()
        if isinstance(image_roots, str):
            image_roots = (image_roots,)
        return cls(
            name=str(payload["name"]),
            adapter=str(payload.get("adapter") or payload.get("type") or payload["name"]),
            path=str(payload["path"]),
            split=str(payload.get("split") or "train"),
            source_profile=str(payload.get("source_profile") or "unknown"),
            enabled=bool(payload.get("enabled", True)),
            image_root=None if payload.get("image_root") is None else str(payload.get("image_root")),
            image_roots=tuple(str(item) for item in image_roots),
            max_qa_per_image=(
                None if payload.get("max_qa_per_image") is None else int(payload["max_qa_per_image"])
            ),
            metadata=dict(payload.get("metadata") or {}),
        )


class SourceAdapter:
    def __init__(self, config: SourceConfig, *, dataset_root: str | Path) -> None:
        self.config = config
        self.dataset_root = Path(dataset_root)

    @property
    def source_dataset(self) -> str:
        return self.config.name

    def annotation_path(self) -> Path:
        path = Path(self.config.path)
        return path if path.is_absolute() else self.dataset_root / path

    def iter_bundles(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError

    def select_bundles(self, *, max_bundles: int | None, seed: int) -> list[dict[str, Any]]:
        bundles = list(self.iter_bundles())
        return _select_ranked_bundles(bundles, max_bundles=max_bundles, seed=seed)

    def estimate_candidate_qa(self) -> int:
        return sum(len(bundle.get("qa_items") or []) for bundle in self.iter_bundles())

    def resolve_path(self, path: str | Path) -> Path:
        raw = Path(path)
        return raw if raw.is_absolute() else self.dataset_root / raw


class VisualGenomeQAAdapter(SourceAdapter):
    def estimate_candidate_qa(self) -> int:
        data = json.loads(self.annotation_path().read_text(encoding="utf-8"))
        return sum(len(record.get("qas") or []) for record in data)

    def iter_bundles(self) -> Iterator[dict[str, Any]]:
        annotation_path = self.annotation_path()
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        for record in data:
            bundle = self._bundle_from_record(record, annotation_path)
            if bundle is not None:
                yield bundle

    def select_bundles(self, *, max_bundles: int | None, seed: int) -> list[dict[str, Any]]:
        annotation_path = self.annotation_path()
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        selected = _select_ranked_records(
            data,
            max_records=max_bundles,
            seed=seed,
            key_fn=lambda record: stable_image_uid(self.source_dataset, str(record.get("id") or "")),
        )
        return [
            bundle
            for record in selected
            for bundle in [self._bundle_from_record(record, annotation_path)]
            if bundle is not None
        ]

    def _bundle_from_record(self, record: dict[str, Any], annotation_path: Path) -> dict[str, Any] | None:
        image_id = str(record.get("id") or "")
        image_path = self._resolve_image_path(image_id)
        qas = _limit_items(
            record.get("qas") if isinstance(record.get("qas"), list) else [],
            self.config.max_qa_per_image,
        )
        qa_items = []
        for index, qa in enumerate(qas):
            question = str(qa.get("question") or "").strip()
            answer = _clean_answer(qa.get("answer"))
            source_record_id = str(qa.get("qa_id") or f"{image_id}:{index}")
            evidence_type = infer_evidence_type(self.source_dataset, self.config.source_profile, question)
            qa_items.append(
                _qa_item(
                    source_dataset=self.source_dataset,
                    source_profile=self.config.source_profile,
                    source_record_id=source_record_id,
                    question=question,
                    answer=answer,
                    aliases=[answer],
                    choices=[],
                    evidence_type=evidence_type,
                    metadata={"vg_qa_id": qa.get("qa_id"), "vg_image_id": qa.get("image_id")},
                )
            )
        if not qa_items:
            return None
        return _bundle_record(
            source_dataset=self.source_dataset,
            source_profile=self.config.source_profile,
            split=self.config.split,
            source_image_id=image_id,
            image_path=str(image_path),
            source_record_id=image_id,
            qa_items=qa_items,
            metadata={"annotation_path": str(annotation_path)},
        )

    def _resolve_image_path(self, image_id: str) -> Path:
        index = self._image_index()
        if image_id in index:
            return index[image_id]
        roots = self.config.image_roots or ("visual_genome/VG_100K", "visual_genome/VG_100K_2")
        candidates: list[Path] = []
        for root in roots:
            base = self.resolve_path(root)
            candidates.extend([base / f"{image_id}.jpg", base / f"{image_id}.png"])
        return candidates[0]

    def _image_index(self) -> dict[str, Path]:
        cached = getattr(self, "_cached_image_index", None)
        if cached is not None:
            return cached
        roots = self.config.image_roots or ("visual_genome/VG_100K", "visual_genome/VG_100K_2")
        index: dict[str, Path] = {}
        for root in roots:
            base = self.resolve_path(root)
            if not base.exists():
                continue
            for path in base.iterdir():
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    index[path.stem] = path
        setattr(self, "_cached_image_index", index)
        return index


class TextVQAAdapter(SourceAdapter):
    def estimate_candidate_qa(self) -> int:
        payload = json.loads(self.annotation_path().read_text(encoding="utf-8"))
        return len(payload.get("data") or [])

    def iter_bundles(self) -> Iterator[dict[str, Any]]:
        annotation_path = self.annotation_path()
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        for row in payload.get("data") or []:
            yield self._bundle_from_row(row, annotation_path)

    def select_bundles(self, *, max_bundles: int | None, seed: int) -> list[dict[str, Any]]:
        annotation_path = self.annotation_path()
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        selected = _select_ranked_records(
            payload.get("data") or [],
            max_records=max_bundles,
            seed=seed,
            key_fn=lambda row: stable_image_uid(self.source_dataset, str(row.get("image_id") or "")),
        )
        return [self._bundle_from_row(row, annotation_path) for row in selected]

    def _bundle_from_row(self, row: dict[str, Any], annotation_path: Path) -> dict[str, Any]:
        image_id = str(row.get("image_id") or "")
        image_path = self._resolve_image_path(image_id)
        question = str(row.get("question") or "").strip()
        answers = [str(item).strip() for item in (row.get("answers") or []) if str(item).strip()]
        answer = _majority_answer(answers)
        evidence_type = infer_evidence_type(self.source_dataset, self.config.source_profile, question)
        source_record_id = str(row.get("question_id") or image_id)
        qa_items = [
            _qa_item(
                source_dataset=self.source_dataset,
                source_profile=self.config.source_profile,
                source_record_id=source_record_id,
                question=question,
                answer=answer,
                aliases=answers,
                choices=[],
                evidence_type=evidence_type,
                metadata={
                    "question_id": row.get("question_id"),
                    "answer_votes": dict(Counter(normalize_answer(item) for item in answers)),
                },
            )
        ]
        return _bundle_record(
            source_dataset=self.source_dataset,
            source_profile=self.config.source_profile,
            split=str(row.get("set_name") or self.config.split),
            source_image_id=image_id,
            image_path=str(image_path),
            source_record_id=source_record_id,
            qa_items=qa_items,
            metadata={"annotation_path": str(annotation_path)},
        )

    def _resolve_image_path(self, image_id: str) -> Path:
        root = self.config.image_root or "textvqa/train_images"
        return self.resolve_path(root) / f"{image_id}.jpg"


class DocVQAAdapter(SourceAdapter):
    def estimate_candidate_qa(self) -> int:
        payload = json.loads(self.annotation_path().read_text(encoding="utf-8"))
        return len(payload.get("data") or [])

    def iter_bundles(self) -> Iterator[dict[str, Any]]:
        annotation_path = self.annotation_path()
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in payload.get("data") or []:
            grouped[str(row.get("image") or "")].append(row)
        for image_rel, rows in grouped.items():
            bundle = self._bundle_from_rows(image_rel, rows, annotation_path)
            if bundle is not None:
                yield bundle

    def select_bundles(self, *, max_bundles: int | None, seed: int) -> list[dict[str, Any]]:
        annotation_path = self.annotation_path()
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in payload.get("data") or []:
            grouped[str(row.get("image") or "")].append(row)
        selected = _select_ranked_records(
            list(grouped.items()),
            max_records=max_bundles,
            seed=seed,
            key_fn=lambda item: stable_image_uid(self.source_dataset, Path(item[0]).stem),
        )
        return [
            bundle
            for image_rel, rows in selected
            for bundle in [self._bundle_from_rows(image_rel, rows, annotation_path)]
            if bundle is not None
        ]

    def _bundle_from_rows(
        self, image_rel: str, rows: list[dict[str, Any]], annotation_path: Path
    ) -> dict[str, Any] | None:
        if not image_rel:
            return None
        image_path = self._resolve_image_path(image_rel)
        source_image_id = Path(image_rel).stem
        qa_items = []
        for row in _limit_items(rows, self.config.max_qa_per_image):
            question = str(row.get("question") or "").strip()
            answers = [str(item).strip() for item in (row.get("answers") or []) if str(item).strip()]
            answer = answers[0] if answers else ""
            source_record_id = str(row.get("questionId") or f"{source_image_id}:{len(qa_items)}")
            evidence_type = infer_evidence_type(self.source_dataset, self.config.source_profile, question)
            qa_items.append(
                _qa_item(
                    source_dataset=self.source_dataset,
                    source_profile=self.config.source_profile,
                    source_record_id=source_record_id,
                    question=question,
                    answer=answer,
                    aliases=answers,
                    choices=[],
                    evidence_type=evidence_type,
                    metadata={
                        "question_id": row.get("questionId"),
                        "question_types": row.get("question_types") or [],
                        "doc_id": row.get("docId"),
                        "document_id": row.get("ucsf_document_id"),
                    },
                )
            )
        return _bundle_record(
            source_dataset=self.source_dataset,
            source_profile=self.config.source_profile,
            split=str(rows[0].get("data_split") or self.config.split),
            source_image_id=source_image_id,
            image_path=str(image_path),
            source_record_id=source_image_id,
            qa_items=qa_items,
            metadata={"annotation_path": str(annotation_path), "source_image": image_rel},
        )

    def _resolve_image_path(self, image_rel: str) -> Path:
        candidate = self.resolve_path(Path("docvqa") / "images" / Path(image_rel).name)
        if candidate.exists():
            return candidate
        root = self.config.image_root or "docvqa/images"
        return self.resolve_path(root) / Path(image_rel).name


class ChartQAAdapter(SourceAdapter):
    def estimate_candidate_qa(self) -> int:
        with self.annotation_path().open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def iter_bundles(self) -> Iterator[dict[str, Any]]:
        annotation_path = self.annotation_path()
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with annotation_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    grouped[str(row.get("image") or row.get("imgname") or "")].append(row)
        for image_path_raw, rows in grouped.items():
            yield self._bundle_from_rows(image_path_raw, rows, annotation_path)

    def select_bundles(self, *, max_bundles: int | None, seed: int) -> list[dict[str, Any]]:
        annotation_path = self.annotation_path()
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with annotation_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    grouped[str(row.get("image") or row.get("imgname") or "")].append(row)
        selected = _select_ranked_records(
            list(grouped.items()),
            max_records=max_bundles,
            seed=seed,
            key_fn=lambda item: stable_image_uid(self.source_dataset, Path(item[0]).stem),
        )
        return [self._bundle_from_rows(image_path_raw, rows, annotation_path) for image_path_raw, rows in selected]

    def _bundle_from_rows(
        self, image_path_raw: str, rows: list[dict[str, Any]], annotation_path: Path
    ) -> dict[str, Any]:
        image_path = self.resolve_path(image_path_raw)
        source_image_id = Path(image_path).stem
        qa_items = []
        for index, row in enumerate(_limit_items(rows, self.config.max_qa_per_image)):
            question = str(row.get("query") or row.get("question") or "").strip()
            labels = row.get("label")
            if isinstance(labels, list):
                answers = [str(item).strip() for item in labels if str(item).strip()]
            else:
                answers = [str(labels).strip()] if labels is not None else []
            answer = answers[0] if answers else ""
            source_record_id = str(row.get("source_index") or f"{source_image_id}:{index}")
            evidence_type = infer_evidence_type(self.source_dataset, self.config.source_profile, question)
            qa_items.append(
                _qa_item(
                    source_dataset=self.source_dataset,
                    source_profile=self.config.source_profile,
                    source_record_id=source_record_id,
                    question=question,
                    answer=answer,
                    aliases=answers,
                    choices=[],
                    evidence_type=evidence_type,
                    metadata={
                        "source_index": row.get("source_index"),
                        "human_or_machine": row.get("human_or_machine"),
                    },
                )
            )
        return _bundle_record(
            source_dataset=self.source_dataset,
            source_profile=self.config.source_profile,
            split=self.config.split,
            source_image_id=source_image_id,
            image_path=str(image_path),
            source_record_id=source_image_id,
            qa_items=qa_items,
            metadata={"annotation_path": str(annotation_path)},
        )


class GenericJsonlAdapter(SourceAdapter):
    """Configurable JSONL adapter for small fixtures or prepared source-QA exports."""

    def estimate_candidate_qa(self) -> int:
        with self.annotation_path().open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def iter_bundles(self) -> Iterator[dict[str, Any]]:
        annotation_path = self.annotation_path()
        meta = self.config.metadata
        image_path_field = str(meta.get("image_path_field", "image_path"))
        image_id_field = str(meta.get("image_id_field", "image_id"))
        question_field = str(meta.get("question_field", "question"))
        answer_field = str(meta.get("answer_field", "answer"))
        answers_field = str(meta.get("answers_field", "answers"))
        choices_field = str(meta.get("choices_field", "choices"))
        split_field = str(meta.get("split_field", "split"))
        source_record_id_field = str(meta.get("source_record_id_field", "id"))
        target_field = str(meta.get("target_field", "target"))
        grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        with annotation_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                image_id = str(_dig(row, image_id_field) or _dig(row, image_path_field) or index)
                uid = stable_image_uid(self.source_dataset, image_id)
                grouped[uid].append((index, row))

        for uid, rows in grouped.items():
            first = rows[0][1]
            raw_image_path = _dig(first, image_path_field)
            image_path = self.resolve_path(str(raw_image_path or ""))
            source_image_id = str(_dig(first, image_id_field) or Path(str(raw_image_path)).stem or uid)
            split = str(_dig(first, split_field) or self.config.split)
            qa_items = []
            for index, row in _limit_items(rows, self.config.max_qa_per_image):
                question = str(_dig(row, question_field) or "").strip()
                raw_answers = _dig(row, answers_field)
                raw_answer = _dig(row, answer_field)
                if isinstance(raw_answers, list):
                    aliases = [str(item).strip() for item in raw_answers if str(item).strip()]
                elif raw_answers:
                    aliases = [str(raw_answers).strip()]
                else:
                    aliases = []
                answer = str(raw_answer or (aliases[0] if aliases else "")).strip()
                if answer and answer not in aliases:
                    aliases.insert(0, answer)
                choices = _coerce_choices(_dig(row, choices_field))
                evidence_type = str(row.get("evidence_type") or meta.get("evidence_type") or "")
                if not evidence_type:
                    evidence_type = infer_evidence_type(
                        self.source_dataset, self.config.source_profile, question
                    )
                answer_type = str(row.get("answer_type") or "")
                if not answer_type:
                    answer_type = infer_answer_type(
                        answer, question=question, choices=choices, evidence_type=evidence_type
                    )
                eval_metric = str(row.get("eval_metric") or "")
                if not eval_metric:
                    eval_metric = infer_eval_metric(answer_type, choices)
                source_record_id = str(_dig(row, source_record_id_field) or f"{source_image_id}:{index}")
                target_text = _dig(row, target_field)
                target_spec = None
                if target_text:
                    target_spec = {
                        "target_text": str(target_text),
                        "focus_type": str(row.get("focus_type") or meta.get("focus_type") or "other"),
                        "entities": [],
                        "attribute_type": None,
                        "relation_type": None,
                        "source": "source_annotation",
                        "confidence": row.get("target_confidence"),
                    }
                qa_items.append(
                    _qa_item(
                        source_dataset=self.source_dataset,
                        source_profile=self.config.source_profile,
                        source_record_id=source_record_id,
                        question=question,
                        answer=answer,
                        aliases=aliases,
                        choices=choices,
                        evidence_type=evidence_type,
                        answer_type=answer_type,
                        eval_metric=eval_metric,
                        difficulty=str(row.get("difficulty") or row.get("difficulty_label") or ""),
                        question_family=str(row.get("question_family") or ""),
                        target_spec=target_spec,
                        metadata={
                            "row_index": index,
                            "stage3_provenance": row.get("provenance") or "source_qa",
                            "teacher_decision": row.get("source_decision")
                            or row.get("teacher_decision"),
                            "tool_need_hint": row.get("tool_need_hint"),
                            "answer_source": row.get("answer_source"),
                            **dict(row.get("metadata") or {}),
                        },
                    )
                )
            yield _bundle_record(
                source_dataset=self.source_dataset,
                source_profile=self.config.source_profile,
                split=split,
                source_image_id=source_image_id,
                image_path=str(image_path),
                source_record_id=source_image_id,
                qa_items=qa_items,
                metadata={"annotation_path": str(annotation_path), "generic_jsonl": True},
            )


def default_source_configs(dataset_root: str | Path = DEFAULT_DATASET_ROOT) -> list[SourceConfig]:
    root = Path(dataset_root)
    return [
        SourceConfig(
            name="visual_genome",
            adapter="visual_genome_qa",
            path=str(root / "visual_genome/annotations/question_answers.json"),
            split="train",
            source_profile="natural_image",
            image_roots=(
                str(root / "visual_genome/VG_100K"),
                str(root / "visual_genome/VG_100K_2"),
            ),
            max_qa_per_image=6,
        ),
        SourceConfig(
            name="textvqa",
            adapter="textvqa",
            path=str(root / "textvqa/annotations/TextVQA_0.5.1_train.json"),
            split="train",
            source_profile="scene_text",
            image_root=str(root / "textvqa/train_images"),
            max_qa_per_image=1,
        ),
        SourceConfig(
            name="docvqa",
            adapter="docvqa",
            path=str(root / "docvqa/annotations/train_v1.0_withQT.json"),
            split="train",
            source_profile="document",
            image_root=str(root / "docvqa/images"),
            max_qa_per_image=6,
        ),
        SourceConfig(
            name="chartqa",
            adapter="chartqa",
            path=str(root / "chartqa/annotations/train.jsonl"),
            split="train",
            source_profile="chart",
            image_root=str(root / "chartqa/images/train"),
            max_qa_per_image=6,
        ),
    ]


def load_source_configs(path: str | Path | None, *, dataset_root: str | Path) -> list[SourceConfig]:
    if path is None:
        return default_source_configs(dataset_root)
    config_path = Path(path)
    payload_text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment.
            raise RuntimeError("YAML source configs require PyYAML; use JSON otherwise") from exc
        payload = yaml.safe_load(payload_text)
    else:
        payload = json.loads(payload_text)
    raw_sources = payload.get("sources", payload) if isinstance(payload, dict) else payload
    if isinstance(raw_sources, dict):
        raw_sources = [
            {"name": name, **(value if isinstance(value, dict) else {"path": value})}
            for name, value in raw_sources.items()
        ]
    return [SourceConfig.from_dict(item) for item in raw_sources]


def adapter_for_config(config: SourceConfig, *, dataset_root: str | Path) -> SourceAdapter:
    adapter_name = config.adapter
    adapter_cls = {
        "visual_genome": VisualGenomeQAAdapter,
        "visual_genome_qa": VisualGenomeQAAdapter,
        "textvqa": TextVQAAdapter,
        "docvqa": DocVQAAdapter,
        "chartqa": ChartQAAdapter,
        "generic_jsonl": GenericJsonlAdapter,
    }.get(adapter_name)
    if adapter_cls is None:
        raise ValueError(f"unsupported Stage3 RL source adapter: {adapter_name}")
    return adapter_cls(config, dataset_root=dataset_root)


def _bundle_record(
    *,
    source_dataset: str,
    source_profile: str,
    split: str,
    source_image_id: str,
    image_path: str,
    source_record_id: str,
    qa_items: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    uid = stable_image_uid(source_dataset, source_image_id, image_path)
    return {
        "bundle_id": stable_bundle_id(source_dataset, uid),
        "stable_image_uid": uid,
        "image_id": source_image_id,
        "image_path": image_path,
        "source_dataset": source_dataset,
        "source_split": split,
        "source_profile": source_profile,
        "source_record_id": source_record_id,
        "metadata": metadata,
        "qa_items": qa_items,
    }


def _qa_item(
    *,
    source_dataset: str,
    source_profile: str,
    source_record_id: str,
    question: str,
    answer: str,
    aliases: list[str],
    choices: list[str],
    evidence_type: str,
    answer_type: str | None = None,
    eval_metric: str | None = None,
    difficulty: str | None = None,
    question_family: str | None = None,
    target_spec: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original_choices = list(choices)
    original_answer = answer
    answer = _open_answer_text(answer, aliases, original_choices)
    if answer and answer not in aliases:
        aliases = [answer, *aliases]
    prompt_choices: list[str] = []
    answer_type = answer_type or infer_answer_type(
        answer, question=question, choices=prompt_choices, evidence_type=evidence_type
    )
    if answer_type == "multiple_choice":
        answer_type = infer_answer_type(
            answer, question=question, choices=prompt_choices, evidence_type=evidence_type
        )
    eval_metric = eval_metric or infer_eval_metric(answer_type, prompt_choices)
    if eval_metric == "mcq":
        eval_metric = infer_eval_metric(answer_type, prompt_choices)
    difficulty = difficulty or infer_difficulty(
        question=question,
        answer_type=answer_type,
        evidence_type=evidence_type,
        source_profile=source_profile,
        choices=prompt_choices,
    )
    built_target = target_spec
    metadata_out = dict(metadata or {})
    if original_choices:
        metadata_out["choice_to_open_answer"] = {
            "original_answer": original_answer,
            "original_answer_format": "multiple_choice",
            "original_choices": original_choices,
            "original_value_span_text": original_answer,
        }
    return {
        "qa_id": stable_qa_id(source_dataset, source_record_id, question, answer),
        "question": question,
        "choices": prompt_choices,
        "original_choices": original_choices,
        "gold_answer": answer,
        "answer_aliases": _unique_aliases(aliases, answer),
        "answer_format": "short_text",
        "answer_type": answer_type,
        "eval_metric": eval_metric,
        "evidence_type": evidence_type,
        "difficulty": difficulty,
        "question_family": question_family or evidence_type,
        "reference_target": None if built_target is None else built_target.get("target_text"),
        "target_spec": built_target,
        "metadata": metadata_out,
    }


def _majority_answer(answers: list[str]) -> str:
    if not answers:
        return ""
    counts = Counter(normalize_answer(item) for item in answers)
    winner = counts.most_common(1)[0][0]
    for item in answers:
        if normalize_answer(item) == winner:
            return item
    return answers[0]


def _clean_answer(value: Any) -> str:
    return str(value or "").strip().strip(".")


def _unique_aliases(aliases: Iterable[str], answer: str) -> list[str]:
    values = [answer, *aliases]
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = normalize_answer(text)
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _dig(row: dict[str, Any], field_path: str) -> Any:
    value: Any = row
    for part in field_path.split("."):
        if not part:
            continue
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _coerce_choices(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [str(item).strip() for item in value.values() if str(item).strip()]
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("answer") or item.get("label")
            else:
                text = item
            if str(text or "").strip():
                out.append(str(text).strip())
        return out
    return [str(value).strip()] if str(value).strip() else []


def _open_answer_text(answer: str, aliases: list[str], choices: list[str]) -> str:
    for candidate in [answer, *aliases]:
        text = str(candidate or "").strip()
        if not text:
            continue
        letter = _answer_choice_letter(text)
        if letter and choices:
            index = ord(letter) - ord("A")
            if 0 <= index < len(choices):
                return choices[index].strip()
        stripped = _strip_choice_letter_prefix(text)
        if stripped and stripped != text:
            return stripped
        if stripped:
            return stripped
    return str(answer or "").strip()


def _answer_choice_letter(answer: str) -> str:
    match = re.match(r"^\s*([A-Z])(?:[.)]|:)?(?:\s+|$)", answer.strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _strip_choice_letter_prefix(text: str) -> str:
    return re.sub(r"^\s*[A-Z](?:[.)]|:)\s+", "", text.strip(), count=1, flags=re.IGNORECASE).strip()


def _limit_items(items: list[Any], limit: int | None) -> list[Any]:
    if limit is None or limit <= 0:
        return items
    return items[:limit]


def _select_ranked_bundles(
    bundles: list[dict[str, Any]],
    *,
    max_bundles: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    return _select_ranked_records(
        bundles,
        max_records=max_bundles,
        seed=seed,
        key_fn=lambda bundle: str(bundle.get("stable_image_uid") or bundle.get("image_path") or ""),
    )


def _select_ranked_records(
    records: list[Any],
    *,
    max_records: int | None,
    seed: int,
    key_fn: Any,
) -> list[Any]:
    if max_records is None or max_records >= len(records):
        return sorted(records, key=lambda item: str(key_fn(item)))
    ranked = sorted(
        records,
        key=lambda item: stable_hash("stage3rl_source_select", seed, key_fn(item), length=40),
    )
    return ranked[:max_records]
