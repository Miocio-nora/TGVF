from __future__ import annotations

import ast
import base64
import binascii
import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tgvf_eval.config import MethodConfig
from tgvf_eval.official_tools import (
    OfficialScorer,
    OfficialToolInfo,
    build_official_scorer,
    resolve_official_tool,
)
from tgvf_eval.parsing import normalize_open_answer, parse_prediction
from tgvf_eval.prompts import build_prompt
from tgvf_eval.sampling import deterministic_sample


BENCHMARK_NAMES = (
    "vstar_bench",
    "hr_bench_4k",
    "ocrbench_v2",
    "blink",
    "mmmu_pro",
    "mathvista",
    "mathverse",
    "ovo_bench",
)


@dataclass
class BenchmarkSample:
    benchmark: str
    sample_id: str
    question: str
    media: list[Any] = field(default_factory=list)
    choices: list[str] = field(default_factory=list)
    gold_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_media(self) -> Any | None:
        for item in self.media:
            if not isinstance(item, str):
                return item
            if item.startswith(("http://", "https://")) or Path(item).exists():
                return item
        return self.media[0] if self.media else None


class BenchmarkAdapter:
    name = "base"
    preferred_files: tuple[str, ...] = ()
    multiple_choice = False

    def __init__(
        self,
        benchmark_root: str | Path,
        tools_root: str | Path | None = None,
        *,
        scoring_backend: str = "project",
        official_llm_mode: str = "disabled",
    ) -> None:
        self.benchmark_root = Path(benchmark_root)
        self.tools_root = Path(tools_root) if tools_root else self.benchmark_root / "_tools"
        self.root = self.benchmark_root / self.name
        self.snapshot = self.root / "snapshot"
        self.tool_info = resolve_official_tool(
            self.name,
            benchmark_root=self.benchmark_root,
            tools_root=self.tools_root,
        )
        self.official_scorer: OfficialScorer | None = build_official_scorer(
            self.name,
            tool_info=self.tool_info,
            scoring_backend=scoring_backend,
            official_llm_mode=official_llm_mode,
        )
        if self.official_scorer is not None:
            self.tool_info = self.official_scorer.tool_info

    def load_samples(self, max_records: int | None = None) -> list[BenchmarkSample]:
        for rel_path in self.preferred_files:
            path = self.root / rel_path
            if path.exists():
                return [
                    self.record_to_sample(record, path=path, index=index)
                    for index, record in enumerate(_read_records(path, max_records=max_records))
                ]
        path = self._discover_annotation_file()
        if path is None:
            raise FileNotFoundError(f"No annotation file found for {self.name} under {self.root}")
        return [
            self.record_to_sample(record, path=path, index=index)
            for index, record in enumerate(_read_records(path, max_records=max_records))
        ]

    def sample(
        self,
        *,
        tier: str,
        limit: int | None,
        seed: int,
        stratify: bool = True,
    ) -> list[BenchmarkSample]:
        max_records = None
        if limit is not None and limit <= 20:
            max_records = max(limit * 20, limit)
        return list(
            deterministic_sample(
                self.load_samples(max_records=max_records),
                benchmark=self.name,
                tier=tier,
                limit=limit,
                seed=seed,
                stratify=stratify,
            )
        )

    def build_prompt(self, sample: BenchmarkSample, method_config: MethodConfig) -> str:
        return build_prompt(sample.question, method_config).prompt

    def parse_prediction(self, raw_output: str, sample: BenchmarkSample) -> str:
        if self.official_scorer is not None:
            return self.official_scorer.parse_prediction(raw_output, sample)
        return parse_prediction(raw_output, choices=sample.choices)

    def score_predictions(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if self.official_scorer is not None:
            return self.official_scorer.score_predictions(rows)
        scored = []
        groups: dict[str, list[float]] = {}
        for row in rows:
            gold = row.get("gold_answer")
            pred = row.get("parsed_answer")
            if gold is None or gold == "":
                continue
            choices = row.get("choices") or []
            score = _score_value(str(pred or ""), str(gold), choices=choices, multiple_choice=bool(choices))
            row["score"] = score
            scored.append(score)
            for key in ("category", "task", "mode", "subset", "subject", "version"):
                value = (row.get("metadata") or {}).get(key)
                if value:
                    groups.setdefault(f"{key}:{value}", []).append(score)
        return {
            "score": _mean(scored),
            "accuracy": _mean(scored),
            "num_scored": len(scored),
            "by_group": {key: _mean(values) for key, values in sorted(groups.items())},
            **self.tool_info.to_dict(),
        }

    def official_tool_info(self) -> OfficialToolInfo:
        return self.tool_info

    def record_to_sample(self, record: dict[str, Any], *, path: Path, index: int) -> BenchmarkSample:
        sample_id = str(
            record.get("question_id")
            or record.get("id")
            or record.get("uid")
            or record.get("sample_id")
            or f"{path.stem}-{index}"
        )
        question = str(
            record.get("question")
            or record.get("text")
            or record.get("query")
            or record.get("prompt")
            or ""
        )
        choices = _extract_choices(record)
        question = _question_with_choices(question, choices)
        gold = _extract_gold(record)
        media = _extract_media(record, base_dir=path.parent, benchmark_dir=self.root)
        metadata = {
            key: value
            for key, value in record.items()
            if key
            in {
                "category",
                "task",
                "task_type",
                "ability",
                "subject",
                "mode",
                "subset",
                "version",
                "source",
                "split",
            }
        }
        return BenchmarkSample(
            benchmark=self.name,
            sample_id=sample_id,
            question=question,
            media=media,
            choices=choices,
            gold_answer=gold,
            metadata=metadata,
            raw=record,
        )

    def _discover_annotation_file(self) -> Path | None:
        roots = [self.root / "data", self.snapshot, self.root]
        suffixes = (".jsonl", ".json", ".tsv", ".csv", ".parquet")
        candidates: list[Path] = []
        for root in roots:
            if not root.exists():
                continue
            for child in root.rglob("*"):
                if child.is_file() and child.suffix.lower() in suffixes:
                    if child.name.startswith(".") or child.name in {"DEPLOYMENT.json", "benchmark_manifest.json"}:
                        continue
                    if any(part.startswith(".") for part in child.relative_to(root).parts):
                        continue
                    candidates.append(child)
        if not candidates:
            return None
        preferred = sorted(
            candidates,
            key=lambda path: (
                0 if "test" in path.name.lower() else 1,
                0 if any(token in path.name.lower() for token in ("annot", "question", "bench")) else 1,
                path.suffix != ".jsonl",
                len(path.parts),
                str(path),
            ),
        )
        return preferred[0]


class VStarBenchAdapter(BenchmarkAdapter):
    name = "vstar_bench"
    preferred_files = ("snapshot/test_questions.jsonl",)
    multiple_choice = True

    def record_to_sample(self, record: dict[str, Any], *, path: Path, index: int) -> BenchmarkSample:
        sample = super().record_to_sample(record, path=path, index=index)
        sample.media = [_resolve_media_path(str(record["image"]), self.snapshot)]
        sample.choices = _choices_from_question(sample.question) or sample.choices
        sample.metadata.setdefault("category", record.get("category"))
        return sample


class HRBench4KAdapter(BenchmarkAdapter):
    name = "hr_bench_4k"
    preferred_files = ("snapshot/hr_bench_4k.parquet", "snapshot/hr_bench_4k.tsv")
    multiple_choice = True

    def record_to_sample(self, record: dict[str, Any], *, path: Path, index: int) -> BenchmarkSample:
        sample = super().record_to_sample(record, path=path, index=index)
        sample.metadata.setdefault("category", record.get("category"))
        sample.metadata.setdefault("cycle_category", record.get("cycle_category"))
        return sample


class OCRBenchV2Adapter(BenchmarkAdapter):
    name = "ocrbench_v2"
    preferred_files = ("snapshot/EN/test-00000-of-00003.parquet", "snapshot/data/test-00000-of-00004.parquet")

    def record_to_sample(self, record: dict[str, Any], *, path: Path, index: int) -> BenchmarkSample:
        sample = super().record_to_sample(record, path=path, index=index)
        for key in ("dataset_name", "type", "answers", "content", "bbox", "image_shape", "raw_text"):
            if key in record:
                sample.metadata.setdefault(key, record.get(key))
        sample.metadata.setdefault("subset", record.get("dataset_name"))
        return sample


class BlinkAdapter(BenchmarkAdapter):
    name = "blink"
    # BLINK test labels are hidden in the local snapshot. Use the validation
    # split for local scored diagnostics.
    preferred_files = ("snapshot/Counting/val-00000-of-00001.parquet",)
    multiple_choice = True

    def record_to_sample(self, record: dict[str, Any], *, path: Path, index: int) -> BenchmarkSample:
        sample = super().record_to_sample(record, path=path, index=index)
        sample.metadata.setdefault("sub_task", record.get("sub_task"))
        return sample


class MMMUProAdapter(BenchmarkAdapter):
    name = "mmmu_pro"
    preferred_files = ("snapshot/standard (10 options)/test-00000-of-00002.parquet",)
    multiple_choice = True


class MathVistaAdapter(BenchmarkAdapter):
    name = "mathvista"
    preferred_files = ("snapshot/data/testmini-00000-of-00001-725687bf7a18d64b.parquet", "snapshot/annot_testmini.json")

    def record_to_sample(self, record: dict[str, Any], *, path: Path, index: int) -> BenchmarkSample:
        sample = super().record_to_sample(record, path=path, index=index)
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            sample.metadata.update(metadata)
        for key in ("pid", "question_type", "answer_type", "precision", "unit", "query"):
            if key in record:
                sample.metadata.setdefault(key, record.get(key))
        if sample.choices:
            sample.metadata.setdefault("choices", sample.choices)
        if isinstance(sample.primary_media, str) and not Path(sample.primary_media).exists():
            sample.media = [_resolve_media_path(Path(sample.primary_media).name, self.snapshot / "images")]
        return sample


class MathVerseAdapter(BenchmarkAdapter):
    name = "mathverse"
    preferred_files = ("snapshot/testmini.json",)

    def record_to_sample(self, record: dict[str, Any], *, path: Path, index: int) -> BenchmarkSample:
        sample = super().record_to_sample(record, path=path, index=index)
        if sample.media:
            raw_image = record.get("image")
            if isinstance(raw_image, str):
                candidate = self.snapshot / "images" / raw_image
                if candidate.exists():
                    sample.media = [str(candidate)]
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            sample.metadata.update(metadata)
        for key in ("sample_index", "problem_index", "problem_version", "question_type"):
            if key in record:
                sample.metadata.setdefault(key, record.get(key))
        sample.metadata.setdefault("version", record.get("problem_version"))
        return sample


class OVOBenchAdapter(BenchmarkAdapter):
    name = "ovo_bench"
    preferred_files = ("data/ovo_bench_new.json", "official_code/data/ovo_bench_new.json")
    multiple_choice = True

    def record_to_sample(self, record: dict[str, Any], *, path: Path, index: int) -> BenchmarkSample:
        sample = super().record_to_sample(record, path=path, index=index)
        raw_video = record.get("video")
        if isinstance(raw_video, str):
            for candidate in (
                self.root / "data" / "src_videos" / raw_video,
                self.root / "data" / "chunked_videos" / Path(raw_video).name,
                path.parent / "src_videos" / raw_video,
                path.parent / "chunked_videos" / Path(raw_video).name,
            ):
                if candidate.exists():
                    sample.media = [str(candidate)]
                    break
        if isinstance(record.get("options"), list):
            sample.choices = [str(choice) for choice in record["options"]]
        if isinstance(record.get("gt"), int):
            sample.gold_answer = chr(ord("A") + int(record["gt"]))
        sample.metadata.setdefault("task", record.get("task"))
        sample.metadata.setdefault("gt", record.get("gt"))
        sample.metadata.setdefault("answer", record.get("answer"))
        return sample


class BenchmarkRegistry:
    adapters = {
        "vstar_bench": VStarBenchAdapter,
        "hr_bench_4k": HRBench4KAdapter,
        "ocrbench_v2": OCRBenchV2Adapter,
        "blink": BlinkAdapter,
        "mmmu_pro": MMMUProAdapter,
        "mathvista": MathVistaAdapter,
        "mathverse": MathVerseAdapter,
        "ovo_bench": OVOBenchAdapter,
    }

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return BENCHMARK_NAMES

    @classmethod
    def get(
        cls,
        name: str,
        *,
        benchmark_root: str | Path,
        tools_root: str | Path | None = None,
        scoring_backend: str = "project",
        official_llm_mode: str = "disabled",
    ) -> BenchmarkAdapter:
        if name not in cls.adapters:
            raise KeyError(f"Unknown benchmark '{name}'. Expected one of {BENCHMARK_NAMES}")
        return cls.adapters[name](
            benchmark_root=benchmark_root,
            tools_root=tools_root,
            scoring_backend=scoring_backend,
            official_llm_mode=official_llm_mode,
        )


def _read_records(path: Path, max_records: int | None = None) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        count = 0
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
                    count += 1
                    if max_records is not None and count >= max_records:
                        break
        return
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            for key in ("data", "questions", "annotations", "examples"):
                if isinstance(payload.get(key), list):
                    for record in payload[key]:
                        yield _ensure_record(record)
                    return
            for key, value in payload.items():
                if isinstance(value, dict):
                    record = dict(value)
                    record.setdefault("id", key)
                    yield record
            return
        for record in payload:
            yield _ensure_record(record)
        return
    if suffix in {".tsv", ".csv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for record in reader:
                yield dict(record)
        return
    if suffix == ".parquet":
        try:
            from datasets import Dataset

            dataset = Dataset.from_parquet(str(path))
            count = 0
            for record in dataset:
                yield dict(record)
                count += 1
                if max_records is not None and count >= max_records:
                    break
            return
        except Exception as exc:
            raise RuntimeError(f"Could not read parquet annotations at {path}: {exc}") from exc
    raise ValueError(f"Unsupported annotation file: {path}")


def _ensure_record(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValueError(f"Expected record object, got {type(value).__name__}")


def _extract_choices(record: dict[str, Any]) -> list[str]:
    for key in ("choices", "options", "answer_options"):
        value = record.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            for parser in (json.loads, ast.literal_eval):
                try:
                    decoded = parser(value)
                    if isinstance(decoded, list):
                        return [str(item) for item in decoded]
                except Exception:
                    pass
    letter_choices = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if letter in record and record[letter] not in (None, ""):
            letter_choices.append(str(record[letter]))
        elif letter_choices:
            break
    if letter_choices:
        return letter_choices
    question = str(record.get("question") or record.get("text") or "")
    return _choices_from_question(question)


def _question_with_choices(question: str, choices: list[str]) -> str:
    if not choices or _choices_from_question(question):
        return question
    lines = [question.rstrip(), ""]
    for index, choice in enumerate(choices):
        lines.append(f"({chr(ord('A') + index)}) {choice}")
    lines.append("Answer only with the option letter.")
    return "\n".join(lines).strip()


def _choices_from_question(question: str) -> list[str]:
    choices: list[str] = []
    for line in question.splitlines():
        stripped = line.strip()
        if len(stripped) > 3 and stripped[0] == "(" and stripped[2] == ")":
            choices.append(stripped[3:].strip())
            continue
        if len(stripped) > 2 and stripped[0].isalpha() and stripped[1] in {":", "."}:
            choices.append(stripped[2:].strip())
    return choices


def _extract_gold(record: dict[str, Any]) -> str | None:
    for key in ("label", "answer", "gold", "gold_answer", "correct_answer", "target"):
        value = record.get(key)
        if value is not None and value != "":
            text = str(value).strip()
            return None if text.lower() == "hidden" else text
    answers = record.get("answers")
    if isinstance(answers, list) and answers:
        return str(answers[0]).strip()
    return None


def _extract_media(record: dict[str, Any], *, base_dir: Path, benchmark_dir: Path) -> list[Any]:
    values: list[Any] = []
    for key, value in record.items():
        if not value:
            continue
        if key in {"image", "decoded_image", "image_path", "img", "images", "video", "video_path", "media"} or key.startswith("image_"):
            values.extend(value if isinstance(value, list) else [value])
    media: list[Any] = []
    for value in values:
        if isinstance(value, str):
            decoded_image = _decode_embedded_image(value)
            if decoded_image is not None:
                media.append(decoded_image)
            elif value.startswith(("http://", "https://")):
                media.append(value)
            else:
                media.append(_resolve_media_path(value, base_dir, benchmark_dir))
        else:
            media.append(value)
    return media


def _decode_embedded_image(value: str) -> Any | None:
    payload = value.strip()
    if not payload:
        return None
    if payload.startswith("data:image") and "," in payload:
        payload = payload.split(",", 1)[1]
    elif not payload.startswith(("/9j/", "iVBOR", "R0lGOD", "UklGR")):
        return None
    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB").copy()
    except Exception:
        return None


def _resolve_media_path(value: str, *roots: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    for root in roots:
        candidate = root / value
        if candidate.exists():
            return str(candidate)
    return str(roots[0] / value) if roots else value


def _score_value(prediction: str, gold: str, *, choices: list[str] | None = None, multiple_choice: bool) -> float:
    if multiple_choice:
        choices = choices or []
        pred = prediction.strip()
        gold_text = gold.strip()
        pred_letter = _normalize_choice_letter(pred)
        gold_letter = _normalize_choice_letter(gold_text)
        if pred_letter and gold_letter:
            return 1.0 if pred_letter == gold_letter else 0.0
        if pred_letter and choices:
            index = ord(pred_letter) - ord("A")
            if 0 <= index < len(choices):
                pred = choices[index]
        if gold_letter and choices:
            index = ord(gold_letter) - ord("A")
            if 0 <= index < len(choices):
                gold_text = choices[index]
        return 1.0 if normalize_open_answer(pred) == normalize_open_answer(gold_text) else 0.0
    return 1.0 if normalize_open_answer(prediction) == normalize_open_answer(gold) else 0.0


def _normalize_choice_letter(text: str) -> str:
    compact = str(text or "").strip().upper()
    compact = compact.strip("()[]{}.: ")
    return compact if len(compact) == 1 and "A" <= compact <= "Z" else ""


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)
