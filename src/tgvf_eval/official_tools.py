from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


SCORING_BACKENDS = ("project", "official", "auto")
OFFICIAL_LLM_MODES = ("disabled", "optional", "required")


@dataclass(frozen=True)
class OfficialToolInfo:
    official_tool_used: bool
    official_tool_path: str | None
    scorer_name: str
    prompt_source: str
    note: str | None = None
    official_compatible: bool = False
    llm_judge_used: bool = False
    llm_judge_required: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "official_tool_used": self.official_tool_used,
            "official_tool_path": self.official_tool_path,
            "scorer_name": self.scorer_name,
            "prompt_source": self.prompt_source,
            "note": self.note,
            "official_compatible": self.official_compatible,
            "llm_judge_used": self.llm_judge_used,
            "llm_judge_required": self.llm_judge_required,
        }


class OfficialScorerUnavailable(RuntimeError):
    pass


class OfficialScorer:
    tool_info: OfficialToolInfo

    def parse_prediction(self, raw_output: str, sample: Any) -> str:
        raise NotImplementedError

    def score_predictions(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError


def validate_scoring_backend(value: str) -> str:
    if value not in SCORING_BACKENDS:
        raise ValueError(f"invalid scoring backend {value!r}; expected one of {SCORING_BACKENDS}")
    return value


def validate_official_llm_mode(value: str) -> str:
    if value not in OFFICIAL_LLM_MODES:
        raise ValueError(f"invalid official LLM mode {value!r}; expected one of {OFFICIAL_LLM_MODES}")
    return value


def resolve_official_tool(
    benchmark: str,
    *,
    benchmark_root: str | Path,
    tools_root: str | Path,
) -> OfficialToolInfo:
    root = Path(benchmark_root)
    tools = Path(tools_root)
    candidates = [
        tools / benchmark,
        tools / f"{benchmark}.py",
        root / benchmark / "official_code",
        root / benchmark / "snapshot" / "eval.py",
        root / benchmark / "snapshot" / "evaluate.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return OfficialToolInfo(
                official_tool_used=False,
                official_tool_path=str(candidate),
                scorer_name="fallback",
                prompt_source="project",
                note="official tool path found; adapter-specific scorer invocation is not wired in v0",
            )
    manifest = tools / "benchmark_manifest.json"
    if manifest.exists():
        try:
            entries = json.loads(manifest.read_text())
            if any(entry.get("name") == benchmark for entry in entries):
                return OfficialToolInfo(
                    official_tool_used=False,
                    official_tool_path=None,
                    scorer_name="fallback",
                    prompt_source="project",
                    note="benchmark is in manifest, but no local official scorer path was found",
                )
        except Exception as exc:  # pragma: no cover - diagnostic branch
            return OfficialToolInfo(
                official_tool_used=False,
                official_tool_path=None,
                scorer_name="fallback",
                prompt_source="project",
                note=f"could not parse tool manifest: {exc}",
            )
    return OfficialToolInfo(
        official_tool_used=False,
        official_tool_path=None,
        scorer_name="fallback",
        prompt_source="project",
        note="no official tool found",
    )


def build_official_scorer(
    benchmark: str,
    *,
    tool_info: OfficialToolInfo,
    scoring_backend: str,
    official_llm_mode: str = "disabled",
) -> OfficialScorer | None:
    scoring_backend = validate_scoring_backend(scoring_backend)
    official_llm_mode = validate_official_llm_mode(official_llm_mode)
    if scoring_backend == "project":
        return None
    try:
        scorer = _build_supported_official_scorer(
            benchmark,
            tool_info=tool_info,
            official_llm_mode=official_llm_mode,
        )
    except OfficialScorerUnavailable:
        if scoring_backend == "official":
            raise
        return None
    return scorer


def _build_supported_official_scorer(
    benchmark: str,
    *,
    tool_info: OfficialToolInfo,
    official_llm_mode: str,
) -> OfficialScorer:
    if benchmark == "mmmu_pro":
        return MMMUProOfficialScorer(tool_info)
    if benchmark == "mathvista":
        return MathVistaOfficialScorer(tool_info, official_llm_mode=official_llm_mode)
    if benchmark == "mathverse":
        return MathVerseOfficialScorer(tool_info, official_llm_mode=official_llm_mode)
    if benchmark == "ocrbench_v2":
        return OCRBenchV2OfficialScorer(tool_info)
    if benchmark == "blink":
        return ChoiceOfficialScorer(tool_info, scorer_name="official_blink_exact_match", note="BLINK official exact-match scoring rules")
    if benchmark == "hr_bench_4k":
        return ChoiceOfficialScorer(
            tool_info,
            scorer_name="official_compatible_hrbench4k_mc",
            note="HRBench4K official-compatible multiple-choice scoring; upstream entrypoint is a full VLMEvalKit runner",
            official_compatible=True,
        )
    if benchmark == "ovo_bench":
        return OVOBenchOfficialScorer(tool_info)
    raise OfficialScorerUnavailable(
        f"official scoring is not wired for {benchmark!r}; use --scoring-backend project "
        "or add a benchmark-specific official scorer wrapper"
    )


class ChoiceOfficialScorer(OfficialScorer):
    def __init__(
        self,
        tool_info: OfficialToolInfo,
        *,
        scorer_name: str,
        note: str,
        official_compatible: bool = False,
    ) -> None:
        self.tool_info = _active_info(
            tool_info,
            scorer_name=scorer_name,
            note=note,
            official_compatible=official_compatible,
        )

    def parse_prediction(self, raw_output: str, sample: Any) -> str:
        return _extract_choice(raw_output, num_choices=len(getattr(sample, "choices", None) or []))

    def score_predictions(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        scored: list[float] = []
        groups: dict[str, list[float]] = {}
        for row in rows:
            gold = row.get("gold_answer") or row.get("label")
            if gold is None or gold == "":
                continue
            pred = row.get("parsed_answer") or row.get("pred_letter") or _extract_choice(
                str(row.get("raw_output") or ""),
                num_choices=len(row.get("choices") or []),
            )
            score = _score_choice(str(pred or ""), str(gold), row.get("choices") or [])
            row["parsed_answer"] = _choice_letter(str(pred or "")) or str(pred or "")
            row["score"] = score
            scored.append(score)
            _add_group_scores(groups, row, score)
        return _score_payload(scored, groups, self.tool_info)


class MMMUProOfficialScorer(OfficialScorer):
    def __init__(self, tool_info: OfficialToolInfo) -> None:
        root = _official_code_root(tool_info)
        path = root / "mmmu-pro" / "evaluate.py" if root.is_dir() else root
        if not path.exists():
            raise OfficialScorerUnavailable(
                "MMMU-Pro official scorer was requested, but "
                f"{path} does not exist"
            )
        self.tool_info = _active_info(
            tool_info,
            official_tool_path=path,
            scorer_name="official_mmmu_pro",
            note="official MMMU-Pro parser/scorer loaded from local official_code",
        )
        self.module = _load_module_from_path(path)

    def parse_prediction(self, raw_output: str, sample: Any) -> str:
        choices = list(getattr(sample, "choices", None) or [])
        if not choices:
            return str(raw_output or "").strip()
        index2ans, all_choices = self.module.get_multi_choice_info(choices)
        return str(self.module.parse_multi_choice_response(str(raw_output or ""), all_choices, index2ans))

    def score_predictions(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        scored: list[float] = []
        groups: dict[str, list[float]] = {}
        random_state = random.getstate()
        try:
            for index, row in enumerate(rows):
                gold = row.get("gold_answer")
                if gold is None or gold == "":
                    continue
                pred = row.get("parsed_answer")
                choices = list(row.get("choices") or [])
                if (pred is None or pred == "") and choices and row.get("raw_output") is not None:
                    index2ans, all_choices = self.module.get_multi_choice_info(choices)
                    random.seed(_stable_seed(row, index))
                    pred = self.module.parse_multi_choice_response(
                        str(row.get("raw_output") or ""),
                        all_choices,
                        index2ans,
                    )
                    row["parsed_answer"] = pred
                correct = bool(self.module.eval_multi_choice(str(gold), str(pred or "")))
                score = 1.0 if correct else 0.0
                row["score"] = score
                scored.append(score)
                _add_group_scores(groups, row, score)
        finally:
            random.setstate(random_state)
        return _score_payload(scored, groups, self.tool_info)


class MathVistaOfficialScorer(OfficialScorer):
    def __init__(self, tool_info: OfficialToolInfo, *, official_llm_mode: str) -> None:
        root = _official_code_root(tool_info)
        self.official_root = root
        self.official_llm_mode = official_llm_mode
        self.tool_info = _active_info(
            tool_info,
            official_tool_path=root / "evaluation" / "calculate_score.py",
            scorer_name="official_mathvista",
            note="MathVista official normalization/scoring; LLM answer extraction is controlled by --official-llm-mode",
            llm_judge_required=official_llm_mode == "required",
        )
        if official_llm_mode == "required":
            _require_mathvista_azure_env()

    def parse_prediction(self, raw_output: str, sample: Any) -> str:
        choices = list(getattr(sample, "choices", None) or [])
        if choices:
            return _extract_choice(raw_output, num_choices=len(choices)) or _clean_answer(raw_output)
        return _extract_final_answer(raw_output)

    def score_predictions(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        scored: list[float] = []
        groups: dict[str, list[float]] = {}
        llm_used = False
        for row in rows:
            gold = row.get("gold_answer")
            if gold is None or gold == "":
                continue
            metadata = row.get("metadata") or {}
            choices = list(row.get("choices") or metadata.get("choices") or [])
            extraction = row.get("parsed_answer") or _extract_final_answer(str(row.get("raw_output") or ""))
            llm_extraction = self._maybe_llm_extract(row, metadata)
            if llm_extraction is not None:
                extraction = llm_extraction
                llm_used = True
            prediction = _mathvista_normalize(
                extraction,
                choices,
                str(metadata.get("question_type") or ("multi_choice" if choices else "free_form")),
                str(metadata.get("answer_type") or "text"),
                metadata.get("precision", 0),
            )
            answer = _mathvista_gold(str(gold), choices)
            score = 1.0 if prediction is not None and _safe_equal(str(prediction), str(answer)) else 0.0
            row["parsed_answer"] = str(extraction or "")
            row["prediction"] = prediction
            row["score"] = score
            scored.append(score)
            _add_group_scores(groups, row, score)
        return _score_payload(scored, groups, self.tool_info, llm_judge_used=llm_used)

    def _maybe_llm_extract(self, row: dict[str, Any], metadata: dict[str, Any]) -> str | None:
        if self.official_llm_mode == "disabled":
            return None
        if not _mathvista_azure_env_present():
            if self.official_llm_mode == "required":
                _require_mathvista_azure_env()
            return None
        try:
            extract_path = self.official_root / "evaluation" / "extract_answer.py"
            module = _load_module_from_path(extract_path, extra_sys_paths=[self.official_root])
            from openai import AzureOpenAI

            client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            )
            model = module.gpt.GPT_Model(client=client, model=os.environ["AZURE_OPENAI_MODEL"])
            problem = {
                "pid": str(row.get("sample_id") or row.get("id") or ""),
                "question_type": metadata.get("question_type") or ("multi_choice" if row.get("choices") else "free_form"),
                "answer_type": metadata.get("answer_type") or "text",
                "choices": row.get("choices") or metadata.get("choices") or [],
                "query": row.get("question") or metadata.get("query") or "",
            }
            return module.extract_answer(model, str(row.get("raw_output") or ""), problem, quick_extract=False)
        except Exception as exc:
            if self.official_llm_mode == "required":
                raise OfficialScorerUnavailable(f"MathVista official LLM extraction failed: {exc}") from exc
            return None


class MathVerseOfficialScorer(OfficialScorer):
    def __init__(self, tool_info: OfficialToolInfo, *, official_llm_mode: str) -> None:
        root = _official_code_root(tool_info)
        self.official_root = root
        self.official_llm_mode = official_llm_mode
        self.tool_info = _active_info(
            tool_info,
            official_tool_path=root / "evaluation" / "score_answer_s2.py",
            scorer_name="official_mathverse",
            note="MathVerse official quick-match scoring; LLM extraction/judge is controlled by --official-llm-mode",
            llm_judge_required=official_llm_mode == "required",
        )
        if official_llm_mode == "required" and not os.environ.get("OPENAI_API_KEY"):
            raise OfficialScorerUnavailable("MathVerse official LLM judge requires OPENAI_API_KEY")

    def parse_prediction(self, raw_output: str, sample: Any) -> str:
        choices = list(getattr(sample, "choices", None) or [])
        return _extract_choice(raw_output, num_choices=len(choices)) or _extract_final_answer(raw_output)

    def score_predictions(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        scored: list[float] = []
        groups: dict[str, list[float]] = {}
        llm_used = False
        for row in rows:
            gold = row.get("gold_answer")
            if gold is None or gold == "":
                continue
            pred = row.get("parsed_answer") or self.parse_prediction(str(row.get("raw_output") or ""), _RowSample(row))
            llm_score = self._maybe_llm_score(row, pred)
            if llm_score is not None:
                score = llm_score
                llm_used = True
            else:
                score = _score_choice(str(pred or ""), str(gold), row.get("choices") or [])
            row["parsed_answer"] = str(pred or "")
            row["score"] = score
            scored.append(score)
            _add_group_scores(groups, row, score)
        return _score_payload(scored, groups, self.tool_info, llm_judge_used=llm_used)

    def _maybe_llm_score(self, row: dict[str, Any], pred: Any) -> float | None:
        if self.official_llm_mode == "disabled":
            return None
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            if self.official_llm_mode == "required":
                raise OfficialScorerUnavailable("MathVerse official LLM judge requires OPENAI_API_KEY")
            return None
        try:
            module = _load_module_from_path(
                self.official_root / "evaluation" / "score_answer_s2.py",
                extra_sys_paths=[self.official_root / "evaluation"],
            )
            metadata = row.get("metadata") or {}
            inst = {
                "question": row.get("question") or "",
                "answer": row.get("gold_answer") or "",
                "extraction": pred,
                "category": metadata.get("category") or metadata,
                "problem_version": metadata.get("version") or metadata.get("problem_version") or "unknown",
            }
            judgement = module.match_answer(inst, api_key, quick_match=False)
            return 1.0 if str(judgement).strip() == "1" else 0.0
        except Exception as exc:
            if self.official_llm_mode == "required":
                raise OfficialScorerUnavailable(f"MathVerse official LLM judge failed: {exc}") from exc
            return None


class OCRBenchV2OfficialScorer(OfficialScorer):
    def __init__(self, tool_info: OfficialToolInfo) -> None:
        root = _official_code_root(tool_info)
        path = root / "OCRBench_v2" / "eval_scripts" / "eval.py"
        if not path.exists():
            raise OfficialScorerUnavailable(f"OCRBench v2 official eval.py not found at {path}")
        self.tool_info = _active_info(
            tool_info,
            official_tool_path=path,
            scorer_name="official_ocrbench_v2",
            note="official OCRBench v2 eval.py::process_predictions over converted prediction rows",
        )
        self.module = _load_module_from_path(path, extra_sys_paths=[path.parent], stub_modules=["ipdb"])

    def parse_prediction(self, raw_output: str, sample: Any) -> str:
        return _extract_final_answer(raw_output)

    def score_predictions(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        items = []
        scored_rows = []
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            answers = metadata.get("answers") or ([row.get("gold_answer")] if row.get("gold_answer") not in (None, "") else [])
            task_type = metadata.get("type") or metadata.get("task")
            if not task_type or not answers:
                continue
            item = dict(metadata)
            item.update(
                type=task_type,
                question=row.get("question") or metadata.get("question") or "",
                answers=answers,
                predict=row.get("parsed_answer") or row.get("raw_output") or "",
            )
            items.append(item)
            scored_rows.append(row)
        if not items:
            return _score_payload([], {}, self.tool_info)
        with tempfile.TemporaryDirectory(prefix="tgvf_ocrbench_official_") as tmp:
            input_path = Path(tmp) / "predictions.json"
            output_path = Path(tmp) / "scores.json"
            input_path.write_text(json.dumps(items, ensure_ascii=False))
            self.module.process_predictions(str(input_path), str(output_path))
            scored_items = json.loads(output_path.read_text())
        scored: list[float] = []
        groups: dict[str, list[float]] = {}
        for row, item in zip(scored_rows, scored_items):
            score = float(item.get("score") or 0.0)
            row["score"] = score
            scored.append(score)
            _add_group_scores(groups, row, score, extra_group=item.get("type"))
        return _score_payload(scored, groups, self.tool_info)


class OVOBenchOfficialScorer(OfficialScorer):
    def __init__(self, tool_info: OfficialToolInfo) -> None:
        self.tool_info = _active_info(
            tool_info,
            scorer_name="official_compatible_ovo_bench_offline",
            note="OVO-Bench offline scoring rules adapted to one-row-per-sample predictions",
            official_compatible=True,
        )

    def parse_prediction(self, raw_output: str, sample: Any) -> str:
        return _extract_choice(raw_output, num_choices=len(getattr(sample, "choices", None) or [])) or _extract_final_answer(raw_output)

    def score_predictions(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        scored: list[float] = []
        groups: dict[str, list[float]] = {}
        for row in rows:
            task = str((row.get("metadata") or {}).get("task") or row.get("task") or "")
            gold = row.get("gold_answer") or row.get("label")
            if gold is None or gold == "":
                continue
            pred = row.get("parsed_answer") or _extract_final_answer(str(row.get("raw_output") or ""))
            if task == "REC":
                score = 1.0 if re.sub(r"\D+", "", str(pred or "")) == str(gold) else 0.0
            elif task in {"SSR", "CRR"}:
                score = _score_yes_no(str(pred or ""), str(gold))
            else:
                score = _score_choice(str(pred or ""), str(gold), row.get("choices") or [])
                if score == 0.0 and not row.get("choices"):
                    score = 1.0 if str(gold).lower() in str(pred or "").lower() else 0.0
            row["parsed_answer"] = str(pred or "")
            row["score"] = score
            scored.append(score)
            _add_group_scores(groups, row, score)
        return _score_payload(scored, groups, self.tool_info)


class _RowSample:
    def __init__(self, row: dict[str, Any]) -> None:
        self.choices = row.get("choices") or []


def _active_info(
    tool_info: OfficialToolInfo,
    *,
    scorer_name: str,
    note: str,
    official_tool_path: Path | str | None = None,
    official_compatible: bool = False,
    llm_judge_required: bool = False,
) -> OfficialToolInfo:
    return OfficialToolInfo(
        official_tool_used=True,
        official_tool_path=str(official_tool_path or tool_info.official_tool_path) if (official_tool_path or tool_info.official_tool_path) else None,
        scorer_name=scorer_name,
        prompt_source=tool_info.prompt_source,
        note=note,
        official_compatible=official_compatible,
        llm_judge_required=llm_judge_required,
    )


def _official_code_root(tool_info: OfficialToolInfo) -> Path:
    path = Path(tool_info.official_tool_path or "")
    if not path.exists():
        raise OfficialScorerUnavailable("official tool path was not found")
    return path if path.is_dir() else path.parent


@contextmanager
def _temporary_sys_path(paths: list[Path] | None):
    if not paths:
        yield
        return
    additions = [str(path) for path in paths]
    old = list(sys.path)
    sys.path[:0] = additions
    try:
        yield
    finally:
        sys.path[:] = old


def _load_module_from_path(
    path: Path,
    *,
    extra_sys_paths: list[Path] | None = None,
    stub_modules: list[str] | None = None,
) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"tgvf_eval_official_{path.stem}_{abs(hash(str(path)))}", path)
    if spec is None or spec.loader is None:
        raise OfficialScorerUnavailable(f"could not load official scorer module from {path}")
    module = importlib.util.module_from_spec(spec)
    previous_stubs: dict[str, ModuleType | None] = {}
    for name in stub_modules or []:
        previous_stubs[name] = sys.modules.get(name)
        if name not in sys.modules:
            sys.modules[name] = ModuleType(name)
    try:
        with _temporary_sys_path(extra_sys_paths):
            spec.loader.exec_module(module)
    except Exception as exc:
        raise OfficialScorerUnavailable(f"could not load official scorer module from {path}: {exc}") from exc
    finally:
        for name, previous in previous_stubs.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def _extract_choice(text: Any, *, num_choices: int = 26) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"<\|[^>]+\|>", " ", cleaned)
    valid = {chr(ord("A") + index) for index in range(max(num_choices, 1))}
    patterns = (
        r"<ANSWER>\s*\(?\s*([A-Z])\s*\)?",
        r"(?i)(?:final\s+answer|answer|option|choice)\s*(?:is|:|=)?\s*\(?\s*([A-Z])\s*\)?",
        r"\(([A-Z])\)",
        r"(?m)^\s*([A-Z])\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned.strip())
        if match:
            letter = match.group(1).upper()
            if letter in valid:
                return letter
    compact = _choice_letter(cleaned)
    return compact if compact in valid else ""


def _choice_letter(text: str) -> str:
    compact = str(text or "").strip().upper().strip("()[]{}.: ")
    return compact if len(compact) == 1 and "A" <= compact <= "Z" else ""


def _score_choice(prediction: str, gold: str, choices: list[str]) -> float:
    pred_letter = _choice_letter(prediction)
    gold_letter = _choice_letter(gold)
    if pred_letter and gold_letter:
        return 1.0 if pred_letter == gold_letter else 0.0
    pred = str(prediction or "").strip()
    gold_text = str(gold or "").strip()
    if pred_letter and choices:
        index = ord(pred_letter) - ord("A")
        if 0 <= index < len(choices):
            pred = choices[index]
    if gold_letter and choices:
        index = ord(gold_letter) - ord("A")
        if 0 <= index < len(choices):
            gold_text = choices[index]
    return 1.0 if _normalize_text(pred) == _normalize_text(gold_text) else 0.0


def _score_yes_no(prediction: str, gold: str) -> float:
    pred = str(prediction or "").strip().lower()
    expected = str(gold or "").strip().lower()
    if expected in {"0", "no", "n"}:
        return 1.0 if pred in {"no", "n"} or "no" in pred else 0.0
    if expected in {"1", "yes", "y"}:
        return 1.0 if pred in {"yes", "y"} or "yes" in pred else 0.0
    return 1.0 if expected and expected in pred else 0.0


def _extract_final_answer(text: Any) -> str:
    cleaned = str(text or "").strip()
    matches = list(re.finditer(r"<ANSWER>\s*(.*?)(?:</ANSWER>|$)", cleaned, flags=re.IGNORECASE | re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    answer_matches = list(re.finditer(r"(?i)(?:final\s+answer|answer)\s*(?:is|:|=)\s*(.+)$", cleaned))
    if answer_matches:
        return answer_matches[-1].group(1).strip().strip(". ")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned.replace(",", ""))
    if numbers:
        return numbers[-1]
    return _clean_answer(cleaned)


def _clean_answer(text: Any) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _mathvista_normalize(
    extraction: Any,
    choices: list[str],
    question_type: str,
    answer_type: str,
    precision: Any,
) -> str | None:
    extraction_text = _clean_answer(extraction)
    if question_type in {"multi_choice", "multi-choice"} or choices:
        letter = _choice_letter(extraction_text)
        if letter and choices:
            index = ord(letter) - ord("A")
            if 0 <= index < len(choices):
                return choices[index]
        if extraction_text in choices:
            return extraction_text
        if choices:
            return min(choices, key=lambda choice: _edit_distance(extraction_text, choice))
        return extraction_text or None
    if answer_type == "integer":
        numbers = re.findall(r"-?\d+(?:\.\d+)?", extraction_text.replace(",", ""))
        if not numbers:
            return None
        try:
            return str(int(float(numbers[-1])))
        except Exception:
            return None
    if answer_type == "float":
        numbers = re.findall(r"-?\d+(?:\.\d+)?", extraction_text.replace(",", ""))
        if not numbers:
            return None
        try:
            return str(round(float(numbers[-1]), int(float(precision or 0))))
        except Exception:
            return None
    if answer_type == "list":
        return extraction_text
    return extraction_text or None


def _mathvista_gold(gold: str, choices: list[str]) -> str:
    letter = _choice_letter(gold)
    if letter and choices:
        index = ord(letter) - ord("A")
        if 0 <= index < len(choices):
            return choices[index]
    return gold


def _safe_equal(prediction: str, answer: str) -> bool:
    return prediction == answer


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())).strip()


def _add_group_scores(
    groups: dict[str, list[float]],
    row: dict[str, Any],
    score: float,
    *,
    extra_group: Any | None = None,
) -> None:
    metadata = row.get("metadata") or {}
    for key in ("type", "sub_task", "subdomain", "subject", "category", "task", "subset", "version", "cycle_category"):
        value = metadata.get(key) or row.get(key)
        if value not in (None, ""):
            groups.setdefault(f"{key}:{value}", []).append(score)
    if extra_group not in (None, ""):
        groups.setdefault(f"type:{extra_group}", []).append(score)


def _score_payload(
    scored: list[float],
    groups: dict[str, list[float]],
    tool_info: OfficialToolInfo,
    *,
    llm_judge_used: bool | None = None,
) -> dict[str, Any]:
    payload = {
        "score": _mean(scored),
        "accuracy": _mean(scored),
        "num_scored": len(scored),
        "by_group": {key: _mean(values) for key, values in sorted(groups.items())},
        **tool_info.to_dict(),
    }
    if llm_judge_used is not None:
        payload["llm_judge_used"] = llm_judge_used
    return payload


def _mathvista_azure_env_present() -> bool:
    return all(
        os.environ.get(name)
        for name in (
            "AZURE_OPENAI_API_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_OPENAI_MODEL",
        )
    )


def _require_mathvista_azure_env() -> None:
    if not _mathvista_azure_env_present():
        raise OfficialScorerUnavailable(
            "MathVista official LLM extraction requires AZURE_OPENAI_API_ENDPOINT, "
            "AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_MODEL"
        )


def _stable_seed(row: dict[str, Any], index: int) -> int:
    key = str(row.get("sample_id") or row.get("id") or index)
    return sum((offset + 1) * ord(char) for offset, char in enumerate(key)) % (2**32)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
