"""OpenAI batch runner for Stage3 RL teacher-triage requests."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from revisit_vlm_clean.schema import _to_jsonable

from .schemas import now_iso, stable_hash
from .sources import DEFAULT_SOURCE_MIX

DEFAULT_BATCH_REQUEST_MB = 90


def prepare_teacher_batch_files(
    *,
    output_root: str | Path,
    limit_requests: int | None = None,
    max_request_file_mb: int = DEFAULT_BATCH_REQUEST_MB,
    stratified_requests: bool = True,
    skip_existing_outputs: bool = True,
    skip_submitted_requests: bool = False,
    source_weights_override: dict[str, float] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    teacher_requests_path = root / "teacher_requests.jsonl"
    if not teacher_requests_path.exists():
        raise FileNotFoundError(f"missing teacher_requests.jsonl: {teacher_requests_path}")
    request_dir = root / "api_requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    for stale in request_dir.glob("batch_requests_part_*.jsonl"):
        stale.unlink()
    index_path = request_dir / "request_index.jsonl"

    plan = _read_json(root / "stage3_rl_data_plan.json")
    source_weights = source_weights_override or (
        ((plan.get("balance_config") or {}).get("source_weights") if plan else None)
        or DEFAULT_SOURCE_MIX
    )
    seed = int((plan or {}).get("random_seed") or 42)
    all_requests = _read_jsonl(teacher_requests_path)
    archived_custom_ids = _archived_teacher_custom_ids(root)
    completed_custom_ids = _completed_teacher_custom_ids(root) if skip_existing_outputs else set()
    submitted_custom_ids = _submitted_teacher_custom_ids(root) if skip_submitted_requests else set()
    skipped_custom_ids = archived_custom_ids | completed_custom_ids | submitted_custom_ids
    pending_requests = [
        request
        for request in all_requests
        if str(request.get("custom_id") or "") not in skipped_custom_ids
    ]
    selected_requests = _select_teacher_requests(
        pending_requests,
        limit_requests=limit_requests,
        source_weights=source_weights,
        seed=seed,
        stratified=stratified_requests,
    )

    request_files: list[Path] = []
    request_index: list[dict[str, Any]] = []
    max_bytes = max_request_file_mb * 1024 * 1024
    part_index = 0
    current_path = request_dir / f"batch_requests_part_{part_index:03d}.jsonl"
    current = current_path.open("w", encoding="utf-8")
    current_bytes = 0
    written = 0
    try:
        for request in selected_requests:
            body = _api_payload_with_image(request)
            line_payload = {
                "custom_id": request["custom_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            line = json.dumps(_to_jsonable(line_payload), ensure_ascii=False, sort_keys=True)
            line_bytes = len(line.encode("utf-8")) + 1
            if current_bytes and current_bytes + line_bytes > max_bytes:
                current.close()
                request_files.append(current_path)
                part_index += 1
                current_path = request_dir / f"batch_requests_part_{part_index:03d}.jsonl"
                current = current_path.open("w", encoding="utf-8")
                current_bytes = 0
            current.write(line + "\n")
            current_bytes += line_bytes
            written += 1
            request_index.append(
                {
                    "custom_id": request["custom_id"],
                    "request_id": request["request_id"],
                    "stable_image_uid": request["stable_image_uid"],
                    "image_path": request["image_path"],
                    "source_dataset": request["source_dataset"],
                    "source_profile": request["source_profile"],
                }
            )
    finally:
        current.close()
    if current_bytes:
        request_files.append(current_path)
    else:
        current_path.unlink(missing_ok=True)
    _write_jsonl(index_path, request_index)
    report = {
        "schema_version": "stage3_rl_api_batch_prepare_report_v0",
        "created_at": now_iso(),
        "output_root": str(root),
        "teacher_requests_path": str(teacher_requests_path),
        "request_count": written,
        "teacher_request_count": len(all_requests),
        "pending_request_count": len(pending_requests),
        "skipped_archived_requests": len(archived_custom_ids),
        "skipped_existing_outputs": len(completed_custom_ids),
        "skipped_submitted_requests": len(submitted_custom_ids - completed_custom_ids),
        "request_file_count": len(request_files),
        "request_files": [str(path) for path in request_files],
        "request_index": str(index_path),
        "max_request_file_mb": max_request_file_mb,
        "limit_requests": limit_requests,
        "selection_mode": "source_mix_stratified_v0" if stratified_requests else "file_order_v0",
        "source_weights": source_weights,
        "source_distribution": dict(Counter(row["source_dataset"] for row in request_index)),
        "source_bucket_distribution": dict(
            Counter(_source_bucket(row, source_weights) for row in request_index)
        ),
    }
    _write_json(request_dir / "batch_prepare_report.json", report)
    return report


def submit_teacher_batches(
    *,
    output_root: str | Path,
    completion_window: str = "24h",
    max_workers: int = 1,
) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in this process environment")
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("OpenAI Python SDK is not installed") from exc

    root = Path(output_root)
    request_dir = root / "api_requests"
    request_files = sorted(request_dir.glob("batch_requests_part_*.jsonl"))
    if not request_files:
        raise FileNotFoundError(f"no batch request files under {request_dir}")
    batches: list[dict[str, Any]] = []
    workers = max(1, int(max_workers))
    if workers == 1 or len(request_files) <= 1:
        for request_file in request_files:
            batches.append(_submit_one_batch_request(request_file, completion_window=completion_window))
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(request_files))) as executor:
            futures = {
                executor.submit(
                    _submit_one_batch_request,
                    request_file,
                    completion_window=completion_window,
                ): request_file
                for request_file in request_files
            }
            for future in as_completed(futures):
                batches.append(future.result())
        batches.sort(key=lambda item: item.get("request_file") or "")
    _append_submitted_requests(root, batches)
    state = {
        "schema_version": "stage3_rl_api_batch_state_v0",
        "created_at": now_iso(),
        "output_root": str(root),
        "completion_window": completion_window,
        "max_workers": workers,
        "batches": batches,
    }
    _write_json(root / "api_run_state.json", state)
    _append_jsonl(
        root / "api_run_history.jsonl",
        [
            {
                "event": "submitted",
                "created_at": state["created_at"],
                "completion_window": completion_window,
                "max_workers": workers,
                "batch_count": len(batches),
                "batch_ids": [row.get("batch_id") for row in batches],
            }
        ],
    )
    return state


def _submit_one_batch_request(request_file: Path, *, completion_window: str) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    with request_file.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    file_id = getattr(uploaded, "id", None) or _to_plain_dict(uploaded).get("id")
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/responses",
        completion_window=completion_window,
    )
    batch_data = _to_plain_dict(batch)
    return {
        "request_file": str(request_file),
        "input_file_id": file_id,
        "batch_id": batch_data.get("id"),
        "status": batch_data.get("status"),
        "created_at": now_iso(),
        "raw_batch": batch_data,
    }


def _append_submitted_requests(root: Path, batches: list[dict[str, Any]]) -> None:
    index_path = root / "api_requests" / "request_index.jsonl"
    index_rows = _read_jsonl(index_path)
    if not index_rows:
        return
    batch_ids = [str(batch.get("batch_id") or "") for batch in batches if batch.get("batch_id")]
    _append_submitted_request_rows(root, index_rows, mode="batch", batch_ids=batch_ids)


def _append_submitted_request_rows(
    root: Path,
    index_rows: list[dict[str, Any]],
    *,
    mode: str,
    batch_ids: list[str] | None = None,
    direct_run_id: str | None = None,
) -> None:
    submitted_at = now_iso()
    rows = []
    for row in index_rows:
        rows.append(
            {
                "submitted_at": submitted_at,
                "mode": mode,
                "custom_id": row.get("custom_id"),
                "request_id": row.get("request_id"),
                "stable_image_uid": row.get("stable_image_uid"),
                "source_dataset": row.get("source_dataset"),
                "batch_ids": batch_ids or [],
                "direct_run_id": direct_run_id,
            }
        )
    _append_jsonl(root / "api_submitted_requests.jsonl", rows)


def run_teacher_requests_direct(
    *,
    output_root: str | Path,
    limit_requests: int | None = None,
    max_workers: int = 4,
    stratified_requests: bool = True,
    skip_existing_outputs: bool = True,
    skip_submitted_requests: bool = False,
    source_weights_override: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in this process environment")
    try:
        from openai import OpenAI  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("OpenAI Python SDK is not installed") from exc

    root = Path(output_root)
    teacher_requests_path = root / "teacher_requests.jsonl"
    if not teacher_requests_path.exists():
        raise FileNotFoundError(f"missing teacher_requests.jsonl: {teacher_requests_path}")
    plan = _read_json(root / "stage3_rl_data_plan.json")
    source_weights = source_weights_override or (
        ((plan.get("balance_config") or {}).get("source_weights") if plan else None)
        or DEFAULT_SOURCE_MIX
    )
    seed = int((plan or {}).get("random_seed") or 42)
    archived_custom_ids = _archived_teacher_custom_ids(root)
    completed_custom_ids = _completed_teacher_custom_ids(root) if skip_existing_outputs else set()
    submitted_custom_ids = _submitted_teacher_custom_ids(root) if skip_submitted_requests else set()
    skipped_custom_ids = archived_custom_ids | completed_custom_ids | submitted_custom_ids
    all_requests = _read_jsonl(teacher_requests_path)
    pending_requests = [
        request
        for request in all_requests
        if str(request.get("custom_id") or "") not in skipped_custom_ids
    ]
    selected_requests = _select_teacher_requests(
        pending_requests,
        limit_requests=limit_requests,
        source_weights=source_weights,
        seed=seed,
        stratified=stratified_requests,
    )

    run_id = f"direct_{now_iso().replace(':', '').replace('-', '').replace('.', '')}"
    run_dir = root / "api_direct_runs" / run_id
    raw_dir = root / "api_raw_responses"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    request_index = [
        {
            "custom_id": request["custom_id"],
            "request_id": request["request_id"],
            "stable_image_uid": request["stable_image_uid"],
            "image_path": request["image_path"],
            "source_dataset": request["source_dataset"],
            "source_profile": request["source_profile"],
        }
        for request in selected_requests
    ]
    _write_jsonl(run_dir / "request_index.jsonl", request_index)

    start = time.monotonic()
    workers = max(1, int(max_workers))
    raw_rows: list[dict[str, Any]] = []
    if selected_requests:
        with ThreadPoolExecutor(max_workers=min(workers, len(selected_requests))) as executor:
            futures = {
                executor.submit(_run_one_direct_request, request): request
                for request in selected_requests
            }
            for future in as_completed(futures):
                raw_rows.append(future.result())
    raw_rows.sort(key=lambda row: str(row.get("custom_id") or ""))
    raw_path = raw_dir / f"{run_id}.jsonl"
    _write_jsonl(raw_path, raw_rows)
    _append_submitted_request_rows(root, request_index, mode="direct", direct_run_id=run_id)

    report = {
        "schema_version": "stage3_rl_api_direct_run_report_v0",
        "created_at": now_iso(),
        "output_root": str(root),
        "direct_run_id": run_id,
        "request_count": len(selected_requests),
        "raw_output_file": str(raw_path),
        "request_index": str(run_dir / "request_index.jsonl"),
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "max_workers": workers,
        "selection_mode": "source_mix_stratified_v0" if stratified_requests else "file_order_v0",
        "source_weights": source_weights,
        "source_distribution": dict(Counter(row["source_dataset"] for row in request_index)),
        "skipped_archived_requests": len(archived_custom_ids),
        "skipped_existing_outputs": len(completed_custom_ids),
        "skipped_submitted_requests": len(submitted_custom_ids - completed_custom_ids),
        "errors": sum(1 for row in raw_rows if row.get("error")),
    }
    _write_json(run_dir / "direct_run_report.json", report)
    return report


def _run_one_direct_request(request: dict[str, Any]) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    try:
        response = client.responses.create(**_api_payload_with_image(request))
        return {
            "id": stable_hash("stage3_rl_direct_response", request.get("custom_id"), now_iso()),
            "custom_id": request.get("custom_id"),
            "response": {"status_code": 200, "request_id": None, "body": _to_plain_dict(response)},
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - API boundary.
        return {
            "id": stable_hash("stage3_rl_direct_error", request.get("custom_id"), now_iso()),
            "custom_id": request.get("custom_id"),
            "response": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def archive_teacher_outputs(
    *,
    output_root: str | Path,
    custom_ids: Iterable[str] | None = None,
    last_submitted_count: int | None = None,
    archive_name: str | None = None,
    reason: str = "smoke_excluded_from_formal",
) -> dict[str, Any]:
    root = Path(output_root)
    selected_custom_ids = _resolve_archive_custom_ids(
        root,
        custom_ids=custom_ids,
        last_submitted_count=last_submitted_count,
    )
    if not selected_custom_ids:
        raise ValueError("no custom_ids resolved for archive")
    archive_id = archive_name or f"archive_{now_iso().replace(':', '').replace('-', '').replace('.', '')}"
    archive_dir = root / "archives" / archive_id
    archive_dir.mkdir(parents=True, exist_ok=True)

    request_rows = [
        row for row in _read_jsonl(root / "teacher_requests.jsonl")
        if str(row.get("custom_id") or "") in selected_custom_ids
    ]
    submitted_rows = [
        row for row in _read_jsonl(root / "api_submitted_requests.jsonl")
        if str(row.get("custom_id") or "") in selected_custom_ids
    ]
    output_rows = [
        row for row in _read_jsonl(root / "teacher_outputs.jsonl")
        if str(row.get("custom_id") or "") in selected_custom_ids
    ]
    raw_rows = []
    for path in sorted((root / "api_raw_responses").glob("*.jsonl")):
        for row in _read_jsonl(path):
            if str(row.get("custom_id") or "") in selected_custom_ids:
                raw = dict(row)
                raw["_source_raw_file"] = str(path)
                raw_rows.append(raw)

    request_by_id = {str(row.get("custom_id") or ""): row for row in request_rows}
    existing_archived = _archived_teacher_custom_ids(root)
    archive_rows = []
    for custom_id in sorted(selected_custom_ids):
        request = request_by_id.get(custom_id) or {}
        archive_rows.append(
            {
                "archived_at": now_iso(),
                "archive_id": archive_id,
                "reason": reason,
                "custom_id": custom_id,
                "request_id": request.get("request_id"),
                "stable_image_uid": request.get("stable_image_uid"),
                "source_dataset": request.get("source_dataset"),
                "source_profile": request.get("source_profile"),
            }
        )
    new_archive_rows = [
        row for row in archive_rows if str(row.get("custom_id") or "") not in existing_archived
    ]
    _append_jsonl(root / "api_archived_custom_ids.jsonl", new_archive_rows)
    _write_jsonl(archive_dir / "archived_custom_ids.jsonl", archive_rows)
    _write_jsonl(archive_dir / "teacher_requests.jsonl", request_rows)
    _write_jsonl(archive_dir / "submitted_requests.jsonl", submitted_rows)
    _write_jsonl(archive_dir / "teacher_outputs.jsonl", output_rows)
    _write_jsonl(archive_dir / "raw_responses.jsonl", raw_rows)
    manifest = {
        "schema_version": "stage3_rl_api_archive_manifest_v0",
        "created_at": now_iso(),
        "archive_id": archive_id,
        "reason": reason,
        "output_root": str(root),
        "custom_ids": sorted(selected_custom_ids),
        "custom_id_count": len(selected_custom_ids),
        "newly_archived_count": len(new_archive_rows),
        "request_rows": len(request_rows),
        "submitted_rows": len(submitted_rows),
        "teacher_output_rows": len(output_rows),
        "raw_response_rows": len(raw_rows),
        "source_distribution": dict(
            Counter(str(row.get("source_dataset") or "unknown") for row in archive_rows)
        ),
        "formal_policy": "archived custom_ids are skipped by prepare, parse, and finalize",
    }
    _write_json(archive_dir / "archive_manifest.json", manifest)
    return manifest


def _resolve_archive_custom_ids(
    root: Path,
    *,
    custom_ids: Iterable[str] | None,
    last_submitted_count: int | None,
) -> set[str]:
    if custom_ids:
        return {str(item) for item in custom_ids if str(item).strip()}
    if last_submitted_count is not None:
        rows = _read_jsonl(root / "api_submitted_requests.jsonl")
        tail = rows[-int(last_submitted_count):] if int(last_submitted_count) > 0 else []
        return {str(row.get("custom_id") or "") for row in tail if row.get("custom_id")}
    request_index = root / "api_requests" / "request_index.jsonl"
    return {str(row.get("custom_id") or "") for row in _read_jsonl(request_index) if row.get("custom_id")}


def poll_teacher_batches(*, output_root: str | Path) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in this process environment")
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("OpenAI Python SDK is not installed") from exc

    root = Path(output_root)
    state_path = root / "api_run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    raw_dir = root / "api_raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    batches: list[dict[str, Any]] = []
    for batch_info in state.get("batches") or []:
        batch = client.batches.retrieve(batch_info["batch_id"])
        batch_data = _to_plain_dict(batch)
        merged = {**batch_info, **batch_data, "polled_at": now_iso()}
        if batch_data.get("output_file_id"):
            output_text = _download_file_text(client, batch_data["output_file_id"])
            output_path = raw_dir / f"{batch_info['batch_id']}.jsonl"
            output_path.write_text(output_text, encoding="utf-8")
            merged["downloaded_output_path"] = str(output_path)
        if batch_data.get("error_file_id"):
            error_text = _download_file_text(client, batch_data["error_file_id"])
            error_path = raw_dir / f"{batch_info['batch_id']}.errors.jsonl"
            error_path.write_text(error_text, encoding="utf-8")
            merged["downloaded_error_path"] = str(error_path)
        batches.append(merged)
    state["batches"] = batches
    state["polled_at"] = now_iso()
    _write_json(state_path, state)
    return state


def parse_teacher_batch_outputs(*, output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    raw_dir = root / "api_raw_responses"
    output_paths = sorted(path for path in raw_dir.glob("*.jsonl") if not path.name.endswith(".errors.jsonl"))
    archived_custom_ids = _archived_teacher_custom_ids(root)
    teacher_outputs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    archived_skipped = 0
    for path in output_paths:
        for row in _read_jsonl(path):
            custom_id = row.get("custom_id")
            if str(custom_id or "") in archived_custom_ids:
                archived_skipped += 1
                continue
            response = row.get("response") or {}
            body = response.get("body") if isinstance(response, dict) else None
            if row.get("error") or not isinstance(body, dict):
                errors.append({"custom_id": custom_id, "error": row.get("error") or row})
                continue
            text = _extract_output_text(body)
            try:
                parsed = json.loads(text)
            except Exception as exc:  # noqa: BLE001 - parsing boundary.
                errors.append({"custom_id": custom_id, "error": f"{type(exc).__name__}: {exc}", "text": text})
                continue
            teacher_outputs.append(
                {
                    "custom_id": custom_id,
                    "response_id": body.get("id"),
                    "output": parsed,
                    "usage": _normalize_usage(body.get("usage")),
                }
            )
    _write_jsonl(root / "teacher_outputs.jsonl", teacher_outputs)
    _write_jsonl(root / "teacher_output_errors.jsonl", errors)
    report = {
        "schema_version": "stage3_rl_teacher_output_parse_report_v0",
        "created_at": now_iso(),
        "output_root": str(root),
        "raw_output_files": [str(path) for path in output_paths],
        "parsed_outputs": len(teacher_outputs),
        "archived_outputs_skipped": archived_skipped,
        "errors": len(errors),
        "usage": _sum_usage(row.get("usage") or {} for row in teacher_outputs),
    }
    _write_json(root / "teacher_output_parse_report.json", report)
    return report


def _api_payload_with_image(request: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(request["api_payload_template"]))
    content = payload["input"][1]["content"]
    for item in content:
        if item.get("type") == "input_image":
            item["image_url"] = image_to_data_url(request["image_path"])
    return payload


def image_to_data_url(path_like: str | Path) -> str:
    path = Path(path_like)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _select_teacher_requests(
    requests: list[dict[str, Any]],
    *,
    limit_requests: int | None,
    source_weights: dict[str, float],
    seed: int,
    stratified: bool,
) -> list[dict[str, Any]]:
    if not stratified:
        return requests if limit_requests is None else requests[:limit_requests]

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        by_source[_source_bucket(request, source_weights)].append(request)
    for rows in by_source.values():
        rows.sort(
            key=lambda request: stable_hash(
                "stage3_rl_api_request_order",
                seed,
                request.get("custom_id"),
                request.get("request_id"),
                request.get("stable_image_uid"),
            )
        )
    total = len(requests) if limit_requests is None else min(limit_requests, len(requests))
    if total <= 0:
        return []
    selected = _interleave_source_mix(by_source, total=total, source_weights=source_weights)
    if len(selected) >= total:
        return selected[:total]
    selected_ids = {str(row.get("custom_id") or row.get("request_id") or id(row)) for row in selected}
    leftovers = [
        request
        for rows in by_source.values()
        for request in rows
        if str(request.get("custom_id") or request.get("request_id") or id(request)) not in selected_ids
    ]
    leftovers.sort(
        key=lambda request: stable_hash(
            "stage3_rl_api_request_leftover",
            seed,
            request.get("custom_id"),
            request.get("request_id"),
            request.get("stable_image_uid"),
        )
    )
    selected.extend(leftovers[: total - len(selected)])
    return selected[:total]


def _interleave_source_mix(
    by_source: dict[str, list[dict[str, Any]]],
    *,
    total: int,
    source_weights: dict[str, float],
) -> list[dict[str, Any]]:
    quotas = _quotas(total, source_weights)
    indices = {source: 0 for source in source_weights}
    selected_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    while len(selected) < total:
        best_source: str | None = None
        best_deficit: float | None = None
        for source, fraction in source_weights.items():
            if selected_counts[source] >= quotas.get(source, 0):
                continue
            if indices.get(source, 0) >= len(by_source.get(source, [])):
                continue
            expected = (len(selected) + 1) * float(fraction)
            deficit = expected - selected_counts[source]
            if best_deficit is None or deficit > best_deficit:
                best_source = source
                best_deficit = deficit
        if best_source is None:
            break
        row = by_source[best_source][indices[best_source]]
        indices[best_source] += 1
        selected_counts[best_source] += 1
        selected.append(row)
    return selected


def _source_bucket(row: dict[str, Any], source_weights: dict[str, float]) -> str:
    source = str(row.get("source_dataset") or "unknown")
    if source in source_weights:
        return source
    if source in {"textvqa", "textocr", "textvqa_textocr"}:
        for alias in ("textvqa", "textvqa_textocr"):
            if alias in source_weights:
                return alias
    return source


def _quotas(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        return {source: 0 for source in weights}
    raw = {source: total * float(weight) for source, weight in weights.items()}
    quotas = {source: int(value) for source, value in raw.items()}
    remainder = total - sum(quotas.values())
    for source, _value in sorted(
        raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True
    )[:remainder]:
        quotas[source] += 1
    return quotas


def _completed_teacher_custom_ids(root: Path) -> set[str]:
    completed: set[str] = set()
    for row in _read_jsonl(root / "teacher_outputs.jsonl"):
        custom_id = str(row.get("custom_id") or "")
        if custom_id:
            completed.add(custom_id)
    return completed


def _submitted_teacher_custom_ids(root: Path) -> set[str]:
    submitted: set[str] = set()
    for row in _read_jsonl(root / "api_submitted_requests.jsonl"):
        custom_id = str(row.get("custom_id") or "")
        if custom_id:
            submitted.add(custom_id)
    return submitted


def _archived_teacher_custom_ids(root: Path) -> set[str]:
    archived: set[str] = set()
    for row in _read_jsonl(root / "api_archived_custom_ids.jsonl"):
        custom_id = str(row.get("custom_id") or "")
        if custom_id:
            archived.add(custom_id)
    return archived


def _download_file_text(client: Any, file_id: str) -> str:
    content = client.files.content(file_id)
    if hasattr(content, "text"):
        return str(content.text)
    if hasattr(content, "read"):
        data = content.read()
        return data.decode("utf-8") if isinstance(data, bytes) else str(data)
    return str(content)


def _extract_output_text(raw_response: dict[str, Any]) -> str:
    output_text = raw_response.get("output_text")
    if output_text:
        return str(output_text)
    texts: list[str] = []
    for output in raw_response.get("output", []) or []:
        for content in output.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                texts.append(str(content["text"]))
    return "\n".join(texts)


def _normalize_usage(usage: Any) -> dict[str, int]:
    data = _to_plain_dict(usage) if usage is not None else {}
    input_tokens = int(data.get("input_tokens") or data.get("prompt_tokens") or 0)
    output_tokens = int(data.get("output_tokens") or data.get("completion_tokens") or 0)
    total_tokens = int(data.get("total_tokens") or input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _sum_usage(usages: Iterable[dict[str, int]]) -> dict[str, int]:
    total = Counter()
    for usage in usages:
        total.update({key: int(value) for key, value in usage.items()})
    return dict(total)


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    try:
        return json.loads(json.dumps(obj, default=lambda value: getattr(value, "__dict__", str(value))))
    except Exception:
        return {"repr": repr(obj)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
