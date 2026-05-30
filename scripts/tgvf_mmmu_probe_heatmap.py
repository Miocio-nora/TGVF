from __future__ import annotations

import argparse
import ast
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from revisit_vlm.tgvf import FOVEATE_END, FOVEATE_START
from revisit_vlm.tgvf_capture import capture_tgvf_single_pass, format_tgvf_debug

DEFAULT_MMMU_PARQUET = (
    "/home/dredvpn009/Flash_Storage/datasets/benchmarks/mmmu_pro/"
    "snapshot/standard (4 options)/test-00000-of-00002.parquet"
)


class ForcedGreedyWrapper:
    def __init__(self, model: Any, forced_ids: list[int]) -> None:
        self.model = model
        self.forced_ids = forced_ids
        self.forward_calls = 0
        self.forward_input_lengths: list[int] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        outputs = self.model(*args, **kwargs)
        self._record_input_length(args, kwargs)

        step = self.forward_calls
        self.forward_calls += 1
        if step < len(self.forced_ids):
            _force_next_token(outputs, self.forced_ids[step])
        return outputs

    def _record_input_length(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None:
            self.forward_input_lengths.append(int(input_ids.shape[-1]))

    def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def parameters(self) -> Any:
        return self.model.parameters()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


class PrefixNaturalSuffixWrapper:
    """Force markers while letting Qwen2-VL produce the target phrase itself."""

    def __init__(
        self,
        model: Any,
        *,
        prefix_ids: list[int],
        suffix_ids: list[int],
        natural_steps: int,
        tokenizer: Any,
        stop_token_ids: set[int],
    ) -> None:
        self.model = model
        self.prefix_ids = prefix_ids
        self.suffix_ids = suffix_ids
        self.natural_steps = natural_steps
        self.tokenizer = tokenizer
        self.stop_token_ids = stop_token_ids
        self.forward_calls = 0
        self.forward_input_lengths: list[int] = []
        self.natural_generated = 0
        self.suffix_index: int | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        outputs = self.model(*args, **kwargs)
        self._record_input_length(args, kwargs)

        step = self.forward_calls
        self.forward_calls += 1
        if step < len(self.prefix_ids):
            _force_next_token(outputs, self.prefix_ids[step])
            return outputs

        if self.suffix_index is not None:
            self._force_suffix(outputs)
            return outputs

        candidate_id = int(torch.argmax(outputs.logits[:, -1, :], dim=-1)[0].item())
        if self.natural_generated >= self.natural_steps or self._should_stop_target(candidate_id):
            self.suffix_index = 0
            self._force_suffix(outputs)
            return outputs

        self.natural_generated += 1
        return outputs

    def _should_stop_target(self, token_id: int) -> bool:
        if token_id in self.stop_token_ids:
            return True
        piece = self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        stop_fragments = ("<|", "|>", "|", "\n", "Question", "Answer", "Human:")
        return any(fragment in piece for fragment in stop_fragments)

    def _force_suffix(self, outputs: Any) -> None:
        if self.suffix_index is None or self.suffix_index >= len(self.suffix_ids):
            return
        _force_next_token(outputs, self.suffix_ids[self.suffix_index])
        self.suffix_index += 1

    def _record_input_length(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None:
            self.forward_input_lengths.append(int(input_ids.shape[-1]))

    def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def parameters(self) -> Any:
        return self.model.parameters()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TGVF heatmap probes on MMMU-Pro samples.")
    parser.add_argument("--parquet", default=DEFAULT_MMMU_PARQUET)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/tgvf_mmmu_probe")
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--target", default=None)
    parser.add_argument("--natural-target-tokens", type=int, default=10)
    parser.add_argument(
        "--query-source",
        choices=("hidden_states", "token_embeddings"),
        default="hidden_states",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(args.parquet)

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=False)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map={"": args.device},
        trust_remote_code=False,
    )
    model.eval()

    visual_capture: dict[str, torch.Tensor] = {}

    def capture_visual_embeddings(_module: Any, _inputs: Any, output: Any) -> None:
        visual_capture.setdefault("embeddings", _extract_visual_embeddings(output))

    hook = _visual_module(model).register_forward_hook(capture_visual_embeddings)
    summaries = []
    try:
        for row_index in range(args.row, args.row + args.count):
            sample_dir = (
                output_dir if args.count == 1 else output_dir / _sample_dir_name(table, row_index)
            )
            sample_dir.mkdir(parents=True, exist_ok=True)
            visual_capture.clear()
            summary = _run_sample(
                table=table,
                row_index=row_index,
                output_dir=sample_dir,
                model=model,
                processor=processor,
                visual_capture=visual_capture,
                target=args.target,
                natural_target_tokens=args.natural_target_tokens,
                device=args.device,
                attn_implementation=args.attn_implementation,
                model_id=args.model_id,
                query_source=args.query_source,
            )
            summaries.append(summary)
            print(json.dumps(summary, indent=2))
    finally:
        hook.remove()

    if args.count > 1:
        (output_dir / "batch_summary.json").write_text(json.dumps(summaries, indent=2))


def _run_sample(
    *,
    table: Any,
    row_index: int,
    output_dir: Path,
    model: Any,
    processor: Any,
    visual_capture: dict[str, torch.Tensor],
    target: str | None,
    natural_target_tokens: int,
    device: str,
    attn_implementation: str,
    model_id: str,
    query_source: str,
) -> dict[str, Any]:
    row = _load_row_from_table(table, row_index)
    image = _row_image(row)
    image_path = output_dir / "mmmu_sample.png"
    image.save(image_path)

    question = _question_with_options(row)
    messages = _target_probe_messages(str(image_path), question)
    start_ids = processor.tokenizer.encode(FOVEATE_START, add_special_tokens=False)
    end_ids = processor.tokenizer.encode(FOVEATE_END, add_special_tokens=False)

    if target:
        forced_ids = processor.tokenizer.encode(
            f"{FOVEATE_START}{target}{FOVEATE_END}", add_special_tokens=False
        )
        wrapped_model = ForcedGreedyWrapper(model, forced_ids)
        max_new_tokens = len(forced_ids) + 8
    else:
        wrapped_model = PrefixNaturalSuffixWrapper(
            model,
            prefix_ids=start_ids,
            suffix_ids=end_ids,
            natural_steps=natural_target_tokens,
            tokenizer=processor.tokenizer,
            stop_token_ids=_target_stop_token_ids(processor.tokenizer),
        )
        max_new_tokens = len(start_ids) + natural_target_tokens + len(end_ids) + 8

    capture = capture_tgvf_single_pass(
        wrapped_model,
        processor,
        image=str(image_path),
        question=question,
        messages=messages,
        max_new_tokens=max_new_tokens,
        device=device,
        eos_token_id=None,
    )

    if not capture.capture_found:
        raise RuntimeError(f"No complete TGVF span captured. Generated: {capture.generated_text!r}")
    if "embeddings" not in visual_capture:
        raise RuntimeError("Visual embeddings were not captured from the visual module hook.")

    visual_embeddings = visual_capture["embeddings"]
    query_embeddings = _target_query_embeddings(
        capture=capture,
        model=model,
        query_source=query_source,
        device=device,
    )
    if query_embeddings.shape[-1] != visual_embeddings.shape[-1]:
        raise RuntimeError(
            "Hidden size mismatch: "
            f"query={query_embeddings.shape[-1]}, visual={visual_embeddings.shape[-1]}"
        )

    scores, token_scores = _cosine_probe_scores(query_embeddings, visual_embeddings)
    grid_shape = _merged_grid_shape(capture.image_grid_thw, model)
    heatmap_path = None
    grid_scores = None
    if grid_shape is not None and grid_shape[0] * grid_shape[1] == scores.numel():
        grid_scores = scores.reshape(grid_shape)
        heatmap_path = output_dir / "heatmap_overlay.png"
        _save_heatmap_overlay(image, grid_scores, heatmap_path, label_text=capture.target_text)
        _save_heatmap_image(grid_scores, output_dir / "heatmap.png")

    second_full_forward_used = any(length > 1 for length in wrapped_model.forward_input_lengths[1:])
    debug = {
        "dataset": "MMMU-Pro",
        "row": row_index,
        "sample_id": row.get("id"),
        "subject": row.get("subject"),
        "question": row.get("question"),
        "options": _parse_options(row.get("options")),
        "answer": row.get("answer"),
        "target_source": "manual" if target else "model_generated_between_forced_markers",
        "target_text": capture.target_text,
        "raw_target_text": capture.raw_target_text,
        "generated_text": capture.generated_text,
        "target_token_count": len(capture.target_token_ids),
        "target_hidden_shape": list(capture.target_hidden_states.shape),
        "query_source": query_source,
        "query_shape": list(query_embeddings.shape),
        "visual_embedding_shape": list(visual_embeddings.shape),
        "score_shape": list(scores.shape),
        "token_score_shape": list(token_scores.shape),
        "score_aggregation": f"mean_over_{query_source}_token_cosine_scores",
        "min_score": float(scores.min().item()),
        "max_score": float(scores.max().item()),
        "argmax_visual_token": int(scores.argmax().item()),
        "argmax_grid_yx": None if grid_shape is None else _argmax_yx(scores, grid_shape),
        "mean_score": float(scores.mean().item()),
        "std_score": float(scores.std().item()),
        "image_size": list(image.size),
        "image_grid_thw": None
        if capture.image_grid_thw is None
        else capture.image_grid_thw.detach().cpu().tolist(),
        "merged_grid_shape": None if grid_shape is None else list(grid_shape),
        "visual_source": "model visual forward hook pooler_output",
        "attn_implementation": attn_implementation,
        "model_id": model_id,
        "stop_reason": capture.stop_reason,
        "forward_input_lengths": wrapped_model.forward_input_lengths,
        "second_full_forward_used": second_full_forward_used,
        "image_path": str(image_path),
        "heatmap_path": None if heatmap_path is None else str(heatmap_path),
    }

    torch.save(scores.cpu(), output_dir / "similarity_scores.pt")
    torch.save(token_scores.cpu(), output_dir / "similarity_token_scores.pt")
    if grid_scores is not None:
        torch.save(grid_scores.cpu(), output_dir / "similarity_grid.pt")
    (output_dir / "probe_debug.json").write_text(json.dumps(debug, indent=2))
    (output_dir / "generated.txt").write_text(format_tgvf_debug(capture))
    return {
        "row": row_index,
        "sample_id": row.get("id"),
        "target_text": capture.target_text,
        "target_hidden_shape": list(capture.target_hidden_states.shape),
        "query_source": query_source,
        "query_shape": list(query_embeddings.shape),
        "visual_embedding_shape": list(visual_embeddings.shape),
        "score_shape": list(scores.shape),
        "token_score_shape": list(token_scores.shape),
        "score_aggregation": f"mean_over_{query_source}_token_cosine_scores",
        "max_score": float(scores.max().item()),
        "argmax_grid_yx": debug["argmax_grid_yx"],
        "heatmap_path": debug["heatmap_path"],
        "second_full_forward_used": second_full_forward_used,
    }


def _target_query_embeddings(
    *,
    capture: Any,
    model: Any,
    query_source: str,
    device: str,
) -> torch.Tensor:
    if query_source == "hidden_states":
        return capture.target_hidden_states
    if query_source != "token_embeddings":
        raise ValueError(f"Unsupported query_source: {query_source}")

    token_ids = torch.tensor(
        capture.target_token_ids,
        dtype=torch.long,
        device=device,
    )
    if token_ids.numel() == 0:
        return capture.target_hidden_states.new_empty((0, capture.target_hidden_states.shape[-1]))
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(token_ids)
    return embeddings.detach().cpu()


def _target_stop_token_ids(tokenizer: Any) -> set[int]:
    stop_ids: set[int] = set()
    for token in ("<|im_end|>", "<|endoftext|>", FOVEATE_END):
        ids = tokenizer.encode(token, add_special_tokens=False)
        stop_ids.update(int(token_id) for token_id in ids)
    for attr in ("eos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            stop_ids.add(int(token_id))
    return stop_ids


def _sample_dir_name(table: Any, row_index: int) -> str:
    row = _load_row_from_table(table, row_index)
    sample_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.get("id", "sample")))
    return f"row_{row_index:04d}_{sample_id}"


def _load_row_from_table(table: Any, row_index: int) -> dict[str, Any]:
    if row_index < 0 or row_index >= table.num_rows:
        raise IndexError(f"row {row_index} outside parquet with {table.num_rows} rows")
    return table.slice(row_index, 1).to_pylist()[0]


def _target_probe_messages(image_path: str, question: str) -> list[dict[str, Any]]:
    prompt = (
        "Look at the image and question. Select one concise visual focus target needed "
        "to answer the question. The target should be a specific region, object, chart "
        "element, label, or text span. Do not describe the whole image and do not answer "
        "the question. Output only one foveation request in this exact format: "
        f"{FOVEATE_START}short visual target phrase{FOVEATE_END}\n\n"
        f"Question:\n{question}"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _load_row(path: Path, row_index: int) -> dict[str, Any]:
    return _load_row_from_table(pq.read_table(path), row_index)


def _row_image(row: dict[str, Any]) -> Image.Image:
    for index in range(1, 8):
        value = row.get(f"image_{index}")
        if value and value.get("bytes"):
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
    value = row.get("image")
    if value and value.get("bytes"):
        return Image.open(BytesIO(value["bytes"])).convert("RGB")
    raise ValueError("No image bytes found in MMMU-Pro row")


def _question_with_options(row: dict[str, Any]) -> str:
    options = _parse_options(row.get("options"))
    option_lines = "\n".join(f"{chr(ord('A') + i)}. {option}" for i, option in enumerate(options))
    return f"{row.get('question', '')}\n\nOptions:\n{option_lines}"


def _parse_options(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [str(value)]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _visual_module(model: Any) -> torch.nn.Module:
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    raise RuntimeError("Could not find Qwen2-VL visual module")


def _extract_visual_embeddings(output: Any) -> torch.Tensor:
    if hasattr(output, "pooler_output"):
        output = output.pooler_output

    if isinstance(output, torch.Tensor):
        embeddings = output
    elif (
        isinstance(output, (list, tuple))
        and output
        and all(isinstance(item, torch.Tensor) for item in output)
    ):
        embeddings = torch.cat([item.reshape(-1, item.shape[-1]) for item in output], dim=0)
    elif isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        embeddings = output[0]
    elif hasattr(output, "last_hidden_state"):
        embeddings = output.last_hidden_state
    else:
        raise RuntimeError(f"Unsupported visual output type: {type(output)!r}")

    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    if embeddings.ndim != 2:
        raise RuntimeError(f"Expected 2D visual embeddings, got {tuple(embeddings.shape)}")
    return embeddings.detach().cpu()


def _cosine_probe_scores(
    target_hidden_states: torch.Tensor,
    visual_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = F.normalize(target_hidden_states.float(), dim=-1)
    visual = F.normalize(visual_embeddings.float(), dim=-1)
    token_scores = target @ visual.T
    scores = token_scores.mean(dim=0)
    return scores, token_scores


def _merged_grid_shape(image_grid_thw: torch.Tensor | None, model: Any) -> tuple[int, int] | None:
    if image_grid_thw is None:
        return None
    grid = image_grid_thw.detach().cpu()[0].tolist()
    if len(grid) != 3 or grid[0] != 1:
        return None
    merge_size = getattr(_visual_module(model), "spatial_merge_size", 2)
    return int(grid[1] // merge_size), int(grid[2] // merge_size)


def _argmax_yx(scores: torch.Tensor, grid_shape: tuple[int, int]) -> list[int]:
    index = int(scores.argmax().item())
    return [index // grid_shape[1], index % grid_shape[1]]


def _save_heatmap_overlay(
    image: Image.Image,
    grid_scores: torch.Tensor,
    path: Path,
    *,
    label_text: str | None = None,
) -> None:
    heat = _scores_to_heatmap(grid_scores)
    heat = heat.resize(image.size, Image.Resampling.BILINEAR)
    overlay = Image.blend(image.convert("RGBA"), heat, alpha=0.45)
    if label_text:
        _draw_label(overlay, label_text)
    overlay.save(path)


def _draw_label(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    max_chars = max(24, min(52, image.size[0] // 8))
    lines = _wrap_text(f"target: {text}", max_chars=max_chars)[:4]
    line_height = 14
    padding = 6
    width = min(image.size[0] - 8, max(draw.textlength(line) for line in lines) + padding * 2)
    height = len(lines) * line_height + padding * 2
    draw.rectangle((4, 4, 4 + width, 4 + height), fill=(0, 0, 0, 230))
    y = 4 + padding
    for line in lines:
        draw.text((4 + padding, y), line, fill=(0, 255, 80, 255))
        y += line_height


def _wrap_text(text: str, *, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def _save_heatmap_image(grid_scores: torch.Tensor, path: Path) -> None:
    heat = _scores_to_heatmap(grid_scores)
    heat = heat.resize(
        (grid_scores.shape[1] * 24, grid_scores.shape[0] * 24),
        Image.Resampling.NEAREST,
    )
    heat.save(path)


def _scores_to_heatmap(grid_scores: torch.Tensor) -> Image.Image:
    values = grid_scores.float()
    values = (values - values.min()) / (values.max() - values.min()).clamp_min(1e-6)
    arr = (values * 255).to(torch.uint8).cpu()
    red = arr
    green = torch.zeros_like(arr)
    blue = 255 - arr
    alpha = torch.full_like(arr, 180)
    rgba = torch.stack([red, green, blue, alpha], dim=-1).numpy()
    return Image.fromarray(rgba, mode="RGBA")


def _force_next_token(outputs: Any, token_id: int) -> None:
    logits = torch.full_like(outputs.logits, -1e4)
    logits[:, -1, token_id] = 1e4
    outputs.logits = logits


if __name__ == "__main__":
    main()
