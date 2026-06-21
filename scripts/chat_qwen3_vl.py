#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revisit_vlm.qwen3_vl_tgvf import (  # noqa: E402
    _decode,
    build_direct_messages,
    build_qwen3_inputs,
    load_qwen3_vl,
)


DEFAULT_MODEL = "/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive raw Qwen3-VL chat without TGVF or LoRA.")
    parser.add_argument("--image", default=None, help="Initial image path. Can be changed with /image PATH.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-image-resolution", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--output-jsonl", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Loading raw Qwen3-VL model={args.model_id} processor={args.processor_id or args.model_id} "
        f"max_image_resolution={args.max_image_resolution}",
        flush=True,
    )
    loaded = load_qwen3_vl(
        args.model_id,
        processor_id=args.processor_id,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    loaded.model.eval()
    image = args.image
    if image:
        _check_image(image)
    out_handle = None
    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_handle = out_path.open("a")
    print("Ready.", flush=True)
    print("Commands: /image PATH | /maxtok N | /quit", flush=True)
    print("Inline image: `/path/to/image.jpg your question`", flush=True)
    if image:
        print(f"Current image: {image}", flush=True)
    else:
        print("No image set. Text-only chat works; send an image with /image PATH or inline path.", flush=True)

    turn = 0
    try:
        while True:
            try:
                text = input("You> ").strip()
            except EOFError:
                break
            if not text:
                continue
            inline_image, inline_question = parse_inline_image_question(text)
            if inline_image is not None:
                _check_image(inline_image)
                image = inline_image
                text = inline_question
                print(f"Current image: {image}", flush=True)
                if not text:
                    continue
            elif text.startswith("/"):
                image = handle_command(text, image, args)
                continue
            turn += 1
            started = time.perf_counter()
            try:
                row = run_turn(loaded.model, loaded.processor, image=image, question=text, args=args)
                row["turn"] = turn
                row["wall_time_sec"] = time.perf_counter() - started
                print(f"Assistant> {row['raw_output'].strip()}", flush=True)
            except Exception as exc:
                row = {
                    "turn": turn,
                    "image": image,
                    "question": text,
                    "error": f"{type(exc).__name__}: {exc}",
                    "wall_time_sec": time.perf_counter() - started,
                }
                print(f"Assistant> [error] {row['error']}", flush=True)
            if out_handle is not None:
                out_handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                out_handle.flush()
    finally:
        if out_handle is not None:
            out_handle.close()


@torch.no_grad()
def run_turn(model: Any, processor: Any, *, image: str | None, question: str, args: argparse.Namespace) -> dict[str, Any]:
    messages = build_messages(image=image, question=question, max_image_resolution=args.max_image_resolution)
    inputs = build_qwen3_inputs(processor, messages)
    device = torch.device(args.device)
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    prompt_len = int(model_inputs["input_ids"].shape[-1])
    generate_kwargs: dict[str, Any] = {
        **model_inputs,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": bool(args.do_sample),
    }
    eos_id = getattr(processor.tokenizer, "eos_token_id", None)
    if eos_id is not None:
        generate_kwargs["eos_token_id"] = eos_id
    if args.do_sample:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p
    generated = model.generate(**generate_kwargs)
    new_ids = generated[0, prompt_len:].detach().cpu().tolist()
    raw = _decode(processor.tokenizer, new_ids)
    return {
        "image": image,
        "question": question,
        "raw_output": raw,
        "generated_ids": new_ids,
        "output_tokens": len(new_ids),
        "do_sample": bool(args.do_sample),
        "max_new_tokens": args.max_new_tokens,
        "max_image_resolution": args.max_image_resolution,
    }


def build_messages(*, image: str | None, question: str, max_image_resolution: int) -> list[dict[str, Any]]:
    if image:
        image_payload = {
            "type": "image",
            "image": image,
            "max_pixels": int(max_image_resolution) * int(max_image_resolution),
        }
        return build_direct_messages(image_payload, question)
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": question}],
        }
    ]


def parse_inline_image_question(text: str) -> tuple[str | None, str]:
    stripped = text.strip()
    if not stripped:
        return None, ""
    if stripped[0] in {"'", '"'}:
        quote = stripped[0]
        end = stripped.find(quote, 1)
        if end > 1:
            candidate = stripped[1:end]
            if looks_like_image_path(candidate) and Path(candidate).exists():
                return candidate, stripped[end + 1 :].strip()
    parts = stripped.split(maxsplit=1)
    candidate = parts[0]
    if looks_like_image_path(candidate) and Path(candidate).exists():
        return candidate, parts[1].strip() if len(parts) > 1 else ""
    return None, stripped


def looks_like_image_path(value: str) -> bool:
    lower = value.lower()
    return lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")) or "/" in value


def handle_command(text: str, image: str | None, args: argparse.Namespace) -> str | None:
    parts = text.split(maxsplit=1)
    cmd = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""
    if cmd in {"/quit", "/q", "/exit"}:
        raise SystemExit(0)
    if cmd == "/image":
        if not rest:
            print("Usage: /image /path/to/image.jpg", flush=True)
            return image
        _check_image(rest)
        print(f"Current image: {rest}", flush=True)
        return rest
    if cmd == "/maxtok":
        try:
            args.max_new_tokens = max(1, int(rest))
        except ValueError:
            print("Usage: /maxtok N", flush=True)
            return image
        print(f"max_new_tokens={args.max_new_tokens}", flush=True)
        return image
    print("Commands: /image PATH | /maxtok N | /quit", flush=True)
    return image


def _check_image(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"image not found: {path}")


if __name__ == "__main__":
    main()
