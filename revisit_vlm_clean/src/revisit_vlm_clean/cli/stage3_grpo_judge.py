"""Offline Stage3 GRPO judge runner CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from revisit_vlm_clean.cli.common import print_json
from revisit_vlm_clean.stage3_grpo.judge_runner import (
    DEFAULT_MODEL_ROOT,
    MODEL_PRESETS,
    OfflineJudgeConfig,
    run_offline_judge,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline Stage3 GRPO visual judges.")
    parser.add_argument("--pending-path", required=True, help="Path to judge_pending.jsonl.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <pending-dir>/offline_judge.")
    parser.add_argument("--focus-cache-path", default=None)
    parser.add_argument("--grounding-cache-path", default=None)
    parser.add_argument("--backend", choices=("local_qwen_vl", "fake"), default="local_qwen_vl")
    parser.add_argument(
        "--model-preset",
        choices=sorted(MODEL_PRESETS),
        default=None,
        help="Convenience preset; --model-id overrides it.",
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--require-local-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require --model-id/--processor-id to resolve under --model-root or an explicit local path.",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-image-resolution", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--append", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-version", default="stage3_grpo_local_judge_v0")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.preflight_only or args.execute):
        raise SystemExit("use --preflight-only or --execute")
    output_dir = args.output_dir or str(Path(args.pending_path).parent / "offline_judge")
    model_id = args.model_id or MODEL_PRESETS.get(args.model_preset or "", None)
    if model_id is None:
        model_id = MODEL_PRESETS["qwen3_vl_32b_thinking"]
    config = OfflineJudgeConfig(
        pending_path=args.pending_path,
        output_dir=output_dir,
        focus_cache_path=args.focus_cache_path,
        grounding_cache_path=args.grounding_cache_path,
        backend=args.backend,
        model_id=model_id,
        processor_id=args.processor_id,
        model_root=args.model_root,
        require_local_model=args.require_local_model,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        max_image_resolution=args.max_image_resolution,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        limit=args.limit,
        skip_existing=args.skip_existing,
        append=args.append,
        prompt_version=args.prompt_version,
    )
    result = run_offline_judge(config, preflight_only=args.preflight_only)
    print_json(result)
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
