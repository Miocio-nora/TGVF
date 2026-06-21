from __future__ import annotations

import argparse
from pathlib import Path

from tgvf_data.generate_teacher import GenerationConfig, OpenAIConfig, resume_sync
from tgvf_data.tgvf_teacher_schema_v4 import SCHEMA_VERSION_V4, TEACHER_VERSION_V4


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TGVF V4 teacher generation.")
    parser.add_argument(
        "--selection",
        default="data/tgvf_teacher/preparation/selections/selected_images_v3_50k_v0.jsonl",
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-id", default="tgvf_v4_teacher_50k")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--image-detail", default="original")
    parser.add_argument("--target-accepted-samples", type=int, default=50_000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=5000)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--allow-regenerate", action="store_true")
    parser.add_argument("--allow-duplicate-items", action="store_true")
    args = parser.parse_args()

    report = resume_sync(
        selection=Path(args.selection),
        project_root=Path(args.project_root),
        run_id=args.run_id,
        openai_config=OpenAIConfig(
            model=args.model,
            image_detail=args.image_detail,
            max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        ),
        generation_config=GenerationConfig(
            target_accepted_samples=args.target_accepted_samples,
            prompt_version=TEACHER_VERSION_V4,
            schema_version=SCHEMA_VERSION_V4,
            confidence_threshold=0.75,
        ),
        limit_images=args.limit_images,
        concurrency=args.concurrency,
        allow_regenerate=args.allow_regenerate,
        allow_duplicate_items=args.allow_duplicate_items,
        fail_fast=False,
    )
    print(report)


if __name__ == "__main__":
    main()
