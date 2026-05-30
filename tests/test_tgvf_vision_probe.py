from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from revisit_vlm.tgvf import FOVEATE_END, FOVEATE_START
from revisit_vlm.tgvf_capture import capture_tgvf_single_pass, format_tgvf_debug

pytestmark = pytest.mark.integration


class ForcedGreedyWrapper:
    def __init__(self, model: Any, forced_ids: list[int]) -> None:
        self.model = model
        self.forced_ids = forced_ids
        self.forward_calls = 0
        self.forward_input_lengths: list[int] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        outputs = self.model(*args, **kwargs)
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None:
            self.forward_input_lengths.append(int(input_ids.shape[-1]))

        step = self.forward_calls
        self.forward_calls += 1
        if step < len(self.forced_ids):
            forced_id = self.forced_ids[step]
            logits = torch.full_like(outputs.logits, -1e4)
            logits[:, -1, forced_id] = 1e4
            outputs.logits = logits
        return outputs

    def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def parameters(self) -> Any:
        return self.model.parameters()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def test_tgvf_vision_probe_qwen2vl(tmp_path: Path) -> None:
    if os.environ.get("TGVF_RUN_VISION_PROBE") != "1":
        pytest.skip("Set TGVF_RUN_VISION_PROBE=1 to run the Qwen2-VL vision probe.")

    if not torch.cuda.is_available():
        pytest.skip("Qwen2-VL vision probe requires CUDA for this integration test.")

    transformers = pytest.importorskip("transformers")
    pytest.importorskip("qwen_vl_utils")
    image_path = _probe_image(tmp_path)
    output_dir = Path(os.environ.get("TGVF_PROBE_OUTPUT_DIR", tmp_path / "tgvf_probe"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = os.environ.get("TGVF_QWEN2VL_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
    device = os.environ.get("TGVF_VISION_PROBE_DEVICE", "cuda:0")
    attn_implementation = os.environ.get("TGVF_ATTN_IMPLEMENTATION", "flash_attention_2")

    processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=False)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = transformers.Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map={"": device},
        trust_remote_code=False,
    )
    model.eval()

    visual_capture: dict[str, torch.Tensor] = {}
    visual_module = _visual_module(model)

    def capture_visual_embeddings(_module: Any, _inputs: Any, output: Any) -> None:
        visual_capture.setdefault("embeddings", _extract_visual_embeddings(output))

    hook = visual_module.register_forward_hook(capture_visual_embeddings)

    target_text = "the date below the barcode"
    forced_text = f"{FOVEATE_START}{target_text}{FOVEATE_END}"
    forced_ids = processor.tokenizer.encode(forced_text, add_special_tokens=False)
    wrapped_model = ForcedGreedyWrapper(model, forced_ids)

    try:
        capture = capture_tgvf_single_pass(
            wrapped_model,
            processor,
            image=str(image_path),
            question="What exact date is printed below the BARCODE label?",
            max_new_tokens=len(forced_ids) + 8,
            device=device,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    finally:
        hook.remove()

    assert capture.capture_found
    assert capture.target_text
    assert capture.target_hidden_states.ndim == 2
    assert capture.target_hidden_states.shape[0] > 0
    assert "embeddings" in visual_capture

    visual_embeddings = visual_capture["embeddings"]
    assert visual_embeddings.ndim == 2
    assert visual_embeddings.shape[0] > 0

    if capture.target_hidden_states.shape[-1] != visual_embeddings.shape[-1]:
        pytest.skip(
            "Target hidden size and visual embedding size differ: "
            f"{capture.target_hidden_states.shape[-1]} vs {visual_embeddings.shape[-1]}"
        )

    scores, token_scores = _cosine_probe_scores(capture.target_hidden_states, visual_embeddings)
    assert token_scores.shape == (capture.target_hidden_states.shape[0], visual_embeddings.shape[0])
    assert scores.ndim == 1
    assert scores.shape[0] == visual_embeddings.shape[0]
    assert torch.isfinite(scores).all()

    prompt_len = wrapped_model.forward_input_lengths[0]
    second_full_forward_used = any(length > 1 for length in wrapped_model.forward_input_lengths[1:])
    assert prompt_len > 1
    assert not second_full_forward_used

    torch.save(scores.cpu(), output_dir / "similarity_scores.pt")
    torch.save(token_scores.cpu(), output_dir / "similarity_token_scores.pt")
    debug = {
        "generated_text": capture.generated_text,
        "target_text": capture.target_text,
        "target_token_count": len(capture.target_token_ids),
        "target_hidden_shape": list(capture.target_hidden_states.shape),
        "visual_embedding_shape": list(visual_embeddings.shape),
        "score_shape": list(scores.shape),
        "token_score_shape": list(token_scores.shape),
        "score_aggregation": "mean_over_target_token_cosine_scores",
        "min_score": float(scores.min().item()),
        "max_score": float(scores.max().item()),
        "stop_reason": capture.stop_reason,
        "capture_layer": -1,
        "visual_source": "model visual forward hook pooler_output",
        "image_grid_thw": None
        if capture.image_grid_thw is None
        else capture.image_grid_thw.detach().cpu().tolist(),
        "attn_implementation": attn_implementation,
        "model_id": model_id,
        "forward_input_lengths": wrapped_model.forward_input_lengths,
        "second_full_forward_used": second_full_forward_used,
    }
    (output_dir / "probe_debug.json").write_text(json.dumps(debug, indent=2))
    (output_dir / "generated.txt").write_text(format_tgvf_debug(capture))

    assert not debug["second_full_forward_used"]


def _probe_image(tmp_path: Path) -> Path:
    image_env = os.environ.get("TGVF_VISION_PROBE_IMAGE")
    if image_env:
        image_path = Path(image_env)
        if not image_path.exists():
            pytest.skip(f"TGVF_VISION_PROBE_IMAGE does not exist: {image_path}")
        return image_path

    pillow = pytest.importorskip("PIL.Image")
    image_draw = pytest.importorskip("PIL.ImageDraw")
    image_path = tmp_path / "tgvf_probe_image.png"
    image = pillow.new("RGB", (640, 360), "white")
    draw = image_draw.Draw(image)
    draw.rectangle((40, 40, 600, 320), outline="black", width=4)
    draw.rectangle((90, 90, 300, 250), fill=(40, 110, 220))
    draw.text((340, 110), "BARCODE", fill="black")
    draw.text((340, 155), "DATE: 2026-05-20", fill="black")
    draw.text((340, 205), "SERIAL: TGVF-2B", fill="black")
    image.save(image_path)
    return image_path


def _visual_module(model: Any) -> torch.nn.Module:
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    raise AssertionError("Could not find Qwen2-VL visual module for the probe hook.")


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
        raise AssertionError(f"Unsupported visual module output type: {type(output)!r}")

    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    if embeddings.ndim != 2:
        raise AssertionError(f"Expected 2D visual embeddings, got shape {tuple(embeddings.shape)}")
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
