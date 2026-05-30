# revisit-vlm

Editable Qwen2-VL scaffold for VLM research. The code keeps model construction,
data processing, module surgery, and fine-tuning in separate files so you can
replace parts of Qwen2-VL without rewriting the training loop.

## Install

```bash
pip install -e ".[dev]"
```

For GPU training, install the PyTorch build that matches your CUDA environment
before installing this package.

## Data format

Training data is JSONL. Each line contains a `messages` list compatible with
Qwen2-VL chat templates:

```json
{"messages":[{"role":"user","content":[{"type":"image","image":"examples/images/demo.jpg"},{"type":"text","text":"What is in the image?"}]},{"role":"assistant","content":[{"type":"text","text":"A small test image."}]}]}
```

Local image paths are resolved relative to the JSONL file unless they are
absolute paths or URLs.

## Fine-tune

LoRA:

```bash
accelerate launch scripts/train_qwen2vl.py \
  --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
  --train_file data/train.jsonl \
  --output_dir outputs/qwen2vl-lora \
  --use_lora true \
  --bf16 true
```

Full fine-tuning:

```bash
accelerate launch scripts/train_qwen2vl.py \
  --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
  --train_file data/train.jsonl \
  --output_dir outputs/qwen2vl-full \
  --use_lora false \
  --gradient_checkpointing true \
  --bf16 true
```

## Edit points

- `src/revisit_vlm/models/qwen2vl.py`: model and processor loading.
- `src/revisit_vlm/models/surgery.py`: replace or freeze modules.
- `src/revisit_vlm/data/qwen2vl_dataset.py`: JSONL loading and image path normalization.
- `src/revisit_vlm/data/collators.py`: chat template processing and label masking.
- `scripts/train_qwen2vl.py`: training entry point.
