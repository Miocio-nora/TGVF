from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, set_seed

from revisit_vlm.data import Qwen2VLCollator, Qwen2VLJsonlDataset
from revisit_vlm.models import apply_research_edits, load_qwen2vl


@dataclass
class TrainConfig:
    model_name_or_path: str
    train_file: str
    output_dir: str
    eval_file: str | None = None
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-5
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int | None = 4096
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int | None = None
    seed: int = 42
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = True
    freeze_vision_tower: bool = False
    train_on_inputs: bool = False
    attn_implementation: str | None = "flash_attention_2"
    torch_dtype: str = "auto"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    for field_name, field_def in TrainConfig.__dataclass_fields__.items():
        default = field_def.default
        arg_name = f"--{field_name}"
        if isinstance(default, bool):
            parser.add_argument(arg_name, type=str_to_bool, default=default)
        elif field_name in {"eval_steps", "max_length"}:
            parser.add_argument(arg_name, type=optional_int, default=default)
        elif field_name in {"eval_file", "attn_implementation"}:
            parser.add_argument(arg_name, type=none_if_string, default=default)
        elif default is None:
            parser.add_argument(arg_name, default=default)
        else:
            parser.add_argument(arg_name, type=type(default), default=default)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    loaded = load_qwen2vl(
        config.model_name_or_path,
        torch_dtype=config.torch_dtype,
        attn_implementation=config.attn_implementation,
    )
    model = apply_research_edits(
        loaded.model,
        {"freeze_vision_tower": config.freeze_vision_tower},
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if config.use_lora:
        model = get_peft_model(model, build_lora_config(config))
        model.print_trainable_parameters()

    train_dataset = Qwen2VLJsonlDataset(config.train_file)
    eval_dataset = Qwen2VLJsonlDataset(config.eval_file) if config.eval_file else None
    collator = Qwen2VLCollator(
        loaded.processor,
        max_length=config.max_length,
        train_on_inputs=config.train_on_inputs,
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        remove_unused_columns=False,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps" if eval_dataset is not None and config.eval_steps else "no",
        save_strategy="steps",
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=loaded.processor.tokenizer,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    loaded.processor.save_pretrained(config.output_dir)
    write_config(config)


def build_lora_config(config: TrainConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def write_config(config: TrainConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "train_config.json"
    path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def optional_int(value: str | int | None) -> int | None:
    if value is None or isinstance(value, int):
        return value
    if value.lower() in {"none", "null", ""}:
        return None
    return int(value)


def none_if_string(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in {"none", "null", ""}:
        return None
    return value


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


if __name__ == "__main__":
    main()
