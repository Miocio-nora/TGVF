from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from qwen_vl_utils import process_vision_info


@dataclass
class Qwen2VLCollator:
    """Build Qwen2-VL batches and mask padding labels."""

    processor: Any
    max_length: int | None = None
    label_pad_token_id: int = -100
    train_on_inputs: bool = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        messages = [feature["messages"] for feature in features]
        texts = [
            self.processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=False,
            )
            for message in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)

        batch = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()

        if not self.train_on_inputs:
            prompt_lengths = self._prompt_lengths(messages)
            for row, prompt_length in enumerate(prompt_lengths):
                labels[row, :prompt_length] = self.label_pad_token_id

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = self.label_pad_token_id

        image_pad_id = self._token_id("<|image_pad|>")
        video_pad_id = self._token_id("<|video_pad|>")
        for token_id in (image_pad_id, video_pad_id):
            if token_id is not None:
                labels[labels == token_id] = self.label_pad_token_id

        batch["labels"] = labels
        return batch

    def _prompt_lengths(self, messages: list[list[dict[str, Any]]]) -> list[int]:
        prompt_messages = [self._drop_last_assistant(message) for message in messages]
        prompt_texts = [
            self.processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in prompt_messages
        ]
        image_inputs, video_inputs = process_vision_info(prompt_messages)
        prompt_batch = self.processor(
            text=prompt_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors="pt",
        )
        attention_mask = prompt_batch.get("attention_mask")
        if attention_mask is None:
            return [len(input_ids) for input_ids in prompt_batch["input_ids"]]
        return attention_mask.sum(dim=1).tolist()

    @staticmethod
    def _drop_last_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if messages and messages[-1].get("role") == "assistant":
            return messages[:-1]
        return messages

    def _token_id(self, token: str) -> int | None:
        token_id = self.processor.tokenizer.convert_tokens_to_ids(token)
        if token_id == self.processor.tokenizer.unk_token_id:
            return None
        return token_id
