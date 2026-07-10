from revisit_vlm_clean.peft_token_rows import (
    protocol_token_peft_kwargs,
    trainable_token_indices_from_checkpoint,
)


def test_protocol_token_peft_kwargs_keeps_default_full_modules() -> None:
    result = protocol_token_peft_kwargs(mode="full_modules", token_ids=[11, 12])
    assert result == {
        "modules_to_save": ["embed_tokens", "lm_head"],
        "trainable_token_indices": None,
    }


def test_protocol_token_peft_kwargs_builds_input_and_output_row_adapters() -> None:
    result = protocol_token_peft_kwargs(mode="row_only", token_ids=[11, 12, 11])
    assert result == {
        "modules_to_save": None,
        "trainable_token_indices": {
            "embed_tokens": [11, 12],
            "lm_head": [11, 12],
        },
    }


def test_trainable_token_indices_reconstructs_checkpoint_modules() -> None:
    state = {
        "base_model.model.model.language_model.embed_tokens.token_adapter.trainable_tokens_delta": object(),
        "base_model.model.lm_head.token_adapter.trainable_tokens_delta": object(),
    }
    assert trainable_token_indices_from_checkpoint(state=state, token_ids=[11, 12]) == {
        "embed_tokens": [11, 12],
        "lm_head": [11, 12],
    }
