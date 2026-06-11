from ft.models.transformer import TransformerLanguageModel


def build_model(model_config: dict, vocab_size: int):
    model_type = model_config["type"].lower()
    kwargs = {key: value for key, value in model_config.items() if key != "type"}

    if model_type == "transformer":
        return TransformerLanguageModel(vocab_size=vocab_size, **kwargs)

    raise ValueError(f"Unsupported model type: {model_type}")


__all__ = [
    "TransformerLanguageModel",
    "build_model",
]
