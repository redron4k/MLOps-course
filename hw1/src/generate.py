"""Генерация ответа на промпт из конфига."""

from src.config import load_params
from src.model import generate, load_model


def main() -> None:
    params = load_params()
    # TODO: зафиксировать seed — сейчас прогоны не воспроизводятся

    tokenizer, model = load_model(params)
    print(f"Модель: Qwen/Qwen3-0.6B")
    text, n_tokens = generate(tokenizer, model, params, params["bench"]["prompt"])

    print(text)
    print(f"\n[{n_tokens} токенов]")


if __name__ == "__main__":
    main()
