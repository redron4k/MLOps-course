"""Генерация ответа на промпт из конфига."""

from src.config import load_params
from src.model import generate, load_model, set_seed


def main() -> None:
    params = load_params()

    set_seed(params["generate"]["seed"])

    tokenizer, model = load_model(params)
    print(f"Модель: Qwen/Qwen3-0.6B")
    text, n_tokens = generate(tokenizer, model, params, params["bench"]["prompt"])

    print(text)
    print(f"\n[{n_tokens} токенов]")


if __name__ == "__main__":
    main()
