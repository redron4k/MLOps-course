"""Загрузка модели и сборка промпта. Общий код для генерации и замеров."""

import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    """Зафиксировать источники случайности, чтобы прогон воспроизводился."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(params: dict):
    """Загрузить токенизатор и модель по имени из конфига."""
    name = params["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=getattr(torch, params["model"]["dtype"]),
        device_map=params["model"]["device"],
    )
    model.eval()
    return tokenizer, model


def build_prompt(tokenizer, params: dict, text: str) -> str:
    """Собрать промпт шаблоном модели.

    Никогда не склеивайте роли вручную: у каждой модели свой формат,
    а расхождение шаблонов обучения и инференса — самая частая тихая ошибка.
    """
    messages = [{"role": "user", "content": text}]
    kwargs = {}
    # Параметр есть только у моделей с режимом рассуждений (Qwen3);
    # остальные шаблоны его молча проигнорируют.
    if params["generate"].get("enable_thinking") is not None:
        kwargs["enable_thinking"] = params["generate"]["enable_thinking"]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )


def generate(tokenizer, model, params: dict, text: str) -> tuple[str, int]:
    """Сгенерировать ответ. Возвращает текст и число новых токенов."""
    prompt = build_prompt(tokenizer, params, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    temperature = params["generate"]["temperature"]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=params["generate"]["max_new_tokens"],
            do_sample=temperature > 0,
            **({"temperature": temperature} if temperature > 0 else {}),
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), len(new_tokens)
