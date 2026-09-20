"""Схема примера и её валидация. Битая строка роняет стадию — это гейт, а не отчёт."""

import json
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

ROLES: tuple[str, ...] = ("system", "user", "assistant")


class SchemaError(ValueError):
    """Ошибка валидации с указанием файла и номера строки."""


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content пустой")
        return v


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    topic: str
    messages: list[Message]

    @model_validator(mode="after")
    def _exact_roles(self) -> "Example":
        got = tuple(m.role for m in self.messages)
        if got != ROLES:
            raise ValueError(
                f"роли должны идти ровно как {ROLES}, получено {got or '()'}"
            )
        if not self.id.strip():
            raise ValueError("id пустой")
        return self

    @property
    def user(self) -> str:
        return self.messages[1].content

    @property
    def assistant(self) -> str:
        return self.messages[2].content


def _explain(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "<корень>"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def iter_examples(path: str | Path) -> Iterator[Example]:
    """Прочитать JSONL с валидацией. Первая же битая строка — исключение с её номером."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                raise SchemaError(f"{path}:{lineno}: пустая строка в JSONL")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path}:{lineno}: не разбирается как JSON — {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise SchemaError(f"{path}:{lineno}: ожидался объект, получен {type(payload).__name__}")
            try:
                yield Example.model_validate(payload)
            except ValidationError as exc:
                raise SchemaError(f"{path}:{lineno}: {_explain(exc)}") from exc


def dump(example: Example) -> str:
    """Одна строка JSONL. ensure_ascii=False — иначе кириллица превращается в \\u04.."""
    return json.dumps(example.model_dump(), ensure_ascii=False)
