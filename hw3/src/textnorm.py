"""Нормализация текста и шинглы — общие для очистки, сплита и проверки контаминации.

Один модуль на все три стадии специально: если нормализация разъедется,
дедупликация и проверка контаминации начнут мерить разные вещи, и проверка
станет зелёной при реальном пересечении.
"""

import re
import unicodedata

_SPACES = re.compile(r"\s+")
_DASHES = str.maketrans({"—": "-", "–": "-", "‑": "-", " ": " "})


def normalize_text(text: str) -> str:
    """Каноническая форма строки: NFKC, единые тире, схлопнутые пробелы, нижний регистр."""
    text = unicodedata.normalize("NFKC", text).translate(_DASHES)
    return _SPACES.sub(" ", text).strip().lower()


def normalize_group(topic: str) -> str:
    """Каноническая форма названия темы — ключ группы для сплита.

    В источнике одна и та же тема встречается в нескольких написаниях:
    «... - 2025» и «... — 2025», плюс склейка алиасов через «|».
    Без нормализации это разные группы, и сплит по группам протекает.
    """
    return normalize_text(topic.split("|")[0])


def shingles(text: str, size: int) -> set[str]:
    """Множество словных n-грамм — вход для MinHash."""
    words = re.findall(r"\w+", text)
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}
