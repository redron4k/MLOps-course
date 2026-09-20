"""Проверка контаминации train/test — общий код для стадии split и для скрипта.

Три уровня, каждый ловит своё:
  1. id      — та же строка попала в оба сплита;
  2. текст   — тот же вопрос под другим id (копипаста источника);
  3. near-dup— парафраз: переставленные варианты ответа, другой порядок слов.
Случайный сплит валится обычно на третьем: первые два он проходит.
"""

from typing import Sequence

from src.dedup import cross_near_duplicates
from src.schema import Example
from src.textnorm import normalize_group, normalize_text


def report(
    train: Sequence[Example],
    test: Sequence[Example],
    shingle_words: int,
    num_perm: int,
    threshold: float,
) -> dict:
    """Сводка пересечений train/test. Ноль по всем ключам — сплит честный."""
    train_ids = {ex.id for ex in train}
    test_ids = {ex.id for ex in test}
    id_overlap = sorted(train_ids & test_ids)

    train_texts = [normalize_text(ex.user) for ex in train]
    test_texts = [normalize_text(ex.user) for ex in test]
    text_overlap = sorted(set(train_texts) & set(test_texts))

    train_groups = {normalize_group(ex.topic) for ex in train}
    test_groups = {normalize_group(ex.topic) for ex in test}
    group_overlap = sorted(train_groups & test_groups)

    pairs = cross_near_duplicates(
        train_texts, test_texts, shingle_words=shingle_words, num_perm=num_perm, threshold=threshold
    )

    return {
        "id_overlap": len(id_overlap),
        "text_overlap": len(text_overlap),
        "group_overlap": len(group_overlap),
        "near_dup_pairs": len(pairs),
        "examples": {
            "id": id_overlap[:3],
            "group": group_overlap[:3],
            "near_dup": [
                {"train": train[i].id, "test": test[j].id} for i, j in pairs[:3]
            ],
        },
    }


def is_clean(rep: dict) -> bool:
    return all(rep[k] == 0 for k in ("id_overlap", "text_overlap", "group_overlap", "near_dup_pairs"))
