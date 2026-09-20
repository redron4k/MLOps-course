"""Дедупликация: точная и near-duplicate.

Точная ловит буквальные повторы, near-dup — переформулировки и перестановки
вариантов ответа. В курсовом источнике буквальных повторов нет, а почти-дублей
несколько процентов: одна только точная дедупликация здесь не делает ничего.
"""

from typing import Sequence

from datasketch import MinHash, MinHashLSH

from src.textnorm import shingles


def build_minhash(text: str, shingle_words: int, num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    mh.update_batch([s.encode("utf-8") for s in shingles(text, shingle_words)])
    return mh


def exact_duplicates(keys: Sequence[str]) -> list[int]:
    """Индексы повторных вхождений. Первое вхождение остаётся."""
    seen: set[str] = set()
    dupes: list[int] = []
    for i, key in enumerate(keys):
        if key in seen:
            dupes.append(i)
        else:
            seen.add(key)
    return dupes


def near_duplicates(
    texts: Sequence[str], shingle_words: int, num_perm: int, threshold: float
) -> list[int]:
    """Индексы почти-дублей: жадный проход, первый представитель кластера остаётся.

    MinHash + LSH дают линейное время вместо O(n^2) полного попарного сравнения.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    dupes: list[int] = []
    for i, text in enumerate(texts):
        mh = build_minhash(text, shingle_words, num_perm)
        if lsh.query(mh):
            dupes.append(i)
        else:
            lsh.insert(str(i), mh)
    return dupes


def cross_near_duplicates(
    left: Sequence[str],
    right: Sequence[str],
    shingle_words: int,
    num_perm: int,
    threshold: float,
) -> list[tuple[int, int]]:
    """Пары (индекс в left, индекс в right) с Жаккаром выше порога.

    Используется проверкой контаминации: точное совпадение текстов ловит
    копипасту, а протекают обычно парафразы.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, text in enumerate(left):
        lsh.insert(str(i), build_minhash(text, shingle_words, num_perm))
    pairs: list[tuple[int, int]] = []
    for j, text in enumerate(right):
        for key in lsh.query(build_minhash(text, shingle_words, num_perm)):
            pairs.append((int(key), j))
    return pairs
