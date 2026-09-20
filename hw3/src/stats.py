"""Персентили и разброс длин — общие для стадий clean и diversity.

Одна реализация на обе стадии специально. Если отчёт о длинах и гейт по
длинам начнут считать p90 по-разному, они будут спорить друг с другом:
в datasheet одно число, в сообщении об ошибке другое. Метод здесь самый
простой (nearest-rank по индексу q·n) — важна не тонкость определения,
а то, что оно везде одно.
"""

from statistics import mean, pstdev
from typing import Sequence


def percentile(values: Sequence[int], q: float) -> int:
    """Персентиль уровня q (0..1) по отсортированной копии. Пустой вход — 0."""
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def spread(values: Sequence[int]) -> dict:
    """Сводка распределения длин: персентили, отношение p90/p10, коэф. вариации.

    p90/p10 — грубая, но честная мера разброса: у набора, сгенерированного
    по одному шаблону, она близка к единице. Коэффициент вариации добавлен
    как второй взгляд на то же самое, гейт его не использует.
    """
    if not values:
        return {"p10": 0, "p50": 0, "p90": 0, "ratio_p90_p10": 0.0, "cv": 0.0, "min": 0, "max": 0}
    p10 = percentile(values, 0.10)
    p50 = percentile(values, 0.50)
    p90 = percentile(values, 0.90)
    average = mean(values)
    return {
        "p10": p10,
        "p50": p50,
        "p90": p90,
        # max(p10, 1) — деления на ноль не бывает: пустые ответы отсеивает clean.
        "ratio_p90_p10": round(p90 / max(p10, 1), 2),
        "cv": round(pstdev(values) / average, 3) if average else 0.0,
        "min": min(values),
        "max": max(values),
    }
