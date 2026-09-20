#!/usr/bin/env python3
"""Проверка контаминации train/test как отдельный запускаемый гейт.

Стадия split уже проверяет себя, но проверка обязана существовать отдельно:
сплит могли собрать руками, получить от соседа или откатить через dvc checkout.
Возвращает 1 при любом пересечении — годится для CI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_params  # noqa: E402
from src.contamination import is_clean, report  # noqa: E402
from src.schema import iter_examples  # noqa: E402


def main() -> int:
    params = load_params()
    paths = params["paths"]
    nd = params["clean"]["near_dup"]

    train = list(iter_examples(paths["train"]))
    test = list(iter_examples(paths["test"]))
    rep = report(
        train,
        test,
        shingle_words=nd["shingle_words"],
        num_perm=nd["num_perm"],
        threshold=params["contamination"]["threshold"],
    )

    print(f"train: {len(train)} строк, test: {len(test)} строк")
    print(f"  пересечение по id:        {rep['id_overlap']}")
    print(f"  пересечение по тексту:    {rep['text_overlap']}")
    print(f"  пересечение по группам:   {rep['group_overlap']}")
    print(f"  near-dup пар train↔test:  {rep['near_dup_pairs']}")

    if is_clean(rep):
        print("контаминации нет")
        return 0

    for kind, items in rep["examples"].items():
        if items:
            print(f"  примеры ({kind}): {items}")
    print("КОНТАМИНАЦИЯ: train и test пересекаются, метрики на test завышены")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
