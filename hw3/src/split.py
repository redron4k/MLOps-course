"""Стадия split: разбиение на train/val/test."""

import json
import random
import time
from pathlib import Path

from src.config import load_params
from src.contamination import is_clean, report
from src.schema import Example, dump, iter_examples
from src.textnorm import normalize_group


def group_split(groups: dict[str, list[Example]], ratios: dict[str, float], seed: int) -> dict[str, list[Example]]:
    """Разделить целые группы, приблизив размеры к заданным долям.

    Строковый сплит смешивает вопросы одной подсистемы между train и test. Здесь
    группа назначается единственному бакету; большой группе нельзя «долить»
    остаток в другой сплит, поэтому точные 80/10/10 не гарантируются.
    """
    total = sum(len(rows) for rows in groups.values())
    targets = {name: total * ratio for name, ratio in ratios.items()}
    buckets: dict[str, list[Example]] = {name: [] for name in ratios}
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda key: len(groups[key]), reverse=True)
    for key in keys:
        name = max(
            ratios,
            key=lambda candidate: (targets[candidate] - len(buckets[candidate]), -len(buckets[candidate])),
        )
        buckets[name].extend(groups[key])
    return buckets


def main() -> None:
    params = load_params()
    paths = params["paths"]
    cfg = params["split"]
    started = time.perf_counter()

    examples: list[Example] = list(iter_examples(paths["clean"]))
    if cfg["group_key"] != "topic":
        raise SystemExit(f"неизвестный split.group_key: {cfg['group_key']!r}")

    grouped: dict[str, list[Example]] = {}
    for ex in examples:
        key = normalize_group(ex.topic)
        grouped.setdefault(key, []).append(ex)

    buckets = group_split(grouped, cfg["ratios"], cfg["seed"])

    for name, rows in buckets.items():
        out = Path(paths[name])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(dump(ex) + "\n")

    nd = params["clean"]["near_dup"]
    rep = report(
        buckets["train"],
        buckets["test"],
        shingle_words=nd["shingle_words"],
        num_perm=nd["num_perm"],
        threshold=params["contamination"]["threshold"],
    )

    metrics = {
        "version": params["collect"]["version"],
        "seed": cfg["seed"],
        "group_key": cfg["group_key"],
        "groups_total": len(grouped),
        "sizes": {name: len(rows) for name, rows in buckets.items()},
        "groups": {
            name: len({normalize_group(ex.topic) for ex in rows}) for name, rows in buckets.items()
        },
        "ratios_actual": {
            name: round(len(rows) / len(examples), 4) for name, rows in buckets.items()
        },
        "contamination": rep,
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_split"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not is_clean(rep):
        raise SystemExit(
            "split: обнаружена контаминация train/test: "
            f"id={rep['id_overlap']}, text={rep['text_overlap']}, "
            f"groups={rep['group_overlap']}, near-dup={rep['near_dup_pairs']}"
        )

    print(
        "split: "
        + ", ".join(f"{name} {len(rows)}" for name, rows in buckets.items())
        + f" (групп {len(grouped)}, {metrics['seconds']} с)"
    )


if __name__ == "__main__":
    main()
