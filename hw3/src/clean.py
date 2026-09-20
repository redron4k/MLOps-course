"""Стадия clean: валидация схемы, фильтр длин, чистка ПДн, дедупликация."""

import json
import time
from pathlib import Path

from src.config import load_params
from src.dedup import exact_duplicates
from src.pii import scrub
from src.schema import Example, dump, iter_examples
from src.stats import percentile
from src.textnorm import normalize_group, normalize_text


def percentiles(values: list[int]) -> dict[str, int]:
    """Сводка длин для отчёта. Считается тем же модулем, что и гейт diversity:
    иначе в datasheet окажется одно число, а в сообщении об ошибке другое."""
    return {
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else 0,
    }


def main() -> None:
    params = load_params()
    cfg = params["clean"]
    paths = params["paths"]
    started = time.perf_counter()

    # 1. Валидация схемы. Битая строка — исключение с номером строки, стадия падает.
    examples: list[Example] = list(iter_examples(paths["raw"]))
    rows_in = len(examples)

    # 2. Фильтр длин.
    kept: list[Example] = []
    dropped_length = 0
    for ex in examples:
        user_len = len(ex.user)
        if not (cfg["min_user_chars"] <= user_len <= cfg["max_user_chars"]):
            dropped_length += 1
            continue
        if len(ex.assistant) < cfg["min_assistant_chars"]:
            dropped_length += 1
            continue
        kept.append(ex)

    # 3. Чистка ПДн — по всем ролям, включая ответ ассистента.
    pii_hits: dict[str, int] = {}
    pii_rows = 0
    if cfg["pii"]["enabled"]:
        for ex in kept:
            touched = False
            for msg in ex.messages:
                cleaned, hits = scrub(msg.content)
                if hits:
                    msg.content = cleaned
                    touched = True
                    for name, count in hits.items():
                        pii_hits[name] = pii_hits.get(name, 0) + count
            pii_rows += touched

    # 4. Точная дедупликация по нормализованному тексту вопроса.
    keys = [normalize_text(ex.user) for ex in kept]
    exact = set(exact_duplicates(keys))
    kept = [ex for i, ex in enumerate(kept) if i not in exact]

    # 5. TODO: сюда просится ещё один шаг дедупликации.
    near: set[int] = set()

    out = Path(paths["clean"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for ex in kept:
            fh.write(dump(ex) + "\n")

    metrics = {
        "version": params["collect"]["version"],
        "rows_in": rows_in,
        "rows_out": len(kept),
        "dropped_length": dropped_length,
        "dropped_exact_dup": len(exact),
        "dropped_near_dup": len(near),
        "pii_rows_masked": pii_rows,
        "pii_hits": {name: pii_hits.get(name, 0) for name in ("phone", "email", "birth_date")},
        "groups": len({normalize_group(ex.topic) for ex in kept}),
        "user_chars": percentiles([len(ex.user) for ex in kept]),
        "assistant_chars": percentiles([len(ex.assistant) for ex in kept]),
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_clean"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"clean: {rows_in} → {len(kept)} строк "
        f"(длина -{dropped_length}, точные -{len(exact)}, near-dup -{len(near)}), "
        f"ПДн замаскировано в {pii_rows} строках, {metrics['seconds']} с"
    )


if __name__ == "__main__":
    main()
