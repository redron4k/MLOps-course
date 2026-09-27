"""Срез датасета курса: parquet -> data/train.jsonl, data/val.jsonl.

Нужен преподавателю, чтобы эталон был воспроизводим. Студенту не нужен:
вход стадии tokenize — его собственный датасет из ДЗ 3, выход стадии split,
положенный в data/train.jsonl и data/val.jsonl. Формат тот же.

Полноценный пайплайн подготовки данных — это ДЗ 3 (DVC, сплит, дедупликация).
Здесь берётся небольшая детерминированная выборка: 500 train / 100 val.

Формат строки JSONL: {"id", "topic", "messages": [{role, content}, ...]},
последняя реплика — ответ ассистента. Ровно этот формат ждёт src/tokenize_data.py.
"""

import json
import random
from pathlib import Path

import pyarrow.parquet as pq

from src.config import load_params

KEEP_FIELDS = ("id", "topic", "messages")


def read_rows(path: Path, limit: int) -> list[dict]:
    """Прочитать не более `limit` строк parquet, не поднимая файл целиком."""
    if not path.exists():
        raise SystemExit(
            f"Нет исходного parquet: {path}\n\n"
            "Так и должно быть: это датасет курса, у вас его нет. Вход стадии tokenize —\n"
            "ВАШ датасет из ДЗ 3. Положите выход стадии split в data/train.jsonl и\n"
            "data/val.jsonl (формат тот же: id, topic, messages) — и `make sample`\n"
            "больше не нужен, `make tokenize` и `make check` увидят файлы сами.\n"
            "Курсовой parquet нужен только преподавателю; путь к нему — params.yaml,\n"
            "data.source_train / data.source_val."
        )
    rows: list[dict] = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=list(KEEP_FIELDS)):
        rows.extend(batch.to_pylist())
        if len(rows) >= limit:
            break
    return rows[:limit]


def take_sample(rows: list[dict], size: int, seed: int) -> list[dict]:
    """Детерминированная выборка без повторов.

    Строки в parquet идут блоками по темам, поэтому берём случайные индексы,
    а не первые N: иначе в выборку попадёт две-три темы из сотен.
    """
    size = min(size, len(rows))
    index = sorted(random.Random(seed).sample(range(len(rows)), size))
    return [rows[i] for i in index]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            record = {
                "id": row["id"],
                "topic": row["topic"],
                "messages": [
                    {"role": m["role"], "content": m["content"]} for m in row["messages"]
                ],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    params = load_params()
    cfg = params["sample"]
    data = params["data"]

    for src_key, dst_key, size in (
        ("source_train", "train_jsonl", cfg["train_size"]),
        ("source_val", "val_jsonl", cfg["val_size"]),
    ):
        rows = read_rows(Path(data[src_key]), cfg["scan_rows"])
        sample = take_sample(rows, size, cfg["seed"])
        out = Path(data[dst_key])
        write_jsonl(sample, out)
        topics = len({r["topic"] for r in sample})
        print(f"{out}: {len(sample)} примеров, тем: {topics}")


if __name__ == "__main__":
    main()
