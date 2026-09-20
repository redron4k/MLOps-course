"""Стадия collect: источник → data/raw.jsonl.

ЗДЕСЬ студент подменяет сбор на свой. Ниже — чтение parquet курсового датасета
НМО; у вас на этом месте будет парсер сайта, выгрузка из БД, экспорт из Notion.
Контракт стадии, а не её внутренности, держит остальной пайплайн:
на выходе JSONL со строками {"id", "topic", "messages": [system, user, assistant]}.

Скачанный чужой набор сам по себе сдачей не является (README, «Готовый датасет
как источник»). Поэтому стадия не перекладывает parquet в JSONL один в один,
а делает три вещи, и каждая видна числом в metrics/collect.json:

  1. сужает набор до перечисленных тем (collect.topics), если это нужно задаче;
  2. сверяет ответ с разметкой источника (collect.verify_answer_index) —
     расхождение выбрасывается, а не переносится в обучение;
  3. разводит единственную инструкцию источника на варианты
     (collect.system_prompts), чтобы модель не заучила её формулировку.
"""

import hashlib
import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from src.config import load_params, source_files

COLUMNS = ["id", "topic", "correct_choice_indices", "messages"]
BATCH = 2000


def pick_prompt(example_id: str, variants: list[str]) -> str:
    """Детерминированно выбрать вариант инструкции по id примера.

    Именно sha1, а не встроенный hash(): тот солится на каждый запуск процесса,
    и raw.jsonl переставал бы быть воспроизводимым.
    """
    digest = hashlib.sha1(example_id.encode("utf-8")).hexdigest()
    return variants[int(digest, 16) % len(variants)]


def answer_matches_source(row: dict) -> bool:
    """Совпадает ли ответ ассистента с correct_choice_indices источника.

    В sft_single правильный вариант ровно один, а ответ начинается с его
    номера. Всё, что не так, — либо другой тип задачи, либо битая разметка.
    """
    indices = list(row["correct_choice_indices"] or [])
    if len(indices) != 1:
        return False
    return row["messages"][2]["content"].startswith(f"Ответ: {indices[0]}")


def main() -> None:
    params = load_params()
    cfg = params["collect"]
    paths = params["paths"]
    n_rows = cfg["n_rows"]
    variants = cfg["system_prompts"]
    if not variants:
        raise SystemExit("collect.system_prompts пуст: инструкцию брать неоткуда")
    topics = cfg["topics"]
    wanted = set(topics) if topics else None

    out = Path(paths["raw"])
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    scanned = written = dropped_topic = dropped_answer = 0
    prompts_used: set[str] = set()

    with out.open("w", encoding="utf-8") as fh:
        for src in source_files(params):
            if not src.exists():
                raise SystemExit(f"нет файла-источника: {src}")
            taken = 0
            # Фильтры применяются ДО отсечки n_rows: иначе «первые 3000 строк»
            # и «3000 строк по теме» — разные вещи, и сужение набора давало бы
            # случайный огрызок вместо заказанного объёма.
            for batch in pq.ParquetFile(src).iter_batches(batch_size=BATCH, columns=COLUMNS):
                for row in batch.to_pylist():
                    if taken >= n_rows:
                        break
                    scanned += 1
                    if wanted is not None and row["topic"] not in wanted:
                        dropped_topic += 1
                        continue
                    if cfg["verify_answer_index"] and not answer_matches_source(row):
                        dropped_answer += 1
                        continue
                    prompt = pick_prompt(row["id"], variants)
                    prompts_used.add(prompt)
                    record = {
                        "id": row["id"],
                        "topic": row["topic"],
                        # messages из parquet уже в формате чата; меняется только
                        # системная реплика — на выбранный вариант инструкции.
                        "messages": [
                            {"role": "system", "content": prompt},
                            *(
                                {"role": m["role"], "content": m["content"]}
                                for m in row["messages"][1:]
                            ),
                        ],
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    taken += 1
                    written += 1
                if taken >= n_rows:
                    break

    metrics = {
        "version": cfg["version"],
        "files": len(source_files(params)),
        "rows_scanned": scanned,
        "rows_written": written,
        "dropped_topic_filter": dropped_topic,
        "dropped_answer_mismatch": dropped_answer,
        "topics_filter": len(wanted) if wanted else 0,
        "system_prompt_variants": len(prompts_used),
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_collect"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"collect: версия {cfg['version']}, файлов {metrics['files']}, "
        f"просмотрено {scanned}, записано {written} "
        f"(фильтр тем -{dropped_topic}, расхождение с разметкой -{dropped_answer}), "
        f"вариантов инструкции {len(prompts_used)}, "
        f"{metrics['seconds']} с → {out}"
    )


if __name__ == "__main__":
    main()
