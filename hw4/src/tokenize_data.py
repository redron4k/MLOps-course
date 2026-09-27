"""Стадия tokenize: JSONL -> токенизированный датасет на диске + метрики + отчёт.

Что здесь происходит с каждым примером: применяется шаблон чата,
текст режется по max_seq_len, собираются input_ids / attention_mask / labels.

Формат на диске: torch.save одного словаря со списком примеров
(input_ids / attention_mask / labels — списки int, тензоры делает коллатор).
Так проще, чем datasets.save_to_disk, и файл целиком годится в `outs`
DVC-стадии:
    deps: data/train.jsonl, data/val.jsonl, src/, params.yaml
    outs: data/tokenized/
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from src.collate import LABEL_PAD_ID
from src.config import load_params
from src.pack import pack_examples, packing_report
from src.prompt import build_chat_text

METRICS_PATH = Path("metrics/tokenize.json")
REPORT_PATH = Path("docs/tokenize_report.md")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Нет {path}.\n"
            "Сюда кладётся ВАШ датасет из ДЗ 3 — выход стадии split, тот же формат "
            "(id, topic, messages). Курсовой срез собирает `make sample`, но parquet "
            "для него есть только у преподавателя."
        )
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def mask_prompt(input_ids: list[int], n_prompt: int) -> list[int]:
    """labels для лосса."""
    # TODO: промпт должен быть замаскирован.
    return list(input_ids)


def encode_example(tokenizer, record: dict, params: dict, max_seq_len: int) -> dict:
    """Один пример -> input_ids / attention_mask / labels + служебная статистика."""
    messages = record["messages"]
    full_text = build_chat_text(tokenizer, messages, params, add_generation_prompt=False)

    encoded = tokenizer(full_text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    # TODO: найти границу промпта и ответа
    n_prompt, used_fallback = 0, False

    full_len = len(input_ids)
    truncated = full_len > max_seq_len
    if truncated:
        input_ids = input_ids[:max_seq_len]

    labels = mask_prompt(input_ids, n_prompt)
    return {
        "id": record.get("id"),
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "_meta": {
            "id": record.get("id"),
            "full_len": full_len,
            "prompt_len": n_prompt,
            "answer_len": full_len - n_prompt,
            "truncated": truncated,
            "bpe_fallback": used_fallback,
            "supervised": sum(1 for x in labels if x != LABEL_PAD_ID),
        },
    }


def describe(values: list[int]) -> dict:
    """Распределение длин: перцентили важнее среднего — хвост решает max_seq_len."""
    a = np.asarray(values)
    return {
        "count": int(a.size),
        "mean": round(float(a.mean()), 1),
        "p50": int(np.percentile(a, 50)),
        "p90": int(np.percentile(a, 90)),
        "p99": int(np.percentile(a, 99)),
        "max": int(a.max()),
    }


def truncation_stats(metas: list[dict], name: str, params: dict) -> dict:
    """Статистика обрезки по max_seq_len."""
    # TODO: посчитать долю обрезанных и предупредить, если она выше
    # tokenize.truncated_warn_ratio.
    return {}


def process_split(
    tokenizer, name: str, path: Path, params: dict
) -> tuple[list[dict], dict, list[dict]]:
    """Токенизировать сплит и собрать по нему статистику."""
    cfg = params["tokenize"]
    max_seq_len = cfg["max_seq_len"]
    records = read_jsonl(path)

    examples: list[dict] = []
    metas: list[dict] = []
    dropped = 0
    for record in records:
        encoded = encode_example(tokenizer, record, params, max_seq_len)
        # Статистика длин считается по ВСЕМ записям, включая выброшенные:
        # иначе доля обрезанных занижается ровно на самые длинные примеры.
        meta = encoded.pop("_meta")
        metas.append(meta)
        # Обрезка съела весь ответ: учить нечему, такой пример только шумит.
        if meta["supervised"] == 0:
            dropped += 1
            continue
        examples.append(encoded)

    stats = {
        "examples_in": len(records),
        "examples_kept": len(examples),
        "dropped_no_supervision": dropped,
        "length_tokens": describe([m["full_len"] for m in metas]),
        "prompt_tokens": describe([m["prompt_len"] for m in metas]),
        "answer_tokens": describe([m["answer_len"] for m in metas]),
        "bpe_boundary_fallback": sum(m["bpe_fallback"] for m in metas),
        "total_tokens": sum(len(e["input_ids"]) for e in examples),
        "supervised_tokens": sum(m["supervised"] for m in metas),
    }
    print(
        f"  {name}: {len(examples)} примеров, токенов {stats['total_tokens']} "
        f"(в лосс идёт {stats['supervised_tokens']}), p50/p90/max = "
        f"{stats['length_tokens']['p50']}/{stats['length_tokens']['p90']}/"
        f"{stats['length_tokens']['max']}"
    )
    stats.update(truncation_stats(metas, name, params))

    if params["packing"]["enabled"]:
        bins = pack_examples(examples, max_seq_len)
        stats["packing"] = packing_report(examples, bins, max_seq_len, params["packing"]["batch_size"])
    else:
        stats["packing"] = None
        bins = []

    return examples, stats, bins


def estimate_train_time(total_tokens: int, params: dict) -> dict:
    """Грубый прогноз времени обучения: токены x эпохи / пропускная способность."""
    cfg = params["train_estimate"]
    tps = cfg["tokens_per_sec"]
    seconds = total_tokens * cfg["epochs"] / tps
    return {
        "epochs": cfg["epochs"],
        "tokens_per_sec": tps,
        "tokens_per_epoch": total_tokens,
        "seconds": round(seconds, 1),
        "hours": round(seconds / 3600, 2),
        "note": (
            "tokens_per_sec взят из замера ГЕНЕРАЦИИ в ДЗ 1. Обучение считает "
            "ещё backward и шаг оптимизатора, поэтому реальная скорость ниже "
            "в 2-3 раза, а прогноз ниже — это оптимистичная граница."
        ),
    }


def mask_line(share: float) -> str:
    """Строка отчёта про долю токенов, попавших в лосс.

    Доля около единицы означает, что промпт не замаскирован: такой отчёт
    обязан сказать об этом прямо, а не молча показать красивое число.
    """
    if share > 0.99:
        return (
            "В лосс идут ПОЧТИ ВСЕ токены последовательности — похоже, промпт "
            "не замаскирован, и модель учится воспроизводить вопрос наравне с ответом."
        )
    return (
        f"Промпт занимает {1 - share:.0%} токенов: без маски лосса именно он "
        "и составил бы большую часть обучающего сигнала."
    )


def truncation_line(metrics: dict, train: dict) -> str:
    """Строка отчёта про обрезку — или честное признание, что её не считали."""
    warn = metrics["truncated_warn_ratio"]
    if "truncated_ratio" not in train:
        return (
            "Доля обрезанных НЕ ПОСЧИТАНА: стадия не знает, сколько ответов "
            f"потеряла на max_seq_len = {metrics['max_seq_len']}."
        )
    ratio = train["truncated_ratio"]
    verdict = "в норме" if ratio <= warn else "ВЫШЕ ПОРОГА"
    return f"Порог предупреждения: {warn:.1%}. Фактически обрезано (train): {ratio:.1%} — {verdict}."


def truncated_cell(s: dict) -> str:
    """Ячейка «обрезано». Если статистики нет — так и пишем, а не молчим."""
    if "truncated_ratio" not in s:
        return "НЕ СЧИТАЛАСЬ"
    return f"{s['truncated']} ({s['truncated_ratio']:.1%})"


def render_report(metrics: dict) -> str:
    """Отчёт по датасету — то, что читают глазами перед запуском обучения."""
    lines = [
        "# Отчёт стадии tokenize",
        "",
        "Сгенерирован `make tokenize`, руками не правится.",
        "",
        f"- модель: `{metrics['model']}`",
        f"- `max_seq_len`: {metrics['max_seq_len']}",
        f"- `enable_thinking`: {str(metrics['enable_thinking']).lower()}",
        f"- `padding_side`: {metrics['padding_side']}",
        "",
        "## Длины в токенах",
        "",
        "| сплит | примеров | p50 | p90 | p99 | max | обрезано | всего токенов | в лосс |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in metrics["splits"].items():
        L = s["length_tokens"]
        lines.append(
            f"| {name} | {s['examples_kept']} | {L['p50']} | {L['p90']} | {L['p99']} | "
            f"{L['max']} | {truncated_cell(s)} | "
            f"{s['total_tokens']} | {s['supervised_tokens']} |"
        )

    train = metrics["splits"]["train"]
    est = metrics["train_time_estimate"]
    share = train["supervised_tokens"] / train["total_tokens"]
    lines += [
        "",
        mask_line(share),
        "",
        "## Обрезка",
        "",
        truncation_line(metrics, train),
        "",
        f"p99 длины в токенах — {train['length_tokens']['p99']} при `max_seq_len` "
        f"{metrics['max_seq_len']}; хвост длиной до {train['length_tokens']['max']} "
        "теряет конец ответа. Примеров, у которых обрезка съела ответ целиком: "
        f"{train['dropped_no_supervision']} (такие выброшены).",
        "",
        "## Граница маски и BPE",
        "",
        f"Запасной путь по символьным офсетам сработал на {train['bpe_boundary_fallback']} "
        "примерах train. Основной путь (токены промпта совпадают с началом токенов "
        "полного текста) держится, потому что шаблон Qwen3 заканчивает промпт "
        "переводом строки — BPE не склеивает его с началом ответа. У шаблона, "
        "который обрывается посреди слова, склейка будет, и тогда границу задаёт "
        "первый токен, начинающийся не раньше конца промпта.",
        "",
        "## Прогноз времени обучения",
        "",
        f"Токенов за эпоху: {est['tokens_per_epoch']}, эпох: {est['epochs']}, "
        f"скорость: {est['tokens_per_sec']} ток/с.",
        "",
        f"Итого {est['seconds']:.0f} с ({est['hours']} ч).",
        "",
        est["note"],
        "",
    ]

    pack = train.get("packing")
    if pack:
        lines += [
            "## Packing",
            "",
            f"Последовательностей до packing: {pack['sequences_before']}, "
            f"бинов по {metrics['max_seq_len']} токенов после: {pack['sequences_after']} "
            f"(заполнение {pack['fill_ratio']:.0%}, до {pack['max_examples_per_bin']} "
            "примеров в бине).",
            "",
            f"Шагов оптимизатора при batch_size {pack['batch_size']}: "
            f"{pack['steps_before']} -> {pack['steps_after']} "
            f"(-{pack['steps_saved_ratio']:.0%}).",
            "",
            "Внимание: без блочно-диагональной маски токены соседних примеров "
            "в бине видят друг друга. Рядом с упакованным датасетом сохранены "
            "`seq_lens` — по ним такая маска строится.",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    params = load_params()
    tokenizer = AutoTokenizer.from_pretrained(params["model"]["name"])
    # TODO: tokenize.padding_side из params.yaml сюда так и не доехал

    out_dir = Path(params["data"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for name, key in (("train", "train_jsonl"), ("val", "val_jsonl")):
        examples, stats, bins = process_split(tokenizer, name, Path(params["data"][key]), params)
        torch.save(
            {
                "examples": examples,
                "model": params["model"]["name"],
                "max_seq_len": params["tokenize"]["max_seq_len"],
                "padding_side": tokenizer.padding_side,
                "pad_token_id": tokenizer.pad_token_id,
            },
            out_dir / f"{name}.pt",
        )
        if bins:
            torch.save({"bins": bins, "max_seq_len": params["tokenize"]["max_seq_len"]},
                       out_dir / f"{name}_packed.pt")
        splits[name] = stats

    metrics = {
        "model": params["model"]["name"],
        "enable_thinking": params["model"].get("enable_thinking"),
        "max_seq_len": params["tokenize"]["max_seq_len"],
        "padding_side": tokenizer.padding_side,
        "truncated_warn_ratio": params["tokenize"]["truncated_warn_ratio"],
        "splits": splits,
        "train_time_estimate": estimate_train_time(splits["train"]["total_tokens"], params),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(metrics), encoding="utf-8")
    print(f"  -> {out_dir}/, {METRICS_PATH}, {REPORT_PATH}")


if __name__ == "__main__":
    main()
