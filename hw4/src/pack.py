"""Packing: несколько коротких примеров в одну последовательность.

Зачем: медианный пример здесь ~220 токенов при max_seq_len 512 — почти
половина каждого батча уходит на паддинг. Packing убирает этот простой
и сокращает число шагов оптимизатора.

Чем платим: внимание «перетекает» между склеенными примерами — токены
второго примера видят первый. Честное решение — блочно-диагональная маска
(FlashAttention varlen / `position_ids` с перезапуском), поэтому вместе
с последовательностью сохраняем `seq_lens`: длины исходных примеров внутри
бина. Без этой маски packing МЕНЯЕТ задачу, а не только ускоряет обучение.

Здесь — greedy first-fit по убыванию длины: дёшево и упаковывает плотно.
"""

LABEL_PAD_ID = -100


def pack_examples(examples: list[dict], max_seq_len: int) -> list[dict]:
    """Сложить примеры в бины длиной не больше max_seq_len."""
    order = sorted(range(len(examples)), key=lambda i: -len(examples[i]["input_ids"]))
    bins: list[dict] = []
    for i in order:
        ex = examples[i]
        size = len(ex["input_ids"])
        for b in bins:
            if len(b["input_ids"]) + size <= max_seq_len:
                target = b
                break
        else:
            target = {"input_ids": [], "attention_mask": [], "labels": [], "seq_lens": []}
            bins.append(target)
        target["input_ids"].extend(ex["input_ids"])
        target["attention_mask"].extend(ex["attention_mask"])
        target["labels"].extend(ex["labels"])
        target["seq_lens"].append(size)
    return bins


def packing_report(examples: list[dict], bins: list[dict], max_seq_len: int, batch_size: int) -> dict:
    """Выигрыш packing в числе шагов оптимизатора."""
    steps_before = -(-len(examples) // batch_size)      # ceil
    steps_after = -(-len(bins) // batch_size)
    used = sum(len(b["input_ids"]) for b in bins)
    return {
        "sequences_before": len(examples),
        "sequences_after": len(bins),
        "batch_size": batch_size,
        "steps_before": steps_before,
        "steps_after": steps_after,
        "steps_saved_ratio": round(1 - steps_after / steps_before, 4) if steps_before else 0.0,
        "fill_ratio": round(used / (len(bins) * max_seq_len), 4) if bins else 0.0,
        "max_examples_per_bin": max((len(b["seq_lens"]) for b in bins), default=0),
    }
