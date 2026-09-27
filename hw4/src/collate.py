"""Коллатор: динамический паддинг батча.

Паддим до самого длинного примера в батче, а не до max_seq_len:
внимание квадратично по длине, лишние 300 паддингов стоят реального времени.
"""

import torch

LABEL_PAD_ID = -100


class DynamicPaddingCollator:
    """Собирает список примеров в батч тензоров."""

    def __init__(self, pad_token_id: int, padding_side: str = "right") -> None:
        if padding_side not in ("left", "right"):
            raise ValueError(f"padding_side должен быть left или right, получено {padding_side!r}")
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def _pad(self, seq: list[int], width: int, value: int) -> list[int]:
        tail = [value] * (width - len(seq))
        return tail + seq if self.padding_side == "left" else seq + tail

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        width = max(len(f["input_ids"]) for f in features)
        batch = {
            "input_ids": [self._pad(f["input_ids"], width, self.pad_token_id) for f in features],
            "attention_mask": [self._pad(f["attention_mask"], width, 0) for f in features],
            "labels": [self._pad(f["labels"], width, LABEL_PAD_ID) for f in features],
        }
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
