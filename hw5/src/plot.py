"""Кривые train/val loss обоих вариантов на одном графике -> docs/curves.png."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import load_params  # noqa: E402

COLORS = {"all_layers": "#7e22ce", "freeze14": "#0e7490"}


def main() -> None:
    params = load_params()
    mdir = Path(params["paths"]["metrics"])
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(mdir.glob("train_*.json"))]
    if not runs:
        raise SystemExit("нет metrics/train_*.json — сначала make train")

    fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for r in runs:
        c = COLORS.get(r["variant"], "#444")
        label = f"{r['variant']} ({r['trainable_share']:.2%} параметров, {r['seconds']:.0f} с)"
        ax_t.plot(*zip(*r["curve_train"]), color=c, alpha=.85, label=label)
        ax_v.plot(*zip(*r["curve_val"]), color=c, marker="o", label=r["variant"])
    ax_t.set_title("train loss (по шагам оптимизатора)")
    ax_v.set_title("val loss (шаг 0 — базовая модель)")
    for ax in (ax_t, ax_v):
        ax.set_xlabel("шаг")
        ax.grid(alpha=.3)
    ax_t.set_ylabel("loss на токен ответа")
    ax_t.legend(fontsize=8)
    fig.tight_layout()
    out = Path(params["paths"]["curves"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
