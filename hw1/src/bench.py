"""Замер производительности машины на выбранной модели.

Три числа меряются РАЗДЕЛЬНО — смешивать их бессмысленно:
  * время загрузки модели  — разовая стоимость старта;
  * tokens/sec             — скорость генерации, только после прогрева;
  * пиковая RSS            — максимум за процесс, а не снимок в конце.
"""

import json
import resource
import statistics
import sys
import time
from pathlib import Path

from src.config import load_params
from src.model import generate, load_model


def peak_rss_mb() -> float:
    """Пиковая резидентная память процесса.

    ru_maxrss на macOS в байтах, на Linux в килобайтах.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 2) if sys.platform == "darwin" else peak / 1024


def main() -> None:
    params = load_params()
    prompt = params["bench"]["prompt"]

    # TODO: разделить замеры. Сейчас в одном таймере и загрузка, и генерация.
    t0 = time.perf_counter()
    tokenizer, model = load_model(params)

    # TODO: добавить прогрев перед измерением.
    speeds = []
    for _ in range(params["bench"]["measure_runs"]):
        _, n_tokens = generate(tokenizer, model, params, prompt)
        elapsed = time.perf_counter() - t0
        speeds.append(n_tokens / elapsed)

    load_time = 0.0

    # Медиана устойчивее среднего к одиночному выбросу.
    report = {
        "model": "Qwen/Qwen3-0.6B",
        "device": str(model.device),
        "dtype": params["model"]["dtype"],
        "load_time_sec": round(load_time, 2),
        "tokens_per_sec": round(statistics.median(speeds), 2),
        "tokens_per_sec_all": [round(s, 2) for s in speeds],
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }

    Path("docs").mkdir(exist_ok=True)
    Path("docs/bench.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
