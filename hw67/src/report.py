"""Таблица из сервера, а не вручную переписанные числа."""
import csv
import io
import json
from pathlib import Path

from mlflow import MlflowClient
from src.common import connect, load_params


def main():
    p = load_params()
    connect(p)
    summary = json.loads(Path("metrics/sweep.json").read_text())
    client = MlflowClient()
    rows = []
    for item in summary["rows"]:
        run = client.get_run(item["run_id"])
        rows.append({"name": item["name"], "run_id": run.info.run_id,
            **{k: run.data.params[k] for k in ("r", "lr", "target_modules")},
            **{k: run.data.metrics[k] for k in ("base_val_loss", "final_val_loss", "seconds", "peak_memory_mb", "adapter_mb")}})
    rows.sort(key=lambda row: row["final_val_loss"])
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    Path("docs").mkdir(exist_ok=True)
    Path("docs/comparison.csv").write_text(out.getvalue())
    print(out.getvalue())


if __name__ == "__main__":
    main()
