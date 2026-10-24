"""Выбор из одного законченного sweep; порог проверяется ДО записи в реестр."""
import argparse
import json
import math
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from src.common import connect, load_params, provenance, save_json


def gate(run, max_ratio):
    m = run.data.metrics
    values = [m.get(k, float("nan")) for k in ("final_val_loss", "base_val_loss")]
    if not math.isfinite(max_ratio) or max_ratio < 0 or not all(math.isfinite(v) for v in values) or values[1] <= 0:
        raise ValueError("Некорректная метрика или порог")
    if run.info.status != "FINISHED" or values[0] / values[1] > max_ratio:
        raise ValueError(f"Порог не пройден: val/base={values[0] / values[1]:.4f}, максимум {max_ratio}")
    return values[0] / values[1]


def select(client, params, summary):
    current = provenance(params)
    parent = client.get_run(summary["parent_id"])
    if parent.info.status != "FINISHED":
        raise ValueError("Sweep не завершён")
    runs = client.search_runs([parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}' and tags.kind = 'candidate'",
        max_results=1000)
    names = {c["name"] for c in params["sweep"]}
    if len(runs) != len(names) or {r.data.tags.get("candidate") for r in runs} != names:
        raise ValueError("Число отдельных runs не совпадает с конфигурациями")
    for run in runs:
        for key in ("code_sha256", "config_sha256", "data_train_sha256", "data_val_sha256"):
            if run.data.tags.get(key) != current[key]:
                raise ValueError(f"Run {run.info.run_id}: отсутствует или устарел {key}; повторите sweep")
        if run.info.status != "FINISHED" or not math.isfinite(run.data.metrics.get("final_val_loss", float("nan"))):
            raise ValueError("Есть незавершённый кандидат или отсутствует val loss")
    if len({r.data.tags.get("base_revision") for r in runs}) != 1:
        raise ValueError("Кандидаты обучены на разных ревизиях базы")
    return min(runs, key=lambda r: (r.data.metrics["final_val_loss"], r.info.run_id))


def promote(client, run, name, alias, threshold):
    ratio = gate(run, threshold)
    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", name)
    client.set_model_version_tag(name, version.version, "val_ratio", str(ratio))
    client.set_model_version_tag(name, version.version, "gate.max_val_ratio", str(threshold))
    client.set_model_version_tag(name, version.version, "validation", "passed")
    client.set_registered_model_alias(name, alias, version.version)
    return version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-val-ratio", type=float)
    args = ap.parse_args()
    p = load_params()
    connect(p)
    client = MlflowClient()
    summary = json.loads(Path("metrics/sweep.json").read_text())
    best = select(client, p, summary)
    threshold = p["registry"]["max_val_ratio"] if args.max_val_ratio is None else args.max_val_ratio
    version = promote(client, best, p["registry"]["name"], p["registry"]["alias"], threshold)
    result = {"name": version.name, "version": version.version, "run_id": best.info.run_id,
        "uri": f"models:/{version.name}@{p['registry']['alias']}", "val_loss": best.data.metrics["final_val_loss"]}
    save_json("metrics/promoted.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
