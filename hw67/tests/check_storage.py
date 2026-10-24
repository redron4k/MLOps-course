"""Проверяет реальное переживание пересоздания контейнера MLflow.

Пересоздаёт ТОЛЬКО сервис mlflow текущего учебного compose-проекта.
Базу и volumes не удаляет. make down также сохраняет volumes.
"""
import os
import sys
import subprocess
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
import mlflow
from mlflow import MlflowClient
from src.common import connect, digest, load_params, save_json


def main():
    connect(load_params())
    with tempfile.TemporaryDirectory(prefix="hw6-storage-") as tmp:
        source = Path(tmp) / "probe.txt"
        source.write_text(str(uuid.uuid4()))
        expected = digest(source)
        with mlflow.start_run(run_name="storage-probe", tags={"kind": "check"}) as run:
            mlflow.log_artifact(str(source), "persistence")
            run_id = run.info.run_id
        project = os.environ.get("HW6_COMPOSE_PROJECT", "mlops-hw6")
        subprocess.run(["docker", "compose", "-p", project, "up", "-d", "--no-deps", "--force-recreate", "--wait", "mlflow"], check=True)
        dest = Path(tmp) / "download"
        dest.mkdir()
        restored = MlflowClient().download_artifacts(run_id, "persistence/probe.txt", str(dest))
        if digest(restored) != expected:
            raise SystemExit("Артефакт потерян или изменён при пересоздании контейнера")
        save_json("metrics/storage-check.json", {"run_id": run_id, "sha256": expected,
            "compose_sha256": digest("compose.yaml"), "project": project})
        print("PASS: новый контейнер вернул тот же артефакт, SHA256 совпал")


if __name__ == "__main__":
    main()
