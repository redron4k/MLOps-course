"""Поведенческая приёмка артефактов; полное обучение повторно не запускает."""
import json
import math
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
import mlflow
from mlflow import MlflowClient
from src.common import connect, digest, load_params, provenance
from src.promote import promote


def main():
    p = load_params()
    if not Path("metrics/sweep.json").exists():
        raise SystemExit("Нет результатов. Порядок: make up → make sweep → make candidate → make check-storage → make check")
    connect(p)
    client = MlflowClient()
    summary = json.loads(Path("metrics/sweep.json").read_text())
    parent = client.get_run(summary["parent_id"])
    runs = client.search_runs([parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}' and tags.kind = 'candidate'", max_results=1000)
    failures = []
    def check(title, fn):
        try:
            fn()
            print("PASS", title, flush=True)
        except Exception as exc:
            failures.append(title)
            print("FAIL", title, str(exc), flush=True)
    def require(condition, message):
        if not condition:
            raise AssertionError(message)
    def separate():
        require(parent.info.status == "FINISHED", "родительский sweep не завершён")
        require(len(runs) == len(p["sweep"]) >= 6, f"отдельных кандидатов: {len(runs)}")
        require({r.data.tags.get("candidate") for r in runs} == {c["name"] for c in p["sweep"]}, "не совпадают кандидаты")
        require(all(r.info.status == "FINISHED" for r in runs), "есть незавершённый run")
        configs = {(c["r"], c["lr"], tuple(c["target_modules"])) for c in p["sweep"]}
        require(len(configs) >= 6, "конфигурации повторяются")
        require(all(len({x[i] for x in configs}) >= 2 for i in range(3)), "нужно варьировать r, lr и target_modules")
    check("1. Минимум шесть отдельных законченных runs", separate)
    def versions():
        now = provenance(p)
        require(bool(runs), "нет кандидатов")
        for r in runs:
            for k in ("code_sha256", "config_sha256", "data_train_sha256", "data_val_sha256"):
                require(r.data.tags.get(k) == now[k], f"{r.info.run_id}: нет актуального {k}")
            require(len(r.data.tags.get("git_sha", "")) == 40, "нет git sha")
            require(len(r.data.tags.get("base_revision", "")) == 40, "не закреплена база HF")
    check("2. Код, данные, конфиг и база имеют версии", versions)
    def curves():
        by_name = {c["name"]: c for c in p["sweep"]}
        require(bool(runs), "нет кандидатов")
        for r in runs:
            c = by_name[r.data.tags["candidate"]]
            for key in ("r", "lr"):
                require(float(r.data.params[key]) == c[key], f"неверный {key}")
            require(r.data.params["target_modules"] == ",".join(c["target_modules"]), "неверные target_modules")
            curve = client.get_metric_history(r.info.run_id, "val_loss")
            require({0, p["train"]["max_steps"]}.issubset({x.step for x in curve}), "нет val до и после обучения")
            require(all(math.isfinite(x.value) for x in curve), "val nan/inf")
            require(len(client.get_metric_history(r.info.run_id, "train_loss")) == p["train"]["max_steps"], "неверное число train-шагов")
    check("3. Конфигурации и кривые соответствуют бюджету", curves)
    with tempfile.TemporaryDirectory(prefix="hw6-check-") as tmp:
        def download(run_id, path):
            dest = Path(tmp) / uuid.uuid4().hex
            dest.mkdir()
            return Path(client.download_artifacts(run_id, path, str(dest)))
        def artifacts():
            require(bool(runs), "нет кандидатов")
            for r in runs:
                import yaml
                model_file = download(r.info.run_id, "model/MLmodel")
                meta = yaml.safe_load(model_file.read_text())
                require("signature" in meta and "saved_input_example_info" in meta, "нет signature / input_example")
                location = meta["flavors"]["python_function"]["artifacts"]["adapter"]["path"]
                cfg = json.loads(download(r.info.run_id, f"model/{location}/adapter_config.json").read_text())
                require(cfg["r"] == int(r.data.params["r"]), "реальный ранг не совпал с логом")
                require(set(cfg["target_modules"]) == set(r.data.params["target_modules"].split(',')), "модули адаптера не совпали с логом")
                download(r.info.run_id, f"model/{location}/tokenizer_config.json")
                download(r.info.run_id, f"model/{location}/manifest.json")
        check("4. Реальный адаптер, токенизатор, signature и пример входа", artifacts)
        def samples():
            require(bool(runs), "нет кандидатов")
            for r in runs:
                s = json.loads(download(r.info.run_id, "samples.json").read_text())
                require(s["columns"] == ["prompt", "answer"], "неверная схема таблицы")
                require([x[0] for x in s["data"]] == p["generation"]["prompts"], "разные промпты")
                require(all(str(x[1]).strip() for x in s["data"]), "пустые ответы")
        check("5. Одинаковые промпты и непустые примеры генерации", samples)
        best = min(runs, key=lambda r: r.data.metrics.get("final_val_loss", float("inf"))) if runs else None
        def negative_gate():
            require(best is not None, "нет кандидата")
            name = "hw6-check-" + uuid.uuid4().hex
            try:
                try:
                    promote(client, best, name, "test", 0.0)
                except ValueError:
                    return
                raise AssertionError("кандидат с положительным loss прошёл нулевой порог")
            finally:
                # Только реестр, созданный этой проверкой. Учебный champion не затрагивается.
                from mlflow.exceptions import MlflowException
                try:
                    client.delete_registered_model(name)
                except MlflowException as exc:
                    if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
                        raise
        check("6. Запрет регистрации при непройденном пороге", negative_gate)
        def reload():
            require(best is not None, "нет кандидата")
            version = client.get_model_version_by_alias(p["registry"]["name"], "candidate")
            require(version.run_id == best.info.run_id, "alias указывает не на лучший run этого sweep")
            expected = json.loads(download(best.info.run_id, "samples.json").read_text())["data"][0]
            env = dict(os.environ, HF_HUB_OFFLINE="1")
            command = [sys.executable, "-I", str(Path("src/predict.py").resolve()),
                "--tracking-uri", mlflow.get_tracking_uri(), "--uri", f"models:/{version.name}@candidate", "--prompt", expected[0]]
            proc = subprocess.run(command, cwd=tmp, env=env, text=True, capture_output=True, timeout=240)
            require(proc.returncode == 0, proc.stderr[-1500:])
            answer = json.loads(proc.stdout.strip().splitlines()[-1])
            require(answer == [expected[1]], f"ответ изменился: {answer}")
        check("7. Загрузка из реестра в новом процессе, без локального конфига", reload)
        def storage():
            proof = json.loads(Path("metrics/storage-check.json").read_text())
            require(proof["compose_sha256"] == digest("compose.yaml"), "compose изменён после проверки")
            require(digest(download(proof["run_id"], "persistence/probe.txt")) == proof["sha256"], "артефакт потерян")
        check("8. Артефакт пережил пересоздание контейнера", storage)
    def report():
        for name in ("docs/defects.md", "docs/comparison.csv", "docs/conclusion.md"):
            require(Path(name).is_file() and Path(name).stat().st_size > 0, f"нет {name}")
    check("9. Есть сравнение, вывод и разбор дефектов (смысл проверяет преподаватель)", report)
    print(f"Итог: {9 - len(failures)}/9", flush=True)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
