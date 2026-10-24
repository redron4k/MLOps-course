"""Нужны tracking URI, имя модели и текст. Локальные параметры обучения не читаются."""
import argparse
import json

import mlflow
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True, help="models:/my-qwen-adapter@champion")
    ap.add_argument("--tracking-uri", default="http://127.0.0.1:5006")
    ap.add_argument("--prompt", required=True)
    args = ap.parse_args()
    if not args.uri.startswith("models:/"):
        raise SystemExit("Ожидается адрес models:/ из реестра")
    mlflow.set_tracking_uri(args.tracking_uri)
    model = mlflow.pyfunc.load_model(args.uri)
    print(json.dumps(model.predict(pd.DataFrame({"prompt": [args.prompt]})), ensure_ascii=False))


if __name__ == "__main__":
    main()
