"""Чтение params.yaml — единственная точка правды о конфигурации."""

from pathlib import Path

import yaml


def load_params(path: str = "params.yaml") -> dict:
    """Загрузить параметры запуска."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def source_files(params: dict) -> list[Path]:
    """Файлы-источники для текущей версии датасета.

    Версия живёт в params, а не в аргументах командной строки: иначе
    dvc.lock не запомнит, из чего собран артефакт.
    """
    version = params["collect"]["version"]
    sources = params["collect"]["sources"]
    if version not in sources:
        raise SystemExit(
            f"collect.version = {version!r}, но в collect.sources "
            f"есть только {sorted(sources)}"
        )
    files = [Path(p) for p in sources[version]]
    missing = [f for f in files if not f.exists()]
    if missing:
        # Первое, обо что спотыкается каждый: пакет приходит настроенным на
        # курсовой датасет, которого у студента нет. Сообщение должно говорить,
        # что делать, а не печатать FileNotFoundError с чужим абсолютным путём.
        raise SystemExit(
            "стадия collect не нашла источник:\n  "
            + "\n  ".join(str(f) for f in missing)
            + "\n\nТак и должно быть, если вы ещё не подключили СВОЙ датасет.\n"
              "Что сделать: переписать src/collect.py под свой источник и\n"
              "указать пути в params.yaml → collect.sources. Остальные стадии\n"
              "работают с контрактом raw.jsonl и правок не требуют."
        )
    return files
