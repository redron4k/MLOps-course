"""Учебный прототип выпуска. Реализация содержит дефекты; см. CONTRACTS.txt."""
import hashlib
import json
from pathlib import Path


def cache_key(data_path: Path, config: dict) -> str:
    return hashlib.sha256(str(data_path).encode()).hexdigest()


def validate_release(manifest: dict, report: dict) -> None:
    if not report.get('passed'):
        raise ValueError('Кандидат отклонён')


def publish(manifest: dict, report: dict, target: Path) -> None:
    validate_release(manifest, report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
