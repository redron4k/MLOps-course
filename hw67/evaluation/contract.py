"""Контракт файлов оценки и проверка утечки; работает без GPU и MLflow."""
import hashlib
import json
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [row['id'] for row in rows]
    if not rows or len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError('Нужны непустые строки с уникальными строковыми id')
    return rows


def validate_golden(rows, training, behavioral):
    if len(rows) < 50:
        raise ValueError('Golden set: минимум 50 примеров')
    for row in rows + behavioral:
        if not isinstance(row.get('input'), str) or not row['input'].strip():
            raise ValueError('Нет input')
        if not isinstance(row.get('group_id'), str) or not row['group_id']:
            raise ValueError('Нет group_id')
        if not isinstance(row.get('slices'), list) or not row['slices'] or any(not isinstance(s, str) or not s for s in row['slices']):
            raise ValueError('Нужны именованные slices')
        if not isinstance(row.get('choices'), list) or not row['choices'] or any(not isinstance(s, str) for s in row['choices']):
            raise ValueError('Нужны допустимые строковые метки choices')
        if row.get('expected') not in row['choices'] or not row.get('source'):
            raise ValueError('Нужны проверенный expected и source')
    normalize = lambda s: ' '.join(s.casefold().split())
    golden_inputs = [normalize(row['input']) for row in rows]
    if len(golden_inputs) != len(set(golden_inputs)):
        raise ValueError('В golden есть точные дубли вопросов с разными id')
    train_inputs = {normalize(row['input']) for row in training}
    train_groups = {row['group_id'] for row in training}
    leaks = [row['id'] for row in rows if normalize(row['input']) in train_inputs or row['group_id'] in train_groups]
    if leaks:
        raise ValueError(f'Пересечение train/golden: {leaks[:5]}')
    slices = {s for row in rows for s in row['slices']}
    if len(slices) < 3:
        raise ValueError('Нужны минимум три среза golden set')
    kinds = {row.get('type') for row in behavioral}
    if len(behavioral) < 3 or not {'format', 'paraphrase', 'refusal'} <= kinds:
        raise ValueError('Behavioral: format, paraphrase и refusal, минимум три кейса')
    if {r['id'] for r in rows} & {r['id'] for r in behavioral}:
        raise ValueError('ID golden и behavioral должны различаться')
    return {'exact_or_group_intersections': 0, 'n_train_index': len(training),
            'n_golden': len(rows), 'n_behavioral': len(behavioral)}
