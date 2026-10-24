"""Строгая оценка меток и отказ при неполном или несовместимом отчёте."""
import json
import math
from collections import defaultdict


def _ids(items):
    if not isinstance(items, list) or not items:
        raise ValueError('Нужен непустой список')
    ids = [r.get('id') for r in items if isinstance(r, dict)]
    if len(ids) != len(items) or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Нужны уникальные строковые id')
    return ids


def _counts(items):
    n = len(items)
    correct = sum(i['correct'] for i in items)
    valid = sum(i['valid'] for i in items)
    return dict(n=n, correct=correct, valid=valid, errors=sum(i['error'] is not None for i in items),
                accuracy=correct / n, format_rate=valid / n)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Повтор JSON-поля')
        result[key] = value
    return result


def score(rows, outputs):
    """Оценить все вопросы, сопоставляя ответы по id, включая ошибки генерации."""
    ids, output_ids = _ids(rows), _ids(outputs)
    if set(ids) != set(output_ids):
        raise ValueError('Наборы id заданий и ответов различаются')
    by_id = {o['id']: o for o in outputs}
    per_item, groups = [], defaultdict(list)
    for row in rows:
        choices, slices = row.get('choices'), row.get('slices')
        if (not isinstance(choices, list) or not choices or
                any(not isinstance(c, str) for c in choices) or row.get('expected') not in choices or
                not isinstance(slices, list) or not slices or
                any(not isinstance(s, str) or not s for s in slices) or len(slices) != len(set(slices))):
            raise ValueError('Некорректные choices, expected или slices')
        out = by_id[row['id']]
        if 'raw' not in out or 'error' not in out or not isinstance(out['raw'], str):
            raise ValueError('Ответ требует строковый raw и error')
        raw, error, parsed, valid = out['raw'], out['error'], None, False
        if error is not None and (not isinstance(error, str) or not error):
            raise ValueError('error должен быть непустой строкой или None')
        try:
            obj = json.loads(raw, object_pairs_hook=_unique_object)
            if (isinstance(obj, dict) and set(obj) == {'answer'} and
                    isinstance(obj['answer'], str) and obj['answer'] in choices):
                parsed = obj['answer']
                valid = error is None
        except (ValueError, TypeError, RecursionError):
            pass
        item = dict(id=row['id'], raw=raw, parsed=parsed, valid=valid,
                    correct=valid and parsed == row['expected'], error=error, slices=list(slices))
        per_item.append(item)
        for name in slices:
            groups[name].append(item)
    return dict(overall=_counts(per_item), slices={s: _counts(g) for s, g in sorted(groups.items())},
                per_item=per_item)


def _validate_metrics(actual, expected):
    if not isinstance(actual, dict):
        raise ValueError('Нет метрик')
    for key, value in expected.items():
        given = actual.get(key)
        if key in ('n', 'correct', 'valid', 'errors'):
            if type(given) is not int or given != value:
                raise ValueError(f'Некорректный count: {key}')
        elif (type(given) not in (int, float) or not math.isfinite(given) or
              not math.isclose(given, value, rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f'Метрика не соответствует counts: {key}')


def _validate_report(report, minimum, min_slices):
    if not isinstance(report, dict):
        raise ValueError('Нет отчёта')
    items = report.get('per_item')
    _ids(items)
    if len(items) < minimum:
        raise ValueError(f'Недостаточно примеров: минимум {minimum}')
    groups = defaultdict(list)
    for item in items:
        if (not {'id', 'raw', 'parsed', 'valid', 'correct', 'error', 'slices'} <= item.keys() or
                not isinstance(item['raw'], str) or type(item['valid']) is not bool or
                type(item['correct']) is not bool or (item['correct'] and not item['valid']) or
                (item['valid'] and not isinstance(item['parsed'], str)) or
                (item['error'] is not None and (not isinstance(item['error'], str) or not item['error'])) or
                (item['error'] is not None and (item['valid'] or item['correct']))):
            raise ValueError('Некорректная строка per_item')
        slices = item['slices']
        if (not isinstance(slices, list) or not slices or
                any(not isinstance(s, str) or not s for s in slices) or len(slices) != len(set(slices))):
            raise ValueError('Некорректные срезы per_item')
        for s in slices:
            groups[s].append(item)
    if len(groups) < min_slices or not isinstance(report.get('slices'), dict) or set(report['slices']) != set(groups):
        raise ValueError('Отсутствуют срезы отчёта')
    _validate_metrics(report.get('overall'), _counts(items))
    for s, group in groups.items():
        _validate_metrics(report['slices'][s], _counts(group))
    if report['overall']['errors']:
        raise ValueError('Есть ошибки генерации')
    return {i['id']: set(i['slices']) for i in items}


def quality_gate(baseline, candidate, critical_slices):
    """Нулевой допуск регрессии; counts сверяются с каждой строкой отчёта."""
    try:
        for key in ('protocol_sha256', 'golden_sha256', 'baseline_spec_sha256'):
            if not isinstance(baseline.get(key), str) or not baseline[key] or candidate.get(key) != baseline[key]:
                raise ValueError(f'Несовместимый {key}')
        if (not isinstance(critical_slices, list) or not critical_slices or
                any(not isinstance(s, str) or not s for s in critical_slices)):
            raise ValueError('Не заданы критические срезы')
        identities = []
        behaviors = []
        for report in (baseline, candidate):
            if report.get('status') != 'complete' or not isinstance(report.get('model'), dict) or not report['model']:
                raise ValueError('Неполный отчёт или отсутствует модель')
            model = report['model']
            if model.get('kind') == 'base':
                if not model.get('name') or not isinstance(model.get('revision'), str) or len(model['revision']) != 40:
                    raise ValueError('Не закреплена ревизия базы')
            elif model.get('kind') == 'adapter':
                if not all(model.get(k) for k in ('name', 'version', 'run_id', 'uri')):
                    raise ValueError('Не закреплена версия адаптера')
            elif model != {'kind': 'fixture'}:
                raise ValueError('Неизвестная модель')
            identities.append(_validate_report(report, 50, 3))
            behaviors.append(_validate_report(report.get('behavioral'), 3, 1))
        if identities[0] != identities[1] or behaviors[0] != behaviors[1]:
            raise ValueError('Различаются id или состав срезов')
        if not set(critical_slices) <= set(baseline['slices']):
            raise ValueError('Нет критического среза')
        for metric in ('accuracy', 'format_rate'):
            if candidate['overall'][metric] < baseline['overall'][metric]:
                raise ValueError(f'Регрессия общей {metric}')
        for s in critical_slices:
            if candidate['slices'][s]['accuracy'] < baseline['slices'][s]['accuracy']:
                raise ValueError(f'Регрессия критического среза {s}')
        behavior = candidate['behavioral']['overall']
        if behavior['correct'] != behavior['n']:
            raise ValueError('Не пройдены обязательные behavioral-кейсы')
        return dict(passed=True, critical_slices=list(critical_slices),
                    protocol_sha256=candidate['protocol_sha256'], golden_sha256=candidate['golden_sha256'],
                    baseline_spec_sha256=candidate['baseline_spec_sha256'])
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('Некорректный или неполный отчёт') from exc
