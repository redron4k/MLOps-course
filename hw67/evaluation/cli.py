"""Готовая обвязка: студент реализует score и quality_gate в solution.py."""
import argparse
import copy
import gc
import json
import os
import time
from pathlib import Path

from evaluation.contract import fingerprint, read_json, read_rows, validate_golden, write_json
from evaluation.solution import quality_gate, score

BASELINE = Path('evaluation/baseline.json')
CANDIDATE = Path('metrics/candidate-model.json')


def tracking():
    from mlflow import MlflowClient
    from src.common import connect, load_params
    p = load_params()
    connect(p)
    return p, MlflowClient()


def manifest_for(spec):
    import yaml
    _, client = tracking()
    version = client.get_model_version(spec['name'], str(spec['version']))
    if version.run_id != spec['run_id']:
        raise ValueError('Версия реестра не соответствует run id')
    root = Path(client.download_artifacts(spec['run_id'], 'model'))
    info = yaml.safe_load((root / 'MLmodel').read_text())
    adapter = (root / info['flavors']['python_function']['artifacts']['adapter']['path']).resolve()
    if not adapter.is_relative_to(root.resolve()):
        raise ValueError('Некорректный путь адаптера')
    return read_json(adapter / 'manifest.json'), adapter


def candidate_model():
    from src.promote import select
    p, client = tracking()
    spec = read_json(CANDIDATE)
    best = select(client, p, read_json('metrics/sweep.json'))
    if best.info.run_id != spec['run_id']:
        raise ValueError('Выбран другой/устаревший кандидат; выполните make candidate')
    return spec


def prepare_candidate():
    from src.promote import promote, select
    p, client = tracking()
    best = select(client, p, read_json('metrics/sweep.json'))
    version = promote(client, best, p['registry']['name'], 'candidate', p['registry']['max_val_ratio'])
    spec = dict(kind='adapter', name=version.name, version=str(version.version), run_id=best.info.run_id,
                uri=f'models:/{version.name}/{version.version}')
    write_json(CANDIDATE, spec)
    print(json.dumps(spec, ensure_ascii=False, indent=2))


def init_baseline():
    if BASELINE.exists():
        raise ValueError('Baseline уже закреплён. Обновление только make baseline-update после принятия')
    manifest, _ = manifest_for(candidate_model())
    write_json(BASELINE, dict(kind='base', name=manifest['model']['name'],
                             revision=manifest['model']['revision']))
    print('Baseline: исходная модель с точной ревизией. Сохраните evaluation/baseline.json в Git')


def inputs():
    config = read_json('evaluation/params.json')
    rows = read_rows(config['golden'])
    training = read_rows(config['training_index'])
    behavioral = read_rows(config['behavioral'])
    proof = validate_golden(rows, training, behavioral)
    if not set(config['critical_slices']) <= {s for r in rows for s in r['slices']}:
        raise ValueError('Критический срез отсутствует')
    baseline = read_json(BASELINE)
    protocol = dict(config=config, golden=rows, behavioral=behavioral, training_index=training,
                    evaluation_code={p.name: p.read_text() for p in sorted(Path('evaluation').glob('*.py'))})
    return config, rows, behavioral, baseline, proof, fingerprint(protocol)


def generate(spec, common_manifest, config, rows):
    """Одинаковый eval prompt и decoding. База и адаптер загружаются последовательно."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.runtime import resolve_device, resolve_dtype, set_seed
    manifest = common_manifest
    adapter = None
    if spec['kind'] == 'adapter':
        manifest, adapter = manifest_for(spec)
    model_cfg = manifest['model']
    base_name = spec['name'] if spec['kind'] == 'base' else model_cfg['name']
    revision = spec['revision'] if spec['kind'] == 'base' else model_cfg['revision']
    if len(revision) != 40:
        raise ValueError('Нужна точная commit-ревизия базы')
    device = resolve_device(os.environ.get('HW6_DEVICE', model_cfg['device']))
    dtype = resolve_dtype(os.environ.get('HW6_DTYPE', model_cfg['dtype']))
    set_seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(str(adapter) if adapter else base_name,
                                              **({'local_files_only': True} if adapter else {'revision': revision}))
    model = AutoModelForCausalLM.from_pretrained(base_name, revision=revision, dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model = model.to(device).eval()
    outputs = []
    for row in rows:
        started = time.monotonic()
        raw, error = '', None
        try:
            text = tokenizer.apply_chat_template(
                [{'role': 'system', 'content': config['system']}, {'role': 'user', 'content': row['input']}],
                tokenize=False, add_generation_prompt=True, enable_thinking=config['enable_thinking'])
            batch = tokenizer(text, return_tensors='pt').to(device)
            with torch.inference_mode():
                out = model.generate(**batch, max_new_tokens=config['max_new_tokens'], do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            raw = tokenizer.decode(out[0, batch['input_ids'].shape[1]:], skip_special_tokens=True)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        outputs.append(dict(id=row['id'], raw=raw, error=error,
                            latency_seconds=time.monotonic() - started))
    del model, tokenizer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    return outputs


def evaluate():
    import mlflow
    config, rows, behavioral, baseline, proof, protocol_hash = inputs()
    candidate = candidate_model()
    manifest, _ = manifest_for(candidate)
    reports = {}
    for role, spec in [('baseline', baseline), ('candidate', candidate)]:
        outputs = generate(spec, manifest, config, rows + behavioral)
        by_id = {o['id']: o for o in outputs}
        write_json(f'metrics/{role}-outputs.json', outputs)
        report = score(rows, [by_id[r['id']] for r in rows])
        report.update(status='complete', model=spec, protocol_sha256=protocol_hash,
                      golden_sha256=fingerprint(rows), baseline_spec_sha256=fingerprint(baseline),
                      behavioral=score(behavioral, [by_id[r['id']] for r in behavioral]), leakage=proof,
                      outputs_sha256=fingerprint(outputs))
        reports[role] = report
    a = {r['id']: r for r in reports['baseline']['per_item']}
    b = {r['id']: r for r in reports['candidate']['per_item']}
    diff = dict(improved=[i for i in a if not a[i]['correct'] and b[i]['correct']],
                regressed=[i for i in a if a[i]['correct'] and not b[i]['correct']])
    mlflow.set_experiment('hw67-evaluation')
    with mlflow.start_run(run_name='baseline-vs-candidate') as run:
        for role, report in reports.items():
            report['evaluation_run_id'] = run.info.run_id
            write_json(f'metrics/{role}-eval.json', report)
            mlflow.log_metrics({f'{role}.{key}': report['overall'][key] for key in ('accuracy', 'format_rate')})
        mlflow.set_tags(dict(kind='evaluation', candidate_run_id=candidate['run_id'],
                             protocol_sha256=protocol_hash, golden_sha256=fingerprint(rows)))
        write_json('metrics/eval-diff.json', diff)
        for path in sorted(Path('metrics').glob('*-eval.json')):
            mlflow.log_artifact(str(path), 'evaluation')
        for path in sorted(Path('metrics').glob('*-outputs.json')):
            mlflow.log_artifact(str(path), 'evaluation')
        mlflow.log_artifact('metrics/eval-diff.json', 'evaluation')
        mlflow.log_artifact('evaluation/params.json', 'evaluation')
        mlflow.log_artifact(str(BASELINE), 'evaluation')
        mlflow.log_dict(proof, 'evaluation/leakage.json')
    print('Отчёты и сырые ответы сохранены. Далее: make eval; только после PASS - make promote')


def current_reports():
    _, rows, behavioral, baseline_spec, _, protocol_hash = inputs()
    baseline, candidate = [read_json(f'metrics/{role}-eval.json') for role in ('baseline', 'candidate')]
    candidate_spec = candidate_model()
    if candidate['model'] != candidate_spec or baseline['model'] != baseline_spec:
        raise ValueError('В отчёте другая модель')
    if any(r['protocol_sha256'] != protocol_hash or r['golden_sha256'] != fingerprint(rows) or
           r['baseline_spec_sha256'] != fingerprint(baseline_spec) for r in (baseline, candidate)):
        raise ValueError('Изменились данные/протокол/код оценки; повторите make evaluate')
    for role, report in [('baseline', baseline), ('candidate', candidate)]:
        outputs = read_json(f'metrics/{role}-outputs.json')
        if fingerprint(outputs) != report['outputs_sha256']:
            raise ValueError('Сырые ответы изменены после оценки')
        by_id = {o['id']: o for o in outputs}
        if len(by_id) != len(outputs) or set(by_id) != {r['id'] for r in rows + behavioral}:
            raise ValueError('Изменились id сырых ответов')
        recalculated = score(rows, [by_id[r['id']] for r in rows])
        if any(report[k] != recalculated[k] for k in ('overall', 'slices', 'per_item')):
            raise ValueError('Метрики не соответствуют сырым ответам')
        if report['behavioral'] != score(behavioral, [by_id[r['id']] for r in behavioral]):
            raise ValueError('Behavioral-отчёт не соответствует ответам')
    return baseline, candidate


def gate():
    baseline, candidate = current_reports()
    result = quality_gate(baseline, candidate, read_json('evaluation/params.json')['critical_slices'])
    if not isinstance(result, dict) or result.get('passed') is not True:
        raise ValueError('Гейт не подтвердил passed=True')
    write_json('metrics/quality-gate.json', result)
    return candidate


def release():
    candidate = gate()
    spec = candidate['model']
    _, client = tracking()
    # Смена champion выполняется только после успешного gate, candidate остаётся отдельным alias.
    client.set_registered_model_alias(spec['name'], 'champion', spec['version'])
    write_json('metrics/released.json', dict(model=spec, evaluation_run_id=candidate['evaluation_run_id'],
                                           protocol_sha256=candidate['protocol_sha256']))
    print('champion -> ' + spec['uri'])


def update_baseline():
    candidate = gate()
    spec = candidate['model']
    _, client = tracking()
    champion = client.get_model_version_by_alias(spec['name'], 'champion')
    if str(champion.version) != spec['version'] or champion.run_id != spec['run_id']:
        raise ValueError('Сначала примите эту версию: make promote')
    old = read_json(BASELINE)
    write_json(f'evaluation/baseline-history/{fingerprint(old)}.json', old)
    write_json(BASELINE, spec)
    print('Baseline обновлён явно; старый сохранён. Следующая оценка сравнит с принятой версией')


def negative():
    baseline, candidate = current_reports()
    rows = read_rows(read_json('evaluation/params.json')['golden'])
    broken_outputs = [dict(id=r['id'], raw='INVALID', error=None) for r in rows]
    broken = copy.deepcopy(candidate)
    broken.update(score(rows, broken_outputs))
    broken['behavioral']['overall']['correct'] = 0
    broken['behavioral']['overall']['accuracy'] = 0.0
    try:
        quality_gate(baseline, broken, read_json('evaluation/params.json')['critical_slices'])
    except ValueError as exc:
        write_json('metrics/negative-gate.json', dict(rejected=True, reason=str(exc),
                                                   note='Испорчены сохранённые ответы; реальные веса и alias не менялись'))
        print('PASS: испорченные ответы отклонены, alias не менялся')
        return
    raise ValueError('FAIL: испорченные ответы прошли gate')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['candidate', 'baseline-init', 'evaluate', 'gate', 'promote', 'baseline-update', 'negative'])
    args = ap.parse_args()
    commands = {'candidate': prepare_candidate, 'baseline-init': init_baseline, 'evaluate': evaluate,
                'gate': gate, 'promote': release, 'baseline-update': update_baseline, 'negative': negative}
    try:
        commands[args.command]()
    except (ValueError, KeyError, FileNotFoundError, NotImplementedError) as exc:
        if args.command in ('gate', 'promote'):
            write_json('metrics/quality-gate.json', dict(passed=False, reason=str(exc)))
        raise SystemExit(f'Отказ: {exc}') from exc


if __name__ == '__main__':
    main()
