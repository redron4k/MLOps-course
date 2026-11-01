"""Reuse HW67 training; evaluate with the exact backend/runtime served by the API.

No sweep, no val-loss shortcut, no change to HW67 files or registry aliases.
"""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
os.environ['NO_PROXY'] = ','.join(filter(None, [os.environ.get('NO_PROXY',''), '127.0.0.1', 'localhost']))

from backend import TransformersBackend, hashes, tokenizer_fingerprint, template_fingerprint
from release import atomic_bytes, canonical, digest, publish_release

HW67 = Path(__file__).resolve().parents[1]/'hw67'
sys.path.insert(0, str(HW67))
from evaluation.solution import quality_gate, score
from evaluation.contract import read_rows, validate_golden


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    atomic_bytes(path, canonical(value)+b'\n')


def manifest_for(spec, artifact, base, config, protocol, golden, device):
    names = sorted(str(p.relative_to(artifact)) for p in artifact.rglob('*')
                   if p.is_file() and p.suffix in ('.json', '.safetensors', '.jinja', '.txt'))
    file_hashes = hashes(artifact, names)
    base_files = hashes(base, sorted(p.name for p in base.iterdir()
                                   if p.is_file() and p.suffix in ('.json', '.safetensors', '.jinja')))
    runtime = dict(base_revision=read(HW67/'evaluation/baseline.json')['revision'],
        tokenizer_sha256=tokenizer_fingerprint(artifact), template_sha256=template_fingerprint(artifact),
        generation=dict(system=config['system'], do_sample=False, seed=config['seed'],
                        max_new_tokens=config['max_new_tokens'], enable_thinking=config['enable_thinking']),
        dtype='bfloat16', device=device, max_input_tokens=512,
        backend_sha256=digest((Path(__file__).parent/'backend.py').read_bytes()),
        packages={p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft')})
    model_id = spec.get('run_id', spec['name']+'@'+spec.get('revision', ''))
    return dict(schema_version=1, kind=spec['kind'], model_id=model_id,
                artifact_uri=str(artifact), file_hashes=file_hashes,
                model_sha256=digest(canonical(file_hashes)), runtime=runtime,
                runtime_sha256=digest(canonical(runtime)), protocol_sha256=protocol,
                golden_sha256=golden, base_artifact_uri=str(base), base_file_hashes=base_files,
                model_spec=spec)


def release_report(manifest, baseline, candidate, critical, sources, role):
    selected = candidate if role=='candidate' else baseline
    try:
        decision = quality_gate(baseline, candidate, critical)
        reason = None
    except ValueError as exc:
        decision, reason = {'passed': False}, str(exc)
    result = {k: manifest[k] for k in ('model_id', 'model_sha256', 'protocol_sha256',
                                     'golden_sha256', 'runtime_sha256')}
    result.update(status=selected['status'], passed=decision['passed'] if role=='candidate' else False,
                  evaluation_run_id=selected['evaluation_run_id'], metrics=selected['overall'],
                  reason=reason if role=='candidate' else 'Retained pre-existing baseline; not a candidate promotion',
                  gate=decision, sources=sources, synthetic=False)
    return result


def run(root, tracking_uri, device):
    import mlflow
    from huggingface_hub import snapshot_download
    import yaml
    import torch
    root.mkdir(parents=True, exist_ok=True)
    evidence = root/'evaluation'
    evidence.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    candidate = read(HW67/'metrics/candidate-model.json')
    baseline = read(HW67/'evaluation/baseline.json')
    training = client.get_run(candidate['run_id'])
    if training.info.status != 'FINISHED':
        raise ValueError('Training run incomplete')
    package = Path(client.download_artifacts(candidate['run_id'], 'model', str(root))).resolve()
    info = yaml.safe_load((package/'MLmodel').read_text())
    adapter = (package/info['flavors']['python_function']['artifacts']['adapter']['path']).resolve()
    if not adapter.is_relative_to(package):
        raise ValueError('Unsafe MLflow adapter path')
    training_manifest = read(adapter/'manifest.json')
    if training_manifest['model']['revision'] != baseline['revision']:
        raise ValueError('Training and baseline revisions differ')
    base = Path(snapshot_download(baseline['name'], revision=baseline['revision'], local_files_only=True))
    config = read(HW67/'evaluation/params.json')
    rows, behaviors, training_rows = [read_rows(HW67/config[k]) for k in ('golden','behavioral','training_index')]
    leakage = validate_golden(rows, training_rows, behaviors)
    # New runtime metadata requires new evaluation, not relabeling old reports.
    protocol_document = dict(config=config, golden=rows, behavioral=behaviors,
        training_index=training_rows, backend_sha256=digest(Path('backend.py').read_bytes()),
        scoring_sha256=digest((HW67/'evaluation/solution.py').read_bytes()))
    protocol = digest(canonical(protocol_document))
    save(evidence/'protocol.json', protocol_document)
    manifests = {role: manifest_for(spec, artifact, base, config, protocol,
                                   digest(canonical(rows)), device)
                 for role,spec,artifact in [('baseline',baseline,base),('candidate',candidate,adapter)]}
    mlflow.set_experiment('hw8910-release-evaluation')
    reports = {}
    with mlflow.start_run(run_name='existing-training-exact-serving-runtime') as evaluation:
        for role in ('baseline','candidate'):
            manifest = manifests[role]
            save(root/f'manifest-{role}.json', manifest)
            backend = TransformersBackend(manifest, root)
            outputs = []
            for row in rows+behaviors:
                started = time.perf_counter()
                raw, error = '', None
                try:
                    raw = backend.generate(row['input'], config['max_new_tokens'])['answer']
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                outputs.append(dict(id=row['id'],raw=raw,error=error,
                                    latency_seconds=time.perf_counter()-started))
                print(role, row['id'], raw, error, flush=True)
            save(evidence/f'{role}-outputs.json', outputs)
            by_id = {out['id']:out for out in outputs}
            report = score(rows,[by_id[row['id']] for row in rows])
            report.update(status='complete', model=manifest['model_spec'], protocol_sha256=protocol,
                golden_sha256=manifest['golden_sha256'], baseline_spec_sha256=digest(canonical(baseline)),
                behavioral=score(behaviors,[by_id[row['id']] for row in behaviors]), leakage=leakage,
                evaluation_run_id=evaluation.info.run_id, runtime_sha256=manifest['runtime_sha256'],
                model_sha256=manifest['model_sha256'], outputs_sha256=digest(canonical(outputs)))
            save(evidence/f'{role}-eval.json',report)
            reports[role] = report
            mlflow.log_metrics({f'{role}.{key}':report['overall'][key] for key in ('accuracy','format_rate')})
            del backend
            gc.collect()
            if device=='mps': torch.mps.empty_cache()
        sources = {path.name: dict(uri=str(path.resolve()),sha256=digest(path.read_bytes()))
                   for path in sorted(evidence.glob('*.json'))}
        for role in ('baseline','candidate'):
            report = release_report(manifests[role], reports['baseline'], reports['candidate'],
                                    config['critical_slices'],sources,role)
            save(root/f'report-{role}.json',report)
        mlflow.set_tags(dict(training_run_id=candidate['run_id'], prior_evaluation_run_id=
            read(HW67/'metrics/candidate-eval.json')['evaluation_run_id'], protocol_sha256=protocol))
        mlflow.log_artifacts(str(evidence),'evaluation')
        mlflow.log_artifact(str(root/'report-candidate.json'))
    # Bootstrap retention of original base is explicit; it is never represented as PASS.
    target = root/'accepted.json'
    if not target.exists():
        save(target,manifests['baseline'])
    before = digest(target.read_bytes())
    candidate_report = read(root/'report-candidate.json')
    rejected = False
    try:
        publish_release(manifests['candidate'],candidate_report,root)
    except ValueError as exc:
        rejected = True
        print('CANDIDATE_REJECTED',str(exc),flush=True)
    save(root/'release-decision.json',dict(candidate_passed=candidate_report['passed'],
        rejected=rejected,accepted_before_sha256=before,accepted_after_sha256=digest(target.read_bytes()),
        served_model_id=read(target)['model_id'],training_run_id=candidate['run_id'],
        prior_evaluation_run_id=read(HW67/'metrics/candidate-eval.json')['evaluation_run_id'],
        evaluation_run_id=reports['candidate']['evaluation_run_id']))
    # Input-validation preparation is a real, content-addressed stage (not training).
    atomic_bytes(root/'input.txt',(HW67/config['golden']).read_bytes())
    save(root/'config.json',dict(tokenizer_revision=baseline['revision'],
        template_sha256=manifests['candidate']['runtime']['template_sha256'],max_length=512,
        code_revision=digest(Path('lab.py').read_bytes()),format='jsonl'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,default=Path('.real'))
    parser.add_argument('--tracking-uri',default='http://127.0.0.1:5006')
    parser.add_argument('--device',default='mps',choices=['cpu','mps','cuda'])
    args=parser.parse_args()
    run(args.workspace.resolve(),args.tracking_uri,args.device)
