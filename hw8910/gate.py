"""Recompute HW67's full decision from hash-bound reports and raw responses."""
import json
from pathlib import Path
import sys
from release import canonical, digest

HW67 = Path(__file__).resolve().parents[1]/'hw67'
sys.path.insert(0, str(HW67))
from evaluation.solution import quality_gate, score


def verify_evaluation(manifest, report):
    if manifest.get('kind') not in ('base', 'adapter'):
        if report.get('synthetic') is not True or not manifest['model_id'].startswith('fixture-'):
            raise ValueError('Unrecognized synthetic fixture')
        return
    if report.get('synthetic') is not False:
        raise ValueError('Synthetic evaluation cannot authorize real weights')
    sources = report.get('sources', {})
    required = {'protocol.json', 'baseline-eval.json', 'candidate-eval.json',
                'baseline-outputs.json', 'candidate-outputs.json'}
    if not required <= sources.keys():
        raise ValueError('Missing complete source evidence')
    documents = {}
    for name in required:
        source = sources[name]
        payload = Path(source['uri']).read_bytes()
        if digest(payload) != source['sha256']:
            raise ValueError('Evaluation source changed: ' + name)
        documents[name] = json.loads(payload)
    protocol = documents['protocol.json']
    if digest(canonical(protocol)) != manifest['protocol_sha256']:
        raise ValueError('Evaluation protocol mismatch')
    if digest(canonical(protocol['golden'])) != manifest['golden_sha256']:
        raise ValueError('Golden set mismatch')
    if digest((HW67/'evaluation/solution.py').read_bytes()) != protocol['scoring_sha256']:
        raise ValueError('Scoring implementation changed; reevaluate')
    if digest((Path(__file__).parent/'backend.py').read_bytes()) != protocol['backend_sha256']:
        raise ValueError('Serving implementation changed; reevaluate')
    for role in ('baseline', 'candidate'):
        full = documents[f'{role}-eval.json']
        outputs = documents[f'{role}-outputs.json']
        if full['outputs_sha256'] != digest(canonical(outputs)):
            raise ValueError('Raw outputs fingerprint mismatch')
        by_id = {o['id']:o for o in outputs}
        rows, behaviors = protocol['golden'], protocol['behavioral']
        if len(by_id) != len(outputs) or set(by_id) != {r['id'] for r in rows+behaviors}:
            raise ValueError('Raw output IDs mismatch')
        metrics = score(rows,[by_id[r['id']] for r in rows])
        if any(metrics[k] != full[k] for k in ('overall','slices','per_item')):
            raise ValueError('Metrics differ from raw outputs')
        if score(behaviors,[by_id[r['id']] for r in behaviors]) != full['behavioral']:
            raise ValueError('Behavioral report differs from raw outputs')
    baseline, candidate = documents['baseline-eval.json'], documents['candidate-eval.json']
    decision = quality_gate(baseline,candidate,protocol['config']['critical_slices'])
    if report['passed'] is not True or decision['passed'] is not True:
        raise ValueError('Full quality gate did not pass')
    for key in ('model_sha256','runtime_sha256','protocol_sha256','golden_sha256'):
        if candidate[key] != manifest[key]:
            raise ValueError('Candidate binding mismatch: ' + key)
    if (candidate['model'] != manifest['model_spec'] or report['metrics'] != candidate['overall']
            or candidate['model'].get('kind') != manifest['kind']):
        raise ValueError('Candidate identity or metrics mismatch')
    if report['evaluation_run_id'] != candidate['evaluation_run_id']:
        raise ValueError('Evaluation run mismatch')

