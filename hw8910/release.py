"""Strict release contracts and recoverable, single-writer publication."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def cache_key(data_path: Path, config: dict) -> str:
    return digest(canonical({'data_sha256': digest(Path(data_path).read_bytes()), 'config': config}))


def validate_release(manifest: dict, report: dict) -> None:
    try:
        if type(manifest['schema_version']) is not int or manifest['schema_version'] != 1:
            raise ValueError('Unsupported schema_version')
        for key in ('model_id', 'artifact_uri'):
            if not isinstance(manifest[key], str) or not manifest[key].strip():
                raise ValueError('Missing ' + key)
        for key in ('model_sha256', 'protocol_sha256', 'golden_sha256', 'runtime_sha256'):
            if not isinstance(manifest[key], str) or not re.fullmatch('[0-9a-f]{64}', manifest[key]):
                raise ValueError('Invalid ' + key)
            if report[key] != manifest[key]:
                raise ValueError('Report mismatch: ' + key)
        if not isinstance(manifest['runtime'], dict) or not manifest['runtime']:
            raise ValueError('Missing runtime')
        if digest(canonical(manifest['runtime'])) != manifest['runtime_sha256']:
            raise ValueError('Runtime fingerprint mismatch')
        if report['model_id'] != manifest['model_id']:
            raise ValueError('Report model mismatch')
        if report['status'] != 'complete' or report['passed'] is not True:
            raise ValueError('Incomplete or rejected evaluation')
        if not isinstance(report['evaluation_run_id'], str) or not report['evaluation_run_id'].strip():
            raise ValueError('Missing evaluation_run_id')
        for key in ('accuracy', 'format_rate'):
            value = report['metrics'][key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid metric: ' + key)
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('Malformed release metadata') from exc


def atomic_bytes(target, payload):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.'+target.name+'.',
                                         suffix='.tmp', delete=False) as stream:
            name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def publish(manifest: dict, report: dict, target: Path) -> None:
    validate_release(manifest, report)
    atomic_bytes(target, canonical(manifest) + b'\n')


def verify_artifact(manifest, workspace):
    """Check real files before publication and before loading weights."""
    files = manifest.get('file_hashes')
    if not isinstance(files, dict) or not files:
        raise ValueError('Missing pinned artifact file list')
    artifact = Path(manifest['artifact_uri'])
    if not artifact.is_absolute():
        artifact = Path(workspace) / artifact
    artifact = artifact.resolve()
    base = artifact if artifact.is_dir() else artifact.parent
    actual = {}
    for relative, expected in files.items():
        path = (base / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(base):
            # HF snapshots contain symlinks into a content-addressed blobs directory.
            # Check lexical traversal; trusted snapshot symlinks are pinned by bytes.
            if Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise ValueError('Unsafe artifact path')
        actual[relative] = digest(path.read_bytes())
        if actual[relative] != expected:
            raise ValueError('Artifact changed: ' + relative)
    if digest(canonical(actual)) != manifest['model_sha256']:
        raise ValueError('Model fingerprint mismatch')
    return artifact


def publish_release(manifest, report, workspace, fail_after_pointer=False):
    """Durable intent -> atomic pointer -> deduplicated atomic journal.

    A retry completes a pending transaction before starting another. One writer;
    process-crash recovery, not a distributed transaction or power-loss guarantee.
    """
    root = Path(workspace)
    validate_release(manifest, report)
    verify_artifact(manifest, root)
    from gate import verify_evaluation
    verify_evaluation(manifest, report)
    intent = root / 'publication-pending.json'
    journal = root / 'publications.jsonl'

    def finish(transaction, inject=False):
        m, r = transaction['manifest'], transaction['report']
        verify_artifact(m, root)
        verify_evaluation(m, r)
        publish(m, r, root / 'accepted.json')
        if inject:
            raise RuntimeError('Injected failure after pointer, before publication journal')
        events = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
        event = transaction['event']
        if not any(item['release_id'] == event['release_id'] for item in events):
            events.append(event)
        atomic_bytes(journal, b''.join(canonical(item)+b'\n' for item in events))
        intent.unlink()

    if intent.exists():
        finish(json.loads(intent.read_text()))
        print('RECOVERED_PUBLICATION')
    identity = digest(canonical({'manifest': manifest, 'report': report}))
    transaction = dict(manifest=manifest, report=report,
                       event=dict(release_id=identity, model_id=manifest['model_id'],
                                  runtime_sha256=manifest['runtime_sha256'],
                                  evaluation_run_id=report['evaluation_run_id']))
    atomic_bytes(intent, canonical(transaction)+b'\n')
    finish(transaction, fail_after_pointer)
    return identity
