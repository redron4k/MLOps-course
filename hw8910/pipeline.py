"""Prefect graph; only small descriptors cross task boundaries, never tensors."""
import os
from pathlib import Path
os.environ.setdefault('PREFECT_SERVER_ANALYTICS_ENABLED', 'false')
os.environ.setdefault('PREFECT_HOME', str(Path(__file__).parent/'.prefect'))
os.environ['NO_PROXY'] = ','.join(filter(None, [os.environ.get('NO_PROXY',''), '127.0.0.1', 'localhost']))

from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from lab import prepare, read
from release import cache_key, canonical, digest, publish_release, validate_release, verify_artifact


def preparation_key(context, parameters):
    root = Path(parameters['root'])
    # Namespace by workspace so descriptors never refer to another experiment.
    return digest(canonical({'workspace': str(root.resolve()),
                             'input': cache_key(root/'input.txt', read(root/'config.json'))}))


@task(persist_result=True, cache_key_fn=preparation_key, result_serializer='json', log_prints=True)
def prepare_task(root):
    return prepare(Path(root))


@task(cache_policy=NO_CACHE)
def artifact_task(root, manifest_name, prepared):
    manifest = read(Path(root)/manifest_name)
    verify_artifact(manifest, root)
    return {'manifest': manifest, 'prepared_sha256': prepared['input_sha256']}


@task(cache_policy=NO_CACHE)
def gate_task(root, report_name, artifact):
    report = read(Path(root)/report_name)
    validate_release(artifact['manifest'], report)
    return report


@task(cache_policy=NO_CACHE, log_prints=True)
def publication_task(root, artifact, report, fail_after_pointer):
    identity = publish_release(artifact['manifest'], report, Path(root), fail_after_pointer)
    print('PUBLISHED', artifact['manifest']['model_id'], identity)
    return identity


@flow(name='verified-model-release', log_prints=True)
def release_flow(root, report_name='report-B.json', fail_after_prepare=False,
                 manifest_name='manifest-B.json', fail_after_pointer=False):
    root = Path(root).resolve()
    prepared = prepare_task(root)
    # The durable disk cache also gives a framework-independent CACHE_HIT audit line.
    # Prefect logs distinguish task Cached from task Completed.
    print('PREPARATION_RESULT', prepared['input_sha256'])
    if fail_after_prepare:
        raise RuntimeError('Injected failure after preparation')
    artifact = artifact_task(root, manifest_name, prepared)
    report = gate_task(root, report_name, artifact)
    return publication_task(root, artifact, report, fail_after_pointer)
