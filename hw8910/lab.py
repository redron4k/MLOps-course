"""Исполняемый прототип для расследования. Только синтетические данные, без модели.

Запускайте в отдельном каталоге .lab. Этот последовательный прототип надо
превратить в корректный Prefect-flow; его зелёный статус не доказывает корректность.
"""
import argparse
import hashlib
import json
import uuid
import time
from pathlib import Path
import release


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2))


def initialize(root):
    if root.exists():
        raise FileExistsError('Рабочий каталог уже существует. Выберите новый --workspace.')
    root.mkdir(parents=True)
    runtime={'base_revision':'fixture-only','tokenizer_sha256':'a'*64,
             'template_sha256':'b'*64,'generation':{'max_new_tokens':32,'do_sample':False},'dtype':'float32'}
    for version in ['A','B']:
        artifact=root/f'model-{version}.txt';artifact.write_text(f'STUB ONLY — version {version}\n')
        files={artifact.name:digest(artifact.read_bytes())}
        m={'schema_version':1,'model_id':f'fixture-{version}',
           'artifact_uri':artifact.name,'model_sha256':digest(canonical(files)),
           'protocol_sha256':'d'*64,'golden_sha256':'e'*64,
           'runtime_sha256':digest(canonical(runtime)),'runtime':runtime,'file_hashes':files}
        report={k:m[k] for k in ['model_id','model_sha256','protocol_sha256','golden_sha256','runtime_sha256']}
        report.update(status='complete',passed=True,evaluation_run_id=f'fixture-eval-{version}',metrics={'accuracy':.8,'format_rate':1.0},synthetic=True)
        write(root/f'manifest-{version}.json',m);write(root/f'report-{version}.json',report)
    write(root/'accepted.json',read(root/'manifest-A.json'))
    report=read(root/'report-B.json');report['status']='running';report['passed']='false'
    write(root/'report-incomplete.json',report)
    report=read(root/'report-B.json');report['passed']=False
    write(root/'report-reject.json',report)
    write(root/'config.json',{'tokenizer_revision':'r1','template_sha256':'b'*64,'max_length':128,'code_revision':'prepare-v1'})
    (root/'input.txt').write_text('Первый пример\nВторой пример\n')
    print('Создан синтетический стенд:',root)


def prepare(root):
    config=read(root/'config.json');key=release.cache_key(root/'input.txt',config)
    cache=root/'cache';cache.mkdir(exist_ok=True);entry=cache/(key+'.json')
    if entry.exists():
        result=read(entry);print('CACHE_HIT',result['input_sha256']);return result
    payload=(root/'input.txt').read_bytes()
    if config.get('format')=='jsonl':
        rows=[json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        if not rows or any(not isinstance(row.get('input'),str) or not row['input'].strip()
                           or not isinstance(row.get('id'),str) for row in rows):
            raise ValueError('Invalid evaluation input schema')
        if len({row['id'] for row in rows})!=len(rows):
            raise ValueError('Duplicate input IDs')
    result={'input_sha256':digest(payload),'rows':len(payload.splitlines()),'config':config}
    release.atomic_bytes(entry, canonical(result));print('PREPARED',result['input_sha256']);return result


def run(root,report_name,fail_after_prepare=False,manifest_name='manifest-B.json',fail_after_pointer=False):
    prepared=prepare(root)
    if fail_after_prepare:raise RuntimeError('Инъекция сбоя после подготовки')
    manifest=read(root/manifest_name);report=read(root/report_name)
    identity=release.publish_release(manifest,report,root,fail_after_pointer)
    print('PUBLISHED',manifest['model_id'],identity)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,default=Path('.lab'))
    sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('init')
    run_parser=sub.add_parser('run');run_parser.add_argument('--report',default='report-B.json');run_parser.add_argument('--fail-after-prepare',action='store_true')
    run_parser.add_argument('--manifest',default='manifest-B.json')
    run_parser.add_argument('--fail-after-pointer',action='store_true')
    run_parser.add_argument('--engine',choices=['stdlib','prefect'],default='stdlib')
    args=parser.parse_args()
    if args.command=='init':initialize(args.workspace)
    else:
        attempt={'attempt_id':str(uuid.uuid4()),'started_at':time.time(),'engine':args.engine}
        try:
            if args.engine=='prefect':
                from pipeline import release_flow
                release_flow(args.workspace,args.report,args.fail_after_prepare,args.manifest,args.fail_after_pointer)
            else:run(args.workspace,args.report,args.fail_after_prepare,args.manifest,args.fail_after_pointer)
            attempt['status']='complete'
        except Exception as exc:
            attempt.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            with (args.workspace/'attempts.jsonl').open('a') as stream:
                stream.write(json.dumps(attempt,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
