"""Separate real-model smoke, restart check, and 8+8 load measurement."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from benchmark import request,measure


def run(workspace,output,with_benchmark=True):
    workspace=workspace.resolve()
    output.mkdir(parents=True,exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
    url=f'http://127.0.0.1:{port}'
    prompts=[json.loads(line)['input'] for line in (PROJECT.parent/'hw67/data/eval/golden.jsonl').read_text().splitlines()][:8]
    snapshots=[]
    for iteration in (0,1):
        with (output/f'server-{iteration}.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(PROJECT/'service.py'),'--workspace',str(workspace),
                '--engine','fastapi','--backend','transformers','--port',str(port),'--warmup-seconds','0'],
                stdout=log,stderr=log,cwd=PROJECT)
            try:
                deadline=time.monotonic()+180
                while True:
                    try:
                        status,ready=request(url,'/ready',timeout=2)
                        if status==200: break
                    except OSError: pass
                    if process.poll() is not None or time.monotonic()>deadline:
                        raise RuntimeError('Model failed to become ready; see server log')
                    time.sleep(.2)
                health=request(url,'/health')
                invalid=request(url,'/generate',dict(prompt='x',max_new_tokens=100000))
                status,answer=request(url,'/generate',dict(prompt=prompts[1],max_new_tokens=32))
                expected=json.loads((workspace/'accepted.json').read_text())
                assert status==200 and answer['model_id']==expected['model_id']
                assert answer['runtime_sha256']==expected['runtime_sha256']
                assert 400<=invalid[0]<500
                assert answer['output_tokens']>0 and not answer['answer'].startswith('STUB ONLY')
                snapshots.append(dict(restart=iteration,health=health,ready=ready,invalid=invalid,
                                      prompt=prompts[1],response=answer))
                (output/'smoke.json').write_text(json.dumps(snapshots,ensure_ascii=False,indent=2)+'\n')
                if iteration==0 and with_benchmark:
                    summaries=measure(url,prompts,output/'benchmark')
                    assert all(s['errors']==0 for s in summaries)
            finally:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
    print(json.dumps(snapshots,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,default=PROJECT/'.real')
    parser.add_argument('--output',type=Path,default=PROJECT/'evidence/real-api')
    parser.add_argument('--skip-benchmark',action='store_true')
    args=parser.parse_args()
    run(args.workspace,args.output,not args.skip_benchmark)
