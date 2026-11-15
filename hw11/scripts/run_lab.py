"""Real Kubernetes experiments; evidence is never generated from invented timings."""
import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import aiohttp
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from client import attempt, execute, schedule, summarize
os.chdir(ROOT)
RESULTS=ROOT/'results'
K=['kubectl','--kubeconfig',str(ROOT/'.tools/kubeconfig')]
H=[str(ROOT/'.tools/helm'),'--kubeconfig',str(ROOT/'.tools/kubeconfig')]
LOCK=threading.Lock()
STOP=threading.Event()


def write(name,data):
    p=RESULTS/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def append(name,data):
    with LOCK:
        p=RESULTS/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('a') as f:f.write(json.dumps(data,ensure_ascii=False)+'\n')


def event(name,**kw):
    row=dict(event=name,unix=time.time(),**kw)
    append('events.jsonl',row);print(json.dumps(row),flush=True)


def cmd(args,check=True,timeout=240):
    result=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    append('commands.jsonl',dict(unix=time.time(),argv=args,code=result.returncode,stdout=result.stdout if not any(x in args for x in ('get','logs')) else '[saved separately]',stderr=result.stderr))
    if check and result.returncode:raise RuntimeError(f'{args}: {result.stderr}\n{result.stdout}')
    return result.stdout


def k(*args,**kwargs):return cmd(K+['-n','hw11']+list(args),**kwargs)

def get(resource,*args):return json.loads(k('get',resource,*args,'-o','json'))


def pods():return get('pods','-l','app=api')['items']


def ready(p):return any(c['type']=='Ready' and c['status']=='True' for c in p.get('status',{}).get('conditions',[]))

def active_pods():return [p for p in pods() if ready(p) and not p['metadata'].get('deletionTimestamp')]


def wait_api(timeout=180):
    k('rollout','status','deployment/api',f'--timeout={timeout}s',timeout=timeout+10)


def upgrade(**values):
    args=H+['upgrade','--install','hw11','chart','-n','hw11','--create-namespace']
    for key,value in values.items():args+=['--set-json',f'{key}={json.dumps(value)}']
    cmd(args)
    event('upgrade',values=values)



LOG_CURSOR={}
def save_pod_log(name,text):
    directory=RESULTS/'pod-logs';directory.mkdir(exist_ok=True)
    path=directory/f'{name}.log'
    if name not in LOG_CURSOR:
        previous=path.read_text().splitlines() if path.exists() else []
        LOG_CURSOR[name]=previous[-1].split(' ',1)[0] if previous else ''
    fresh=[line for line in text.splitlines() if line.split(' ',1)[0]>LOG_CURSOR[name]]
    if fresh:
        with path.open('a') as stream:stream.write('\n'.join(fresh)+'\n')
        LOG_CURSOR[name]=fresh[-1].split(' ',1)[0]

def collect_pod_logs():
    for p in pods():
        name=p['metadata']['name']
        log=k('logs',name,'--timestamps=true',check=False)
        if log:save_pod_log(name,log)


def observer():
    while not STOP.is_set():
        try:
            pp=pods()
            for p in pp:
                name=p['metadata']['name']
                text=k('logs',name,'--timestamps=true',check=False)
                if text:
                    save_pod_log(name,text)
                for c in p.get('status',{}).get('containerStatuses',[]):
                    if c.get('restartCount',0)>0:
                        previous=k('logs',name,'--previous','--timestamps=true',check=False)
                        if previous:save_pod_log(name+'-previous-'+str(c['restartCount']),previous)
            row=dict(unix=time.time(),pods=pp,endpoints=get('endpointslices','-l','kubernetes.io/service-name=api'),hpa=get('hpa'),deployment=get('deployment','api'))
            append('timeline.jsonl',row)
        except Exception as exc:append('observer-errors.jsonl',dict(unix=time.time(),error=str(exc)))
        STOP.wait(2)


async def json_http(url,method='GET',payload=None):
    async with aiohttp.ClientSession(trust_env=False) as s:
        async with s.request(method,url,json=payload,timeout=aiohttp.ClientTimeout(total=15)) as r:
            return r.status,await r.json()


def forward(resource,port,target):
    log=(RESULTS/f'forward-{port}.log').open('a')
    proc=subprocess.Popen(K+['-n','hw11','port-forward',resource,f'{port}:{target}'],stdout=log,stderr=log)
    time.sleep(1)
    return proc


async def one(url,run,payload=None,abort=False,timeout=35):
    e=dict(id=0,at=0,client='host',payload=payload or {'prompt':'probe','max_chunks':4})
    e['payload'].setdefault('max_chunks',4)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True),trust_env=False) as s:
        row=await attempt(s,url,e,time.perf_counter(),run,timeout,abort)
    append('requests.jsonl',row)
    return row


async def traffic(url,run,duration=75,rate=3,chunks=4):
    entries=schedule(duration,rate,False)
    for e in entries:e['payload']['max_chunks']=chunks
    rows=await execute(entries,url,run,duration)
    for row in rows:append('requests.jsonl',row)
    write(f'summaries/{run}.json',summarize(rows))
    return rows


async def probes_experiment():
    for broken in (True,False):
        label='A-warmup-broken' if broken else 'A-warmup-fixed'
        upgrade(brokenProbes=broken,warmupSeconds=20)
        started=time.time()
        rows=await traffic('http://127.0.0.1:18000',label,75,3)
        wait_api()
        write(f'checks/{label}.json',dict(start=started,**summarize(rows)))
        event(label,**summarize(rows))


async def rollout_experiment():
    upgrade(chunkSeconds=.25);await asyncio.to_thread(wait_api)
    # Direct stream to both old Pods guarantees at least one deleted old replica has an active stream.
    procs=[];tasks=[]
    try:
        for i,p in enumerate(active_pods()):
            procs.append(forward('pod/'+p['metadata']['name'],18100+i,8000))
            tasks.append(asyncio.create_task(one(f'http://127.0.0.1:{18100+i}','A-rollout-long-'+str(i),{'prompt':'long-answer','max_chunks':128})))
        load=asyncio.create_task(traffic('http://127.0.0.1:18000','A-rollout',90,3))
        await asyncio.sleep(2)
        event('rollout_B_begin')
        upgrade(modelVersion='B',chunkSeconds=.25)
        await asyncio.to_thread(wait_api)
        results=await asyncio.gather(*tasks)
        rows=await load
        if any(r['outcome']!='completed' for r in results+rows):raise AssertionError('clean rollout not clean')
        event('rollout_B_pass',long_streams=[r['request_id'] for r in results])
    finally:
        for p in procs:p.terminate()
    # Roll back to A before deploying an intentionally unready B.
    upgrade(modelVersion='A');await asyncio.to_thread(wait_api)
    upgrade(modelVersion='B',neverReady=True)
    rows=await traffic('http://127.0.0.1:18000','A-bad-B',35,3)
    write('checks/unready-B.json',dict(pods=pods(),responses=sorted(set(r['version'] for r in rows if r['version']))))
    cmd(H+['rollback','hw11','0','-n','hw11'])
    event('rollback_A_begin')
    await asyncio.to_thread(wait_api)
    r=await one('http://127.0.0.1:18000','A-after-rollback')
    assert r['version']=='A' and r['outcome']=='completed'
    write('checks/rollback-version.json',r)


async def recovery_experiment():
    # Kill the task from the node namespace: Linux protects namespace PID 1 from child signals.
    victim=active_pods()[0]['metadata']['name']
    proc=forward('pod/'+victim,18110,8000)
    try:
        stream=asyncio.create_task(one('http://127.0.0.1:18110','A-forced-kill',{'prompt':'kill-me','max_chunks':128}))
        load=asyncio.create_task(traffic('http://127.0.0.1:18000','A-kill-recovery',65,3))
        await asyncio.sleep(2)
        event('forced_kill',pod=victim)
        container=next(p for p in pods() if p['metadata']['name']==victim)['status']['containerStatuses'][0]['containerID'].split('://',1)[1]
        await asyncio.to_thread(cmd,['docker','exec','hw11-control-plane','ctr','-n','k8s.io','tasks','kill','--signal','SIGKILL',container])
        r=await stream
        assert r['outcome']=='interrupted',r
        write('checks/interrupted-accounting.json',r)
        await load
    finally:proc.terminate()
    await asyncio.to_thread(wait_api)
    # Two disjoint dialogues, client history survives Pod loss.
    original=active_pods()[0]['metadata']['name']
    p=forward('pod/'+original,18111,8000)
    histories=[[{'role':'user','content':marker}] for marker in ('CANARY-ALPHA-9381','CANARY-BETA-5726')]
    try:
        for i,h in enumerate(histories):
            row=await one('http://127.0.0.1:18111',f'A-isolation-before-{i}',{'messages':h,'max_chunks':2})
            histories[i]+=[{'role':'assistant','content':row['content'][0]},{'role':'user','content':'continue'}]
    finally:p.terminate()
    await asyncio.to_thread(k,'delete','pod',original,'--wait=false')
    await asyncio.sleep(2);await asyncio.to_thread(wait_api)
    rows=[]
    for i,h in enumerate(histories):
        row=await one('http://127.0.0.1:18000',f'A-isolation-after-{i}',{'messages':h,'max_chunks':2})
        text=' '.join(row['content'])
        assert histories[i][0]['content'] in text and histories[1-i][0]['content'] not in text
        assert row['version']=='A' and row['pod']!=original
        rows.append(row)
    write('checks/replacement-dialogues.json',dict(deleted_pod=original,rows=rows))
    event('A_complete')


def image_name():
    values=yaml.safe_load((ROOT/'chart/values.yaml').read_text())
    return values['image']['repository']+':'+values['image']['tag']


def make_clients(inventory='clients.json'):
    existing={p['metadata']['name']:p for p in get('pods','-l','app=load-client')['items']}
    for i in range(4):
        old=existing.get(f'client-{i}')
        if old and old['spec']['containers'][0]['image']!=image_name():
            k('delete','pod',f'client-{i}','--wait=true')
        obj={'apiVersion':'v1','kind':'Pod','metadata':{'name':f'client-{i}','namespace':'hw11','labels':{'app':'load-client'}},'spec':{'containers':[{'name':'client','image':image_name(),'imagePullPolicy':'IfNotPresent','command':['python','-c','import time; time.sleep(86400)'],'resources':{'requests':{'cpu':'50m','memory':'48Mi'},'limits':{'cpu':'500m','memory':'128Mi'}}}],'restartPolicy':'Never'}}
        path=ROOT/'.tools'/f'client-{i}.yaml';path.write_text(yaml.safe_dump(obj));k('apply','-f',str(path))
    k('wait','--for=condition=Ready','pod','-l','app=load-client','--timeout=90s')
    write(inventory,get('pods','-l','app=load-client'))


async def reset_cache():
    for i,p in enumerate(active_pods()):
        proc=forward('pod/'+p['metadata']['name'],18200+i,8000)
        try:
            status,data=await json_http(f'http://127.0.0.1:{18200+i}/reset-cache','POST')
            assert status==200,data
        finally:proc.terminate()


async def run_clients(run,service,skew,duration=120,rate=8):
    start=time.time()+5
    entries=schedule(duration,rate,skew)
    write(f'schedules/{run}.json',dict(start_unix=start,duration=duration,rate=rate,skew=skew,entries=entries))
    event('load_start',run=run,service=service,skew=skew,start_unix=start,duration=duration,rate=rate)
    async def client(i):
        args=K+['-n','hw11','exec',f'client-{i}','--','python','client.py','--client',str(i),'--run',run,'--service',service,'--duration',str(duration),'--rate',str(rate),'--start',str(start)]
        if skew:args+=['--skew']
        proc=await asyncio.create_subprocess_exec(*args,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        stdout,stderr=await proc.communicate()
        if proc.returncode:raise RuntimeError(stderr.decode())
        (RESULTS/'client-logs').mkdir(exist_ok=True)
        (RESULTS/'client-logs'/f'{run}-{i}.jsonl').write_bytes(stdout)
        return [json.loads(line) for line in stdout.splitlines() if line.strip()]
    groups=await asyncio.gather(*(client(i) for i in range(4)))
    rows=[row for g in groups for row in g]
    for row in rows:append('requests.jsonl',row)
    write(f'summaries/{run}.json',summarize(rows))
    event('load_end',run=run,**summarize(rows))
    return rows


async def balance_experiment():
    upgrade();await asyncio.to_thread(wait_api)
    make_clients()
    write('network.json',dict(services=get('services'),nodes=json.loads(cmd(K+['get','nodes','-o','json'])),kubeproxy=cmd(K+['-n','kube-system','get','configmap','kube-proxy','-o','yaml'])))
    # Fixed order alternates policies across repeats. Cold local cache, fully warmed backend.
    for repeat in range(1,4):
        for skew in (False,True):
            services=('api','api-sticky') if repeat%2 else ('api-sticky','api')
            for service in services:
                await reset_cache()
                run=f'B-{service}-{"skew" if skew else "uniform"}-{repeat}'
                await run_clients(run,service,skew,rate=12)
    event('B_complete')


async def prometheus_snapshot(name):
    async with aiohttp.ClientSession(trust_env=False) as s:
        for q in ('ALERTS','inference_requests_total','inference_ttfc_seconds_bucket','inference_duration_seconds_bucket','inference_running','inference_waiting'):
            async with s.get('http://127.0.0.1:19090/api/v1/query',params={'query':q}) as r:
                write(f'prometheus/{name}-{q}.json',await r.json())


async def hpa_experiment():
    upgrade(**{'hpa.enabled':True});await asyncio.to_thread(wait_api)
    write('checks/metrics-api.json',json.loads(cmd(K+['get','--raw','/apis/metrics.k8s.io/v1beta1/pods'])))
    await reset_cache()
    await run_clients('C-hpa-high','api',False,180,26)
    await asyncio.sleep(5)
    write('checks/hpa-scaled.json',get('hpa'))
    await prometheus_snapshot('high')
    event('hpa_load_drop')
    start=time.time()
    while time.time()-start<300:
        pp=pods()
        if len(pp)==2 and all(ready(p) for p in pp):break
        await asyncio.sleep(5)
    assert len(pods())==2,'HPA did not scale back to 2'
    event('hpa_minimum_restored',seconds_since_drop=time.time()-start)
    await run_clients('C-hpa-hot-sticky','api-sticky',True,120,8)
    await prometheus_snapshot('hot-sticky')
    # Overload a single replica with long streams: waiting raises TTFC and bounded queue rejects requests.
    await run_clients('C-alerts-overload','api-sticky',True,100,40)
    await prometheus_snapshot('overload')
    await asyncio.sleep(65)
    await prometheus_snapshot('recovered')
    write('checks/hpa-final.json',get('hpa'))
    event('C_complete')


async def prom_observer():
    async with aiohttp.ClientSession(trust_env=False) as s:
        while not STOP.is_set():
            try:
                async with s.get('http://127.0.0.1:19090/api/v1/alerts',timeout=aiohttp.ClientTimeout(total=5)) as r:
                    append('alerts.jsonl',dict(unix=time.time(),data=await r.json()))
            except Exception:pass
            await asyncio.sleep(2)


async def main(phases):
    RESULTS.mkdir(exist_ok=True)
    procs=[]
    upgrade();wait_api()
    procs.append(forward('service/api',18000,8000))
    procs.append(forward('service/prometheus',19090,9090))
    # Service port-forward pins a Pod. For rollout traffic use a persistent in-cluster HTTP proxy client.
    # Forward to a stable bridge Pod whose requests go through the ClusterIP Service on each connection.
    bridge={'apiVersion':'v1','kind':'Pod','metadata':{'name':'bridge','namespace':'hw11'},'spec':{'containers':[{'name':'bridge','image':image_name(),'command':['python','-c',BRIDGE], 'resources':{'requests':{'cpu':'30m','memory':'48Mi'},'limits':{'cpu':'500m','memory':'128Mi'}}}]}}
    path=ROOT/'.tools/bridge.yaml';path.write_text(yaml.safe_dump(bridge));k('apply','-f',str(path));k('wait','--for=condition=Ready','pod/bridge','--timeout=90s')
    procs[0].terminate();procs[0]=forward('pod/bridge',18000,8001)
    t=threading.Thread(target=observer,daemon=True);t.start()
    promtask=asyncio.create_task(prom_observer())
    try:
        if 'A' in phases:await probes_experiment()
        if 'R' in phases or 'A' in phases:
            await rollout_experiment();await recovery_experiment()
        elif 'K' in phases:await recovery_experiment()
        if 'B' in phases:await balance_experiment()
        if 'C' in phases:await hpa_experiment()
    finally:
        STOP.set();promtask.cancel();t.join(timeout=10)
        with contextlib.suppress(Exception):collect_pod_logs()
        with contextlib.suppress(Exception):await export_prometheus()
        for p in procs:p.terminate()


async def export_prometheus():
    if not (RESULTS/'timeline.jsonl').exists():return
    times=[json.loads(l)['unix'] for l in (RESULTS/'timeline.jsonl').read_text().splitlines()]
    queries={
        'ttfc_p95':'histogram_quantile(0.95,sum by(le,pod)(rate(inference_ttfc_seconds_bucket[30s])))',
        'running':'inference_running','waiting':'inference_waiting',
        'cpu':'sum by(pod)(rate(container_cpu_usage_seconds_total{namespace="hw11",container="api"}[30s]))',
        'memory':'sum by(pod)(container_memory_working_set_bytes{namespace="hw11",container="api"})',
        'alerts':'ALERTS','ttfc_buckets':'inference_ttfc_seconds_bucket',
        'completed':'inference_requests_total{outcome="completed"}'}
    async with aiohttp.ClientSession(trust_env=False) as s:
        for name,q in queries.items():
            async with s.get('http://127.0.0.1:19090/api/v1/query_range',params=dict(query=q,start=min(times),end=max(times),step=2)) as r:
                write('prometheus/range-'+name+'.json',await r.json())


BRIDGE='''import asyncio
from aiohttp import web,ClientSession,TCPConnector
async def handle(r):
    async with ClientSession(connector=TCPConnector(force_close=True),trust_env=False) as s:
        async with s.request(r.method,'http://api:8000'+r.path_qs,data=await r.read(),headers={k:v for k,v in r.headers.items() if k.lower() in ('content-type','x-request-id')}) as upstream:
            out=web.StreamResponse(status=upstream.status,headers={'Content-Type':upstream.headers.get('Content-Type','application/json')})
            await out.prepare(r)
            async for data in upstream.content.iter_any(): await out.write(data)
            await out.write_eof()
            return out
app=web.Application();app.router.add_route('*','/{tail:.*}',handle);web.run_app(app,port=8001,access_log=None)
'''

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phases',default='ABC');a=p.parse_args()
    try:asyncio.run(main(a.phases))
    except Exception as exc:
        event('FAILED',error=str(exc),traceback=traceback.format_exc());raise
