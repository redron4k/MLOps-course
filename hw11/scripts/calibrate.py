"""Match histogram counter deltas to an isolated client cohort, through a Kubernetes Pod."""
import asyncio
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_lab import active_pods,forward,k,upgrade,wait_api,one,write,event,json_http,collect_pod_logs
from client import percentile
import aiohttp

async def metrics(url):
    async with aiohttp.ClientSession(trust_env=False) as s:
        async with s.get(url+'/metrics') as r:text=await r.text()
    values={}
    for line in text.splitlines():
        if line.startswith('inference_ttfc_seconds_bucket'):
            left,value=line.split(' ');le=left.split('le="')[1].split('"')[0];values[le]=float(value)
        if line.startswith('inference_ttfc_seconds_count '):values['count']=float(line.split()[1])
    return values,text

def bucket_quantile(buckets,q):
    pairs=sorted((float(le),n) for le,n in buckets.items() if le!='count')
    rank=pairs[-1][1]*q;prev_bound=prev_count=0
    for bound,count in pairs:
        if count>=rank:
            if bound==float('inf'):return prev_bound
            return prev_bound+(bound-prev_bound)*(rank-prev_count)/(count-prev_count) if count>prev_count else bound
        prev_bound,prev_count=bound,count

async def main():
    # No other traffic during this cohort; HPA disabled and backend stable.
    upgrade();await asyncio.to_thread(wait_api)
    p=max(active_pods(),key=lambda item:item['metadata']['creationTimestamp']);proc=forward('pod/'+p['metadata']['name'],18300,8000)
    prom=forward('service/prometheus',19090,9090)
    try:
        from datetime import datetime
        previous=json.loads((ROOT/'results/checks/replacement-dialogues.json').read_text())
        status,ready=await json_http('http://127.0.0.1:18300/ready')
        recovery=await one('http://127.0.0.1:18300','D-replacement-fresh',{'prompt':'fresh replacement version check','max_chunks':2})
        before_dialogue=[json.loads(line) for line in (ROOT/'results/requests.jsonl').read_text().splitlines() if 'A-isolation-before-' in line]
        created=datetime.fromisoformat(p['metadata']['creationTimestamp'].replace('Z','+00:00')).timestamp()
        expected=json.loads((ROOT/'artifacts/manifest-A.json').read_text())['model_sha256']
        assert created>max(x['sent_unix'] for x in before_dialogue) and recovery['version']=='A' and recovery['outcome']=='completed' and ready['artifact_sha256']==expected
        write('checks/replacement-fresh-pod.json',dict(deleted=previous['deleted_pod'],replacement=p['metadata']['name'],created_unix=created,ready=ready,response=recovery))
        await asyncio.sleep(4)
        before,rawbefore=await metrics('http://127.0.0.1:18300')
        start=time.time()
        rows=[]
        for i in range(80):
            row=await one('http://127.0.0.1:18300',f'D-histogram-{i}',{'messages':[{'role':'user','content':f'calibration-{i%3}'}],'max_chunks':2})
            rows.append(row)
        await asyncio.sleep(4)
        end=time.time()
        after,rawafter=await metrics('http://127.0.0.1:18300')
        delta={key:after[key]-before.get(key,0) for key in after}
        (ROOT/'results/prometheus/calibration-before.txt').write_text(rawbefore)
        (ROOT/'results/prometheus/calibration-after.txt').write_text(rawafter)
        good=[r for r in rows if r['outcome']=='completed']
        expr=f'histogram_quantile(0.95, sum by(le) (inference_ttfc_seconds_bucket{{pod="{p["metadata"]["name"]}"}} @ {end} - inference_ttfc_seconds_bucket{{pod="{p["metadata"]["name"]}"}} @ {start}))'
        async with aiohttp.ClientSession(trust_env=False) as session:
            async with session.get('http://127.0.0.1:19090/api/v1/query',params={'query':expr}) as response: actual=await response.json()
        write('histogram-promql.json',dict(query=expr,result=actual))
        write('histogram-comparison.json',dict(start_unix=start,end_unix=end,pod=p['metadata']['name'],n_client=len(good),n_server=delta['count'],cohort_matches=len(good)==delta['count']==80,
            client_p50=percentile([r['ttfc_seconds'] for r in good],.5),client_p95=percentile([r['ttfc_seconds'] for r in good],.95),
            histogram_p50=bucket_quantile(delta,.5),histogram_p95=bucket_quantile(delta,.95),buckets=delta,
            promql=f'histogram_quantile(0.95, sum by (le) (increase(inference_ttfc_seconds_bucket{{pod="{p["metadata"]["name"]}"}}[{int(end-start)+1}s])))',
            method='Exact before/after cumulative bucket deltas; same successful response cohort. PromQL range estimate has scrape/extrapolation boundary uncertainty. Client includes HTTP and port-forward overhead.'))
        collect_pod_logs()
        event('histogram_cohort_complete',count=len(good))
    finally:proc.terminate();prom.terminate()

if __name__=='__main__':asyncio.run(main())
