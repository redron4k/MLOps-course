"""Open-loop load generator; monotonic client timings, one connection per attempt."""
import argparse
import asyncio
from collections import Counter
import json
import math
from pathlib import Path
import time
import aiohttp


def schedule(duration=120, rate=8, skew=False, seed=11):
    # Exactly 80% client 0 when skew=True; identical content/timing across policies.
    rows=[]
    assignment=[0]*12+[1,2,3] if skew else [0,1,2,3]
    for i in range(int(duration*rate)):
        client=assignment[i % len(assignment)]
        rows.append(dict(id=i, at=i/rate, client=client,
            payload={'messages':[{'role':'system','content':f'PREFIX-{i%8}-'+('x'*128)},
                                 {'role':'user','content':f'question-{i%37}'}],
                     'session_id':f'dialog-{client}-{i%8}', 'max_chunks':4}))
    return rows


async def attempt(session, url, entry, start, run, timeout=35, abort=False):
    planned=start+entry['at']
    await asyncio.sleep(max(0, planned-time.perf_counter()))
    sent=time.perf_counter()
    row=dict(run=run, id=entry['id'], client=entry['client'], scheduled_mono=planned,
        sent_mono=sent, sent_unix=time.time(), schedule_lag_seconds=sent-planned,
        outcome='error', status=None, first_mono=None, end_mono=None, chunks=0,
        pod=None, version=None, backend_mode=None, queue_seconds=None, request_id=f'{run}-{entry["client"]}-{entry["id"]}',
        inter_chunk_seconds=[], content=[], completed=False, retry=0)
    previous=None
    try:
        async with session.post(url+'/generate', json=entry['payload'],
            headers={'X-Request-ID':row['request_id']}, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            row['status']=response.status
            if response.status != 200:
                data=await response.json()
                row.update(pod=data.get('pod'),version=data.get('version'),backend_mode=data.get('backend_mode'))
                row['outcome']='rejected' if response.status in (429,503) else 'http_error'
                row['error']=data.get('error')
            else:
                row['outcome']='interrupted'
                async for line in response.content:
                    if not line.strip(): continue
                    data=json.loads(line)
                    if data.get('done'):
                        row['completed']=True
                        continue
                    now=time.perf_counter()
                    if row['first_mono'] is None: row['first_mono']=now
                    if previous is not None: row['inter_chunk_seconds'].append(now-previous)
                    previous=now
                    row['chunks']+=1
                    row['content'].append(data['content'])
                    for key in ('pod','version','backend_mode','queue_seconds','cache_hit'):
                        row[key]=data.get(key)
                    if abort:
                        response.close()
                        break
                if row['completed'] and row['chunks']==entry['payload']['max_chunks']:
                    row['outcome']='completed'
    except asyncio.TimeoutError:
        row['outcome']='timed_out'
    except (aiohttp.ClientError, OSError, ValueError) as exc:
        row['error']=str(exc)
        row['outcome']='interrupted' if row['status']==200 else 'connection_error'
    finally:
        row['end_mono']=time.perf_counter()
        row['duration_seconds']=row['end_mono']-sent
        row['ttfc_seconds']=row['first_mono']-sent if row['first_mono'] is not None else None
    return row


async def execute(entries, url, run, duration, timeout=35, start_unix=None):
    # Each Pod owns an independent client/IP; shared absolute start is mapped to local monotonic clock.
    start=time.perf_counter()+max(0, (start_unix or time.time()+1)-time.time())
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True, limit=0), trust_env=False) as session:
        tasks=[asyncio.create_task(attempt(session,url,e,start,run,timeout)) for e in entries]
        results=await asyncio.gather(*tasks)
        await asyncio.sleep(max(0,start+duration-time.perf_counter()))
    return results


def percentile(values,p):
    if not values: return None
    values=sorted(values)
    at=(len(values)-1)*p
    return values[math.floor(at)]*(1-at%1)+values[math.ceil(at)]*(at%1)


def summarize(rows):
    good=[r for r in rows if r['outcome']=='completed']
    return dict(planned=len(rows), sent=sum(r['sent_mono'] is not None for r in rows), outcomes=dict(Counter(r['outcome'] for r in rows)),
        observations=len(good), error_fraction=1-len(good)/len(rows) if rows else None,
        timeout_fraction=sum(r['outcome']=='timed_out' for r in rows)/len(rows) if rows else None,
        ttfc_p50=percentile([r['ttfc_seconds'] for r in good],.5),ttfc_p95=percentile([r['ttfc_seconds'] for r in good],.95),
        duration_p95=percentile([r['duration_seconds'] for r in good],.95),
        schedule_lag_p95=percentile([r['schedule_lag_seconds'] for r in rows],.95),
        pods={p:dict(n=sum(r['pod']==p for r in rows),ttfc_p95=percentile([r['ttfc_seconds'] for r in good if r['pod']==p],.95),queue_p95=percentile([r['queue_seconds'] for r in good if r['pod']==p],.95)) for p in set(r['pod'] for r in rows)})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--client',type=int,required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--service',default='api')
    p.add_argument('--duration',type=float,default=120)
    p.add_argument('--rate',type=float,default=8)
    p.add_argument('--skew',action='store_true')
    p.add_argument('--start',type=float,required=True)
    a=p.parse_args()
    entries=[e for e in schedule(a.duration,a.rate,a.skew) if e['client']==a.client]
    rows=asyncio.run(execute(entries,f'http://{a.service}:8000',a.run,a.duration,start_unix=a.start))
    for row in rows: print(json.dumps(row),flush=True)
