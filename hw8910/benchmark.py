"""Complete-response latency benchmark; retain every failure and timeout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request
import uuid


def request(url, path, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url+path, data=data, headers={'Content-Type':'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def measure(url, prompts, output, synthetic=False):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    summaries = []
    with (output/'requests.jsonl').open('w') as stream:
        for concurrency in (1,2):
            def one(item):
                index,prompt = item
                started = time.perf_counter()
                record = dict(client_request_id=str(uuid.uuid4()), concurrency=concurrency,
                              index=index, prompt=prompt, max_new_tokens=32, synthetic=synthetic)
                try:
                    status,body = request(url,'/generate',dict(prompt=prompt,max_new_tokens=32))
                    record.update(status=status, response=body, error=None if status==200 else str(body))
                except Exception as exc:
                    record.update(status=None,error=f'{type(exc).__name__}: {exc}')
                record['client_seconds'] = time.perf_counter()-started
                return record
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                records=list(pool.map(one,enumerate(prompts)))
            elapsed = time.perf_counter()-started
            for record in records:
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                stream.flush()
            successes=[r for r in records if r['status']==200]
            failures=[r for r in records if r['status']!=200]
            def mean(rows):
                return statistics.mean(r['client_seconds'] for r in rows) if rows else None
            def maximum(rows):
                return max(r['client_seconds'] for r in rows) if rows else None
            summaries.append(dict(concurrency=concurrency,n=len(records),successes=len(successes),
                errors=len(failures),batch_seconds=elapsed,successful_requests_per_second=len(successes)/elapsed,
                success_mean_seconds=mean(successes),success_max_seconds=maximum(successes),
                error_mean_seconds=mean(failures),error_max_seconds=maximum(failures),
                generation_mean_seconds=statistics.mean(r['response']['generation_seconds'] for r in successes) if successes else None,
                queue_mean_seconds=statistics.mean(r['response']['queue_seconds'] for r in successes) if successes else None))
    (output/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))
    return summaries


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8091')
    parser.add_argument('--prompts',type=Path,default=Path('../hw67/data/eval/golden.jsonl'))
    parser.add_argument('--output',type=Path,default=Path('evidence/benchmark'))
    parser.add_argument('--n',type=int,default=8)
    parser.add_argument('--synthetic',action='store_true')
    args=parser.parse_args()
    prompts=[json.loads(line)['input'] for line in args.prompts.read_text().splitlines()][:args.n]
    if request(args.url,'/ready')[0]!=200:
        raise SystemExit('Service is not ready')
    summaries=measure(args.url,prompts,args.output,args.synthetic)
    raise SystemExit(any(s['errors'] for s in summaries))
