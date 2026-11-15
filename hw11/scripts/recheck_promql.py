"""Evaluate the corrected PromQL over the already recorded, identical cohort."""
import asyncio
import json
from pathlib import Path
import sys
import shutil
import aiohttp
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from scripts.run_lab import forward,write
async def main():
    p=root/'results/histogram-promql.json'
    old=json.loads(p.read_text())
    if old['result'].get('status')=='error':shutil.copy2(p,root/'results/histogram-promql-invalid.json')
    cohort=json.loads((root/'results/histogram-comparison.json').read_text())
    pod=cohort['pod'];start=cohort['start_unix'];end=cohort['end_unix']
    query=f'histogram_quantile(0.95, sum by(le) (inference_ttfc_seconds_bucket{{pod="{pod}"}} @ {end} - inference_ttfc_seconds_bucket{{pod="{pod}"}} @ {start}))'
    proc=forward('service/prometheus',19090,9090)
    try:
        async with aiohttp.ClientSession(trust_env=False) as s:
            async with s.get('http://127.0.0.1:19090/api/v1/query',params={'query':query}) as r:result=await r.json()
        assert result['status']=='success',result
        write('histogram-promql.json',dict(query=query,result=result))
        print(json.dumps(result))
    finally:proc.terminate()
asyncio.run(main())
