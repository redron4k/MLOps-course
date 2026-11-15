"""Explicit exploratory counterexample: ClientIP can balance four equal clients 2:2.

All mapping probes are retained. This selects a balanced initial network mapping,
then repeats the already-defined uniform workload three times, without changing
backend, requests, cache policy or compute parameters. This is not a randomized
estimate of the probability of a balanced mapping.
"""
import asyncio
from collections import Counter
from pathlib import Path
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_lab import upgrade,wait_api,make_clients,run_clients,reset_cache,k,write,event

async def main():
    upgrade();await asyncio.to_thread(wait_api);make_clients("counterexample-clients.json")
    probes=[];chosen=None
    for i in range(16):
        name=f'api-sticky-counter-{i}'
        obj={'apiVersion':'v1','kind':'Service','metadata':{'name':name,'namespace':'hw11'},'spec':{'selector':{'app':'api'},'ports':[{'port':8000,'targetPort':8000}],'sessionAffinity':'ClientIP','sessionAffinityConfig':{'clientIP':{'timeoutSeconds':10800}}}}
        p=ROOT/'.tools'/f'{name}.yaml';p.write_text(yaml.safe_dump(obj));k('apply','-f',str(p))
        rows=await run_clients(f'B-mapping-probe-{i}',name,False,1,4)
        counts=Counter(r['pod'] for r in rows if r['outcome']=='completed')
        probes.append(dict(service=name,pod_counts=dict(counts),client_mapping={r['client']:r['pod'] for r in rows},service_manifest=obj))
        if sorted(counts.values())==[2,2]:chosen=name;break
    write('counterexample-selection.json',dict(chosen=chosen,probes=probes,selection='Explicitly selected balanced 2:2 mapping; not a random prevalence estimate.'))
    assert chosen,'No balanced mapping found; all exploratory outcomes retained'
    for rep in range(1,4):
        await reset_cache()
        rows=await run_clients(f'B-counterexample-uniform-{rep}',chosen,False,120,12)
        assert sorted(Counter(r['pod'] for r in rows).values())==[720,720]
    event('counterexample_complete',service=chosen)

if __name__=='__main__':asyncio.run(main())
