"""A quick summary of the measured lab progress; no cluster connection required."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'results/events.jsonl'
if not p.exists():raise SystemExit('No run started')
for e in [json.loads(l) for l in p.read_text().splitlines()]:
    if e['event']=='load_end':
        print(e['run'],e['outcomes'],'p95=',round(e['ttfc_p95'],3) if e['ttfc_p95'] is not None else None)
    elif e['event'] in ('A-warmup-broken','A-warmup-fixed'):
        print(e['event'],e['outcomes'])
    elif e['event'].endswith('complete') or e['event']=='FAILED':print(e['event'],e.get('error',''))
print('Latest event:',json.loads(p.read_text().splitlines()[-1]))
