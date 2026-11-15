"""Derive reviewable evidence points from raw, measured traces."""
from collections import Counter,defaultdict
from datetime import datetime
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
def lines(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def load(name):
 p=R/name;return json.loads(p.read_text()) if p.exists() else {}
rows=lines(R/'requests.jsonl');timeline=lines(R/'timeline.jsonl')
logs=[]
for p in (R/'pod-logs').glob('*.log'):
 for line in p.read_text().splitlines():
  try:logs.append(json.loads(line.split(' ',1)[1]))
  except (ValueError,IndexError):pass
current={r['request_id']:r for r in rows}
unique={}
for x in logs:
 if x.get('event')=='received' and x.get('request_id') in current and x.get('pod')==current[x['request_id']].get('pod'):
  unique[(x['request_id'],x.get('unix'),x['pod'])]=x
received=list(unique.values())
fixed=[x for x in received if str(x.get('request_id')).startswith('A-warmup-fixed-')]
broken=[x for x in received if str(x.get('request_id')).startswith('A-warmup-broken-') and not x.get('warm')]

def ready(p):return any(c['type']=='Ready' and c['status']=='True' for c in p.get('status',{}).get('conditions',[]))
examples=[]
for log in broken:
 snapshots=sorted(timeline,key=lambda t:abs(t['unix']-log['unix']))[:3]
 for snap in snapshots:
  pod=next((p for p in snap['pods'] if p['metadata']['name']==log['pod']),None)
  eps=[e for item in snap['endpoints']['items'] for e in item.get('endpoints',[]) if e.get('targetRef',{}).get('name')==log['pod']]
  if pod and ready(pod) and any(e.get('conditions',{}).get('ready') for e in eps):
   examples.append(dict(request=log,snapshot_unix=snap['unix'],pod_ready=True,endpoints=eps));break
 if len(examples)>=3:break

client_map=defaultdict(Counter)
for x in received:
 if str(x.get('request_id')).startswith('B-'):client_map[x.get('peer')][x['pod']]+=1
# Quantitative simultaneous hot waiting / spare slot proof, using aligned query_range times.
def series_map(name):
 out=defaultdict(dict)
 for s in load('prometheus/range-'+name+'.json').get('data',{}).get('result',[]):
  for t,v in s['values']:out[float(t)][s['metric'].get('pod')]=float(v)
 return out
waiting=series_map('waiting');running=series_map('running');hot=[]
for t,values in waiting.items():
 for pod,n in values.items():
  if n<1:continue
  for other,active in running.get(t,{}).items():
   if other!=pod and active<2 and other in values:
    nearest=min(timeline,key=lambda snap:abs(snap['unix']-t))
    available={p['metadata']['name'] for p in nearest['pods'] if ready(p) and not p['metadata'].get('deletionTimestamp')}
    if pod not in available or other not in available:continue
    # Restrict to B windows, so rollout/unready remnants are not mistaken for spare capacity.
    relevant=[r for r in rows if r['run'].startswith('B-api-sticky-skew') and abs(r['sent_unix']-t)<1]
    if relevant:
     hot.append(dict(unix=t,hot=pod,waiting=n,other=other,other_running=active,other_waiting=values[other]));break
  if len(hot)>=5:break
 if len(hot)>=5:break

summary=load('summary.json');counterexamples=[]
for run,s in summary.items():
 if (run.startswith('B-api-sticky-uniform') or run.startswith('B-counterexample-uniform')) and len(s['pods'])==2:
  counts={p:v['n'] for p,v in s['pods'].items()}
  if max(counts.values())/sum(counts.values())<.6:counterexamples.append(dict(run=run,pod_counts=counts,explanation='Four equal clients happened to bind 2:2; ClientIP is not a guarantee of imbalance.'))
expected_hash=json.loads((ROOT/'artifacts/manifest-A.json').read_text())['model_sha256']
recovered=[x for x in received if str(x.get('request_id')).startswith('A-isolation-after-')]
result=dict(recovery_artifact_hashes={x['pod']:x.get('artifact_sha256') for x in recovered},recovery_artifact_matches_manifest=bool(recovered) and all(x.get('artifact_sha256')==expected_hash for x in recovered),broken_cold_requests=len(broken),fixed_cold_requests=sum(not x['warm'] for x in fixed),fixed_observed_requests=len(fixed),
 broken_ready_endpoint_examples=examples,client_peer_to_pod={p:dict(c) for p,c in client_map.items()},
 hot_waiting_spare_capacity_examples=hot,no_skew_counterexamples=counterexamples)
(R/'trace-proofs.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if not isinstance(v,(list,dict))}))
