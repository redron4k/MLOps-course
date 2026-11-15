"""Strict delivery gate: validate genuine recorded Kubernetes experiments."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
r=root/'results'
rows=[json.loads(x) for x in (r/'requests.jsonl').read_text().splitlines()]
events=[json.loads(x) for x in (r/'events.jsonl').read_text().splitlines()]
checks=[]
def check(name,condition,detail=None):
    checks.append(dict(name=name,passed=bool(condition),detail=detail))
    print(('PASS ' if condition else 'FAIL ')+name)

groups=defaultdict(list)
for row in rows:groups[row['run']].append(row)
for run,g in groups.items():
    check('unique_attempts:'+run,len({x['request_id'] for x in g})==len(g))
    check('monotonic_times:'+run,all(x['sent_mono']<=x['end_mono'] and (x['first_mono'] is None or x['sent_mono']<=x['first_mono']<=x['end_mono']) for x in g))
    check('stream_accounting:'+run,all(x['completed'] and x['chunks']>0 and len(x['inter_chunk_seconds'])==x['chunks']-1 for x in g if x['outcome']=='completed'))

# These tests concern actual requests through Kubernetes from the experiment controller.
check('kubernetes_warmup_original_failure',sum(x['status']==503 for x in groups['A-warmup-broken'])>0)
check('kubernetes_warmup_fixed',len(groups['A-warmup-fixed'])==225 and all(x['outcome']=='completed' for x in groups['A-warmup-fixed']))
check('kubernetes_version_after_rollback',all(x['version']=='A' and x['outcome']=='completed' for x in groups['A-after-rollback']) and bool(groups['A-after-rollback']))
check('rollout_both_actual_versions',set(x['version'] for x in groups['A-rollout'])=={'A','B'})
check('unready_B_excluded',len(groups['A-bad-B'])==105 and all(x['version']=='A' and x['outcome']=='completed' for x in groups['A-bad-B']))
check('rollout_zero_errors',bool(groups['A-rollout']) and all(x['outcome']=='completed' for x in groups['A-rollout']))
long=[x for x in rows if x['run'].startswith('A-rollout-long-')]
begin=next((e['unix'] for e in events if e['event']=='rollout_B_begin'),None)
check('rollout_preexisting_long_stream',bool(long) and begin is not None and all(x['outcome']=='completed' and x['sent_unix']<begin<x['sent_unix']+x['duration_seconds'] for x in long))
check('forced_kill_counted_as_interrupted',any(x['outcome']=='interrupted' and not x['completed'] for x in groups['A-forced-kill']))
replacement=json.loads((r/'checks/replacement-dialogues.json').read_text()) if (r/'checks/replacement-dialogues.json').exists() else {}
check('kubernetes_replacement_same_version',bool(replacement) and all(x['version']=='A' and x['pod']!=replacement['deleted_pod'] for x in replacement.get('rows',[])))
isolated=groups['A-isolation-after-0']+groups['A-isolation-after-1']
check('dialogue_isolation',len(isolated)==2 and 'CANARY-ALPHA-9381' in ''.join(isolated[0]['content']) and 'CANARY-BETA-5726' not in ''.join(isolated[0]['content']) and 'CANARY-BETA-5726' in ''.join(isolated[1]['content']) and 'CANARY-ALPHA-9381' not in ''.join(isolated[1]['content']))

for rep in range(1,4):
    for traffic in ('uniform','skew'):
        comparable=[]
        for policy in ('api','api-sticky'):
            name=f'B-{policy}-{traffic}-{rep}';g=groups[name]
            schedulefile=r/f'schedules/{name}.json'
            if not schedulefile.exists():check('full_B_run:'+name,False);continue
            sched=json.loads(schedulefile.read_text());entries=sched['entries']
            check('full_B_run:'+name,sched['duration']>=120 and len(g)==len(entries) and len({x['client'] for x in g})==4)
            check('generator_schedule_lag:'+name,bool(g) and max(x['schedule_lag_seconds'] for x in g)<1)
            comparable.append(entries)
        check(f'identical_load:{traffic}-{rep}',len(comparable)==2 and comparable[0]==comparable[1])

clients=json.loads((r/'clients.json').read_text()) if (r/'clients.json').exists() else {'items':[]}
ips=[p.get('status',{}).get('podIP') for p in clients['items']]
check('four_distinct_client_pod_ips',len(ips)==4 and len(set(ips))==4 and None not in ips)
timeline=[json.loads(l) for l in (r/'timeline.jsonl').read_text().splitlines()]
ready_counts=[sum(any(c['type']=='Ready' and c['status']=='True' for c in p.get('status',{}).get('conditions',[])) for p in t['pods']) for t in timeline]
comparison_rows=[x for x in rows if x['run'].startswith('B-') and not x['run'].startswith('B-mapping-probe')]
check('same_backend_version_in_comparisons',bool(comparison_rows) and all(x['version']=='A' and x['backend_mode']=='simulated' for x in comparison_rows))
main_comparison=[x for x in comparison_rows if x['run'].startswith(('B-api-uniform','B-api-skew','B-api-sticky-'))]
if main_comparison:
    first=min(x['sent_unix'] for x in main_comparison);last=max(x['sent_unix'] for x in main_comparison)
    image_ids={c.get('imageID') for snap in timeline if first<=snap['unix']<=last for p in snap['pods'] for c in p.get('status',{}).get('containerStatuses',[]) if c.get('ready')}
else:image_ids=set()
check('same_image_all_twelve_B_runs',len(image_ids)==1,sorted(image_ids))
cstart=next((e['unix'] for e in events if e['event']=='load_start' and e.get('run')=='C-hpa-high'),None)
cend=next((e['unix'] for e in events if e['event']=='hpa_minimum_restored'),None)
sequence=[]
for snap in timeline:
    if cstart is not None and cend is not None and cstart-3<=snap['unix']<=cend+3:
        count=sum(any(c['type']=='Ready' and c['status']=='True' for c in p.get('status',{}).get('conditions',[])) for p in snap['pods'])
        if not sequence or sequence[-1]!=count:sequence.append(count)
check('hpa_2_3_2',sequence==[2,3,2],sequence)
alerts=[json.loads(l) for l in (r/'alerts.jsonl').read_text().splitlines()] if (r/'alerts.jsonl').exists() else []
for name in ('HighTTFC','HighFailureFraction'):
    seen={a['state'] for snap in alerts for a in snap.get('data',{}).get('data',{}).get('alerts',[]) if a['labels']['alertname']==name}
    check('alert_pending_firing:'+name,{'pending','firing'}<=seen,sorted(seen))
    final=alerts[-1]['data']['data']['alerts'] if alerts else []
    check('alert_recovered:'+name,not any(a['labels']['alertname']==name for a in final))
# A long stream must overlap termination of its original Pod, not only the upgrade command.
def as_unix(value):
    from datetime import datetime
    return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
terminations={p['metadata']['name']:as_unix(p['metadata']['deletionTimestamp'])-p['metadata'].get('deletionGracePeriodSeconds',0) for t in timeline for p in t['pods'] if p['metadata'].get('deletionTimestamp')}
check('long_stream_drained_during_termination',any(x['pod'] in terminations and x['sent_unix'] < terminations[x['pod']] < x['sent_unix']+x['duration_seconds'] for x in long))
trace=json.loads((r/'trace-proofs.json').read_text()) if (r/'trace-proofs.json').exists() else {}
check('same_artifact_after_replacement',trace.get('recovery_artifact_matches_manifest') is True)
check('cold_request_joined_to_ready_endpoint',bool(trace.get('broken_ready_endpoint_examples')))
check('no_cold_working_traffic_fixed',trace.get('fixed_observed_requests')==225 and trace.get('fixed_cold_requests')==0)
check('hot_queue_and_spare_capacity_observed',bool(trace.get('hot_waiting_spare_capacity_examples')))
check('no_skew_counterexample_saved',bool(trace.get('no_skew_counterexamples')))
for rep in range(1,4):
    g=groups.get(f'B-counterexample-uniform-{rep}',[])
    check('counterexample_full_repeat:'+str(rep),len(g)==1440 and sorted(Counter(x['pod'] for x in g).values())==[720,720])
fresh=json.loads((r/'checks/replacement-fresh-pod.json').read_text()) if (r/'checks/replacement-fresh-pod.json').exists() else {}
check('fresh_replacement_http_version_and_hash',bool(fresh) and fresh['response']['version']=='A' and fresh['response']['pod']==fresh['replacement'] and fresh['replacement']!=fresh['deleted'] and fresh['ready']['artifact_sha256']==json.loads((root/'artifacts/manifest-A.json').read_text())['model_sha256'])
check('four_figures',all((root/'figures'/f'{i}.svg').exists() for i in range(1,5)))
check('matching_histogram_cohort',(r/'histogram-comparison.json').exists() and json.loads((r/'histogram-comparison.json').read_text()).get('cohort_matches') is True)
prom=json.loads((r/'histogram-promql.json').read_text()) if (r/'histogram-promql.json').exists() else {}
check('promql_actually_evaluated',prom.get('result',{}).get('status')=='success' and bool(prom.get('result',{}).get('data',{}).get('result')))
report={'checks':checks,'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks)}
(r/'verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(f"{report['passed']} passed; {report['failed']} failed")
sys.exit(bool(report['failed']))
