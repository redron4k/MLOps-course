"""Keep the previous A run before replaying the corrected lifecycle implementation."""
import json
from pathlib import Path
import shutil
r=Path(__file__).resolve().parents[1]/'results'
a=r/'before-drain-fix';a.mkdir(exist_ok=True)
p=r/'requests.jsonl';items=[json.loads(l) for l in p.read_text().splitlines()]
(a/'requests.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items if x['run'].startswith('A-')))
p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items if not x['run'].startswith('A-')))
p=r/'events.jsonl';shutil.copy2(p,a/p.name)
items=[json.loads(l) for l in p.read_text().splitlines()]
remove={'A-warmup-broken','A-warmup-fixed','A_complete','rollout_B_begin','rollout_B_pass','rollback_A_begin','forced_kill'}
p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items if x['event'] not in remove))
for directory in ('checks','summaries'):
 for p in (r/directory).glob('*'):
  if p.name.startswith(('A-','rollback-','interrupted-','replacement-','unready-')):
   (a/directory).mkdir(exist_ok=True);shutil.move(str(p),str(a/directory/p.name))
print('Archived previous A; B and C unchanged')
