"""Record the exact backend/server bytes of running containers and source tree."""
import hashlib
import json
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
k=['kubectl','--kubeconfig',str(root/'.tools/kubeconfig'),'-n','hw11']
pods=json.loads(subprocess.check_output(k+['get','pods','-l','app=api','-o','json']))['items']
script="import hashlib,json,pathlib;print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in ['server.py','backend.py','client.py','requirements-runtime.lock']}))"
out={'source':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['server.py','backend.py','client.py','requirements-runtime.lock']},'pods':{}}
for p in pods:
 if p['metadata'].get('deletionTimestamp'):continue
 name=p['metadata']['name']
 out['pods'][name]={'imageID':p['status']['containerStatuses'][0]['imageID'],'sha256':json.loads(subprocess.check_output(k+['exec',name,'--','python','-c',script]))}
out['backend_matches_source']=all(p['sha256']['backend.py']==out['source']['backend.py'] and p['sha256']['server.py']==out['source']['server.py'] for p in out['pods'].values())
(root/'results/runtime-source-hashes.json').write_text(json.dumps(out,indent=2)+'\n')
print('Backend and server match submitted source:',out['backend_matches_source'])
