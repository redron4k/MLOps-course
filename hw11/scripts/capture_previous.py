"""Keep previous-container logs before a crash-restarted Pod is replaced."""
import json
from pathlib import Path
import subprocess
import time
root=Path(__file__).resolve().parents[1]
k=['kubectl','--kubeconfig',str(root/'.tools/kubeconfig'),'-n','hw11']
end=time.time()+300
seen=set()
while time.time()<end:
 p=subprocess.run(k+['get','pods','-l','app=api','-o','json'],text=True,capture_output=True)
 if p.returncode==0:
  for pod in json.loads(p.stdout)['items']:
   n=pod['metadata']['name']
   for c in pod.get('status',{}).get('containerStatuses',[]):
    key=(n,c.get('restartCount',0))
    if key[1]>0 and key not in seen:
     logs=subprocess.run(k+['logs',n,'--previous','--timestamps'],capture_output=True,text=True)
     if logs.returncode==0:
      out=root/'results/pod-logs'/f'{n}-previous-{key[1]}.log';out.parent.mkdir(exist_ok=True);out.write_text(logs.stdout);seen.add(key);print('Saved',out.name,flush=True)
 time.sleep(3)
