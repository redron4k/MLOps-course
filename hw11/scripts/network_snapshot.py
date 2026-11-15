"""Read actual networking implementation and affinity rules of the isolated kind node."""
import json
from pathlib import Path
import subprocess
import time
import re
root=Path(__file__).resolve().parents[1]
def run(*args):
 p=subprocess.run(args,text=True,capture_output=True)
 return {'code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
k=['kubectl','--kubeconfig',str(root/'.tools/kubeconfig')]
out=dict(unix=time.time(),iptables_version=run('docker','exec','hw11-control-plane','iptables','--version'),
    nat=run('docker','exec','hw11-control-plane','iptables-save','-t','nat'),
    system_pods=run(*k,'-n','kube-system','get','pods','-o','json'),
    network_links=run('docker','exec','hw11-control-plane','ip','-brief','address'))
nat=out['nat']['stdout']
chains={m.group(1) for line in nat.splitlines() if 'hw11/api-sticky:' in line and 'cluster IP' in line for m in [re.search(r'-j (KUBE-SVC-\S+)',line)] if m}
recent={m.group(1) for line in nat.splitlines() if any(line.startswith('-A '+chain+' ') for chain in chains) for m in [re.search(r'--name (\S+)',line)] if m}
out['affinity_entries']={name:run('docker','exec','hw11-control-plane','cat','/proc/net/xt_recent/'+name) for name in recent}
(root/'results/network-implementation.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['iptables_version']['stdout'])
