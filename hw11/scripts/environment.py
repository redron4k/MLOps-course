"""Read-only environment inventory; run before and after the lab."""
from pathlib import Path
import json
import platform
import shutil
import subprocess
import sys
import time
root=Path(__file__).resolve().parents[1]
def run(*args):
    p=subprocess.run(args,text=True,capture_output=True)
    return {'code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
record={'unix':time.time(),'platform':platform.platform(),'python':sys.version,'disk':dict(zip(('total','used','free'),shutil.disk_usage(root))),
 'docker':run('docker','info','--format','{{json .}}'), 'docker_df':run('docker','system','df','-v'),
 'cluster':run('kubectl','--kubeconfig',str(root/'.tools/kubeconfig'),'version','-o','json'),
 'node_limits':run('docker','inspect','hw11-control-plane','--format','{{json .HostConfig}}'),
 'node_size':run('docker','inspect','--size','hw11-control-plane','--format','{{json .SizeRw}}'),
 'node_storage_KiB':run('docker','exec','hw11-control-plane','du','-sk','/var/lib/containerd','/var/log','/var/lib/kubelet'),
 'images':run('docker','image','inspect','hw11-api:v1','hw11-api:v2','kindest/node:v1.32.2'),
 'pods':run('kubectl','--kubeconfig',str(root/'.tools/kubeconfig'),'-n','hw11','get','pods','-o','json'),
 'git':run('git','rev-parse','HEAD'),
 'workspace_bytes':sum(p.stat().st_size for p in (root/'results').rglob('*') if p.is_file())}
(root/'results'/f'environment-{sys.argv[1] if len(sys.argv)>1 else "after"}.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'disk_free_GiB':record['disk']['free']/2**30,'evidence_MiB':record['workspace_bytes']/2**20}))
