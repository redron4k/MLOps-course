#!/bin/sh
# Reproduce a complete submission without discarding any previous evidence.
set -eu
cd "$(dirname "$0")/.."
docker info --format 'Docker={{.ServerVersion}} CPU={{.NCPU}} RAM={{.MemTotal}}'
uv sync --frozen
[ -x .tools/kind ] && [ -x .tools/helm ] || ./scripts/bootstrap.sh
.venv/bin/python - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,shutil
r=Path('results');r.mkdir(exist_ok=True)
existing=[p for p in r.iterdir() if p.name!='archives']
if existing:
 dest=r/'archives'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');dest.mkdir(parents=True)
 for p in existing:shutil.move(str(p),str(dest/p.name))
free=shutil.disk_usage('.').free
(r/'preflight.json').write_text(json.dumps({'free_bytes_before_cluster':free,'unix_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
if free<6*2**30:raise SystemExit('Need at least 6 GiB free for the lab')
PY
if ! .tools/kind get clusters | rg -qx hw11; then
 .tools/kind create cluster --name hw11 --image kindest/node:v1.32.2 --kubeconfig .tools/kubeconfig --wait 120s
fi
.tools/kind export kubeconfig --name hw11 --kubeconfig .tools/kubeconfig
docker build -f Dockerfile -t hw11-api:v2 ..
./scripts/setup.sh
.venv/bin/python scripts/environment.py before
.venv/bin/python scripts/network_snapshot.py
.venv/bin/python -m pytest -q > results/tests.txt 2>&1 || { cat results/tests.txt; exit 1; }
cat results/tests.txt
.tools/helm lint chart > results/helm-lint.txt 2>&1 || { cat results/helm-lint.txt; exit 1; }
cat results/helm-lint.txt
.venv/bin/python scripts/run_lab.py
.venv/bin/python scripts/counterexample.py
.venv/bin/python scripts/calibrate.py
.venv/bin/python scripts/environment.py after
.venv/bin/python scripts/runtime_hashes.py
MPLCONFIGDIR=.tools/matplotlib .venv/bin/python scripts/analyze.py
.venv/bin/python scripts/proofs.py
.venv/bin/python scripts/verify_evidence.py
.venv/bin/python scripts/checklist.py
