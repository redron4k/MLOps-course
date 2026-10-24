#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
uv run python tests/check.py
status=$?
uv run python - <<'PY'
import hashlib
from pathlib import Path
h = hashlib.sha256()
for p in sorted(Path('tests').glob('*')):
    if p.is_file():
        h.update(p.name.encode())
        h.update(p.read_bytes())
print('Отпечаток всех файлов tests/:', h.hexdigest()[:12])
PY
exit "$status"
