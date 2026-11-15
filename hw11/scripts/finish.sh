#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
.venv/bin/python scripts/archive_a.py
.venv/bin/python scripts/run_lab.py --phases A
.venv/bin/python scripts/counterexample.py
.venv/bin/python scripts/calibrate.py
.venv/bin/python scripts/environment.py after
.venv/bin/python scripts/runtime_hashes.py
MPLCONFIGDIR=.tools/matplotlib .venv/bin/python scripts/analyze.py
.venv/bin/python scripts/proofs.py
.venv/bin/python scripts/verify_evidence.py > results/verification-output.txt
.venv/bin/python scripts/checklist.py
