"""Revalidate preserved real evidence without weights, training or MLflow server."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gate import verify_evaluation
from release import validate_release

parser=argparse.ArgumentParser()
parser.add_argument('--workspace',type=Path,default=Path('.real'))
args=parser.parse_args()
manifest=json.loads((args.workspace/'manifest-candidate.json').read_text())
report=json.loads((args.workspace/'report-candidate.json').read_text())
try:
    # Check sources and full gate even when the compact report already says FAIL.
    verify_evaluation(manifest,report)
    validate_release(manifest,report)
except ValueError as exc:
    print('REJECTED:',str(exc))
    raise SystemExit(1)
print('PASS: full gate and metadata agree')
