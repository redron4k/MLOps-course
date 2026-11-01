"""Verify tracked evidence hashes without local weights or the original MLflow server."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from release import canonical,digest
from gate import verify_evaluation

root=ROOT/'evidence/real-release'
manifest=json.loads((root/'manifest-candidate.json').read_text())
report=json.loads((root/'report-candidate.json').read_text())
# Relocate source URIs in-memory only. Original reports and their bytes stay intact.
for name,source in report['sources'].items():
    source['uri']=str(root/'evaluation'/name)
try:
    verify_evaluation(manifest,report)
    expected_pass=True
except ValueError as exc:
    if str(exc)!='Регрессия общей accuracy':
        raise
    print('Verified honest rejection:',exc)
    expected_pass=False
assert report['passed'] is expected_pass
for role in ('baseline','candidate'):
    m=json.loads((root/f'manifest-{role}.json').read_text())
    full=json.loads((root/f'evaluation/{role}-eval.json').read_text())
    assert digest(canonical(m['runtime']))==m['runtime_sha256']==full['runtime_sha256']
    assert digest(canonical(m['file_hashes']))==m['model_sha256']==full['model_sha256']
decision=json.loads((root/'release-decision.json').read_text())
assert decision['accepted_before_sha256']==decision['accepted_after_sha256']
assert decision['accepted_after_sha256']==digest((root/'accepted.json').read_bytes())
smoke=json.loads((ROOT/'evidence/real-api/smoke.json').read_text())
for result in smoke:
    assert result['response']['model_id']==decision['served_model_id']
    assert result['response']['runtime_sha256']==json.loads((root/'accepted.json').read_text())['runtime_sha256']
print('PASS: sources, runtime, model fingerprints, pointer preservation and real API agree')
