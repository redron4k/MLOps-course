"""Deliberately reintroduce a truthiness defect in a temporary copy, never in cwd."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='hw8910-negative-') as directory:
    temp=Path(directory)
    source=(root/'release.py').read_text()
    original="report['passed'] is not True"
    assert original in source
    (temp/'release.py').write_text(source.replace(original,"not report['passed']",1))
    shutil.copytree(root/'tests',temp/'tests')
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],
                          cwd=temp,capture_output=True,text=True)
    log=f'Injected mutation: {original} -> not report[passed]\n'+result.stdout+result.stderr
    (root/'evidence/intentional-negative.log').write_text(log)
    print(log)
    raise SystemExit(result.returncode)
