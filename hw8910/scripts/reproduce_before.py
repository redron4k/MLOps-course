"""Run the SAME seven regression scenarios on an isolated original checkout."""
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

PROJECT=Path(__file__).resolve().parents[1]
archive=subprocess.check_output(['git','archive','1837801','hw8910'],cwd=PROJECT.parent)
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(root,filter='data')
    original=root/'hw8910'
    shutil.copytree(PROJECT/'integration_tests',original/'integration_tests')
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','integration_tests',
                           '-p','test_incidents.py','-v'],cwd=original,capture_output=True,text=True)
    (PROJECT/'evidence/before.log').write_text(result.stdout+result.stderr)
    print('Original regression exit code:',result.returncode)
    raise SystemExit(result.returncode)
