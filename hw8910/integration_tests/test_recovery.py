"""Additional failure windows beyond the seven original incident regressions."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import lab
import release
from service import ServiceState, StubBackend


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'workspace'
        lab.initialize(self.root)
        self.manifest = lab.read(self.root/'manifest-B.json')
        self.report = lab.read(self.root/'report-B.json')

    def test_pointer_journal_crash_recovers_once(self):
        with self.assertRaisesRegex(RuntimeError, 'after pointer'):
            release.publish_release(self.manifest,self.report,self.root,fail_after_pointer=True)
        self.assertTrue((self.root/'publication-pending.json').exists())
        self.assertEqual(lab.read(self.root/'accepted.json')['model_id'],'fixture-B')
        release.publish_release(self.manifest,self.report,self.root)
        release.publish_release(self.manifest,self.report,self.root)
        events = (self.root/'publications.jsonl').read_text().splitlines()
        self.assertEqual(len(events),1)
        self.assertFalse((self.root/'publication-pending.json').exists())

    def test_pointer_journal_recovery_in_new_process(self):
        project=Path(__file__).resolve().parents[1]
        command=[sys.executable,str(project/'lab.py'),'--workspace',str(self.root),
                 'run','--engine',os.getenv('FLOW_ENGINE','stdlib')]
        failed=subprocess.run([*command,'--fail-after-pointer'],capture_output=True,text=True,timeout=120)
        print('AFTER_POINTER_FAILURE',failed.returncode,failed.stdout,failed.stderr)
        self.assertNotEqual(failed.returncode,0)
        for _ in range(2):
            result=subprocess.run(command,capture_output=True,text=True,timeout=120)
            print('RECOVERY_ATTEMPT',result.returncode,result.stdout,result.stderr)
            self.assertEqual(result.returncode,0)
        self.assertEqual(len((self.root/'publications.jsonl').read_text().splitlines()),1)
        attempts=[json.loads(line) for line in (self.root/'attempts.jsonl').read_text().splitlines()]
        self.assertEqual(len({a['attempt_id'] for a in attempts}),3)
        self.assertEqual([a['status'] for a in attempts],['failed','complete','complete'])

    def test_new_evaluation_or_runtime_is_a_distinct_release(self):
        release.publish_release(self.manifest,self.report,self.root)
        report = {**self.report,'evaluation_run_id':'second-evaluation'}
        release.publish_release(self.manifest,report,self.root)
        manifest = copy.deepcopy(self.manifest)
        manifest['runtime']['dtype']='float16'
        manifest['runtime_sha256']=release.digest(release.canonical(manifest['runtime']))
        report = {**report,'runtime_sha256':manifest['runtime_sha256']}
        release.publish_release(manifest,report,self.root)
        events = [json.loads(line) for line in (self.root/'publications.jsonl').read_text().splitlines()]
        self.assertEqual(len({event['release_id'] for event in events}),3)

    def test_real_file_tampering_blocks_publish_and_load(self):
        before = (self.root/'accepted.json').read_bytes()
        (self.root/'model-B.txt').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'Artifact changed'):
            release.publish_release(self.manifest,self.report,self.root)
        self.assertEqual((self.root/'accepted.json').read_bytes(),before)
        release.atomic_bytes(self.root/'accepted.json',release.canonical(self.manifest))
        state = ServiceState(self.root,0,StubBackend)
        state.loader.join(5)
        self.assertFalse(state.ready)
        self.assertIsNone(state.backend)
        self.assertIn('Artifact changed',state.error)

    def test_fsync_failure_preserves_pointer_and_cleans_temp(self):
        before = (self.root/'accepted.json').read_bytes()
        with patch('os.fsync',side_effect=OSError('injected write error')):
            with self.assertRaises(OSError):
                release.publish(self.manifest,self.report,self.root/'accepted.json')
        self.assertEqual((self.root/'accepted.json').read_bytes(),before)
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_backend_load_failure_never_sets_ready(self):
        def broken(manifest,root):
            raise ValueError('injected load failure')
        state = ServiceState(self.root,0,broken)
        state.loader.join(5)
        self.assertFalse(state.ready)
        self.assertFalse(state.warmup_finished.is_set())
        self.assertIn('injected load failure',state.error)

    def test_synthetic_report_cannot_authorize_real_weights(self):
        manifest={**self.manifest,'kind':'adapter'}
        with self.assertRaisesRegex(ValueError,'Synthetic evaluation'):
            release.publish_release(manifest,self.report,self.root)

    def test_real_manifest_cannot_accidentally_use_stub(self):
        manifest={**self.manifest,'kind':'base'}
        release.atomic_bytes(self.root/'accepted.json',release.canonical(manifest))
        with self.assertRaisesRegex(ValueError,'explicit real backend'):
            ServiceState(self.root,0,StubBackend)


if __name__=='__main__':
    unittest.main(verbosity=2)
