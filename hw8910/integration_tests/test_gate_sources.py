"""Contract fixtures prove the wrapper actually recomputes scores and the HW67 gate."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from gate import HW67, verify_evaluation
from evaluation.solution import score
from release import canonical,digest


class SourceGateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.rows=[dict(id=f'q{i}',input='Question',choices=['A','B'],expected='A',
                        slices=['critical','second','third']) for i in range(50)]
        self.behaviors=[{**row,'id':f'b{i}'} for i,row in enumerate(self.rows[:3])]
        self.outputs=[dict(id=r['id'],raw='{"answer":"A"}',error=None) for r in self.rows+self.behaviors]
        self.protocol=dict(config={'critical_slices':['critical']},golden=self.rows,
            behavioral=self.behaviors,scoring_sha256=digest((HW67/'evaluation/solution.py').read_bytes()),
            backend_sha256=digest((Path(__file__).resolve().parents[1]/'backend.py').read_bytes()))
        self.spec=dict(kind='adapter',name='test',version='1',run_id='training-fixture',uri='fixture://adapter')
        self.manifest=dict(kind='adapter',model_spec=self.spec,model_sha256='a'*64,
            runtime_sha256='b'*64,protocol_sha256=digest(canonical(self.protocol)),
            golden_sha256=digest(canonical(self.rows)))
        full=score(self.rows,self.outputs[:50])
        full.update(status='complete',model=self.spec,behavioral=score(self.behaviors,self.outputs[50:]),
            protocol_sha256=self.manifest['protocol_sha256'],golden_sha256=self.manifest['golden_sha256'],
            baseline_spec_sha256='c'*64,model_sha256='a'*64,runtime_sha256='b'*64,
            evaluation_run_id='evaluation-fixture',outputs_sha256=digest(canonical(self.outputs)))
        baseline=copy.deepcopy(full)
        baseline['model']=dict(kind='base',name='base-fixture',revision='f'*40)
        self.documents={'protocol.json':self.protocol,'baseline-eval.json':baseline,'candidate-eval.json':full,
                        'baseline-outputs.json':self.outputs,'candidate-outputs.json':copy.deepcopy(self.outputs)}
        self.report=dict(synthetic=False,passed=True,metrics=full['overall'],evaluation_run_id='evaluation-fixture')
        self.save()

    def save(self):
        self.report['sources']={}
        for name,document in self.documents.items():
            path=self.root/name
            path.write_bytes(canonical(document))
            self.report['sources'][name]=dict(uri=str(path),sha256=digest(path.read_bytes()))

    def test_complete_raw_evidence_passes(self):
        verify_evaluation(self.manifest,self.report)

    def test_changed_source_rejected(self):
        (self.root/'candidate-outputs.json').write_text('[]')
        with self.assertRaisesRegex(ValueError,'source changed'):
            verify_evaluation(self.manifest,self.report)

    def test_rehashed_raw_cannot_hide_stale_scores(self):
        self.documents['candidate-outputs.json'][0]['raw']='{"answer":"B"}'
        full=self.documents['candidate-eval.json']
        full['outputs_sha256']=digest(canonical(self.documents['candidate-outputs.json']))
        self.save()
        with self.assertRaisesRegex(ValueError,'Metrics differ'):
            verify_evaluation(self.manifest,self.report)

    def test_fabricated_pass_cannot_override_full_quality_gate(self):
        outputs=self.documents['candidate-outputs.json']
        outputs[0]['raw']='{"answer":"B"}'
        full=self.documents['candidate-eval.json']
        full.update(score(self.rows,outputs[:50]))
        full['outputs_sha256']=digest(canonical(outputs))
        self.report['metrics']=full['overall']
        self.save()
        with self.assertRaisesRegex(ValueError,'Регрессия общей accuracy'):
            verify_evaluation(self.manifest,self.report)

    def test_runtime_binding_cannot_be_relabelled(self):
        self.manifest['runtime_sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'binding mismatch'):
            verify_evaluation(self.manifest,self.report)
