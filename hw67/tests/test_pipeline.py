"""Интеграция на синтетических ответах: реальное обучение здесь не выполняется."""
import copy
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evaluation import cli
from evaluation.contract import read_json, write_json


class PipelineContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_cwd = Path.cwd()
        os.chdir(self.temp.name)
        self.stack = ExitStack()
        self.spec = dict(kind='adapter', name='fixture-model', version='2', run_id='fixture-run', uri='models:/fixture-model/2')
        self.base = dict(kind='base', name='fixture-base', revision='a' * 40)
        self.manifest = dict(model=dict(name='fixture-base', revision='a' * 40))
        self.alias = '1'
        self.mutations = []
        def set_alias(name, alias, version):
            self.mutations.append((alias, version)); self.alias = str(version)
        self.client = SimpleNamespace(set_registered_model_alias=set_alias,
                                      get_model_version_by_alias=lambda *a: SimpleNamespace(version=self.alias, run_id='fixture-run'))
        self.stack.enter_context(patch.object(cli, 'tracking', return_value=({}, self.client)))
        self.stack.enter_context(patch.object(cli, 'candidate_model', return_value=self.spec))
        self.stack.enter_context(patch.object(cli, 'manifest_for', return_value=(self.manifest, None)))
        self.fail_candidate = False
        def generate(spec, manifest, config, rows):
            out = [dict(id=r['id'], raw='{"answer":"B"}', error=None, latency_seconds=0.0) for r in rows]
            if self.fail_candidate and spec['kind'] == 'adapter':
                out[0]['raw'] = '{"answer":"A"}'
            return out
        self.stack.enter_context(patch.object(cli, 'generate', side_effect=generate))
        fake_mlflow = Mock()
        fake_mlflow.start_run.return_value.__enter__ = Mock(return_value=SimpleNamespace(info=SimpleNamespace(run_id='fixture-eval')))
        fake_mlflow.start_run.return_value.__exit__ = Mock(return_value=False)
        self.stack.enter_context(patch.dict(sys.modules, {'mlflow': fake_mlflow}))
        cfg = dict(golden='data/golden.jsonl', training_index='data/train.jsonl', behavioral='data/behavior.jsonl',
                   critical_slices=['s0'], system='fixture', max_new_tokens=8, enable_thinking=False, seed=42)
        write_json('evaluation/params.json', cfg)
        write_json('evaluation/baseline.json', self.base)
        rows = [dict(id=f'q{i}', group_id=f'g{i}', input=f'Question {i}', expected='B', choices=['A', 'B'],
                     slices=[f's{i % 3}'], source='fixture') for i in range(50)]
        behavior = [dict(id=f'b{i}', group_id=f'b{i}', input=f'Behavior {i}', expected='B', choices=['A', 'B'],
                         slices=['behavior'], source='fixture', type=kind)
                    for i, kind in enumerate(['format', 'paraphrase', 'refusal'])]
        import json
        for name, content in [('golden', rows), ('train', [dict(id='train', group_id='train', input='Train only')]), ('behavior', behavior)]:
            path = Path(f'data/{name}.jsonl'); path.parent.mkdir(exist_ok=True)
            path.write_text('\n'.join(json.dumps(row) for row in content))

    def tearDown(self):
        self.stack.close()
        os.chdir(self.old_cwd)
        self.temp.cleanup()

    def test_pass_release_and_explicit_baseline_update(self):
        cli.evaluate()
        self.assertEqual(self.mutations, [])
        cli.release()
        self.assertEqual(self.mutations, [('champion', '2')])
        cli.update_baseline()
        self.assertEqual(read_json('evaluation/baseline.json'), self.spec)
        self.assertEqual(len(list(Path('evaluation/baseline-history').glob('*.json'))), 1)
        with self.assertRaises(ValueError):
            cli.gate()  # новый baseline требует новой оценки, а не принятия старого отчёта

    def test_refused_candidate_keeps_alias(self):
        self.fail_candidate = True
        cli.evaluate()
        with self.assertRaises(ValueError):
            cli.release()
        self.assertEqual((self.alias, self.mutations), ('1', []))

    def test_protocol_or_raw_changes_rejected(self):
        cli.evaluate()
        raw = read_json('metrics/candidate-outputs.json')
        raw[0]['raw'] = 'changed'
        write_json('metrics/candidate-outputs.json', raw)
        with self.assertRaises(ValueError):
            cli.release()
        self.assertEqual(self.mutations, [])

    def test_negative_check_does_not_mutate_model_or_alias(self):
        cli.evaluate()
        before = read_json('metrics/candidate-outputs.json')
        cli.negative()
        self.assertIs(read_json('metrics/negative-gate.json')['rejected'], True)
        self.assertEqual(read_json('metrics/candidate-outputs.json'), before)
        self.assertEqual(self.mutations, [])

    def test_baseline_init_refuses_overwrite(self):
        with self.assertRaises(ValueError):
            cli.init_baseline()


if __name__ == '__main__':
    unittest.main()
