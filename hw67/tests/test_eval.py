"""Контракт оценки на синтетических ответах. Не заменяет оценку реальной модели."""
import copy
import unittest
from unittest.mock import patch

from evaluation.contract import validate_golden
from evaluation.solution import quality_gate, score


def rows(n=50):
    return [dict(id=f'q{i}', group_id=f'g{i}', input=f'Учебный вопрос {i}', expected='B',
                 choices=['A', 'B', 'REFUSE'], slices=[f's{i % 3}'], source='synthetic fixture') for i in range(n)]


def outputs(items, correct=None):
    correct = len(items) if correct is None else correct
    return [dict(id=r['id'], raw='{"answer":"B"}' if i < correct else '{"answer":"A"}', error=None)
            for i, r in enumerate(items)]


def report(correct=50):
    items = rows()
    result = score(items, outputs(items, correct))
    result.update(status='complete', model={'kind': 'fixture'}, protocol_sha256='p',
                  golden_sha256='g', baseline_spec_sha256='b',
                  behavioral=score(rows(3), outputs(rows(3))))
    return result


class ScoreContract(unittest.TestCase):
    def test_accuracy_uses_all_questions(self):
        items = rows(3)
        out = outputs(items)
        out[1]['raw'] = 'Правильный ответ B'
        out[2].update(raw='', error='timeout')
        result = score(items, out)
        self.assertEqual(result['overall']['n'], 3)
        self.assertEqual(result['overall']['correct'], 1)
        self.assertAlmostEqual(result['overall']['accuracy'], 1 / 3)
        self.assertEqual(result['overall']['errors'], 1)

    def test_strict_json_and_label(self):
        bad = ['{"answer": "B", "extra": 1}', '{"answer": 2}', '{"answer": "C"}', '```json\n{"answer":"B"}\n```', '', 'null']
        for raw in bad:
            with self.subTest(raw=raw):
                self.assertEqual(score(rows(1), [dict(id='q0', raw=raw, error=None)])['overall']['valid'], 0)

    def test_wrong_label_is_valid_but_incorrect(self):
        result = score(rows(1), outputs(rows(1), 0))
        self.assertEqual((result['overall']['valid'], result['overall']['correct']), (1, 0))

    def test_slice_counts_and_per_item(self):
        result = score(rows(), outputs(rows()))
        self.assertEqual([result['slices'][s]['n'] for s in ['s0', 's1', 's2']], [17, 17, 16])
        self.assertEqual(len(result['per_item']), 50)

    def test_reordered_outputs_are_matched_by_id(self):
        self.assertEqual(score(rows(), outputs(rows()))['overall'], score(rows(), list(reversed(outputs(rows()))))['overall'])

    def test_missing_duplicate_empty_rejected(self):
        for items, out in [(rows(2), outputs(rows(1))), (rows(2), [outputs(rows(1))[0]] * 2), ([], [])]:
            with self.assertRaises(ValueError):
                score(items, out)


class GateContract(unittest.TestCase):
    def test_equal_passes(self):
        self.assertIs(quality_gate(report(), report(), ['s0'])['passed'], True)

    def test_regression_rejected(self):
        with self.assertRaises(ValueError):
            quality_gate(report(), report(49), ['s0'])

    def test_improvement_passes(self):
        self.assertTrue(quality_gate(report(49), report(), ['s0'])['passed'])

    def test_critical_regression_despite_average_growth(self):
        a, b = report(30), report(40)
        # Только fixture для ветки gate; реальные отчёты обвязка пересчитывает из raw.
        a['slices']['s0'].update(correct=17, accuracy=1.0)
        b['slices']['s0'].update(correct=16, accuracy=16 / 17)
        with self.assertRaises(ValueError):
            quality_gate(a, b, ['s0'])

    def test_missing_nan_empty_error_and_incompatible_rejected(self):
        variants = []
        for key in ['protocol_sha256', 'golden_sha256', 'baseline_spec_sha256']:
            x = report(); x[key] = 'different'; variants.append(x)
        x = report(); del x['overall']; variants.append(x)
        x = report(); x['overall']['accuracy'] = float('nan'); variants.append(x)
        x = report(); x['overall']['n'] = 0; variants.append(x)
        x = report(); x['overall']['errors'] = 1; variants.append(x)
        x = report(); x['status'] = 'failed'; variants.append(x)
        x = report(); x['per_item'][0]['id'] = 'not-the-same-id'; variants.append(x)
        x = report(); x['overall']['correct'] = 49; variants.append(x)
        for candidate in variants:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                quality_gate(report(), candidate, ['s0'])

    def test_behavioral_failure_rejected(self):
        x = report(); x['behavioral']['overall'].update(correct=2, accuracy=2 / 3)
        with self.assertRaises(ValueError):
            quality_gate(report(), x, ['s0'])

    def test_missing_critical_slice_rejected(self):
        with self.assertRaises(ValueError):
            quality_gate(report(), report(), ['missing'])


class LeakageContract(unittest.TestCase):
    def setUp(self):
        self.golden = rows()
        self.training = [dict(id='train-1', group_id='train-group', input='Только train')]
        self.behavior = [dict(id=f'b{i}', group_id=f'b{i}', input=f'Behavior {i}', choices=['B'],
                              expected='B', slices=['behavior'], source='fixture', type=kind)
                         for i, kind in enumerate(['format', 'paraphrase', 'refusal'])]

    def test_clean_data(self):
        self.assertEqual(validate_golden(self.golden, self.training, self.behavior)['exact_or_group_intersections'], 0)

    def test_exact_overlap(self):
        self.training[0]['input'] = '  УЧЕБНЫЙ   вопрос 0 '
        with self.assertRaises(ValueError):
            validate_golden(self.golden, self.training, self.behavior)

    def test_group_overlap(self):
        self.training[0]['group_id'] = 'g0'
        with self.assertRaises(ValueError):
            validate_golden(self.golden, self.training, self.behavior)

    def test_duplicate_golden_questions_rejected(self):
        self.golden[1]['input'] = self.golden[0]['input']
        with self.assertRaises(ValueError):
            validate_golden(self.golden, self.training, self.behavior)

    def test_too_small_or_missing_behavior(self):
        with self.assertRaises(ValueError):
            validate_golden(self.golden[:10], self.training, self.behavior)
        with self.assertRaises(ValueError):
            validate_golden(self.golden, self.training, self.behavior[:2])


class PromotionContract(unittest.TestCase):
    def test_no_registry_mutation_when_gate_rejects(self):
        from evaluation import cli
        with patch.object(cli, 'gate', side_effect=ValueError('quality failed')), patch.object(cli, 'tracking') as tracking:
            with self.assertRaises(ValueError):
                cli.release()
            tracking.assert_not_called()


if __name__ == '__main__':
    unittest.main()
