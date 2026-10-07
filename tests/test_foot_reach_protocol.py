"""Exercise the evaluator's actual trial loop without starting Isaac Sim."""
import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from scripts.eval_foot_reach import write_reports


class ProtocolTests(unittest.TestCase):
    def rollout(self, fail_at=None, count=1, continue_on_instability=False):
        source = Path('scripts/eval_foot_reach.py').read_text()
        tree = ast.parse(source)
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                    and ast.unparse(node.target) == '(point, target)')
        commands = []
        anchor = np.zeros(3)
        target = np.array([1., 0., 0.])

        def step(command):
            commands.append(command.copy())
            return dict(target=command, actual=anchor, error=float(np.linalg.norm(command)),
                        root_drift=0., support_drift=0., hand_drift=0.,
                        failure='tilt' if len(commands) == fail_at else ''), None

        with tempfile.TemporaryDirectory() as directory:
            scope = dict(np=np, args=SimpleNamespace(continue_on_instability=continue_on_instability, repeats=1, settle=1., reach=2., hold=3., output=Path(directory)),
                         dt=1., targets=[target * (i+1) for i in range(count)], total=0, anchor=anchor, reset=Mock(), step=step,
                         render_frame=Mock(), csv_writer=Mock(), f=Mock(), summaries=[],
                         write_reports=write_reports, json=SimpleNamespace(dumps=lambda _: ''), print=Mock())
            exec(compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), '<trial>', 'exec'), scope)
        scope['reset'].assert_not_called()
        self.results = scope['summaries']
        return commands, scope['summaries'][0]

    def test_step_target_and_final_window_with_large_error_still_successful(self):
        commands, result = self.rollout()
        np.testing.assert_array_equal(commands, [[0, 0, 0]] + [[1, 0, 0]] * 5)
        self.assertEqual(result['hold_samples'], 3)
        self.assertEqual(result['mean_error_m'], 1.)
        self.assertTrue(result['success'])
        self.assertNotIn('within_threshold_fraction', result)

    def test_instability_during_reaching_fails_without_accuracy_samples(self):
        commands, result = self.rollout(fail_at=2)
        self.assertEqual(len(commands), 2)
        self.assertFalse(result['success'])
        self.assertFalse(result['completed_hold'])
        self.assertIsNone(result['mean_error_m'])

    def test_twenty_targets_without_reset_or_repeated_settle(self):
        commands, _ = self.rollout(count=20)
        expected = [[0, 0, 0]] + [[i, 0, 0] for i in range(1, 21) for _ in range(5)]
        np.testing.assert_array_equal(commands, expected)
        self.assertEqual(len(self.results), 20)
        self.assertTrue(all(r['hold_samples'] == 3 for r in self.results))

    def test_failure_stops_remaining_targets(self):
        commands, _ = self.rollout(fail_at=8, count=20)
        self.assertEqual(len(commands), 8)
        self.assertEqual(len(self.results), 2)
        self.assertFalse(self.results[-1]['success'])

    def test_continue_records_failure_but_completes_all_targets(self):
        commands, result = self.rollout(fail_at=2, count=20, continue_on_instability=True)
        self.assertEqual(len(commands), 101)
        self.assertEqual(len(self.results), 20)
        self.assertFalse(result['success'])
        self.assertEqual(result['failure'], 'tilt')
        self.assertTrue(result['completed_hold'])
        self.assertTrue(all(r['hold_samples'] == 3 for r in self.results))

    def test_partial_hold_is_failure(self):
        _, result = self.rollout(fail_at=5)
        self.assertEqual(result['hold_samples'], 2)
        self.assertFalse(result['success'])
        self.assertFalse(result['completed_hold'])


if __name__ == '__main__':
    unittest.main()
