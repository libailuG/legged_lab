"""CPU regression tests for v3 paired assistance; no Isaac Sim startup required."""

import importlib.util
from pathlib import Path
import unittest

import torch


HELPERS = (Path(__file__).resolve().parents[1] / "legged_lab/tasks/locomotion/amp/config/"
           "g1_assist_exoskeleton_v2_v3/mdp/paired_assist.py")
spec = importlib.util.spec_from_file_location("v3_paired_assist", HELPERS)
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)


class PairedAssistTests(unittest.TestCase):
    def reference(self, velocity, support):
        velocity = torch.tensor(velocity, dtype=torch.float32)
        return paired.support_aware_reference(
            velocity, torch.zeros_like(velocity), torch.tensor(support, dtype=torch.float32),
            10.0, 2.0, 0.04, 1.5, 0.15, 0.8,
        )

    def test_stationary_support_leg_does_not_disable_pair(self):
        gate = paired.shared_motion_gate(torch.tensor([[-1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]), 0.15, 0.8)
        torch.testing.assert_close(gate[:, 0], torch.tensor([1.0, 1.0, 0.0]))

    def test_bounds_balance_slew_and_zero_crossing(self):
        previous = torch.zeros(5, 1)
        targets = torch.tensor([[-100.0], [-10.0], [0.0], [10.0], [100.0]])
        for step in range(100):
            target = targets if step < 40 else -targets
            value = paired.slew_paired_scalar(target, previous, 10.0, 80.0, 0.01)
            torque = paired.paired_torques(value)
            self.assertLessEqual(float(torque.abs().max()), 10.0)
            self.assertLessEqual(float((value - previous).abs().max()), 0.800001)
            self.assertTrue(torch.equal(torque.sum(-1), torch.zeros(5)))
            self.assertTrue(torch.all(previous * value >= 0.0), "Reversal skipped zero")
            previous = value
            if step == 39:
                torch.testing.assert_close(value[:, 0], torch.tensor([-10.0, -10.0, 0.0, 10.0, 10.0]))
        torch.testing.assert_close(previous[:, 0], torch.tensor([10.0, 10.0, 0.0, -10.0, -10.0]))

    def test_supported_lift_and_left_right_symmetry(self):
        result = self.reference([[-1.0, 0.0], [0.0, -1.0]], [[0.0, 1.0], [1.0, 0.0]])
        self.assertLess(float(result[0, 0]), -8.0)
        self.assertGreater(float(result[0, 1]), 8.0)
        torch.testing.assert_close(result[0].flip(0), result[1])
        torch.testing.assert_close(result.sum(-1), torch.zeros(2))

    def test_no_reference_when_unsupported_lowering_or_standing(self):
        result = self.reference(
            [[-1.0, 0.0], [-1.0, -1.0], [1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
        )
        torch.testing.assert_close(result, torch.zeros(4, 2))

    def test_support_requires_load_and_contact_age_then_releases(self):
        contact = torch.zeros(1, 2, dtype=torch.bool)
        age = torch.zeros(1, 2)
        force = torch.tensor([[200.0, 0.0]])
        contact, age, initial = paired.update_support(force, contact, age, .01, 10, 20, 200, .05)
        self.assertTrue(0 < initial[0, 0] < 1)
        for _ in range(4):
            contact, age, support = paired.update_support(force, contact, age, .01, 10, 20, 200, .05)
        torch.testing.assert_close(support, torch.tensor([[1.0, 0.0]]))
        contact, age, support = paired.update_support(torch.zeros_like(force), contact, age, .01, 10, 20, 200, .05)
        self.assertFalse(contact.any())
        torch.testing.assert_close(age, torch.zeros_like(age))
        torch.testing.assert_close(support, torch.zeros_like(support))

    def test_support_hysteresis(self):
        force = torch.tensor([[15.0, 15.0]])
        contact, _, support = paired.update_support(
            force, torch.tensor([[True, False]]), torch.ones(1, 2), .01, 10, 20, 200, .05,
        )
        self.assertTrue(contact[0, 0])
        self.assertFalse(contact[0, 1])
        self.assertEqual(float(support[0, 1]), 0.0)

    def test_press_only_penalized_on_unsupported_leg(self):
        torque = torch.tensor([[-10.0, 10.0], [-10.0, 10.0]])
        support = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        cost = paired.unsupported_press_cost(torque, support, 10.0)
        torch.testing.assert_close(cost, torch.tensor([0.0, 0.5]))

    def test_rest_uses_both_legs_together(self):
        torque = torch.tensor([[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]])
        velocity = torch.tensor([[0.0, 0.0], [-2.0, 0.0], [0.0, -2.0]])
        cost = paired.bilateral_rest_cost(torque, velocity, torch.zeros_like(velocity), 10.0, .5, 3.0)
        self.assertEqual(float(cost[0]), 1.0)
        self.assertLess(float(cost[1]), .06)
        torch.testing.assert_close(cost[1], cost[2])


if __name__ == "__main__":
    unittest.main()
