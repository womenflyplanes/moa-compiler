
"""Regression tests for resource-constrained MoA Omega planning.

Tests heterogeneous memory capacities, unequal ownership,
idle machine components, storage invariants, infeasible
partitions, invalid inputs, and corrupted certificates.

Also checks agreement with the existing attention storage model.
"""

import unittest
from dataclasses import replace

from moa_machine import Machine, Work

from moa_typed_ir import (
    specification,
    stable_online_fold,
    gamma_lower,
    schedule_omega,
)

from moa_attention_resource_planner import (
    modeled_attention_bytes,
    plan_attention_omega,
)

from moa_attention_storage import derive_attention_storage


class TestAttentionResourcePlanner(unittest.TestCase):

    def setUp(self):
        self.expr = gamma_lower(
            stable_online_fold(specification())
        )
        self.work = Work(7, 5, 4, 3)
        self.machine = Machine((2, 3))

    def test_heterogeneous_capacities(self):
        capacities = tuple(
            modeled_attention_bytes(self.work, q)
            for q in (1, 2, 1, 1, 1, 1)
        )

        plan = plan_attention_omega(
            self.expr,
            self.work,
            self.machine,
            capacities,
        )

        self.assertTrue(plan.verify())

        self.assertEqual(
            plan.query_counts,
            (1, 2, 1, 1, 1, 1),
        )

        self.assertEqual(
            plan.intervals,
            (
                (0, 1),
                (1, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
            ),
        )

    def test_tau_output_preserved(self):
        capacities = (
            modeled_attention_bytes(self.work, 7),
        ) * self.machine.tau

        plan = plan_attention_omega(
            self.expr,
            self.work,
            self.machine,
            capacities,
        )

        self.assertEqual(
            sum(plan.query_counts),
            self.work.n,
        )

        self.assertEqual(
            sum(plan.certificate.output_components),
            self.work.n * self.work.dv,
        )

    def test_idle_machine_components(self):
        capacities = (
            modeled_attention_bytes(self.work, 7),
        ) * self.machine.tau

        plan = plan_attention_omega(
            self.expr,
            self.work,
            self.machine,
            capacities,
        )

        self.assertEqual(
            plan.query_counts,
            (7, 0, 0, 0, 0, 0),
        )

        self.assertEqual(
            plan.query_counts.count(0),
            5,
        )

        self.assertTrue(plan.verify())

    def test_insufficient_capacity_rejected(self):
        idle = modeled_attention_bytes(self.work, 0)

        with self.assertRaises(ValueError):
            plan_attention_omega(
                self.expr,
                self.work,
                self.machine,
                (idle,) * self.machine.tau,
            )

    def test_replication_capacity_rejected(self):
        idle = modeled_attention_bytes(self.work, 0)

        capacities = (
            idle - 1,
        ) + (idle + 1000,) * 5

        with self.assertRaises(ValueError):
            plan_attention_omega(
                self.expr,
                self.work,
                self.machine,
                capacities,
            )

    def test_wrong_capacity_count_rejected(self):
        with self.assertRaises(ValueError):
            plan_attention_omega(
                self.expr,
                self.work,
                self.machine,
                (10000,),
            )

    def test_invalid_capacity_rejected(self):
        with self.assertRaises(ValueError):
            plan_attention_omega(
                self.expr,
                self.work,
                self.machine,
                (True,) * self.machine.tau,
            )

    def test_corrupted_storage_rejected(self):
        capacities = (
            modeled_attention_bytes(self.work, 7),
        ) * self.machine.tau

        plan = plan_attention_omega(
            self.expr,
            self.work,
            self.machine,
            capacities,
        )

        corrupted = replace(
            plan,
            modeled_bytes=tuple(
                b + 8 for b in plan.modeled_bytes
            ),
        )

        with self.assertRaises(ValueError):
            corrupted.verify()

    def test_corrupted_capacity_rejected(self):
        capacities = (
            modeled_attention_bytes(self.work, 7),
        ) * self.machine.tau

        plan = plan_attention_omega(
            self.expr,
            self.work,
            self.machine,
            capacities,
        )

        corrupted = replace(
            plan,
            capacities=(1,) + plan.capacities[1:],
        )

        with self.assertRaises(ValueError):
            corrupted.verify()

    def test_storage_matches_existing_model(self):
        canonical = schedule_omega(
            self.expr,
            self.work,
            self.machine,
        )

        storage = derive_attention_storage(canonical)

        expected = tuple(
            modeled_attention_bytes(self.work, end - begin)
            for begin, end in canonical.schedule.intervals
        )

        actual = tuple(
            rank.modeled_bytes
            for rank in storage.ranks
        )

        self.assertEqual(actual, expected)

    def test_deterministic_planning(self):
        capacities = tuple(
            modeled_attention_bytes(self.work, q)
            for q in (1, 2, 1, 1, 1, 1)
        )

        first = plan_attention_omega(
            self.expr,
            self.work,
            self.machine,
            capacities,
        )

        second = plan_attention_omega(
            self.expr,
            self.work,
            self.machine,
            capacities,
        )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
