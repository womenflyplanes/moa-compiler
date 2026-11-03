
"""Adversarial tests for MoA Omega communication audits.

Verifies ownership-to-communication consistency independently
of the resource-derived communication certificate's own checks.

All tests are additive. No existing modules are modified.
"""

from dataclasses import replace
import unittest

from moa_machine import Machine, Work

from moa_typed_ir import (
    specification,
    stable_online_fold,
    gamma_lower,
)

from moa_attention_resource_planner import (
    modeled_attention_bytes,
    plan_attention_omega,
)

from moa_attention_resource_communication import (
    derive_resource_communication,
)

from moa_attention_communication_audit import (
    audit_resource_communication,
    audit_resource_communication_plan,
)


def make_plan(work, shape, counts):
    """Construct a verified resource-constrained Omega plan."""

    machine = Machine(shape)

    if len(counts) != machine.tau:
        raise ValueError("machine component count mismatch")

    if sum(counts) != work.n:
        raise ValueError("query count mismatch")

    expr = gamma_lower(
        stable_online_fold(specification())
    )

    capacities = tuple(
        modeled_attention_bytes(work, q)
        for q in counts
    )

    plan = plan_attention_omega(
        expr,
        work,
        machine,
        capacities,
    )

    if plan.query_counts != counts:
        raise AssertionError("unexpected Omega ownership")

    return plan


class TestCommunicationAudit(unittest.TestCase):

    def make_certificate(self):
        plan = make_plan(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
        )

        cert = derive_resource_communication(plan)

        return plan, cert

    def test_valid_heterogeneous_ownership(self):
        plan, cert = self.make_certificate()

        self.assertTrue(
            audit_resource_communication(cert, 3)
        )

        self.assertTrue(
            audit_resource_communication_plan(plan, cert)
        )

    def test_valid_idle_components(self):
        plan = make_plan(
            Work(3, 6, 2, 4),
            (2, 3),
            (3, 0, 0, 0, 0, 0),
        )

        cert = derive_resource_communication(plan)

        self.assertTrue(
            audit_resource_communication_plan(plan, cert)
        )

    def test_valid_unequal_ownership(self):
        plan = make_plan(
            Work(11, 7, 8, 4),
            (3, 2),
            (1, 2, 3, 1, 2, 2),
        )

        cert = derive_resource_communication(plan)

        self.assertTrue(
            audit_resource_communication_plan(plan, cert)
        )

    def test_valid_single_component(self):
        plan = make_plan(
            Work(3, 6, 2, 1),
            (1,),
            (3,),
        )

        cert = derive_resource_communication(plan)

        self.assertTrue(
            audit_resource_communication_plan(plan, cert)
        )

        self.assertEqual(
            cert.aggregate_nonlocal_components,
            0,
        )

    def test_wrong_partition_rejected_by_plan_audit(self):
        """Internally valid metadata must match the original plan."""

        plan, cert = self.make_certificate()

        # Different ownership of the same seven queries.
        wrong_intervals = (
            (0, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
        )

        wrong_counts = (6, 3, 3, 3, 3, 3)
        wrong_displacements = (0, 6, 9, 12, 15, 18)

        wrong_nonlocal = tuple(
            cert.output_components - c
            for c in wrong_counts
        )

        corrupted = replace(
            cert,
            intervals=wrong_intervals,
            counts=wrong_counts,
            displacements=wrong_displacements,
            per_rank_nonlocal_components=wrong_nonlocal,
        )

        # Its own arithmetic is consistent.
        self.assertTrue(corrupted.verify())

        # Its intervals and metadata agree with each other.
        self.assertTrue(
            audit_resource_communication(corrupted, 3)
        )

        # But it does not represent the verified resource plan.
        with self.assertRaises(ValueError):
            audit_resource_communication_plan(
                plan,
                corrupted,
            )

    def test_counts_disagree_with_ownership(self):
        plan, cert = self.make_certificate()

        # A different, internally consistent partition.
        wrong_counts = (6, 3, 3, 3, 3, 3)
        wrong_displacements = (0, 6, 9, 12, 15, 18)

        wrong_nonlocal = tuple(
            cert.output_components - c
            for c in wrong_counts
        )

        corrupted = replace(
            cert,
            counts=wrong_counts,
            displacements=wrong_displacements,
            per_rank_nonlocal_components=wrong_nonlocal,
        )

        self.assertTrue(corrupted.verify())

        with self.assertRaises(ValueError):
            audit_resource_communication(corrupted, 3)

    def test_gap_rejected(self):
        _, cert = self.make_certificate()

        corrupted = replace(
            cert,
            intervals=(
                (0, 1),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
            ),
        )

        with self.assertRaises(ValueError):
            audit_resource_communication(corrupted, 3)

    def test_overlap_rejected(self):
        _, cert = self.make_certificate()

        corrupted = replace(
            cert,
            intervals=(
                (0, 2),
                (1, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
            ),
        )

        with self.assertRaises(ValueError):
            audit_resource_communication(corrupted, 3)

    def test_incomplete_ownership_rejected(self):
        _, cert = self.make_certificate()

        corrupted = replace(
            cert,
            intervals=(
                (0, 1),
                (1, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 6),
            ),
        )

        with self.assertRaises(ValueError):
            audit_resource_communication(corrupted, 3)

    def test_noninteger_interval_rejected(self):
        _, cert = self.make_certificate()

        corrupted = replace(
            cert,
            intervals=(
                (0, 1.0),
                (1, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
            ),
        )

        with self.assertRaises(ValueError):
            audit_resource_communication(corrupted, 3)

    def test_invalid_value_extent_rejected(self):
        _, cert = self.make_certificate()

        for value_extent in (0, -1, 2.5, True):
            with self.subTest(value_extent=value_extent):
                with self.assertRaises(ValueError):
                    audit_resource_communication(
                        cert,
                        value_extent,
                    )

    def test_wrong_value_extent_rejected(self):
        _, cert = self.make_certificate()

        with self.assertRaises(ValueError):
            audit_resource_communication(cert, 4)

    def test_corrupted_nonlocal_bytes_rejected(self):
        _, cert = self.make_certificate()

        corrupted = replace(
            cert,
            aggregate_nonlocal_bytes=0,
        )

        with self.assertRaises(ValueError):
            audit_resource_communication(corrupted, 3)

    def test_reject_noncertificate(self):
        with self.assertRaises(TypeError):
            audit_resource_communication(None, 3)

    def test_reject_nonplan(self):
        _, cert = self.make_certificate()

        with self.assertRaises(TypeError):
            audit_resource_communication_plan(None, cert)


if __name__ == "__main__":
    unittest.main()
