
"""Tests for resource-derived MoA Omega communication certificates.

Verifies logical MPI_Allgatherv requirements derived from
resource-constrained Omega ownership.

These are logical communication quantities, NOT physical
network traffic, measured bandwidth, or execution-time predictions.
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
    ResourceCommunicationCertificate,
    derive_resource_communication,
)


def make_plan(work, shape, counts):
    """Build a resource-constrained Omega plan."""

    if len(counts) != Machine(shape).tau:
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
        Machine(shape),
        capacities,
    )

    if plan.query_counts != counts:
        raise AssertionError("unexpected Omega ownership")

    return plan


class TestResourceCommunication(unittest.TestCase):

    def check_case(self, work, shape, counts):
        plan = make_plan(work, shape, counts)

        cert = derive_resource_communication(plan)

        self.assertIsInstance(
            cert,
            ResourceCommunicationCertificate,
        )

        self.assertTrue(cert.verify())

        p = Machine(shape).tau
        total = work.n * work.dv

        expected_counts = tuple(
            q * work.dv for q in counts
        )

        expected_displacements = tuple(
            sum(expected_counts[:r])
            for r in range(p)
        )

        expected_nonlocal = tuple(
            total - c for c in expected_counts
        )

        self.assertEqual(cert.machine_components, p)
        self.assertEqual(cert.output_components, total)

        self.assertEqual(cert.counts, expected_counts)
        self.assertEqual(
            cert.displacements,
            expected_displacements,
        )

        self.assertEqual(
            cert.per_rank_receive_components,
            (total,) * p,
        )

        self.assertEqual(
            cert.per_rank_nonlocal_components,
            expected_nonlocal,
        )

        self.assertEqual(
            cert.aggregate_receive_components,
            p * total,
        )

        self.assertEqual(
            cert.aggregate_nonlocal_components,
            (p - 1) * total,
        )

        self.assertEqual(
            cert.aggregate_receive_bytes,
            8 * p * total,
        )

        self.assertEqual(
            cert.aggregate_nonlocal_bytes,
            8 * (p - 1) * total,
        )

        self.assertEqual(sum(cert.counts), total)

        self.assertEqual(
            sum(cert.per_rank_nonlocal_components),
            cert.aggregate_nonlocal_components,
        )

        print(
            "PASS logical MoA Omega communication:",
            "shape=", shape,
            "counts=", counts,
            "output=", total,
            "nonlocal_bytes=", cert.aggregate_nonlocal_bytes,
        )

        return cert

    def test_heterogeneous_ownership(self):
        self.check_case(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
        )

    def test_idle_machine_components(self):
        cert = self.check_case(
            Work(3, 6, 2, 4),
            (2, 3),
            (3, 0, 0, 0, 0, 0),
        )

        self.assertEqual(
            cert.counts,
            (12, 0, 0, 0, 0, 0),
        )

        self.assertEqual(
            cert.per_rank_nonlocal_components,
            (0, 12, 12, 12, 12, 12),
        )

    def test_unequal_ownership(self):
        self.check_case(
            Work(11, 7, 8, 4),
            (3, 2),
            (1, 2, 3, 1, 2, 2),
        )

    def test_single_machine_component(self):
        cert = self.check_case(
            Work(3, 6, 2, 1),
            (1,),
            (3,),
        )

        self.assertEqual(
            cert.aggregate_nonlocal_components,
            0,
        )

        self.assertEqual(
            cert.aggregate_nonlocal_bytes,
            0,
        )

    def test_component_byte_width(self):
        plan = make_plan(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
        )

        cert = derive_resource_communication(
            plan,
            element_bytes=4,
        )

        self.assertTrue(cert.verify())

        self.assertEqual(cert.element_bytes, 4)
        self.assertEqual(cert.aggregate_receive_bytes, 504)
        self.assertEqual(cert.aggregate_nonlocal_bytes, 420)

    def test_invalid_component_byte_width(self):
        plan = make_plan(
            Work(3, 6, 2, 4),
            (2, 3),
            (3, 0, 0, 0, 0, 0),
        )

        for width in (0, -1, 2.5, True):
            with self.subTest(width=width):
                with self.assertRaises(ValueError):
                    derive_resource_communication(
                        plan,
                        element_bytes=width,
                    )

    def test_reject_non_resource_plan(self):
        with self.assertRaises(TypeError):
            derive_resource_communication(None)

    def test_corrupted_counts_rejected(self):
        cert = self.check_case(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
        )

        corrupted = replace(
            cert,
            counts=(4,) + cert.counts[1:],
        )

        with self.assertRaises(ValueError):
            corrupted.verify()

    def test_corrupted_displacements_rejected(self):
        cert = self.check_case(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
        )

        corrupted = replace(
            cert,
            displacements=(0,) * cert.machine_components,
        )

        with self.assertRaises(ValueError):
            corrupted.verify()

    def test_corrupted_receive_rejected(self):
        cert = self.check_case(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
        )

        corrupted = replace(
            cert,
            per_rank_receive_components=(0,)
            + cert.per_rank_receive_components[1:],
        )

        with self.assertRaises(ValueError):
            corrupted.verify()

    def test_corrupted_nonlocal_rejected(self):
        cert = self.check_case(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
        )

        corrupted = replace(
            cert,
            per_rank_nonlocal_components=(0,)
            + cert.per_rank_nonlocal_components[1:],
        )

        with self.assertRaises(ValueError):
            corrupted.verify()

    def test_corrupted_aggregate_rejected(self):
        cert = self.check_case(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
        )

        corrupted = replace(
            cert,
            aggregate_nonlocal_components=0,
        )

        with self.assertRaises(ValueError):
            corrupted.verify()

    def test_corrupted_bytes_rejected(self):
        cert = self.check_case(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
        )

        corrupted = replace(
            cert,
            aggregate_nonlocal_bytes=0,
        )

        with self.assertRaises(ValueError):
            corrupted.verify()

    def test_corrupted_machine_count_rejected(self):
        cert = self.check_case(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
        )

        corrupted = replace(
            cert,
            machine_components=5,
        )

        with self.assertRaises(ValueError):
            corrupted.verify()


if __name__ == "__main__":
    unittest.main()
