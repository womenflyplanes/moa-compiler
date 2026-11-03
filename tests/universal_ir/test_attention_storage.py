
"""Regression tests for MoA attention storage certificates.

The certificate models application-level storage for the Gate 11
MPI backend, not physical peak memory or MPI-internal allocations.
"""

import unittest
from dataclasses import replace

from moa_typed_ir import (
    specification,
    stable_online_fold,
    gamma_lower,
    schedule_omega,
)
from moa_machine import Work, Machine
from moa_attention_storage import derive_attention_storage


def scheduled(n, m, d, dv, machine_shape):
    """Construct a verified typed MoA attention Omega schedule."""
    return schedule_omega(
        gamma_lower(stable_online_fold(specification())),
        Work(n, m, d, dv),
        Machine(machine_shape),
    )


class TestAttentionStorage(unittest.TestCase):

    def test_six_component_machine(self):
        ir = scheduled(7, 5, 4, 3, (2, 3))
        cert = derive_attention_storage(ir)

        self.assertTrue(cert.verify())
        self.assertEqual(cert.machine_shape, (2, 3))
        self.assertEqual(cert.output_shape, (7, 3))
        self.assertEqual(len(cert.ranks), 6)

        self.assertEqual(
            [r.modeled_bytes for r in cert.ranks],
            [736, 736, 736, 736, 736, 760],
        )
        self.assertEqual(cert.aggregate_modeled_bytes, 4440)
        self.assertEqual(cert.maximum_rank_modeled_bytes, 760)
        self.assertIsNone(cert.all_feasible)

    def test_capacity_violation(self):
        ir = scheduled(7, 5, 4, 3, (2, 3))
        cert = derive_attention_storage(
            ir, capacity_bytes=(750,) * 6
        )

        self.assertTrue(cert.verify())
        self.assertEqual(
            [r.feasible for r in cert.ranks],
            [True, True, True, True, True, False],
        )
        self.assertFalse(cert.all_feasible)

    def test_sufficient_capacity(self):
        ir = scheduled(7, 5, 4, 3, (2, 3))
        cert = derive_attention_storage(
            ir, capacity_bytes=(760,) * 6
        )

        self.assertTrue(cert.verify())
        self.assertTrue(cert.all_feasible)
        self.assertTrue(all(r.feasible for r in cert.ranks))

    def test_idle_machine_components(self):
        ir = scheduled(2, 5, 3, 4, (2, 3))
        cert = derive_attention_storage(ir)

        self.assertTrue(cert.verify())

        self.assertEqual(
            [r.query_count for r in cert.ranks],
            [0, 0, 1, 0, 0, 1],
        )
        self.assertEqual(
            [r.modeled_bytes for r in cert.ranks],
            [400, 400, 472, 400, 400, 472],
        )

        for rank in cert.ranks:
            if rank.query_count == 0:
                self.assertEqual(
                    rank.local_output_allocated_components, 1
                )
                self.assertEqual(rank.online_vector_components, 0)
                self.assertEqual(rank.online_scalar_components, 0)
            else:
                self.assertEqual(rank.online_vector_components, 4)
                self.assertEqual(rank.online_scalar_components, 2)

        self.assertEqual(sum(r.query_count for r in cert.ranks), 2)

    def test_component_size(self):
        ir = scheduled(7, 5, 4, 3, (2, 3))

        cert8 = derive_attention_storage(ir, element_bytes=8)
        cert4 = derive_attention_storage(ir, element_bytes=4)

        self.assertEqual(
            cert8.aggregate_modeled_bytes,
            2 * cert4.aggregate_modeled_bytes,
        )
        self.assertTrue(cert4.verify())

    def test_invalid_element_sizes(self):
        ir = scheduled(7, 5, 4, 3, (2, 3))

        for bad in (0, -1, 2.5, True):
            with self.subTest(element_bytes=bad):
                with self.assertRaises(ValueError):
                    derive_attention_storage(
                        ir, element_bytes=bad
                    )

    def test_invalid_capacities(self):
        ir = scheduled(7, 5, 4, 3, (2, 3))

        for bad in (
            (100,),
            (100,) * 5,
            (100,) * 7,
            (100, 100, 100, 100, 100, -1),
            (100, 100, 100, 100, 100, True),
        ):
            with self.subTest(capacity_bytes=bad):
                with self.assertRaises(ValueError):
                    derive_attention_storage(
                        ir, capacity_bytes=bad
                    )

    def test_reject_non_scheduled_ir(self):
        with self.assertRaises(ValueError):
            derive_attention_storage("not a typed MoA schedule")

    def test_mutated_aggregate_rejected(self):
        ir = scheduled(7, 5, 4, 3, (2, 3))
        cert = derive_attention_storage(ir)

        bad = replace(
            cert,
            aggregate_modeled_bytes=cert.aggregate_modeled_bytes + 8,
        )

        with self.assertRaises(ValueError):
            bad.verify()

    def test_mutated_rank_storage_rejected(self):
        ir = scheduled(7, 5, 4, 3, (2, 3))
        cert = derive_attention_storage(ir)

        bad_rank = replace(
            cert.ranks[0],
            modeled_bytes=cert.ranks[0].modeled_bytes + 8,
        )

        bad = replace(
            cert,
            ranks=(bad_rank,) + cert.ranks[1:],
        )

        with self.assertRaises(ValueError):
            bad.verify()


if __name__ == "__main__":
    unittest.main()
