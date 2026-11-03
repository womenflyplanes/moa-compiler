
"""Regression tests for unified MoA attention resource feasibility.

Verifies typed Omega ownership, multidimensional Universal IR mapping,
and modeled application storage for the Gate 11 MPI backend.

No physical peak-memory or network-performance claims are made.
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
from moa_v3 import MachineArray
from moa_multidimensional_lift import ResourceArray
from moa_attention_multidim_bridge import bridge_attention
from moa_attention_resource_bridge import bridge_attention_resources


def make_bridge(
    capacities,
    work=(7, 5, 4, 3),
    machine_shape=(2, 3),
    dtype_bytes=8,
    max_chunk_elements=100,
):
    """Construct typed attention and its verified Omega machine mapping."""
    scheduled = schedule_omega(
        gamma_lower(stable_online_fold(specification())),
        Work(*work),
        Machine(machine_shape),
    )

    machines = tuple(
        MachineArray(
            str(i),
            (0,) * 10,
            capacity,
            max_chunk_elements,
        )
        for i, capacity in enumerate(capacities)
    )

    resources = ResourceArray(machine_shape, machines)

    return bridge_attention(
        scheduled,
        resources,
        dtype_bytes=dtype_bytes,
    )


class TestAttentionResourceBridge(unittest.TestCase):

    def test_output_valid_but_mpi_storage_infeasible(self):
        attention = make_bridge((100,) * 6)
        cert = bridge_attention_resources(attention)

        self.assertTrue(attention.verify())
        self.assertTrue(cert.verify())
        self.assertFalse(cert.all_feasible)

        self.assertEqual(
            [r.modeled_bytes for r in cert.storage.ranks],
            [736, 736, 736, 736, 736, 760],
        )

    def test_sufficient_capacity(self):
        cert = bridge_attention_resources(
            make_bridge((760,) * 6)
        )

        self.assertTrue(cert.verify())
        self.assertTrue(cert.all_feasible)
        self.assertEqual(
            cert.maximum_rank_modeled_bytes, 760
        )

    def test_heterogeneous_capacities(self):
        cert = bridge_attention_resources(
            make_bridge((736, 740, 750, 800, 736, 759))
        )

        self.assertTrue(cert.verify())
        self.assertEqual(
            [r.feasible for r in cert.storage.ranks],
            [True, True, True, True, True, False],
        )
        self.assertFalse(cert.all_feasible)

    def test_idle_machine_components(self):
        attention = make_bridge(
            (500,) * 6,
            work=(2, 5, 3, 4),
        )
        cert = bridge_attention_resources(attention)

        self.assertTrue(cert.verify())
        self.assertTrue(cert.all_feasible)

        self.assertEqual(
            [r.query_count for r in cert.storage.ranks],
            [0, 0, 1, 0, 0, 1],
        )

        self.assertEqual(
            [r.modeled_bytes for r in cert.storage.ranks],
            [400, 400, 472, 400, 400, 472],
        )

    def test_output_partition_limit_rejected(self):
        with self.assertRaises(ValueError):
            make_bridge(
                (1000,) * 6,
                max_chunk_elements=3,
            )

    def test_output_capacity_rejected(self):
        with self.assertRaises(ValueError):
            make_bridge((8,) * 6)

    def test_non_double_component_size_rejected(self):
        attention = make_bridge(
            (1000,) * 6,
            dtype_bytes=4,
        )

        with self.assertRaises(ValueError):
            bridge_attention_resources(attention)

    def test_non_bridge_input_rejected(self):
        with self.assertRaises(TypeError):
            bridge_attention_resources("not an attention bridge")

    def test_corrupted_rank_capacity_rejected(self):
        cert = bridge_attention_resources(
            make_bridge((760,) * 6)
        )

        bad_rank = replace(
            cert.storage.ranks[0],
            capacity_bytes=999,
        )

        bad_storage = replace(
            cert.storage,
            ranks=(bad_rank,) + cert.storage.ranks[1:],
        )

        bad_cert = replace(cert, storage=bad_storage)

        with self.assertRaises(ValueError):
            bad_cert.verify()

    def test_corrupted_rank_coordinate_rejected(self):
        cert = bridge_attention_resources(
            make_bridge((760,) * 6)
        )

        bad_rank = replace(
            cert.storage.ranks[0],
            coordinate=(1, 2),
        )

        bad_storage = replace(
            cert.storage,
            ranks=(bad_rank,) + cert.storage.ranks[1:],
        )

        bad_cert = replace(cert, storage=bad_storage)

        with self.assertRaises(ValueError):
            bad_cert.verify()

    def test_corrupted_ownership_rejected(self):
        cert = bridge_attention_resources(
            make_bridge((760,) * 6)
        )

        bad_rank = replace(
            cert.storage.ranks[0],
            query_begin=1,
        )

        bad_storage = replace(
            cert.storage,
            ranks=(bad_rank,) + cert.storage.ranks[1:],
        )

        bad_cert = replace(cert, storage=bad_storage)

        with self.assertRaises(ValueError):
            bad_cert.verify()


if __name__ == "__main__":
    unittest.main()
