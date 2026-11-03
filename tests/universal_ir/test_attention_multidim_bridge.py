
"""Regression tests for Gate 11 -> Universal IR v3.3."""

import unittest
from dataclasses import replace

from moa_typed_ir import (
    specification,
    stable_online_fold,
    gamma_lower,
    schedule_omega,
)
from moa_machine import Machine, Work
from moa_v3 import MachineArray
from moa_multidimensional_lift import ResourceArray, Rectangle
from moa_attention_multidim_bridge import bridge_attention


def resources(shape, capacity=1024, chunk=1024):
    fields = (
        1, 1, 0, 1.0, chunk, capacity,
        0, "scalar", ("explicit",), 0
    )
    count = 1
    for dimension in shape:
        count *= dimension
    return ResourceArray(
        shape,
        tuple(
            MachineArray(str(i), fields, capacity, chunk)
            for i in range(count)
        ),
    )


def scheduled(n, machine_shape, dv=4):
    return schedule_omega(
        gamma_lower(stable_online_fold(specification())),
        Work(n, 5, 3, dv),
        Machine(machine_shape),
    )


class AttentionMultidimensionalBridgeTests(unittest.TestCase):

    def test_two_by_three_machine(self):
        typed = scheduled(7, (2, 3))
        result = bridge_attention(typed, resources((2, 3)))

        self.assertTrue(result.verify())
        self.assertIs(result.typed, typed)
        self.assertEqual(result.lifted.sequential.dnf.shape, (7, 4))
        self.assertEqual(len(result.lifted.mapped_indices()), 28)
        self.assertEqual(
            result.lifted.mapped_indices()[-1],
            ((6, 3), (1, 2), 27),
        )

    def test_uneven_query_ownership(self):
        typed = scheduled(7, (2, 3))
        result = bridge_attention(typed, resources((2, 3)))

        self.assertEqual(
            typed.schedule.intervals,
            ((0, 1), (1, 2), (2, 3),
             (3, 4), (4, 5), (5, 7)),
        )
        self.assertEqual(
            tuple(r.elements for r in result.lifted.rectangles),
            (4, 4, 4, 4, 4, 8),
        )

    def test_idle_machine_components(self):
        typed = scheduled(2, (2, 3))
        result = bridge_attention(typed, resources((2, 3)))

        self.assertTrue(result.verify())
        self.assertEqual(len(result.lifted.mapped_indices()), 8)
        self.assertEqual(
            sum(r.elements for r in result.lifted.rectangles),
            8,
        )
        self.assertGreater(
            sum(r.elements == 0 for r in result.lifted.rectangles),
            0,
        )

    def test_insufficient_capacity_rejected(self):
        typed = scheduled(7, (2, 3))

        with self.assertRaisesRegex(ValueError, "capacity"):
            bridge_attention(
                typed,
                resources((2, 3), capacity=32),
            )

    def test_machine_shape_mismatch_rejected(self):
        typed = scheduled(7, (2, 3))

        with self.assertRaisesRegex(ValueError, "machine shape"):
            bridge_attention(typed, resources((3, 2)))

    def test_corrupted_ownership_rejected(self):
        typed = scheduled(7, (2, 3))
        result = bridge_attention(typed, resources((2, 3)))

        rectangles = list(result.lifted.rectangles)
        original = rectangles[0]
        rectangles[0] = Rectangle(
            original.lower,
            original.upper,
            (1, 2),
        )

        corrupted_lift = replace(
            result.lifted,
            rectangles=tuple(rectangles),
        )
        corrupted_bridge = replace(
            result,
            lifted=corrupted_lift,
        )

        with self.assertRaises(ValueError):
            corrupted_bridge.verify()


if __name__ == "__main__":
    unittest.main()
