"""Step 108: integrated MoA attention regression tests."""

import unittest
from dataclasses import replace

from moa_machine import Machine, Work
from moa_attention_resource_planner import modeled_attention_bytes
from moa_attention_communication_time import OmegaLinkParameters
from moa_attention_integrated_108 import compile_attention_108


class TestIntegratedAttention108(unittest.TestCase):

    def setUp(self):
        self.work = Work(3, 5, 4, 3)
        self.machine = Machine((3,))

        self.capacities = (
            modeled_attention_bytes(self.work, 1),
        ) * self.machine.tau

        self.links = {
            ((0,), (1,)): OmegaLinkParameters(1e-6, 1e9),
            ((1,), (0,)): OmegaLinkParameters(1e-6, 1e9),
            ((1,), (2,)): OmegaLinkParameters(1e-6, 1e9),
            ((2,), (1,)): OmegaLinkParameters(1e-6, 1e9),
        }

    def compile(self, mode):
        return compile_attention_108(
            self.work,
            self.machine,
            self.capacities,
            self.links,
            outgoing_ports=2,
            mode=mode,
        )

    def test_valid_heuristic_and_exact(self):
        for mode in ("heuristic", "exact"):
            with self.subTest(mode=mode):
                result = self.compile(mode)

                self.assertTrue(result.verify())
                self.assertEqual(
                    result.plan.query_counts,
                    (1, 1, 1),
                )
                self.assertEqual(
                    result.communication.aggregate_nonlocal_bytes,
                    144,
                )
                self.assertEqual(
                    result.topology.total_link_bytes,
                    144,
                )

                if mode == "heuristic":
                    schedule = result.scheduling.selected
                    prediction = result.scheduling.selected_prediction
                else:
                    schedule = result.scheduling.optimal
                    prediction = result.scheduling.optimal_prediction

                self.assertEqual(schedule.stage_count, 2)
                self.assertAlmostEqual(
                    prediction.total_seconds,
                    2.048e-6,
                    places=12,
                )

    def test_corrupted_certificates_rejected(self):
        for mode in ("heuristic", "exact"):
            result = self.compile(mode)

            corrupted_communication = replace(
                result.communication,
                aggregate_nonlocal_bytes=145,
            )

            corrupted_topology = replace(
                result.topology,
                total_link_bytes=145,
            )

            invalid_results = (
                replace(
                    result,
                    communication=corrupted_communication,
                ),
                replace(
                    result,
                    topology=corrupted_topology,
                ),
                replace(
                    result,
                    outgoing_ports=1,
                ),
            )

            for invalid in invalid_results:
                with self.subTest(mode=mode):
                    with self.assertRaises(
                        (ValueError, TypeError, AssertionError)
                    ):
                        invalid.verify()


if __name__ == "__main__":
    unittest.main()
