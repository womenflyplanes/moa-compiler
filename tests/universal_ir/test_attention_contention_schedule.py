
"""Step 104H: regression tests for contention-aware Omega scheduling.

Verify deterministic candidate selection, preservation of the exact
Omega transfer multiset, independent audits, and predicted no-worse
selection under the Step 103 directed-link contention model.

These tests establish properties of the logical communication model,
not measured hardware performance or global scheduling optimality.
"""

from collections import Counter
from dataclasses import replace
import math
import unittest

from moa_machine import Work
from test_attention_omega_topology import make_plan
from moa_attention_omega_topology import derive_omega_topology
from moa_attention_port_schedule import schedule_omega_ports
from moa_attention_port_schedule_audit import audit_omega_port_schedule
from moa_attention_communication_time import (
    OmegaLinkParameters,
    uniform_omega_links,
)
from moa_attention_contention import predict_omega_contention
from moa_attention_contention_audit import audit_omega_contention
from moa_attention_contention_schedule import (
    OmegaContentionScheduleResult,
    schedule_omega_contention,
)


def transfer_multiset(schedule):
    """Count all transfers independently of stage assignment."""
    return Counter(
        transfer
        for stage in schedule.stages
        for transfer in stage
    )


def stage_link_bytes(schedule, stage_index, edge):
    """Count bytes carried by one directed link in a stage."""
    return sum(
        transfer.payload_bytes
        for transfer in schedule.stages[stage_index]
        if (transfer.parent, transfer.child) == edge
    )


class TestOmegaContentionScheduling(unittest.TestCase):

    def setUp(self):
        self.plan = make_plan(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
        )
        self.topology = derive_omega_topology(self.plan)
        self.ports = 2
        self.slow_edge = ((1, 1), (1, 2))
        self.links = uniform_omega_links(
            self.topology, 2e-6, 10e9
        )
        self.links[self.slow_edge] = OmegaLinkParameters(
            2e-6, 1e9
        )

    def schedule(self, links=None, ports=None):
        if links is None:
            links = self.links
        if ports is None:
            ports = self.ports
        return schedule_omega_contention(
            self.plan,
            self.topology,
            links,
            ports,
        )

    def check_result(
        self, plan, topology, links, result
    ):
        """Independently audit both schedules and predictions."""
        self.assertTrue(
            audit_omega_port_schedule(
                plan, topology, result.baseline
            )
        )
        self.assertTrue(
            audit_omega_port_schedule(
                plan, topology, result.selected
            )
        )

        self.assertTrue(
            audit_omega_contention(
                plan,
                topology,
                result.baseline,
                links,
                result.baseline_prediction,
            )
        )
        self.assertTrue(
            audit_omega_contention(
                plan,
                topology,
                result.selected,
                links,
                result.selected_prediction,
            )
        )

        self.assertTrue(result.verify())
        self.assertLessEqual(
            result.selected_seconds,
            result.baseline_seconds,
        )
        self.assertEqual(
            result.baseline.total_link_bytes,
            topology.total_link_bytes,
        )
        self.assertEqual(
            result.selected.total_link_bytes,
            topology.total_link_bytes,
        )
        self.assertEqual(
            transfer_multiset(result.baseline),
            transfer_multiset(result.selected),
        )

    def test_heterogeneous_improvement(self):
        result = self.schedule()
        self.check_result(
            self.plan, self.topology, self.links, result
        )

        self.assertEqual(
            result.strategy,
            "slow_first:shared_links_allowed",
        )
        self.assertEqual(result.candidates_examined, 13)

        self.assertTrue(
            math.isclose(
                result.baseline_seconds,
                12.0648e-6,
                rel_tol=1e-10,
                abs_tol=1e-15,
            )
        )
        self.assertTrue(
            math.isclose(
                result.selected_seconds,
                12.0576e-6,
                rel_tol=1e-10,
                abs_tol=1e-15,
            )
        )
        self.assertGreater(result.improvement_seconds, 0)

    def test_exact_stage_rearrangement(self):
        result = self.schedule()

        self.assertEqual(result.baseline.stage_count, 6)
        self.assertEqual(result.selected.stage_count, 6)

        self.assertEqual(
            tuple(len(stage) for stage in result.baseline.stages),
            (12, 8, 4, 3, 2, 1),
        )
        self.assertEqual(
            tuple(len(stage) for stage in result.selected.stages),
            (12, 8, 4, 3, 2, 1),
        )

        self.assertEqual(
            stage_link_bytes(
                result.baseline, 1, self.slow_edge
            ),
            48,
        )
        self.assertEqual(
            stage_link_bytes(
                result.selected, 1, self.slow_edge
            ),
            24,
        )

    def test_stage_time_tradeoff(self):
        result = self.schedule()

        baseline = result.baseline_prediction.stage_times
        selected = result.selected_prediction.stage_times

        self.assertEqual(len(baseline), 6)
        self.assertEqual(len(selected), 6)

        expected_baseline = (
            2.0048, 2.048, 2.0048,
            2.0024, 2.0024, 2.0024,
        )
        expected_selected = (
            2.024, 2.024, 2.0024,
            2.0024, 2.0024, 2.0024,
        )

        for actual, expected in zip(
            baseline, expected_baseline
        ):
            self.assertAlmostEqual(
                actual * 1e6, expected, places=8
            )

        for actual, expected in zip(
            selected, expected_selected
        ):
            self.assertAlmostEqual(
                actual * 1e6, expected, places=8
            )

        self.assertGreater(selected[0], baseline[0])
        self.assertLess(selected[1], baseline[1])
        self.assertLess(selected[2], baseline[2])

    def test_deterministic_schedule(self):
        first = self.schedule()
        second = self.schedule()

        self.assertEqual(first, second)
        self.check_result(
            self.plan, self.topology, self.links, first
        )

    def test_original_scheduler_preserved(self):
        result = self.schedule()

        original = schedule_omega_ports(
            self.topology, self.ports
        )
        self.assertEqual(result.baseline, original)

    def test_no_worse_than_baseline(self):
        for ports in (1, 2, 3):
            with self.subTest(ports=ports):
                result = self.schedule(ports=ports)
                self.check_result(
                    self.plan,
                    self.topology,
                    self.links,
                    result,
                )

    def test_uniform_links(self):
        links = uniform_omega_links(
            self.topology, 2e-6, 10e9
        )
        result = self.schedule(links=links)
        self.check_result(
            self.plan, self.topology, links, result
        )

    def test_zero_latency_model(self):
        links = uniform_omega_links(
            self.topology, 0.0, 10e9
        )
        result = self.schedule(links=links)
        self.check_result(
            self.plan, self.topology, links, result
        )

    def test_large_latency_model(self):
        links = uniform_omega_links(
            self.topology, 1e-3, 10e9
        )
        result = self.schedule(links=links)
        self.check_result(
            self.plan, self.topology, links, result
        )

    def test_alternative_machine_shape(self):
        plan = make_plan(
            Work(7, 5, 4, 3),
            (3, 2),
            (1, 2, 1, 1, 1, 1),
        )
        topology = derive_omega_topology(plan)
        links = uniform_omega_links(
            topology, 2e-6, 10e9
        )

        result = schedule_omega_contention(
            plan, topology, links, 2
        )
        self.check_result(
            plan, topology, links, result
        )

    def test_single_coordinate_machine(self):
        plan = make_plan(
            Work(7, 5, 4, 3),
            (1,),
            (7,),
        )
        topology = derive_omega_topology(plan)
        links = uniform_omega_links(
            topology, 2e-6, 10e9
        )

        result = schedule_omega_contention(
            plan, topology, links, 1
        )
        self.check_result(
            plan, topology, links, result
        )

        self.assertEqual(result.baseline.stage_count, 0)
        self.assertEqual(result.selected.stage_count, 0)
        self.assertEqual(result.baseline_seconds, 0.0)
        self.assertEqual(result.selected_seconds, 0.0)

    def test_reject_invalid_outgoing_ports(self):
        for ports in (0, -1, 1.5, True):
            with self.subTest(ports=ports):
                with self.assertRaises(
                    (TypeError, ValueError)
                ):
                    self.schedule(ports=ports)

    def test_reject_missing_link_parameters(self):
        links = dict(self.links)
        links.pop(self.slow_edge)

        with self.assertRaises(
            (KeyError, TypeError, ValueError)
        ):
            self.schedule(links=links)

    def test_result_rejects_worse_selection(self):
        result = self.schedule()

        corrupted = replace(
            result,
            baseline_prediction=result.selected_prediction,
            selected_prediction=result.baseline_prediction,
        )

        with self.assertRaises(ValueError):
            corrupted.verify()

    def test_result_rejects_invalid_candidate_count(self):
        result = self.schedule()

        corrupted = replace(
            result,
            candidates_examined=0,
        )

        with self.assertRaises(ValueError):
            corrupted.verify()

    def test_result_rejects_changed_communication_volume(self):
        result = self.schedule()

        corrupted_schedule = replace(
            result.selected,
            total_link_bytes=(
                result.selected.total_link_bytes + 8
            ),
        )
        corrupted = replace(
            result,
            selected=corrupted_schedule,
        )

        with self.assertRaises(ValueError):
            corrupted.verify()


if __name__ == "__main__":
    unittest.main()
