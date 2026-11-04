
"""Step 106: independently verify an adversarial MoA Omega schedule.

Fixed reproducible counterexample:
    machine shape: <3>
    ownership:     (1, 1, 1)
    outgoing ports: 2
    transfers:      6

Compare the Step 104 greedy candidate scheduler with the
Step 105 exact scheduler and an independently implemented
memoized exhaustive reference solver.

All timings refer to the Step 103 logical directed-link
contention model, not measured physical hardware performance.
"""

from math import isclose
import unittest

from moa_machine import Work
from test_attention_omega_topology import make_plan
from moa_attention_omega_topology import derive_omega_topology
from moa_attention_communication_time import (
    OmegaLinkParameters,
)
from moa_attention_port_schedule_audit import (
    audit_omega_port_schedule,
)
from moa_attention_contention import (
    predict_omega_contention,
)
from moa_attention_contention_audit import (
    audit_omega_contention,
)
from moa_attention_contention_schedule import (
    schedule_omega_contention,
)
from moa_attention_exact_schedule import (
    schedule_omega_exact,
)
from test_attention_exact_schedule_memoized import (
    reference_optimum_memoized,
)


SHAPE = (3,)
COUNTS = (1, 1, 1)
PORTS = 2

# Parameters copied from the deterministic Step 106G result.
LINK_PARAMETERS = (
    ((0,), (1,), 1e-7, 50e9),
    ((1,), (0,), 1e-7, 1e9),
    ((1,), (2,), 20e-6, 1e9),
    ((2,), (1,), 0.0, 5e9),
)

EXPECTED_HEURISTIC_US = 40.048
EXPECTED_EXACT_US = 20.2008


def make_counterexample():
    """Reconstruct the exact Step 106G MoA machine."""

    plan = make_plan(
        Work(sum(COUNTS), 5, 4, 3),
        SHAPE,
        COUNTS,
    )

    topology = derive_omega_topology(plan)

    links = {
        (parent, child): OmegaLinkParameters(alpha, beta)
        for parent, child, alpha, beta in LINK_PARAMETERS
    }

    expected_edges = {
        (parent, child)
        for route in topology.routes
        for parent, child in route.edges
    }

    if set(links) != expected_edges:
        raise AssertionError(
            "fixed link parameters do not match topology"
        )

    return plan, topology, links


def independently_audited_time(
    plan, topology, links, schedule
):
    """Independently audit legality and communication prediction."""

    audit_omega_port_schedule(
        plan,
        topology,
        schedule,
    )

    prediction = predict_omega_contention(
        plan,
        topology,
        schedule,
        links,
    )

    audit_omega_contention(
        plan,
        topology,
        schedule,
        links,
        prediction,
    )

    return prediction.total_seconds


class TestOmegaAdversarialSchedule(unittest.TestCase):

    def test_exact_strictly_improves_heuristic(self):
        plan, topology, links = make_counterexample()

        heuristic = schedule_omega_contention(
            plan,
            topology,
            links,
            outgoing_ports=PORTS,
        )

        exact = schedule_omega_exact(
            plan,
            topology,
            links,
            outgoing_ports=PORTS,
            max_transfers=12,
            max_states=100000,
            max_transitions=1000000,
        )

        heuristic_time = independently_audited_time(
            plan,
            topology,
            links,
            heuristic.selected,
        )

        exact_time = independently_audited_time(
            plan,
            topology,
            links,
            exact.optimal,
        )

        # Independently solve the entire legal search space.
        reference_time, states, transitions = (
            reference_optimum_memoized(
                topology,
                links,
                PORTS,
                max_transfers=12,
                max_states=100000,
                max_transitions=1000000,
            )
        )

        self.assertTrue(exact.optimality_proven)
        self.assertTrue(exact.verify())

        self.assertTrue(
            isclose(
                exact_time,
                reference_time,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )

        self.assertTrue(
            isclose(
                heuristic_time * 1e6,
                EXPECTED_HEURISTIC_US,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )

        self.assertTrue(
            isclose(
                exact_time * 1e6,
                EXPECTED_EXACT_US,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )

        self.assertLess(
            exact_time,
            heuristic_time - 1e-15,
        )

        self.assertEqual(heuristic.selected.stage_count, 2)
        self.assertEqual(exact.optimal.stage_count, 3)

        self.assertGreater(states, 0)
        self.assertGreater(transitions, 0)

        improvement = (
            heuristic_time - exact_time
        ) / heuristic_time

        print(
            "\nADVERSARIAL_OMEGA_PASS",
            "shape=", SHAPE,
            "ports=", PORTS,
            "heuristic_us=", heuristic_time * 1e6,
            "exact_us=", exact_time * 1e6,
            "improvement_percent=", 100 * improvement,
            "reference_states=", states,
            "reference_transitions=", transitions,
        )

    def test_deterministic_counterexample(self):
        plan, topology, links = make_counterexample()

        first = schedule_omega_exact(
            plan, topology, links,
            outgoing_ports=PORTS,
        )

        second = schedule_omega_exact(
            plan, topology, links,
            outgoing_ports=PORTS,
        )

        self.assertEqual(first, second)

    def test_slow_link_is_paid_once_in_exact_schedule(self):
        plan, topology, links = make_counterexample()

        heuristic = schedule_omega_contention(
            plan, topology, links,
            outgoing_ports=PORTS,
        )

        exact = schedule_omega_exact(
            plan, topology, links,
            outgoing_ports=PORTS,
        )

        slow_edge = ((1,), (2,))

        heuristic_stages = sum(
            any(
                (transfer.parent, transfer.child) == slow_edge
                for transfer in stage
            )
            for stage in heuristic.selected.stages
        )

        exact_stages = sum(
            any(
                (transfer.parent, transfer.child) == slow_edge
                for transfer in stage
            )
            for stage in exact.optimal.stages
        )

        self.assertEqual(heuristic_stages, 2)
        self.assertEqual(exact_stages, 1)

        # Both schedules communicate the same total payload
        # over the slow directed link.
        def slow_link_bytes(schedule):
            return sum(
                transfer.payload_bytes
                for stage in schedule.stages
                for transfer in stage
                if (
                    transfer.parent,
                    transfer.child,
                ) == slow_edge
            )

        self.assertEqual(
            slow_link_bytes(heuristic.selected),
            slow_link_bytes(exact.optimal),
        )

        self.assertEqual(
            slow_link_bytes(exact.optimal),
            48,
        )

    def test_reference_optimum_is_deterministic(self):
        plan, topology, links = make_counterexample()

        first = reference_optimum_memoized(
            topology, links, PORTS
        )

        second = reference_optimum_memoized(
            topology, links, PORTS
        )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
