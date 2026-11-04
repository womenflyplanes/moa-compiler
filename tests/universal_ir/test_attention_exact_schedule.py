
"""Step 105: independent tests of exact MoA Omega scheduling.

Validate exact schedule optimization against a separately implemented
exhaustive reference search.

The reference solver does not call the optimizer's internal
eligibility, stage enumeration, or cost functions.

These tests concern the abstract MoA machine and Step 103 logical
directed-link contention model, not measured MPI/DGX execution.
"""

from collections import Counter
from dataclasses import replace
from itertools import combinations
from math import fsum, isclose
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
from moa_attention_contention_schedule import schedule_omega_contention
from moa_attention_exact_schedule import (
    OmegaExactSearchLimit,
    schedule_omega_exact,
)


def fixture(shape=(2,), counts=(1, 1), ports=1):
    """Construct a small, verified MoA Omega communication instance."""
    plan = make_plan(Work(sum(counts), 5, 4, 3), shape, counts)
    topology = derive_omega_topology(plan)
    links = uniform_omega_links(topology, 2e-6, 10e9)
    return plan, topology, links, ports


def transfer_counter(schedule):
    return Counter(
        transfer
        for stage in schedule.stages
        for transfer in stage
    )


def reference_stage_cost(stage, links):
    """Independent Step 103 stage-time calculation."""
    traffic = {}

    for transfer in stage:
        edge = (transfer.parent, transfer.child)
        traffic[edge] = traffic.get(edge, 0) + transfer.payload_bytes

    if not traffic:
        return 0.0

    return max(
        links[edge].alpha + nbytes / links[edge].beta
        for edge, nbytes in traffic.items()
    )


def reference_optimum(plan, topology, links, ports):
    """Independently enumerate every legal nonempty-stage schedule.

    No optimizer-private helpers or memoized optimizer states are used.
    Intended only for small instances.
    """
    baseline = schedule_omega_ports(topology, ports)

    transfers = tuple(
        transfer
        for stage in baseline.stages
        for transfer in stage
    )

    n = len(transfers)
    if n > 12:
        raise ValueError("reference instance too large")

    owners = tuple(route.owner for route in topology.routes)
    full = (1 << n) - 1
    complete_paths = 0
    best = float("inf")

    def visit(done, reached, accumulated):
        nonlocal complete_paths, best

        if done == full:
            complete_paths += 1
            if accumulated < best:
                best = accumulated
            return

        available = {
            owner: set(nodes)
            for owner, nodes in reached.items()
        }

        choices = []

        for index, transfer in enumerate(transfers):
            if done & (1 << index):
                continue
            if transfer.parent not in available[transfer.owner]:
                continue
            if transfer.child in available[transfer.owner]:
                continue
            choices.append(index)

        for size in range(1, len(choices) + 1):
            for indices in combinations(choices, size):
                senders = Counter()
                deliveries = set()
                legal = True

                for index in indices:
                    transfer = transfers[index]
                    senders[transfer.parent] += 1
                    delivery = (transfer.owner, transfer.child)

                    if (
                        senders[transfer.parent] > ports
                        or delivery in deliveries
                    ):
                        legal = False
                        break

                    deliveries.add(delivery)

                if not legal:
                    continue

                next_done = done
                next_reached = {
                    owner: set(nodes)
                    for owner, nodes in reached.items()
                }
                stage = []

                for index in indices:
                    transfer = transfers[index]
                    next_done |= 1 << index
                    next_reached[transfer.owner].add(transfer.child)
                    stage.append(transfer)

                visit(
                    next_done,
                    next_reached,
                    accumulated + reference_stage_cost(stage, links),
                )

    visit(
        0,
        {owner: {owner} for owner in owners},
        0.0,
    )

    if complete_paths == 0:
        raise AssertionError("reference found no complete schedule")

    return best, complete_paths


class TestOmegaExactScheduling(unittest.TestCase):

    def check_exact(self, plan, topology, links, ports, result):
        self.assertTrue(result.verify())
        self.assertTrue(result.optimality_proven)

        self.assertTrue(
            audit_omega_port_schedule(
                plan, topology, result.optimal
            )
        )
        self.assertTrue(
            audit_omega_contention(
                plan,
                topology,
                result.optimal,
                links,
                result.optimal_prediction,
            )
        )

        baseline = schedule_omega_ports(topology, ports)

        self.assertEqual(
            transfer_counter(result.optimal),
            transfer_counter(baseline),
        )
        self.assertEqual(
            result.optimal.total_link_bytes,
            topology.total_link_bytes,
        )
        self.assertLessEqual(
            result.optimal_seconds,
            result.heuristic_seconds + 1e-15,
        )

    def test_two_coordinate_reference_optimum(self):
        plan, topology, links, ports = fixture()

        result = schedule_omega_exact(
            plan, topology, links, ports
        )
        expected, paths = reference_optimum(
            plan, topology, links, ports
        )

        self.check_exact(plan, topology, links, ports, result)
        self.assertGreater(paths, 0)
        self.assertTrue(
            isclose(
                result.optimal_seconds,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )

    def test_determinism(self):
        plan, topology, links, ports = fixture()

        first = schedule_omega_exact(
            plan, topology, links, ports
        )
        second = schedule_omega_exact(
            plan, topology, links, ports
        )

        self.assertEqual(first.optimal, second.optimal)
        self.assertEqual(
            first.states_examined,
            second.states_examined,
        )
        self.assertEqual(
            first.transitions_examined,
            second.transitions_examined,
        )

    def test_heterogeneous_links(self):
        plan, topology, links, ports = fixture(
            (2,), (1, 1), 2
        )

        edge = sorted(links)[0]
        links[edge] = OmegaLinkParameters(3e-6, 1e9)

        result = schedule_omega_exact(
            plan, topology, links, ports
        )
        expected, _ = reference_optimum(
            plan, topology, links, ports
        )

        self.check_exact(plan, topology, links, ports, result)
        self.assertTrue(
            isclose(
                result.optimal_seconds,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )

    def test_zero_latency(self):
        plan, topology, links, ports = fixture()
        links = uniform_omega_links(topology, 0.0, 10e9)

        result = schedule_omega_exact(
            plan, topology, links, ports
        )
        expected, _ = reference_optimum(
            plan, topology, links, ports
        )

        self.check_exact(plan, topology, links, ports, result)
        self.assertTrue(
            isclose(
                result.optimal_seconds,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )

    def test_single_coordinate(self):
        plan, topology, links, ports = fixture(
            (1,), (7,), 1
        )

        result = schedule_omega_exact(
            plan, topology, links, ports
        )

        self.check_exact(plan, topology, links, ports, result)
        self.assertEqual(result.optimal.stage_count, 0)
        self.assertEqual(result.optimal_seconds, 0.0)

        expected, paths = reference_optimum(
            plan, topology, links, ports
        )
        self.assertEqual(expected, 0.0)
        self.assertEqual(paths, 1)

    def test_transfer_limit(self):
        plan, topology, links, ports = fixture()

        with self.assertRaises(OmegaExactSearchLimit):
            schedule_omega_exact(
                plan, topology, links, ports,
                max_transfers=1,
            )

    def test_state_limit(self):
        plan, topology, links, ports = fixture()

        with self.assertRaises(OmegaExactSearchLimit):
            schedule_omega_exact(
                plan, topology, links, ports,
                max_states=1,
            )

    def test_transition_limit(self):
        plan, topology, links, ports = fixture()

        with self.assertRaises(OmegaExactSearchLimit):
            schedule_omega_exact(
                plan, topology, links, ports,
                max_transitions=1,
            )

    def test_invalid_search_limits(self):
        plan, topology, links, ports = fixture()

        for invalid in (0, -1, 1.5, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    schedule_omega_exact(
                        plan, topology, links, ports,
                        max_states=invalid,
                    )

    def test_missing_link_parameters(self):
        plan, topology, links, ports = fixture()
        links.pop(next(iter(links)))

        with self.assertRaises(ValueError):
            schedule_omega_exact(
                plan, topology, links, ports
            )

    def test_result_rejects_false_optimality(self):
        plan, topology, links, ports = fixture()

        result = schedule_omega_exact(
            plan, topology, links, ports
        )

        damaged = replace(
            result,
            optimality_proven=False,
        )

        with self.assertRaises(ValueError):
            damaged.verify()

    @unittest.skip("Expensive brute-force reference; four-coordinate case covered by independent memoized verifier")
    def test_four_coordinate_reference_optimum(self):
        plan, topology, links, ports = fixture(
            (2, 2), (1, 1, 1, 1), 2
        )

        baseline = schedule_omega_ports(topology, ports)
        transfer_count = sum(
            len(stage) for stage in baseline.stages
        )

        # This test uses the small independent reference only
        # if the topology is within its explicit bound.
        if transfer_count > 12:
            self.skipTest(
                "four-coordinate topology exceeds reference bound"
            )

        result = schedule_omega_exact(
            plan, topology, links, ports,
            max_transfers=16,
            max_states=100000,
            max_transitions=1000000,
        )

        expected, paths = reference_optimum(
            plan, topology, links, ports
        )

        self.check_exact(plan, topology, links, ports, result)
        self.assertGreater(paths, 0)
        self.assertTrue(
            isclose(
                result.optimal_seconds,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )


if __name__ == "__main__":
    unittest.main()

