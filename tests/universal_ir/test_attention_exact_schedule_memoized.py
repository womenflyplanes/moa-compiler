
"""Step 105: independent memoized MoA Omega optimality verifier.

Independently enumerate every legal nonempty communication stage
and compute the minimum predicted completion time.

The reference solver:
  - reconstructs transfers from the Omega topology;
  - independently checks stage-entry payload availability;
  - independently enforces outgoing-port and delivery constraints;
  - independently evaluates directed-link contention;
  - memoizes completed-transfer states;
  - enforces explicit search limits.

It does not use the exact optimizer's private enumeration,
eligibility, stage-cost, or dynamic-programming functions.

This verifies optimality only under the fixed Step 103 logical
communication model, not physical MPI/DGX execution.
"""

from collections import Counter
from functools import lru_cache
from itertools import combinations
from math import fsum, isclose
import unittest

from moa_machine import Work
from test_attention_omega_topology import make_plan
from moa_attention_omega_topology import derive_omega_topology
from moa_attention_port_schedule_audit import (
    audit_omega_port_schedule,
)
from moa_attention_communication_time import (
    OmegaLinkParameters,
    uniform_omega_links,
)
from moa_attention_contention_audit import (
    audit_omega_contention,
)
from moa_attention_exact_schedule import (
    schedule_omega_exact,
)


class ReferenceSearchLimit(RuntimeError):
    """Independent reference search exceeded its stated budget."""


def reference_optimum_memoized(
    topology,
    links,
    outgoing_ports,
    max_transfers=16,
    max_states=100000,
    max_transitions=1000000,
):
    """Independently compute the exact logical schedule optimum.

    Every legal nonempty subset of eligible transfers is examined.
    Repeated completed-transfer states are solved only once.

    Returns (optimal_seconds, states, transitions).
    """

    for name, value in (
        ("max_transfers", max_transfers),
        ("max_states", max_states),
        ("max_transitions", max_transitions),
        ("outgoing_ports", outgoing_ports),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be positive integer")

    # Reconstruct the expected transfers from the topology,
    # independently of the baseline scheduler.
    transfers = []

    for route in topology.routes:
        for parent, child in route.edges:
            transfers.append(
                (
                    route.owner,
                    parent,
                    child,
                    route.payload_bytes,
                )
            )

    transfers = tuple(sorted(transfers))
    n = len(transfers)

    if n > max_transfers:
        raise ReferenceSearchLimit(
            "reference transfer limit exceeded"
        )

    if len(set(transfers)) != n:
        raise ValueError(
            "duplicate transfer occurrences are unsupported"
        )

    owners = tuple(
        sorted(route.owner for route in topology.routes)
    )

    final_state = (1 << n) - 1
    states = 0
    transitions = 0

    @lru_cache(maxsize=None)
    def minimum_remaining(done):
        nonlocal states, transitions

        states += 1

        if states > max_states:
            raise ReferenceSearchLimit(
                "reference state limit exceeded"
            )

        if done == final_state:
            return 0.0

        # Reconstruct payload availability at stage entry.
        available = {
            owner: {owner}
            for owner in owners
        }

        for i, transfer in enumerate(transfers):
            if done & (1 << i):
                owner, parent, child, payload = transfer
                available[owner].add(child)

        ready = []

        for i, transfer in enumerate(transfers):
            if done & (1 << i):
                continue

            owner, parent, child, payload = transfer

            if parent not in available[owner]:
                continue

            if child in available[owner]:
                continue

            ready.append(i)

        if not ready:
            raise ValueError(
                "incomplete state has no eligible transfers"
            )

        best = float("inf")

        # Exhaustively enumerate all nonempty legal stages.
        for stage_size in range(1, len(ready) + 1):
            for indices in combinations(ready, stage_size):
                outgoing = Counter()
                delivered = set()
                grouped_bytes = {}
                legal = True

                for i in indices:
                    owner, parent, child, payload = transfers[i]

                    outgoing[parent] += 1
                    delivery = (owner, child)

                    if (
                        outgoing[parent] > outgoing_ports
                        or delivery in delivered
                    ):
                        legal = False
                        break

                    delivered.add(delivery)

                    edge = (parent, child)
                    grouped_bytes[edge] = (
                        grouped_bytes.get(edge, 0) + payload
                    )

                if not legal:
                    continue

                transitions += 1

                if transitions > max_transitions:
                    raise ReferenceSearchLimit(
                        "reference transition limit exceeded"
                    )

                stage_time = max(
                    (
                        links[edge].alpha
                        + payload / links[edge].beta
                        for edge, payload in grouped_bytes.items()
                    ),
                    default=0.0,
                )

                next_done = done

                for i in indices:
                    next_done |= 1 << i

                candidate = fsum(
                    (
                        stage_time,
                        minimum_remaining(next_done),
                    )
                )

                if candidate < best:
                    best = candidate

        if best == float("inf"):
            raise ValueError("no legal reference continuation")

        return best

    optimum = minimum_remaining(0)

    return optimum, states, transitions


def make_case(shape, counts, ports, alpha=2e-6, beta=10e9):
    """Construct a small MoA machine test instance."""

    plan = make_plan(
        Work(sum(counts), 5, 4, 3),
        shape,
        counts,
    )

    topology = derive_omega_topology(plan)
    links = uniform_omega_links(
        topology,
        alpha,
        beta,
    )

    return plan, topology, links, ports


class TestOmegaMemoizedOptimality(unittest.TestCase):

    def compare_with_reference(
        self,
        plan,
        topology,
        links,
        ports,
        max_transfers=16,
    ):
        result = schedule_omega_exact(
            plan,
            topology,
            links,
            outgoing_ports=ports,
            max_transfers=max_transfers,
        )

        expected, states, transitions = (
            reference_optimum_memoized(
                topology,
                links,
                ports,
                max_transfers=max_transfers,
            )
        )

        self.assertTrue(result.optimality_proven)
        self.assertTrue(result.verify())

        self.assertTrue(
            audit_omega_port_schedule(
                plan,
                topology,
                result.optimal,
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

        self.assertTrue(
            isclose(
                result.optimal_seconds,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )

        self.assertLessEqual(
            result.optimal_seconds,
            result.heuristic_seconds + 1e-15,
        )

        self.assertGreater(states, 0)
        self.assertGreaterEqual(transitions, 0)

        return result, states, transitions

    def test_two_coordinate_reference(self):
        case = make_case(
            (2,), (1, 1), 1
        )
        self.compare_with_reference(*case)

    def test_four_coordinate_reference(self):
        case = make_case(
            (2, 2), (1, 1, 1, 1), 2
        )

        result, states, transitions = (
            self.compare_with_reference(*case)
        )

        print(
            "\nFOUR_COORDINATE_EXACT",
            "states=", states,
            "transitions=", transitions,
            "optimal_us=", result.optimal_seconds * 1e6,
            "heuristic_us=", result.heuristic_seconds * 1e6,
        )

    def test_four_coordinate_heterogeneous(self):
        plan, topology, links, ports = make_case(
            (2, 2), (1, 1, 1, 1), 2
        )

        edge = sorted(links)[0]

        links[edge] = OmegaLinkParameters(
            2e-6,
            1e9,
        )

        self.compare_with_reference(
            plan, topology, links, ports
        )

    def test_single_coordinate_reference(self):
        case = make_case(
            (1,), (7,), 1
        )

        result, states, transitions = (
            self.compare_with_reference(*case)
        )

        self.assertEqual(result.optimal_seconds, 0.0)
        self.assertEqual(result.optimal.stage_count, 0)
        self.assertEqual(states, 1)
        self.assertEqual(transitions, 0)

    def test_reference_transfer_limit(self):
        plan, topology, links, ports = make_case(
            (2, 2), (1, 1, 1, 1), 2
        )

        with self.assertRaises(ReferenceSearchLimit):
            reference_optimum_memoized(
                topology,
                links,
                ports,
                max_transfers=2,
            )

    def test_reference_state_limit(self):
        plan, topology, links, ports = make_case(
            (2,), (1, 1), 1
        )

        with self.assertRaises(ReferenceSearchLimit):
            reference_optimum_memoized(
                topology,
                links,
                ports,
                max_states=1,
            )

    def test_reference_transition_limit(self):
        plan, topology, links, ports = make_case(
            (2,), (1, 1), 1
        )

        with self.assertRaises(ReferenceSearchLimit):
            reference_optimum_memoized(
                topology,
                links,
                ports,
                max_transitions=1,
            )

    def test_reference_determinism(self):
        plan, topology, links, ports = make_case(
            (2, 2), (1, 1, 1, 1), 2
        )

        first = reference_optimum_memoized(
            topology, links, ports
        )
        second = reference_optimum_memoized(
            topology, links, ports
        )

        self.assertEqual(first, second)

    def test_invalid_reference_limits(self):
        plan, topology, links, ports = make_case(
            (2,), (1, 1), 1
        )

        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    reference_optimum_memoized(
                        topology,
                        links,
                        ports,
                        max_states=value,
                    )


if __name__ == "__main__":
    unittest.main()

