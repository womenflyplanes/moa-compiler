
"""Regression and adversarial tests for MoA Omega topology.

Tests owner-rooted nearest-neighbor routing derived from verified
resource-constrained Omega ownership.

All topology audits are independent of route construction.

These tests concern an abstract shaped machine, not measured
network traffic or MPI collective implementation details.
"""

from dataclasses import replace
import unittest

from moa_machine import Machine, Work, iota

from moa_typed_ir import (
    specification,
    stable_online_fold,
    gamma_lower,
)

from moa_attention_resource_planner import (
    modeled_attention_bytes,
    plan_attention_omega,
)

from moa_attention_omega_topology import (
    owner_rooted_tree,
    derive_omega_topology,
)

from moa_attention_topology_audit import (
    audit_omega_topology,
)


def make_plan(work, shape, counts):
    """Construct a verified resource-constrained Omega plan."""

    machine = Machine(shape)

    if len(counts) != machine.tau:
        raise ValueError("incorrect machine component count")

    if sum(counts) != work.n:
        raise ValueError("incorrect query extent")

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


class TestOmegaTopology(unittest.TestCase):

    def test_heterogeneous_ownership(self):
        plan = make_plan(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
        )

        topology = derive_omega_topology(plan)

        self.assertTrue(
            audit_omega_topology(plan, topology)
        )

        self.assertEqual(len(topology.routes), 6)
        self.assertEqual(topology.total_link_bytes, 840)

    def test_idle_machine_components(self):
        plan = make_plan(
            Work(3, 6, 2, 4),
            (2, 3),
            (3, 0, 0, 0, 0, 0),
        )

        topology = derive_omega_topology(plan)

        self.assertTrue(
            audit_omega_topology(plan, topology)
        )

        self.assertEqual(len(topology.routes), 1)
        self.assertEqual(topology.total_link_bytes, 480)

    def test_unequal_ownership(self):
        plan = make_plan(
            Work(11, 7, 8, 4),
            (3, 2),
            (1, 2, 3, 1, 2, 2),
        )

        topology = derive_omega_topology(plan)

        self.assertTrue(
            audit_omega_topology(plan, topology)
        )

        self.assertEqual(
            topology.total_link_bytes,
            5 * 11 * 4 * 8,
        )

    def test_single_machine_component(self):
        plan = make_plan(
            Work(4, 3, 2, 5),
            (1,),
            (4,),
        )

        topology = derive_omega_topology(plan)

        self.assertTrue(
            audit_omega_topology(plan, topology)
        )

        self.assertEqual(len(topology.routes), 1)
        self.assertEqual(topology.routes[0].edges, ())
        self.assertEqual(topology.total_link_bytes, 0)

    def test_multidimensional_machine(self):
        plan = make_plan(
            Work(8, 4, 3, 2),
            (2, 2, 2),
            (1, 1, 1, 1, 1, 1, 1, 1),
        )

        topology = derive_omega_topology(plan)

        self.assertTrue(
            audit_omega_topology(plan, topology)
        )

        self.assertEqual(len(topology.routes), 8)

    def test_every_coordinate_can_be_root(self):
        machine = Machine((2, 3))

        for owner in iota(machine.shape):
            with self.subTest(owner=owner):
                edges, hops = owner_rooted_tree(
                    machine,
                    owner,
                )

                self.assertEqual(
                    len(edges),
                    machine.tau - 1,
                )

                self.assertGreaterEqual(hops, 0)

    def test_invalid_root(self):
        with self.assertRaises(ValueError):
            owner_rooted_tree(
                Machine((2, 3)),
                (2, 0),
            )


class TestCorruptedOmegaTopology(unittest.TestCase):

    def setUp(self):
        self.plan = make_plan(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
        )

        self.topology = derive_omega_topology(
            self.plan
        )

        self.assertTrue(
            audit_omega_topology(
                self.plan,
                self.topology,
            )
        )

    def replace_first_route(self, **changes):
        original = self.topology.routes[0]

        corrupted = replace(
            original,
            **changes,
        )

        return replace(
            self.topology,
            routes=(
                corrupted,
                *self.topology.routes[1:],
            ),
        )

    def assert_rejected(self, topology):
        with self.assertRaises(ValueError):
            audit_omega_topology(
                self.plan,
                topology,
            )

    def test_missing_owner_route(self):
        corrupted = replace(
            self.topology,
            routes=self.topology.routes[1:],
        )

        self.assert_rejected(corrupted)

    def test_duplicate_owner_route(self):
        corrupted = replace(
            self.topology,
            routes=(
                *self.topology.routes,
                self.topology.routes[0],
            ),
        )

        self.assert_rejected(corrupted)

    def test_wrong_owner(self):
        corrupted = self.replace_first_route(
            owner=(9, 9),
        )

        self.assert_rejected(corrupted)

    def test_wrong_interval(self):
        corrupted = self.replace_first_route(
            interval=(0, 2),
        )

        self.assert_rejected(corrupted)

    def test_wrong_payload_components(self):
        route = self.topology.routes[0]

        corrupted = self.replace_first_route(
            payload_components=(
                route.payload_components + 1
            ),
        )

        self.assert_rejected(corrupted)

    def test_wrong_payload_bytes(self):
        route = self.topology.routes[0]

        corrupted = self.replace_first_route(
            payload_bytes=route.payload_bytes + 8,
        )

        self.assert_rejected(corrupted)

    def test_missing_edge(self):
        route = self.topology.routes[0]

        corrupted = self.replace_first_route(
            edges=route.edges[:-1],
        )

        self.assert_rejected(corrupted)

    def test_duplicate_edge(self):
        route = self.topology.routes[0]

        corrupted = self.replace_first_route(
            edges=(
                route.edges[0],
                *route.edges,
            ),
        )

        self.assert_rejected(corrupted)

    def test_non_neighbor_edge(self):
        route = self.topology.routes[0]

        corrupted_edges = (
            ((0, 0), (1, 2)),
            *route.edges[1:],
        )

        corrupted = self.replace_first_route(
            edges=corrupted_edges,
        )

        self.assert_rejected(corrupted)

    def test_receive_before_forward(self):
        route = self.topology.routes[0]

        corrupted_edges = (
            route.edges[2],
            route.edges[0],
            route.edges[1],
            *route.edges[3:],
        )

        corrupted = self.replace_first_route(
            edges=corrupted_edges,
        )

        self.assert_rejected(corrupted)

    def test_wrong_hop_count(self):
        route = self.topology.routes[0]

        corrupted = self.replace_first_route(
            maximum_hops=route.maximum_hops + 1,
        )

        self.assert_rejected(corrupted)

    def test_wrong_link_bytes(self):
        route = self.topology.routes[0]

        corrupted = self.replace_first_route(
            link_bytes=route.link_bytes + 8,
        )

        self.assert_rejected(corrupted)

    def test_wrong_total_link_bytes(self):
        corrupted = replace(
            self.topology,
            total_link_bytes=(
                self.topology.total_link_bytes + 8
            ),
        )

        self.assert_rejected(corrupted)

    def test_wrong_nonlocal_bytes(self):
        corrupted = replace(
            self.topology,
            aggregate_nonlocal_bytes=(
                self.topology.aggregate_nonlocal_bytes + 8
            ),
        )

        self.assert_rejected(corrupted)

    def test_wrong_machine_shape(self):
        corrupted = replace(
            self.topology,
            machine_shape=(3, 2),
        )

        self.assert_rejected(corrupted)

    def test_wrong_element_width(self):
        corrupted = replace(
            self.topology,
            element_bytes=4,
        )

        self.assert_rejected(corrupted)

    def test_self_consistent_wrong_ownership(self):
        """Correct-looking accounting must not hide wrong Omega."""

        first = self.topology.routes[0]
        second = self.topology.routes[1]

        # Swap ownership metadata and payload sizes while retaining
        # the original route roots and edge structures.
        # The total volume remains unchanged.
        changed_first = replace(
            first,
            interval=second.interval,
            payload_components=second.payload_components,
            payload_bytes=second.payload_bytes,
            link_bytes=second.link_bytes,
        )

        changed_second = replace(
            second,
            interval=first.interval,
            payload_components=first.payload_components,
            payload_bytes=first.payload_bytes,
            link_bytes=first.link_bytes,
        )

        corrupted = replace(
            self.topology,
            routes=(
                changed_first,
                changed_second,
                *self.topology.routes[2:],
            ),
        )

        # Aggregate traffic still agrees, but ownership is wrong.
        self.assertEqual(
            corrupted.total_link_bytes,
            self.topology.total_link_bytes,
        )

        self.assert_rejected(corrupted)


if __name__ == "__main__":
    unittest.main()
