
"""Regression and adversarial tests for MoA Omega port scheduling.

Tests deterministic, port-constrained communication schedules derived
from verified resource-constrained Omega ownership.

The independent auditor checks schedules against the original
Omega topology and ownership certificate.

These tests concern an abstract shaped machine, not measured MPI
collective behavior or physical DGX network performance.
"""

from dataclasses import replace
import unittest

from moa_machine import Work

from moa_attention_omega_topology import derive_omega_topology

from moa_attention_port_schedule import (
    OmegaTransfer,
    OmegaPortSchedule,
    schedule_omega_ports,
)

from moa_attention_port_schedule_audit import (
    audit_omega_port_schedule,
)

from test_attention_omega_topology import make_plan


def make_case(work, shape, counts, ports=1):
    """Construct and independently audit a valid schedule."""

    plan = make_plan(work, shape, counts)
    topology = derive_omega_topology(plan)
    schedule = schedule_omega_ports(topology, ports)

    if not audit_omega_port_schedule(plan, topology, schedule):
        raise AssertionError("valid Omega schedule rejected")

    return plan, topology, schedule


def with_stages(schedule, stages):
    """Replace stages and recompute accounting for mutation tests."""

    return replace(
        schedule,
        stages=tuple(stages),
        stage_count=len(stages),
        total_link_bytes=sum(
            transfer.payload_bytes
            for stage in stages
            for transfer in stage
        ),
    )


class TestOmegaPortSchedule(unittest.TestCase):

    def test_heterogeneous_ownership(self):
        for ports, expected_stages in (
            (1, 11),
            (2, 6),
            (100, 3),
        ):
            with self.subTest(ports=ports):
                plan, topology, schedule = make_case(
                    Work(7, 5, 4, 3),
                    (2, 3),
                    (1, 2, 1, 1, 1, 1),
                    ports,
                )

                self.assertEqual(
                    schedule.stage_count,
                    expected_stages,
                )

                self.assertEqual(
                    schedule.total_link_bytes,
                    840,
                )

                self.assertTrue(
                    audit_omega_port_schedule(
                        plan, topology, schedule
                    )
                )

    def test_idle_machine_components(self):
        for ports in (1, 2, 100):
            with self.subTest(ports=ports):
                plan, topology, schedule = make_case(
                    Work(3, 6, 2, 4),
                    (2, 3),
                    (3, 0, 0, 0, 0, 0),
                    ports,
                )

                self.assertEqual(schedule.stage_count, 3)
                self.assertEqual(schedule.total_link_bytes, 480)

    def test_multidimensional_machine(self):
        for ports, expected_stages in (
            (1, 12),
            (2, 6),
            (100, 3),
        ):
            with self.subTest(ports=ports):
                plan, topology, schedule = make_case(
                    Work(8, 4, 3, 2),
                    (2, 2, 2),
                    (1, 1, 1, 1, 1, 1, 1, 1),
                    ports,
                )

                self.assertEqual(
                    schedule.stage_count,
                    expected_stages,
                )

                self.assertEqual(
                    schedule.total_link_bytes,
                    896,
                )

    def test_singleton_machine(self):
        for ports in (1, 2, 100):
            with self.subTest(ports=ports):
                plan, topology, schedule = make_case(
                    Work(4, 3, 2, 5),
                    (1,),
                    (4,),
                    ports,
                )

                self.assertEqual(schedule.stages, ())
                self.assertEqual(schedule.stage_count, 0)
                self.assertEqual(schedule.total_link_bytes, 0)

    def test_deterministic_schedule(self):
        plan, topology, schedule = make_case(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
            2,
        )

        repeated = schedule_omega_ports(topology, 2)

        self.assertEqual(schedule, repeated)

    def test_communication_volume_invariant(self):
        plan = make_plan(
            Work(11, 7, 8, 4),
            (3, 2),
            (1, 2, 3, 1, 2, 2),
        )

        topology = derive_omega_topology(plan)

        for ports in (1, 2, 3, 100):
            with self.subTest(ports=ports):
                schedule = schedule_omega_ports(
                    topology, ports
                )

                self.assertEqual(
                    schedule.total_link_bytes,
                    topology.total_link_bytes,
                )

                self.assertTrue(
                    audit_omega_port_schedule(
                        plan, topology, schedule
                    )
                )

    def test_reject_invalid_port_counts(self):
        plan = make_plan(
            Work(3, 6, 2, 4),
            (2, 3),
            (3, 0, 0, 0, 0, 0),
        )

        topology = derive_omega_topology(plan)

        for ports in (0, -1, True, 1.5, "2"):
            with self.subTest(ports=ports):
                with self.assertRaises(ValueError):
                    schedule_omega_ports(topology, ports)


class TestOmegaPortScheduleAdversarial(unittest.TestCase):

    def setUp(self):
        self.plan, self.topology, self.schedule = make_case(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
            1,
        )

    def reject(self, candidate):
        with self.assertRaises(ValueError):
            audit_omega_port_schedule(
                self.plan,
                self.topology,
                candidate,
            )

    def test_duplicate_transfer(self):
        stages = list(self.schedule.stages)
        first = stages[0][0]

        stages[0] = stages[0] + (first,)

        self.reject(with_stages(self.schedule, stages))

    def test_missing_transfer(self):
        stages = list(self.schedule.stages)
        stages[0] = stages[0][1:]

        self.reject(with_stages(self.schedule, stages))

    def test_wrong_payload_bytes(self):
        stages = list(self.schedule.stages)
        first = stages[0][0]

        stages[0] = (
            replace(
                first,
                payload_bytes=first.payload_bytes + 8,
            ),
        ) + stages[0][1:]

        self.reject(with_stages(self.schedule, stages))

    def test_wrong_owner(self):
        stages = list(self.schedule.stages)
        first = stages[0][0]

        other_owner = next(
            route.owner
            for route in self.topology.routes
            if route.owner != first.owner
        )

        stages[0] = (
            replace(first, owner=other_owner),
        ) + stages[0][1:]

        self.reject(with_stages(self.schedule, stages))

    def test_wrong_parent(self):
        stages = list(self.schedule.stages)
        first = stages[0][0]

        stages[0] = (
            replace(first, parent=first.child),
        ) + stages[0][1:]

        self.reject(with_stages(self.schedule, stages))

    def test_wrong_child(self):
        stages = list(self.schedule.stages)
        first = stages[0][0]

        stages[0] = (
            replace(first, child=first.parent),
        ) + stages[0][1:]

        self.reject(with_stages(self.schedule, stages))

    def test_wrong_machine_shape(self):
        candidate = replace(
            self.schedule,
            machine_shape=(3, 2),
        )

        self.reject(candidate)

    def test_wrong_stage_count(self):
        candidate = replace(
            self.schedule,
            stage_count=self.schedule.stage_count + 1,
        )

        self.reject(candidate)

    def test_wrong_total_bytes(self):
        candidate = replace(
            self.schedule,
            total_link_bytes=(
                self.schedule.total_link_bytes + 8
            ),
        )

        self.reject(candidate)

    def test_invalid_outgoing_ports(self):
        candidate = replace(
            self.schedule,
            outgoing_ports=0,
        )

        self.reject(candidate)

    def test_excess_outgoing_transfers(self):
        # Relabel the port capacity without changing the stages.
        # A schedule generated for 100 ports may contain multiple
        # outgoing transfers from one parent in one stage.
        plan, topology, schedule = make_case(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
            100,
        )

        candidate = replace(
            schedule,
            outgoing_ports=1,
        )

        with self.assertRaises(ValueError):
            audit_omega_port_schedule(
                plan, topology, candidate
            )

    def test_receive_before_forward_violation(self):
        # Move a non-root forwarding transfer into stage zero,
        # before its parent could have received the payload.
        stages = [
            list(stage)
            for stage in self.schedule.stages
        ]

        transfer = None
        source_stage = None

        for index, stage in enumerate(stages):
            for item in stage:
                if item.parent != item.owner:
                    transfer = item
                    source_stage = index
                    break

            if transfer is not None:
                break

        self.assertIsNotNone(transfer)

        stages[source_stage].remove(transfer)
        stages[0].append(transfer)

        candidate = with_stages(
            self.schedule,
            tuple(tuple(stage) for stage in stages),
        )

        self.reject(candidate)

    def test_same_stage_forwarding_violation(self):
        # Construct a schedule where an intermediate component
        # receives and forwards the same payload in one stage.
        plan, topology, schedule = make_case(
            Work(3, 6, 2, 4),
            (2, 3),
            (3, 0, 0, 0, 0, 0),
            100,
        )

        route = topology.routes[0]
        root = route.owner

        first = next(
            (parent, child)
            for parent, child in route.edges
            if parent == root
        )

        second = next(
            (parent, child)
            for parent, child in route.edges
            if parent == first[1]
        )

        first_transfer = OmegaTransfer(
            owner=root,
            parent=first[0],
            child=first[1],
            payload_bytes=route.payload_bytes,
        )

        second_transfer = OmegaTransfer(
            owner=root,
            parent=second[0],
            child=second[1],
            payload_bytes=route.payload_bytes,
        )

        stages = [
            list(stage)
            for stage in schedule.stages
        ]

        for stage in stages:
            if second_transfer in stage:
                stage.remove(second_transfer)

        stages[0].append(second_transfer)

        candidate = with_stages(
            schedule,
            tuple(tuple(stage) for stage in stages),
        )

        with self.assertRaises(ValueError):
            audit_omega_port_schedule(
                plan, topology, candidate
            )

    def test_empty_schedule_rejected(self):
        candidate = with_stages(
            self.schedule,
            (),
        )

        self.reject(candidate)

    def test_noninteger_payload_rejected(self):
        stages = list(self.schedule.stages)
        first = stages[0][0]

        stages[0] = (
            replace(first, payload_bytes=8.0),
        ) + stages[0][1:]

        self.reject(with_stages(self.schedule, stages))

    def test_noninteger_coordinate_rejected(self):
        stages = list(self.schedule.stages)
        first = stages[0][0]

        stages[0] = (
            replace(first, parent=(0.0, 0)),
        ) + stages[0][1:]

        self.reject(with_stages(self.schedule, stages))


if __name__ == "__main__":
    unittest.main()
