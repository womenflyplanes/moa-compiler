
"""Independent audit of MoA Omega port-constrained schedules.

Verifies a communication schedule against an independently audited
Omega topology.

The auditor does not invoke the scheduler or its verify() method.

Assumptions:
  - Each directed transfer occupies one outgoing port per stage.
  - A payload received in stage s may be forwarded only in stage s+1
    or later.
  - Incoming-port and physical-link contention are not modeled.
  - The topology is an abstract shaped machine, not a physical
    MPI or DGX network model.
"""

from collections import Counter

from moa_machine import Machine, iota

from moa_attention_omega_topology import OmegaTopology

from moa_attention_port_schedule import (
    OmegaPortSchedule,
    OmegaTransfer,
)

from moa_attention_topology_audit import (
    audit_omega_topology,
)


def _positive_integer(value, description):
    if type(value) is not int or value < 1:
        raise ValueError(description)


def _coordinate(value, shape):
    """Independently validate a machine coordinate."""

    if not isinstance(value, tuple):
        raise ValueError("machine coordinate must be a tuple")

    if len(value) != len(shape):
        raise ValueError("machine coordinate rank mismatch")

    for component, extent in zip(value, shape):
        if type(component) is not int:
            raise ValueError("noninteger machine coordinate")

        if not 0 <= component < extent:
            raise ValueError("machine coordinate outside machine")

    return value


def audit_omega_port_schedule(plan, topology, schedule):
    """Audit a staged schedule against its MoA Omega derivation.

    The original resource-constrained Omega plan is independently
    checked against the topology before examining any transfers.

    The schedule must contain exactly the directed transfers in
    the topology, with their original owners and payload sizes.

    Every stage must respect outgoing-port limits and strict
    receive-before-forward dependencies.
    """

    if not isinstance(topology, OmegaTopology):
        raise TypeError("OmegaTopology required")

    if not isinstance(schedule, OmegaPortSchedule):
        raise TypeError("OmegaPortSchedule required")

    # Establish that the topology agrees with the original
    # resource-constrained Omega ownership certificate.
    audit_omega_topology(plan, topology)

    machine = Machine(topology.machine_shape)
    shape = machine.shape
    coordinates = set(iota(shape))

    if tuple(schedule.machine_shape) != shape:
        raise ValueError("schedule machine shape mismatch")

    _positive_integer(
        schedule.outgoing_ports,
        "invalid outgoing-port count",
    )

    if (
        type(schedule.stage_count) is not int
        or schedule.stage_count != len(schedule.stages)
    ):
        raise ValueError("incorrect schedule stage count")

    # Independently reconstruct the expected transfer multiset.
    expected = Counter()
    payload_owners = set()
    reached = {}

    for route in topology.routes:
        owner = _coordinate(route.owner, shape)

        if owner in payload_owners:
            raise ValueError("duplicate topology owner")

        payload_owners.add(owner)
        reached[owner] = {owner}

        for parent, child in route.edges:
            parent = _coordinate(parent, shape)
            child = _coordinate(child, shape)

            key = (
                owner,
                parent,
                child,
                route.payload_bytes,
            )

            expected[key] += 1

    observed = Counter()
    total_bytes = 0

    for stage in schedule.stages:
        if not isinstance(stage, tuple):
            raise ValueError("schedule stage must be a tuple")

        outgoing = Counter()

        # Snapshot of payload availability at stage entry.
        # Receiving during this stage cannot authorize forwarding
        # until the following stage.
        available = {
            owner: set(nodes)
            for owner, nodes in reached.items()
        }

        deliveries = {
            owner: set()
            for owner in payload_owners
        }

        for transfer in stage:
            if not isinstance(transfer, OmegaTransfer):
                raise ValueError("invalid scheduled transfer")

            owner = _coordinate(transfer.owner, shape)
            parent = _coordinate(transfer.parent, shape)
            child = _coordinate(transfer.child, shape)

            _positive_integer(
                transfer.payload_bytes,
                "invalid transfer payload size",
            )

            if owner not in payload_owners:
                raise ValueError("unexpected payload owner")

            key = (
                owner,
                parent,
                child,
                transfer.payload_bytes,
            )

            if expected[key] == 0:
                raise ValueError(
                    "transfer absent from Omega topology"
                )

            observed[key] += 1

            if observed[key] > expected[key]:
                raise ValueError(
                    "duplicate scheduled transfer"
                )

            if parent not in available[owner]:
                raise ValueError(
                    "receive-before-forward violation"
                )

            if child in available[owner]:
                raise ValueError(
                    "payload delivered more than once"
                )

            if child in deliveries[owner]:
                raise ValueError(
                    "duplicate delivery in one stage"
                )

            outgoing[parent] += 1

            if outgoing[parent] > schedule.outgoing_ports:
                raise ValueError(
                    "outgoing-port limit exceeded"
                )

            deliveries[owner].add(child)
            total_bytes += transfer.payload_bytes

        # Commit deliveries only after every transfer in the
        # stage has been checked against entry-time availability.
        for owner, children in deliveries.items():
            reached[owner].update(children)

    if observed != expected:
        raise ValueError(
            "scheduled transfer multiset differs from topology"
        )

    for owner in payload_owners:
        if reached[owner] != coordinates:
            raise ValueError(
                "incomplete Omega payload reachability"
            )

    if total_bytes != topology.total_link_bytes:
        raise ValueError(
            "schedule changed topology communication volume"
        )

    if schedule.total_link_bytes != total_bytes:
        raise ValueError(
            "incorrect schedule link-byte accounting"
        )

    return True
