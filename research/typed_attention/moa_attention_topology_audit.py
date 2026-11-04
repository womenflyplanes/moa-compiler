
"""Independent audit of MoA Omega-derived communication topology.

Checks owner-rooted nearest-neighbor routing against the original
verified resource plan.

Does not call the topology constructor or its internal verifier.

Scope: abstract shaped-machine communication, not physical MPI
network traffic, scheduling optimality, or measured execution time.
"""

from moa_machine import Machine, iota, gamma
from moa_attention_resource_planner import ResourceOmegaPlan


def _integer(value, minimum, description):
    if type(value) is not int or value < minimum:
        raise ValueError(description)


def _coordinate(value, shape):
    """Independently validate a machine psi coordinate."""

    if not isinstance(value, tuple):
        raise ValueError("machine coordinate must be a tuple")

    if len(value) != len(shape):
        raise ValueError("machine coordinate rank mismatch")

    for component, extent in zip(value, shape):
        if type(component) is not int:
            raise ValueError("noninteger machine coordinate")

        if not 0 <= component < extent:
            raise ValueError("machine coordinate out of bounds")

    return value


def _adjacent(parent, child):
    """Check nearest-neighbor adjacency in machine index space."""

    return sum(
        abs(a - b)
        for a, b in zip(parent, child)
    ) == 1


def audit_omega_topology(plan, topology):
    """Independently verify an Omega-derived allgather topology.

    Verifies:
      - original resource-constrained Omega plan
      - machine shape and all component coordinates
      - exact ownership intervals and output payload sizes
      - one route for every nonempty owner, none for idle owners
      - unique, directed nearest-neighbor tree edges
      - parent-before-child ordering and reachability
      - maximum hop count
      - exact logical link-byte accounting

    The topology is not assumed to be time-optimal.
    """

    if not isinstance(plan, ResourceOmegaPlan):
        raise TypeError("ResourceOmegaPlan required")

    plan.verify()

    machine_shape = tuple(
        plan.certificate.scheduled.schedule.machine_shape
    )

    machine = Machine(machine_shape)
    coordinates = iota(machine.shape)

    if tuple(topology.machine_shape) != machine.shape:
        raise ValueError("topology machine shape mismatch")

    _integer(
        topology.element_bytes,
        1,
        "invalid element byte width",
    )

    work_shape = plan.certificate.scheduled.schedule

    value_extent = work_shape.value_extent

    _integer(
        value_extent,
        1,
        "invalid output value extent",
    )

    if len(plan.intervals) != machine.tau:
        raise ValueError("Omega interval count mismatch")

    expected = {}

    for coordinate, interval in zip(
        coordinates,
        plan.intervals,
    ):
        try:
            begin, end = interval
        except (TypeError, ValueError):
            raise ValueError(
                "invalid Omega interval"
            ) from None

        _integer(begin, 0, "invalid interval beginning")
        _integer(end, 0, "invalid interval ending")

        if end < begin:
            raise ValueError("reversed Omega interval")

        count = (end - begin) * value_extent

        if count > 0:
            expected[coordinate] = (
                (begin, end),
                count,
                count * topology.element_bytes,
            )

    seen_owners = set()
    total_link_bytes = 0

    for route in topology.routes:
        owner = _coordinate(
            route.owner,
            machine.shape,
        )

        if owner not in expected:
            raise ValueError(
                "unexpected or idle Omega owner"
            )

        if owner in seen_owners:
            raise ValueError(
                "duplicate Omega owner route"
            )

        seen_owners.add(owner)

        interval, components, payload_bytes = expected[owner]

        if tuple(route.interval) != interval:
            raise ValueError(
                "Omega ownership interval mismatch"
            )

        if route.payload_components != components:
            raise ValueError(
                "Omega payload component mismatch"
            )

        if route.payload_bytes != payload_bytes:
            raise ValueError(
                "Omega payload byte mismatch"
            )

        # Every tree starts with exactly its owner reached.
        reached = {owner}
        depths = {owner: 0}
        used_edges = set()

        for edge in route.edges:
            try:
                parent, child = edge
            except (TypeError, ValueError):
                raise ValueError(
                    "invalid directed routing edge"
                ) from None

            parent = _coordinate(
                parent,
                machine.shape,
            )

            child = _coordinate(
                child,
                machine.shape,
            )

            if not _adjacent(parent, child):
                raise ValueError(
                    "non-neighbor machine transfer"
                )

            if (parent, child) in used_edges:
                raise ValueError(
                    "duplicate routing edge"
                )

            if parent not in reached:
                raise ValueError(
                    "receive-before-forward violation"
                )

            if child in reached:
                raise ValueError(
                    "duplicate delivery or routing cycle"
                )

            used_edges.add((parent, child))

            depths[child] = depths[parent] + 1
            reached.add(child)

        if reached != set(coordinates):
            raise ValueError(
                "incomplete Omega payload reachability"
            )

        if len(used_edges) != machine.tau - 1:
            raise ValueError(
                "incorrect spanning-tree edge count"
            )

        actual_hops = max(depths.values())

        if route.maximum_hops != actual_hops:
            raise ValueError(
                "incorrect maximum hop count"
            )

        expected_link_bytes = (
            len(used_edges) * payload_bytes
        )

        if route.link_bytes != expected_link_bytes:
            raise ValueError(
                "incorrect owner link-byte count"
            )

        total_link_bytes += expected_link_bytes

    if seen_owners != set(expected):
        raise ValueError(
            "missing nonempty Omega owner route"
        )

    if topology.total_link_bytes != total_link_bytes:
        raise ValueError(
            "incorrect total link-byte count"
        )

    expected_nonlocal_bytes = (
        (machine.tau - 1)
        * sum(item[2] for item in expected.values())
    )

    if topology.aggregate_nonlocal_bytes != (
        expected_nonlocal_bytes
    ):
        raise ValueError(
            "incorrect aggregate nonlocal bytes"
        )

    if total_link_bytes != expected_nonlocal_bytes:
        raise ValueError(
            "tree traffic and logical nonlocal demand disagree"
        )

    return True
