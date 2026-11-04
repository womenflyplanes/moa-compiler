
"""MoA Omega-derived nearest-neighbor communication topology.

Additive module: does not modify existing compiler components.

Constructs one deterministic, owner-rooted spanning tree for each
nonempty Omega output ownership interval.

This is a candidate allgather realization on an abstract shaped
machine array, not an implementation or prediction of MPI's actual
collective algorithm or the physical DGX network.
"""

from dataclasses import dataclass

from moa_machine import Machine, iota, gamma

from moa_attention_resource_planner import ResourceOmegaPlan

from moa_attention_resource_communication import (
    derive_resource_communication,
)

from moa_attention_communication_audit import (
    audit_resource_communication_plan,
)


@dataclass(frozen=True)
class OmegaOwnerRoute:
    """A nearest-neighbor spanning tree for one output owner."""

    owner: tuple
    interval: tuple
    payload_components: int
    payload_bytes: int
    edges: tuple
    link_bytes: int
    maximum_hops: int


@dataclass(frozen=True)
class OmegaTopology:
    """Complete owner-rooted routing description."""

    machine_shape: tuple
    element_bytes: int
    routes: tuple
    total_link_bytes: int
    aggregate_nonlocal_bytes: int

    def verify(self):
        """Check internal accounting; independent audit comes later."""

        machine = Machine(self.machine_shape)
        p = machine.tau

        if type(self.element_bytes) is not int:
            raise ValueError("invalid element byte width")

        if self.element_bytes <= 0:
            raise ValueError("invalid element byte width")

        total = 0
        seen_owners = set()

        for route in self.routes:
            if route.owner in seen_owners:
                raise ValueError("duplicate Omega owner")

            seen_owners.add(route.owner)

            if route.owner not in iota(machine.shape):
                raise ValueError("owner outside machine shape")

            if route.payload_components <= 0:
                raise ValueError("empty owner route")

            if route.payload_bytes != (
                route.payload_components * self.element_bytes
            ):
                raise ValueError("payload byte mismatch")

            if len(route.edges) != p - 1:
                raise ValueError("incorrect spanning-tree edge count")

            if route.link_bytes != (
                (p - 1) * route.payload_bytes
            ):
                raise ValueError("incorrect route link bytes")

            total += route.link_bytes

        if self.total_link_bytes != total:
            raise ValueError("incorrect total link bytes")

        if self.aggregate_nonlocal_bytes != total:
            raise ValueError("logical volume mismatch")

        return True


def neighboring_coordinates(coordinate, shape):
    """Generate valid nearest neighbors in deterministic gamma order."""

    neighbors = []

    for axis in range(len(shape)):
        for direction in (-1, 1):
            candidate = list(coordinate)
            candidate[axis] += direction

            if 0 <= candidate[axis] < shape[axis]:
                neighbors.append(tuple(candidate))

    return tuple(
        sorted(
            neighbors,
            key=lambda c: gamma(c, shape),
        )
    )


def owner_rooted_tree(machine, owner):
    """Construct a deterministic breadth-first nearest-neighbor tree.

    Edges are ordered so that each parent is reached before
    any of its children are processed.
    """

    if not isinstance(machine, Machine):
        raise TypeError("Machine required")

    coordinates = iota(machine.shape)

    if owner not in coordinates:
        raise ValueError("invalid owner coordinate")

    visited = {owner}
    frontier = [owner]

    edges = []
    maximum_hops = 0
    depth = {owner: 0}

    while frontier:
        next_frontier = []

        for parent in frontier:
            for child in neighboring_coordinates(
                parent,
                machine.shape,
            ):
                if child in visited:
                    continue

                visited.add(child)

                depth[child] = depth[parent] + 1

                maximum_hops = max(
                    maximum_hops,
                    depth[child],
                )

                edges.append((parent, child))
                next_frontier.append(child)

        frontier = next_frontier

    if len(visited) != machine.tau:
        raise ValueError("incomplete machine reachability")

    if len(edges) != machine.tau - 1:
        raise ValueError("invalid spanning tree")

    return tuple(edges), maximum_hops


def derive_omega_topology(plan, element_bytes=8):
    """Derive all owner-rooted routes from verified Omega ownership."""

    if not isinstance(plan, ResourceOmegaPlan):
        raise TypeError("ResourceOmegaPlan required")

    plan.verify()

    if type(element_bytes) is not int or element_bytes <= 0:
        raise ValueError("invalid element byte width")

    communication = derive_resource_communication(
        plan,
        element_bytes,
    )

    audit_resource_communication_plan(
        plan,
        communication,
    )

    machine_shape = (
        plan.certificate.scheduled.schedule.machine_shape
    )

    machine = Machine(tuple(machine_shape))

    if len(plan.intervals) != machine.tau:
        raise ValueError("Omega ownership and machine mismatch")

    coordinates = iota(machine.shape)

    routes = []

    for ordinal, interval in enumerate(plan.intervals):
        begin, end = interval

        payload_components = communication.counts[ordinal]

        if payload_components == 0:
            continue

        owner = coordinates[ordinal]

        edges, maximum_hops = owner_rooted_tree(
            machine,
            owner,
        )

        payload_bytes = (
            payload_components * element_bytes
        )

        routes.append(
            OmegaOwnerRoute(
                owner=owner,
                interval=(begin, end),
                payload_components=payload_components,
                payload_bytes=payload_bytes,
                edges=edges,
                link_bytes=len(edges) * payload_bytes,
                maximum_hops=maximum_hops,
            )
        )

    total_link_bytes = sum(
        route.link_bytes
        for route in routes
    )

    topology = OmegaTopology(
        machine_shape=machine.shape,
        element_bytes=element_bytes,
        routes=tuple(routes),
        total_link_bytes=total_link_bytes,
        aggregate_nonlocal_bytes=(
            communication.aggregate_nonlocal_bytes
        ),
    )

    topology.verify()

    return topology
