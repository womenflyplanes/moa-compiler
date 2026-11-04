
"""Port-constrained scheduling of MoA Omega communication topology.

Additive compiler module.

Transforms a verified abstract Omega communication topology into
deterministic, discrete communication stages.

Model assumptions:
  - Every directed transfer occupies one outgoing port for one stage.
  - A receiver can forward a payload only in a subsequent stage.
  - Incoming ports and link contention are not constrained here.
  - Transfers of different payloads may occur concurrently.
  - Stages are logical, not measured network time.

This is a candidate communication realization, not a model of
the actual MPI collective algorithm or physical DGX topology.
"""

from dataclasses import dataclass

from moa_machine import Machine, iota, gamma

from moa_attention_omega_topology import OmegaTopology


@dataclass(frozen=True)
class OmegaTransfer:
    """One directed transfer of an Omega-owned payload."""

    owner: tuple
    parent: tuple
    child: tuple
    payload_bytes: int


@dataclass(frozen=True)
class OmegaPortSchedule:
    """Certificate for port-constrained Omega communication."""

    machine_shape: tuple
    outgoing_ports: int
    stages: tuple
    stage_count: int
    total_link_bytes: int

    def verify(self):
        """Check certificate accounting and basic structure.

        Full dependency, topology, and reachability verification
        belongs to a separate independent auditor.
        """

        machine = Machine(self.machine_shape)
        coordinates = set(iota(machine.shape))

        if (
            type(self.outgoing_ports) is not int
            or self.outgoing_ports < 1
        ):
            raise ValueError("invalid outgoing port count")

        if self.stage_count != len(self.stages):
            raise ValueError("incorrect stage count")

        total_bytes = 0

        for stage in self.stages:
            outgoing = {}

            for transfer in stage:
                if not isinstance(transfer, OmegaTransfer):
                    raise ValueError("invalid transfer")

                if transfer.owner not in coordinates:
                    raise ValueError("invalid payload owner")

                if transfer.parent not in coordinates:
                    raise ValueError("invalid parent coordinate")

                if transfer.child not in coordinates:
                    raise ValueError("invalid child coordinate")

                if (
                    type(transfer.payload_bytes) is not int
                    or transfer.payload_bytes <= 0
                ):
                    raise ValueError("invalid payload byte count")

                outgoing[transfer.parent] = (
                    outgoing.get(transfer.parent, 0) + 1
                )

                if outgoing[transfer.parent] > self.outgoing_ports:
                    raise ValueError("outgoing port limit exceeded")

                total_bytes += transfer.payload_bytes

        if total_bytes != self.total_link_bytes:
            raise ValueError("incorrect total link bytes")

        return True


def schedule_omega_ports(topology, outgoing_ports=1):
    """Schedule the exact transfers of a verified Omega topology.

    The algorithm processes all routes in deterministic owner order.

    At each logical stage, it selects every currently eligible
    transfer that fits the outgoing-port constraints.

    Eligibility requires:
      1. The parent already possesses the owner's payload.
      2. The child has not already received that payload.
      3. The parent has an available outgoing port.

    Payloads received in stage s become available for forwarding
    in stage s+1.

    This is a deterministic greedy schedule, not an optimality
    certificate for minimum stage count.
    """

    if not isinstance(topology, OmegaTopology):
        raise TypeError("OmegaTopology required")

    topology.verify()

    if (
        type(outgoing_ports) is not int
        or outgoing_ports < 1
    ):
        raise ValueError("outgoing_ports must be a positive integer")

    machine = Machine(topology.machine_shape)
    coordinates = iota(machine.shape)

    ordered_routes = tuple(
        sorted(
            topology.routes,
            key=lambda route: gamma(
                route.owner,
                machine.shape,
            ),
        )
    )

    # The independent topology auditor must be applied by the
    # calling verification pipeline before this scheduling step.
    # Here we additionally check edge ordering and dependencies
    # to avoid scheduling malformed topology certificates.

    remaining = []
    reached = {}

    for route in ordered_routes:
        if route.owner not in coordinates:
            raise ValueError("invalid route owner")

        reached[route.owner] = {route.owner}

        for parent, child in route.edges:
            remaining.append(
                OmegaTransfer(
                    owner=route.owner,
                    parent=parent,
                    child=child,
                    payload_bytes=route.payload_bytes,
                )
            )

    stages = []

    while remaining:
        stage = []
        deferred = []

        outgoing_used = {}

        # Snapshot prevents forwarding a payload received
        # in the current stage.
        available = {
            owner: set(owners_reached)
            for owner, owners_reached in reached.items()
        }

        for transfer in remaining:
            owner = transfer.owner
            parent = transfer.parent
            child = transfer.child

            if parent not in available[owner]:
                deferred.append(transfer)
                continue

            if child in available[owner]:
                raise ValueError("duplicate payload delivery")

            used = outgoing_used.get(parent, 0)

            if used >= outgoing_ports:
                deferred.append(transfer)
                continue

            stage.append(transfer)
            outgoing_used[parent] = used + 1

        if not stage:
            raise ValueError(
                "no legal communication progress; "
                "topology may contain invalid dependencies"
            )

        for transfer in stage:
            reached[transfer.owner].add(transfer.child)

        stages.append(tuple(stage))

        selected = set(stage)

        remaining = [
            transfer
            for transfer in remaining
            if transfer not in selected
        ]

    expected = set(coordinates)

    for route in ordered_routes:
        if reached[route.owner] != expected:
            raise ValueError(
                "incomplete Omega payload delivery"
            )

    schedule = OmegaPortSchedule(
        machine_shape=machine.shape,
        outgoing_ports=outgoing_ports,
        stages=tuple(stages),
        stage_count=len(stages),
        total_link_bytes=sum(
            transfer.payload_bytes
            for stage in stages
            for transfer in stage
        ),
    )

    schedule.verify()

    if schedule.total_link_bytes != topology.total_link_bytes:
        raise ValueError(
            "scheduling changed communication volume"
        )

    return schedule
