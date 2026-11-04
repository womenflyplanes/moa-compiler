
"""Conditional communication-time prediction for MoA Omega schedules.

Additive compiler module.

Predicts communication time from an independently verified,
port-constrained Omega schedule and abstract directed-link parameters.

For each scheduled transfer e:

    t(e) = alpha(e) + payload_bytes(e) / beta(e)

For each logical stage s:

    T(s) = max(t(e) for e in s), or zero for an empty stage

Total predicted communication time:

    T = sum(T(s) for s in stages)

Assumptions:
  - Scheduled transfers within a stage execute concurrently.
  - A stage finishes when its slowest transfer finishes.
  - Stages execute sequentially.
  - Link parameters describe effective transfer behavior.
  - Incoming-port contention, shared bandwidth, synchronization
    overhead, and physical-network routing are not modeled.

This is a conditional prediction, not measured MPI or DGX time.
"""

from dataclasses import dataclass
from math import isfinite

from moa_attention_omega_topology import OmegaTopology

from moa_attention_port_schedule import OmegaPortSchedule

from moa_attention_port_schedule_audit import (
    audit_omega_port_schedule,
)


@dataclass(frozen=True)
class OmegaLinkParameters:
    """Latency and effective bandwidth for one directed link."""

    alpha: float
    beta: float


@dataclass(frozen=True)
class OmegaCommunicationPrediction:
    """Predicted time certificate for an Omega communication schedule."""

    machine_shape: tuple
    outgoing_ports: int
    stage_times: tuple
    stage_count: int
    total_time: float
    total_link_bytes: int

    def verify(self):
        """Check numerical and structural accounting.

        Independent validation against the original schedule
        and link parameters belongs to a separate auditor.
        """

        if (
            type(self.stage_count) is not int
            or self.stage_count < 0
        ):
            raise ValueError("invalid prediction stage count")

        if self.stage_count != len(self.stage_times):
            raise ValueError("prediction stage count mismatch")

        if (
            type(self.outgoing_ports) is not int
            or self.outgoing_ports < 1
        ):
            raise ValueError("invalid outgoing-port count")

        if (
            type(self.total_link_bytes) is not int
            or self.total_link_bytes < 0
        ):
            raise ValueError("invalid communication volume")

        for value in self.stage_times:
            if not _nonnegative_finite(value):
                raise ValueError("invalid stage time")

        if not _nonnegative_finite(self.total_time):
            raise ValueError("invalid total communication time")

        expected = sum(self.stage_times)

        if not isfinite(expected):
            raise ValueError("nonfinite total communication time")

        if not _close(expected, self.total_time):
            raise ValueError("total time accounting mismatch")

        return True


def _positive_finite(value):
    """Reject booleans, nonnumeric values, zero, NaN, and infinity."""

    if type(value) not in (int, float):
        return False

    try:
        return isfinite(value) and value > 0
    except (OverflowError, TypeError, ValueError):
        return False


def _nonnegative_finite(value):
    if type(value) not in (int, float):
        return False

    try:
        return isfinite(value) and value >= 0
    except (OverflowError, TypeError, ValueError):
        return False


def _close(a, b):
    """Tolerance for floating-point accumulation of stage times."""

    return abs(a - b) <= (
        1.0e-12 * max(abs(a), abs(b), 1.0e-12)
    )


def _validate_link_parameters(parameters):
    if not isinstance(parameters, OmegaLinkParameters):
        raise TypeError("OmegaLinkParameters required")

    if not _nonnegative_finite(parameters.alpha):
        raise ValueError("invalid link latency")

    if not _positive_finite(parameters.beta):
        raise ValueError("invalid link bandwidth")


def uniform_omega_links(topology, alpha, beta):
    """Construct parameters for every directed topology edge.

    Keys are (parent_coordinate, child_coordinate).

    Multiple Omega payload owners may share the same directed link.
    """

    if not isinstance(topology, OmegaTopology):
        raise TypeError("OmegaTopology required")

    topology.verify()

    parameters = OmegaLinkParameters(alpha, beta)
    _validate_link_parameters(parameters)

    links = {}

    for route in topology.routes:
        for parent, child in route.edges:
            links[(parent, child)] = parameters

    return links


def predict_omega_communication_time(
    plan,
    topology,
    schedule,
    links,
):
    """Derive a conditional time prediction from verified Omega data.

    Link parameters must be provided for exactly the directed
    links used by the topology.

    Each link may carry multiple payloads in different stages,
    or multiple payloads within one stage when outgoing-port
    capacity permits. Shared-link contention is not modeled.
    """

    if not isinstance(topology, OmegaTopology):
        raise TypeError("OmegaTopology required")

    if not isinstance(schedule, OmegaPortSchedule):
        raise TypeError("OmegaPortSchedule required")

    # Independent legality check, grounded in original Omega
    # ownership rather than scheduler self-verification.
    audit_omega_port_schedule(plan, topology, schedule)

    if not isinstance(links, dict):
        raise TypeError("link parameter dictionary required")

    expected_links = set()

    for route in topology.routes:
        for parent, child in route.edges:
            expected_links.add((parent, child))

    if set(links) != expected_links:
        raise ValueError(
            "link parameters do not match Omega topology"
        )

    for parameters in links.values():
        _validate_link_parameters(parameters)

    stage_times = []

    for stage in schedule.stages:
        transfer_times = []

        for transfer in stage:
            key = (transfer.parent, transfer.child)
            parameters = links[key]

            transfer_time = (
                parameters.alpha
                + transfer.payload_bytes / parameters.beta
            )

            if not _nonnegative_finite(transfer_time):
                raise ValueError(
                    "nonfinite predicted transfer time"
                )

            transfer_times.append(transfer_time)

        stage_times.append(
            max(transfer_times, default=0.0)
        )

    total_time = sum(stage_times)

    if not _nonnegative_finite(total_time):
        raise ValueError(
            "nonfinite predicted communication time"
        )

    prediction = OmegaCommunicationPrediction(
        machine_shape=topology.machine_shape,
        outgoing_ports=schedule.outgoing_ports,
        stage_times=tuple(stage_times),
        stage_count=schedule.stage_count,
        total_time=total_time,
        total_link_bytes=schedule.total_link_bytes,
    )

    prediction.verify()

    return prediction
