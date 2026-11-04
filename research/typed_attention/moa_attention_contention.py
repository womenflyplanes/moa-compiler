
"""Directed-link contention model for MoA Omega communication.

Step 103: additive, conditional communication-time model.

For each logical stage, transfers using the same directed link
share one effective bandwidth. Their payload bytes are aggregated,
and the link incurs one effective stage latency.

Different directed links operate concurrently.

This model does not yet account for physical network routing,
bidirectional link sharing, switch contention, or measured MPI time.
"""

from dataclasses import dataclass
from math import isfinite, fsum

from moa_attention_omega_topology import OmegaTopology
from moa_attention_port_schedule import OmegaPortSchedule
from moa_attention_port_schedule_audit import (
    audit_omega_port_schedule,
)
from moa_attention_communication_time import (
    OmegaLinkParameters,
)


@dataclass(frozen=True)
class OmegaContendedLink:
    """Aggregate traffic on one directed link during one stage."""

    parent: tuple
    child: tuple
    transfer_count: int
    payload_bytes: int
    alpha: float
    beta: float
    predicted_seconds: float


@dataclass(frozen=True)
class OmegaContentionPrediction:
    """Certificate for stage-by-stage directed-link contention."""

    machine_shape: tuple
    outgoing_ports: int
    stages: tuple
    stage_times: tuple
    stage_count: int
    total_link_bytes: int
    total_seconds: float

    def verify(self):
        """Basic internal accounting, not an independent audit."""

        if (
            type(self.stage_count) is not int
            or self.stage_count < 0
            or self.stage_count != len(self.stages)
            or self.stage_count != len(self.stage_times)
        ):
            raise ValueError("contention stage count mismatch")

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

        if not _nonnegative_finite(self.total_seconds):
            raise ValueError("invalid total time")

        counted_bytes = 0
        expected_times = []

        for stage in self.stages:
            longest = 0.0

            for link in stage:
                if (
                    type(link.transfer_count) is not int
                    or link.transfer_count < 1
                    or type(link.payload_bytes) is not int
                    or link.payload_bytes < 1
                ):
                    raise ValueError("invalid link aggregation")

                if not _nonnegative_finite(
                    link.predicted_seconds
                ):
                    raise ValueError("invalid link duration")

                counted_bytes += link.payload_bytes
                longest = max(
                    longest,
                    link.predicted_seconds,
                )

            expected_times.append(longest)

        if counted_bytes != self.total_link_bytes:
            raise ValueError("communication bytes mismatch")

        for actual, expected in zip(
            self.stage_times,
            expected_times,
        ):
            if not _near(actual, expected):
                raise ValueError("stage duration mismatch")

        if not _near(
            self.total_seconds,
            fsum(expected_times),
        ):
            raise ValueError("total duration mismatch")

        return True


def _nonnegative_finite(value):
    if type(value) not in (int, float):
        return False

    try:
        return isfinite(value) and value >= 0
    except (TypeError, ValueError, OverflowError):
        return False


def _positive_finite(value):
    return _nonnegative_finite(value) and value > 0


def _near(a, b):
    if not (
        _nonnegative_finite(a)
        and _nonnegative_finite(b)
    ):
        return False

    return abs(a - b) <= (
        1.0e-12 * max(abs(b), 1.0e-12)
    )


def predict_omega_contention(
    plan,
    topology,
    schedule,
    links,
):
    """Predict communication time with shared directed-link capacity.

    Inputs:
      plan: original Omega ownership plan
      topology: independently auditable Omega routing
      schedule: port-constrained logical communication stages
      links: dictionary mapping (parent, child) to
             OmegaLinkParameters(alpha, beta)

    Output:
      Immutable prediction certificate containing aggregated
      per-link traffic and predicted stage durations.
    """

    if not isinstance(topology, OmegaTopology):
        raise TypeError("OmegaTopology required")

    if not isinstance(schedule, OmegaPortSchedule):
        raise TypeError("OmegaPortSchedule required")

    audit_omega_port_schedule(
        plan,
        topology,
        schedule,
    )

    if not isinstance(links, dict):
        raise TypeError("link parameter dictionary required")

    required = set()

    for route in topology.routes:
        for parent, child in route.edges:
            required.add((parent, child))

    if set(links) != required:
        raise ValueError(
            "directed-link parameter domain mismatch"
        )

    for parameters in links.values():
        if not isinstance(
            parameters,
            OmegaLinkParameters,
        ):
            raise TypeError("OmegaLinkParameters required")

        if not _nonnegative_finite(parameters.alpha):
            raise ValueError("invalid link latency")

        if not _positive_finite(parameters.beta):
            raise ValueError("invalid link bandwidth")

    contended_stages = []
    stage_times = []
    total_bytes = 0

    for stage in schedule.stages:
        grouped = {}

        for transfer in stage:
            key = (
                transfer.parent,
                transfer.child,
            )

            if key not in grouped:
                grouped[key] = [0, 0]

            grouped[key][0] += 1
            grouped[key][1] += transfer.payload_bytes

        contended_links = []

        for key in sorted(grouped):
            count, payload_bytes = grouped[key]
            parameters = links[key]

            duration = (
                parameters.alpha
                + payload_bytes / parameters.beta
            )

            if not _nonnegative_finite(duration):
                raise ValueError(
                    "nonfinite contended link duration"
                )

            contended_links.append(
                OmegaContendedLink(
                    parent=key[0],
                    child=key[1],
                    transfer_count=count,
                    payload_bytes=payload_bytes,
                    alpha=parameters.alpha,
                    beta=parameters.beta,
                    predicted_seconds=duration,
                )
            )

            total_bytes += payload_bytes

        contended_stages.append(
            tuple(contended_links)
        )

        stage_times.append(
            max(
                (
                    link.predicted_seconds
                    for link in contended_links
                ),
                default=0.0,
            )
        )

    total_seconds = fsum(stage_times)

    if not _nonnegative_finite(total_seconds):
        raise ValueError(
            "nonfinite contended communication time"
        )

    prediction = OmegaContentionPrediction(
        machine_shape=topology.machine_shape,
        outgoing_ports=schedule.outgoing_ports,
        stages=tuple(contended_stages),
        stage_times=tuple(stage_times),
        stage_count=schedule.stage_count,
        total_link_bytes=total_bytes,
        total_seconds=total_seconds,
    )

    prediction.verify()

    return prediction
