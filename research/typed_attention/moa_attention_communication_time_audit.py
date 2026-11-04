
"""Independent audit of MoA Omega communication-time predictions.

Checks a conditional timing certificate against:
  - Original resource-constrained Omega ownership
  - Independently audited communication topology
  - Independently audited port-constrained schedule
  - Supplied directed-link latency and bandwidth parameters

Timing model:
    transfer_time = alpha + payload_bytes / beta
    stage_time = maximum transfer_time within the stage
    total_time = sum of stage times

The auditor does not invoke the prediction function or the
prediction certificate's verify() method.

These predictions concern an abstract shaped machine, not measured
MPI or physical DGX network behavior.
"""

from math import isfinite, fsum

from moa_attention_omega_topology import OmegaTopology

from moa_attention_port_schedule import OmegaPortSchedule

from moa_attention_port_schedule_audit import (
    audit_omega_port_schedule,
)

from moa_attention_communication_time import (
    OmegaLinkParameters,
    OmegaCommunicationPrediction,
)


def _finite_nonnegative(value):
    """Validate a nonnegative, finite numerical quantity."""

    if type(value) not in (int, float):
        return False

    try:
        return isfinite(value) and value >= 0
    except (OverflowError, TypeError, ValueError):
        return False


def _finite_positive(value):
    """Validate a strictly positive, finite numerical quantity."""

    if type(value) not in (int, float):
        return False

    try:
        return isfinite(value) and value > 0
    except (OverflowError, TypeError, ValueError):
        return False


def _times_agree(observed, expected):
    """Compare independently computed floating-point times.

    Relative tolerance applies to ordinary magnitudes; a small
    absolute tolerance accommodates rounding near zero.
    """

    if not (
        _finite_nonnegative(observed)
        and _finite_nonnegative(expected)
    ):
        return False

    tolerance = (
        1.0e-12 * max(abs(expected), 1.0e-12)
    )

    return abs(observed - expected) <= tolerance


def audit_omega_communication_time(
    plan,
    topology,
    schedule,
    links,
    prediction,
):
    """Independently audit a conditional Omega time prediction."""

    if not isinstance(topology, OmegaTopology):
        raise TypeError("OmegaTopology required")

    if not isinstance(schedule, OmegaPortSchedule):
        raise TypeError("OmegaPortSchedule required")

    if not isinstance(
        prediction,
        OmegaCommunicationPrediction,
    ):
        raise TypeError("OmegaCommunicationPrediction required")

    # Validate the schedule against the original Omega plan.
    # This transitively invokes the independent topology audit.
    audit_omega_port_schedule(plan, topology, schedule)

    if not isinstance(links, dict):
        raise TypeError("link parameter dictionary required")

    if prediction.machine_shape != topology.machine_shape:
        raise ValueError("prediction machine shape mismatch")

    if prediction.outgoing_ports != schedule.outgoing_ports:
        raise ValueError("prediction port count mismatch")

    if (
        type(prediction.stage_count) is not int
        or prediction.stage_count != len(schedule.stages)
    ):
        raise ValueError("prediction stage count mismatch")

    if not isinstance(prediction.stage_times, tuple):
        raise ValueError("stage times must be a tuple")

    if len(prediction.stage_times) != len(schedule.stages):
        raise ValueError("incorrect number of stage times")

    if (
        type(prediction.total_link_bytes) is not int
        or prediction.total_link_bytes != topology.total_link_bytes
    ):
        raise ValueError("prediction communication volume mismatch")

    # Reconstruct the required directed-link domain independently.
    required_links = set()

    for route in topology.routes:
        for parent, child in route.edges:
            required_links.add((parent, child))

    if set(links.keys()) != required_links:
        raise ValueError(
            "link parameters do not match topology"
        )

    for parameters in links.values():
        if not isinstance(parameters, OmegaLinkParameters):
            raise ValueError("invalid link parameter object")

        if not _finite_nonnegative(parameters.alpha):
            raise ValueError("invalid link latency")

        if not _finite_positive(parameters.beta):
            raise ValueError("invalid link bandwidth")

    expected_stage_times = []

    for stage in schedule.stages:
        longest = 0.0

        for transfer in stage:
            key = (transfer.parent, transfer.child)
            parameters = links[key]

            duration = (
                parameters.alpha
                + transfer.payload_bytes / parameters.beta
            )

            if not _finite_nonnegative(duration):
                raise ValueError(
                    "nonfinite transfer-time prediction"
                )

            if duration > longest:
                longest = duration

        expected_stage_times.append(longest)

    for observed, expected in zip(
        prediction.stage_times,
        expected_stage_times,
    ):
        if not _times_agree(observed, expected):
            raise ValueError(
                "predicted stage time disagrees with model"
            )

    expected_total = fsum(expected_stage_times)

    if not _finite_nonnegative(expected_total):
        raise ValueError("nonfinite total predicted time")

    if not _times_agree(
        prediction.total_time,
        expected_total,
    ):
        raise ValueError(
            "predicted total time disagrees with model"
        )

    # Also verify internal accounting independently.
    if not _times_agree(
        prediction.total_time,
        fsum(prediction.stage_times),
    ):
        raise ValueError(
            "prediction total disagrees with its stage times"
        )

    return True
