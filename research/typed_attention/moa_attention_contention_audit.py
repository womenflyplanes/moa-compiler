
"""Independent audit of MoA Omega directed-link contention.

Step 103: verify conditional communication-time predictions
against original Omega ownership, topology, schedule, and
directed-link parameters.

The auditor independently reconstructs:
  - Directed-link transfer counts for each stage
  - Aggregated payload bytes for each occupied link
  - Link latency and bandwidth
  - Predicted duration of each occupied link
  - Maximum link duration per stage
  - Total predicted communication time
  - Total logical communication volume

It does not call the contention predictor or the prediction
certificate's verify() method.

This is an abstract machine model, not a physical-network
or MPI timing validation.
"""

from math import isfinite, fsum

from moa_attention_omega_topology import OmegaTopology

from moa_attention_port_schedule import OmegaPortSchedule

from moa_attention_port_schedule_audit import (
    audit_omega_port_schedule,
)

from moa_attention_communication_time import (
    OmegaLinkParameters,
)

from moa_attention_contention import (
    OmegaContendedLink,
    OmegaContentionPrediction,
)


def _nonnegative_finite(value):
    """Accept only finite nonnegative int or float values."""

    if type(value) not in (int, float):
        return False

    try:
        return isfinite(value) and value >= 0
    except (TypeError, ValueError, OverflowError):
        return False


def _positive_finite(value):
    return _nonnegative_finite(value) and value > 0


def _near(actual, expected):
    """Floating-point comparison with tight relative tolerance."""

    if not (
        _nonnegative_finite(actual)
        and _nonnegative_finite(expected)
    ):
        return False

    tolerance = (
        1.0e-12 * max(abs(expected), 1.0e-12)
    )

    return abs(actual - expected) <= tolerance


def audit_omega_contention(
    plan,
    topology,
    schedule,
    links,
    prediction,
):
    """Independently verify an Omega contention certificate.

    Raises TypeError or ValueError for invalid inputs.
    Returns True only after every audit condition passes.
    """

    if not isinstance(topology, OmegaTopology):
        raise TypeError("OmegaTopology required")

    if not isinstance(schedule, OmegaPortSchedule):
        raise TypeError("OmegaPortSchedule required")

    if not isinstance(
        prediction,
        OmegaContentionPrediction,
    ):
        raise TypeError("OmegaContentionPrediction required")

    # Independent schedule verification against original
    # Omega ownership and topology.
    audit_omega_port_schedule(
        plan,
        topology,
        schedule,
    )

    if not isinstance(links, dict):
        raise TypeError("link parameter dictionary required")

    if prediction.machine_shape != topology.machine_shape:
        raise ValueError("machine shape mismatch")

    if prediction.outgoing_ports != schedule.outgoing_ports:
        raise ValueError("outgoing-port count mismatch")

    if (
        type(prediction.stage_count) is not int
        or prediction.stage_count != len(schedule.stages)
    ):
        raise ValueError("stage count mismatch")

    if not isinstance(prediction.stages, tuple):
        raise ValueError("contention stages must be a tuple")

    if not isinstance(prediction.stage_times, tuple):
        raise ValueError("stage times must be a tuple")

    if len(prediction.stages) != len(schedule.stages):
        raise ValueError("contention stage count mismatch")

    if len(prediction.stage_times) != len(schedule.stages):
        raise ValueError("stage-time count mismatch")

    if (
        type(prediction.total_link_bytes) is not int
        or prediction.total_link_bytes != topology.total_link_bytes
        or prediction.total_link_bytes != schedule.total_link_bytes
    ):
        raise ValueError("total logical link bytes mismatch")

    # Reconstruct the directed-link parameter domain
    # independently from the Omega topology.
    required_links = set()

    for route in topology.routes:
        for parent, child in route.edges:
            required_links.add((parent, child))

    if set(links) != required_links:
        raise ValueError("directed-link domain mismatch")

    for parameters in links.values():
        if not isinstance(parameters, OmegaLinkParameters):
            raise ValueError("invalid link parameter object")

        if not _nonnegative_finite(parameters.alpha):
            raise ValueError("invalid link latency")

        if not _positive_finite(parameters.beta):
            raise ValueError("invalid link bandwidth")

    expected_stage_times = []
    independently_counted_bytes = 0

    for stage_index, original_stage in enumerate(
        schedule.stages
    ):
        # Reconstruct exact transfer multiplicity and bytes.
        expected_links = {}

        for transfer in original_stage:
            key = (
                transfer.parent,
                transfer.child,
            )

            if key not in expected_links:
                expected_links[key] = [0, 0]

            expected_links[key][0] += 1
            expected_links[key][1] += transfer.payload_bytes

            independently_counted_bytes += (
                transfer.payload_bytes
            )

        certified_stage = prediction.stages[stage_index]

        if not isinstance(certified_stage, tuple):
            raise ValueError(
                "contention stage must be a tuple"
            )

        # Certificate order is not mathematically significant.
        certified_links = {}

        for certified in certified_stage:
            if not isinstance(certified, OmegaContendedLink):
                raise ValueError(
                    "invalid contended-link certificate"
                )

            key = (
                certified.parent,
                certified.child,
            )

            if key in certified_links:
                raise ValueError(
                    "duplicate directed link in stage"
                )

            certified_links[key] = certified

        if set(certified_links) != set(expected_links):
            raise ValueError(
                "contended-link domain mismatch in stage"
            )

        longest = 0.0

        for key, (count, payload_bytes) in (
            expected_links.items()
        ):
            certified = certified_links[key]
            parameters = links[key]

            if (
                type(certified.transfer_count) is not int
                or certified.transfer_count != count
            ):
                raise ValueError(
                    "transfer multiplicity mismatch"
                )

            if (
                type(certified.payload_bytes) is not int
                or certified.payload_bytes != payload_bytes
            ):
                raise ValueError(
                    "aggregated payload bytes mismatch"
                )

            if not _near(
                certified.alpha,
                parameters.alpha,
            ):
                raise ValueError(
                    "certified link latency mismatch"
                )

            if not _near(
                certified.beta,
                parameters.beta,
            ):
                raise ValueError(
                    "certified link bandwidth mismatch"
                )

            expected_duration = (
                parameters.alpha
                + payload_bytes / parameters.beta
            )

            if not _nonnegative_finite(expected_duration):
                raise ValueError(
                    "nonfinite reconstructed link duration"
                )

            if not _near(
                certified.predicted_seconds,
                expected_duration,
            ):
                raise ValueError(
                    "contended-link duration mismatch"
                )

            longest = max(
                longest,
                expected_duration,
            )

        expected_stage_times.append(longest)

        if not _near(
            prediction.stage_times[stage_index],
            longest,
        ):
            raise ValueError(
                "predicted stage duration mismatch"
            )

    if independently_counted_bytes != topology.total_link_bytes:
        raise ValueError(
            "reconstructed communication volume mismatch"
        )

    expected_total = fsum(expected_stage_times)

    if not _nonnegative_finite(expected_total):
        raise ValueError(
            "nonfinite reconstructed total time"
        )

    if not _near(
        prediction.total_seconds,
        expected_total,
    ):
        raise ValueError(
            "predicted total communication time mismatch"
        )

    if not _near(
        prediction.total_seconds,
        fsum(prediction.stage_times),
    ):
        raise ValueError(
            "certificate internal stage-time mismatch"
        )

    return True
