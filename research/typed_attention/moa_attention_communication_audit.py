
"""Independent audit of resource-derived MoA Omega communication.

Additive verification layer. No existing compiler files are changed.

Reconstructs MPI collective metadata from Omega ownership intervals
and checks all logical communication quantities independently.

Does not model physical network traffic, routing, latency,
bandwidth, power, energy, or execution time.
"""

from moa_attention_resource_communication import (
    ResourceCommunicationCertificate,
)


def audit_resource_communication(certificate, value_extent):
    """Independently audit an Omega communication certificate.

    Ownership intervals are expressed in query coordinates.
    The value extent converts query ownership into output
    component ownership through gamma addressing.
    """

    if not isinstance(
        certificate,
        ResourceCommunicationCertificate,
    ):
        raise TypeError(
            "ResourceCommunicationCertificate required"
        )

    if type(value_extent) is not int or value_extent <= 0:
        raise ValueError("invalid value extent")

    p = certificate.machine_components
    total = certificate.output_components
    intervals = certificate.intervals

    if type(p) is not int or p <= 0:
        raise ValueError("invalid machine component count")

    if type(total) is not int or total < 0:
        raise ValueError("invalid output component count")

    if len(intervals) != p:
        raise ValueError("Omega interval count mismatch")

    if total % value_extent != 0:
        raise ValueError("output shape inconsistent with value extent")

    query_extent = total // value_extent

    expected_counts = []
    expected_displacements = []

    cursor = 0

    for interval in intervals:
        if not isinstance(interval, tuple) or len(interval) != 2:
            raise ValueError("malformed Omega ownership interval")

        begin, end = interval

        if type(begin) is not int or type(end) is not int:
            raise ValueError("noninteger Omega ownership")

        if begin < 0 or end < begin:
            raise ValueError("invalid Omega interval")

        if begin != cursor:
            raise ValueError("Omega ownership gap or overlap")

        if end > query_extent:
            raise ValueError("Omega interval exceeds query extent")

        expected_counts.append(
            (end - begin) * value_extent
        )

        expected_displacements.append(
            begin * value_extent
        )

        cursor = end

    if cursor != query_extent:
        raise ValueError("incomplete Omega ownership")

    expected_counts = tuple(expected_counts)
    expected_displacements = tuple(expected_displacements)

    if certificate.counts != expected_counts:
        raise ValueError(
            "collective counts disagree with Omega ownership"
        )

    if certificate.displacements != expected_displacements:
        raise ValueError(
            "collective displacements disagree with Omega ownership"
        )

    if sum(expected_counts) != total:
        raise ValueError("tau output conservation failure")

    expected_receive = (total,) * p

    expected_nonlocal = tuple(
        total - count
        for count in expected_counts
    )

    if certificate.per_rank_receive_components != expected_receive:
        raise ValueError("incorrect per-rank receive requirement")

    if certificate.per_rank_nonlocal_components != expected_nonlocal:
        raise ValueError("incorrect per-rank nonlocal requirement")

    expected_aggregate_receive = p * total
    expected_aggregate_nonlocal = (p - 1) * total

    if (
        certificate.aggregate_receive_components
        != expected_aggregate_receive
    ):
        raise ValueError("aggregate receive mismatch")

    if (
        certificate.aggregate_nonlocal_components
        != expected_aggregate_nonlocal
    ):
        raise ValueError("aggregate nonlocal mismatch")

    width = certificate.element_bytes

    if type(width) is not int or width <= 0:
        raise ValueError("invalid element byte width")

    if (
        certificate.aggregate_receive_bytes
        != expected_aggregate_receive * width
    ):
        raise ValueError("aggregate receive byte mismatch")

    if (
        certificate.aggregate_nonlocal_bytes
        != expected_aggregate_nonlocal * width
    ):
        raise ValueError("aggregate nonlocal byte mismatch")

    # Check the certificate's own consistency as well.
    certificate.verify()

    return True


def audit_resource_communication_plan(plan, certificate):
    """Bind the independent audit to a verified resource Omega plan."""

    from moa_attention_resource_planner import ResourceOmegaPlan

    if not isinstance(plan, ResourceOmegaPlan):
        raise TypeError("ResourceOmegaPlan required")

    plan.verify()

    schedule = plan.certificate.scheduled.schedule

    if certificate.intervals != plan.intervals:
        raise ValueError(
            "communication ownership differs from resource plan"
        )

    if certificate.machine_components != len(plan.query_counts):
        raise ValueError("machine component count mismatch")

    if (
        certificate.output_components
        != schedule.query_extent * schedule.value_extent
    ):
        raise ValueError("output extent mismatch")

    return audit_resource_communication(
        certificate,
        schedule.value_extent,
    )
