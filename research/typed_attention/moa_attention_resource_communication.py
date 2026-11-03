
"""MoA resource-derived Omega communication certificate.

Additive module: does not modify existing schedulers or backends.

Derives exact logical MPI_Allgatherv metadata and communication
volumes from verified resource-constrained Omega ownership.

Contract:
  - complete read-only Q, K, V replicated on every machine component
  - each component computes its disjoint owned portion of O
  - MPI_Allgatherv makes complete O available to every component

This module does NOT predict:
  - physical network bytes or messages
  - routing, congestion, latency, or bandwidth
  - collective algorithm selection
  - execution time, energy, or power
  - the cost of initially replicating Q, K, V
"""

from dataclasses import dataclass

from moa_attention_resource_planner import ResourceOmegaPlan
from moa_attention_resource_mpi import (
    collective_from_intervals,
    emit_resource_owned_mpi,
)


@dataclass(frozen=True)
class ResourceCommunicationCertificate:
    """Verified logical communication for resource-derived Omega."""

    intervals: tuple[tuple[int, int], ...]
    counts: tuple[int, ...]
    displacements: tuple[int, ...]

    machine_components: int
    output_components: int
    element_bytes: int

    per_rank_receive_components: tuple[int, ...]
    per_rank_nonlocal_components: tuple[int, ...]

    aggregate_receive_components: int
    aggregate_nonlocal_components: int

    aggregate_receive_bytes: int
    aggregate_nonlocal_bytes: int

    contract: str = (
        "replicated Q,K,V; disjoint Omega output ownership; "
        "complete O via MPI_Allgatherv"
    )

    def verify(self):
        """Independently check the logical communication certificate."""

        p = self.machine_components
        total = self.output_components
        width = self.element_bytes

        if type(p) is not int or p <= 0:
            raise ValueError("invalid machine component count")

        if type(total) is not int or total < 0:
            raise ValueError("invalid output component count")

        if type(width) is not int or width <= 0:
            raise ValueError("invalid component byte width")

        if len(self.intervals) != p:
            raise ValueError("Omega interval count mismatch")

        if len(self.counts) != p:
            raise ValueError("collective count length mismatch")

        if len(self.displacements) != p:
            raise ValueError("collective displacement length mismatch")

        if len(self.per_rank_receive_components) != p:
            raise ValueError("receive vector length mismatch")

        if len(self.per_rank_nonlocal_components) != p:
            raise ValueError("nonlocal vector length mismatch")

        if any(
            type(x) is not int or x < 0
            for x in self.counts + self.displacements
        ):
            raise ValueError("invalid collective metadata")

        if sum(self.counts) != total:
            raise ValueError("tau output mismatch")

        expected_displacements = tuple(
            sum(self.counts[:r])
            for r in range(p)
        )

        if self.displacements != expected_displacements:
            raise ValueError("noncontiguous gamma layout")

        if any(
            self.displacements[r] + self.counts[r] > total
            for r in range(p)
        ):
            raise ValueError("collective output bounds exceeded")

        if any(
            type(x) is not int
            for x in self.per_rank_receive_components
        ):
            raise ValueError("invalid receive component type")

        if any(
            type(x) is not int
            for x in self.per_rank_nonlocal_components
        ):
            raise ValueError("invalid nonlocal component type")

        expected_receive = (total,) * p

        expected_nonlocal = tuple(
            total - self.counts[r]
            for r in range(p)
        )

        if self.per_rank_receive_components != expected_receive:
            raise ValueError("incorrect per-rank receive requirement")

        if self.per_rank_nonlocal_components != expected_nonlocal:
            raise ValueError("incorrect per-rank nonlocal requirement")

        expected_aggregate_receive = p * total
        expected_aggregate_nonlocal = (p - 1) * total

        if self.aggregate_receive_components != expected_aggregate_receive:
            raise ValueError("incorrect aggregate receive components")

        if self.aggregate_nonlocal_components != expected_aggregate_nonlocal:
            raise ValueError("incorrect aggregate nonlocal components")

        if self.aggregate_receive_bytes != (
            expected_aggregate_receive * width
        ):
            raise ValueError("incorrect aggregate receive bytes")

        if self.aggregate_nonlocal_bytes != (
            expected_aggregate_nonlocal * width
        ):
            raise ValueError("incorrect aggregate nonlocal bytes")

        if not isinstance(self.contract, str) or not self.contract:
            raise ValueError("missing collective contract")

        return True


def derive_resource_communication(plan, element_bytes=8):
    """Derive logical communication from a verified resource Omega plan.

    The result describes logical output requirements, not the
    physical traffic generated by a particular MPI implementation.
    """

    if not isinstance(plan, ResourceOmegaPlan):
        raise TypeError("ResourceOmegaPlan required")

    plan.verify()

    if type(element_bytes) is not int or element_bytes <= 0:
        raise ValueError("invalid component byte width")

    intervals = plan.intervals
    p = len(intervals)

    schedule = plan.certificate.scheduled.schedule

    n = schedule.query_extent
    dv = schedule.value_extent

    counts, displacements = collective_from_intervals(
        intervals,
        n,
        dv,
    )

    total = n * dv

    # Cross-check against the independently verified MPI emitter.
    emitted = emit_resource_owned_mpi(plan)
    emitted.verify()

    if emitted.intervals != intervals:
        raise ValueError("MPI Omega ownership mismatch")

    if emitted.counts != counts:
        raise ValueError("MPI collective counts mismatch")

    if emitted.displacements != displacements:
        raise ValueError("MPI collective displacements mismatch")

    if emitted.output_components != total:
        raise ValueError("MPI output extent mismatch")

    receive = (total,) * p

    nonlocal_components = tuple(
        total - count
        for count in counts
    )

    aggregate_receive = sum(receive)
    aggregate_nonlocal = sum(nonlocal_components)

    result = ResourceCommunicationCertificate(
        intervals=intervals,
        counts=counts,
        displacements=displacements,
        machine_components=p,
        output_components=total,
        element_bytes=element_bytes,
        per_rank_receive_components=receive,
        per_rank_nonlocal_components=nonlocal_components,
        aggregate_receive_components=aggregate_receive,
        aggregate_nonlocal_components=aggregate_nonlocal,
        aggregate_receive_bytes=aggregate_receive * element_bytes,
        aggregate_nonlocal_bytes=aggregate_nonlocal * element_bytes,
    )

    result.verify()

    return result
