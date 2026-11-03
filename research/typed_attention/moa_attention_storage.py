
"""MoA attention storage certificate for the Gate 11 MPI backend.

Derives application-level storage requirements from a verified typed
MoA Omega schedule and the generated MPI implementation contract.

This is not a prediction of physical peak memory, MPI-internal storage,
network traffic, time, power, or energy.
"""

from dataclasses import dataclass
from moa_typed_ir import ScheduledAttention, verify_schedule


@dataclass(frozen=True)
class RankStorage:
    coordinate: tuple[int, ...]
    query_begin: int
    query_end: int
    query_count: int

    replicated_q_components: int
    replicated_k_components: int
    replicated_v_components: int

    complete_output_components: int
    local_output_allocated_components: int
    online_vector_components: int
    online_scalar_components: int

    modeled_components: int
    modeled_bytes: int
    capacity_bytes: int | None
    feasible: bool | None


@dataclass(frozen=True)
class AttentionStorageCertificate:
    machine_shape: tuple[int, ...]
    output_shape: tuple[int, int]
    element_bytes: int
    ranks: tuple[RankStorage, ...]
    aggregate_modeled_bytes: int
    maximum_rank_modeled_bytes: int
    all_feasible: bool | None

    def verify(self):
        if not self.ranks:
            raise ValueError("empty machine")

        if self.aggregate_modeled_bytes != sum(
            r.modeled_bytes for r in self.ranks
        ):
            raise ValueError("aggregate storage mismatch")

        if self.maximum_rank_modeled_bytes != max(
            r.modeled_bytes for r in self.ranks
        ):
            raise ValueError("maximum storage mismatch")

        for r in self.ranks:
            expected = (
                r.replicated_q_components
                + r.replicated_k_components
                + r.replicated_v_components
                + r.complete_output_components
                + r.local_output_allocated_components
                + r.online_vector_components
                + r.online_scalar_components
            )

            if r.modeled_components != expected:
                raise ValueError("rank component mismatch")

            if r.modeled_bytes != expected * self.element_bytes:
                raise ValueError("rank byte mismatch")

            if r.capacity_bytes is None:
                if r.feasible is not None:
                    raise ValueError("unexpected feasibility result")
            elif r.feasible != (r.modeled_bytes <= r.capacity_bytes):
                raise ValueError("incorrect feasibility result")

        statuses = tuple(r.feasible for r in self.ranks)
        expected_all = (
            None if any(x is None for x in statuses)
            else all(statuses)
        )

        if self.all_feasible != expected_all:
            raise ValueError("aggregate feasibility mismatch")

        return True


def derive_attention_storage(
    ir,
    element_bytes=8,
    capacity_bytes=None,
):
    """Derive storage for replicated Q,K,V and MPI_Allgatherv output.

    capacity_bytes, when supplied, contains one capacity per machine
    component in the same gamma order as the Omega schedule.

    A separate local output buffer is allocated on every rank, with
    at least one component even for an empty query interval.

    The online vector is allocated for one query at a time. The two
    online scalars are included as logical scalar storage.

    The modeled total includes externally supplied Q,K,V,O storage.
    It excludes allocator overhead, MPI-internal buffers, and other
    implementation-dependent memory.
    """
    if not isinstance(ir, ScheduledAttention):
        raise ValueError("typed scheduled attention required")

    verify_schedule(ir)

    if type(element_bytes) is not int or element_bytes <= 0:
        raise ValueError("positive integer element size required")

    s = ir.schedule
    intervals = s.intervals
    p = len(intervals)

    if capacity_bytes is not None:
        if len(capacity_bytes) != p:
            raise ValueError("capacity count must match machine")
        if any(
            type(c) is not int or c < 0
            for c in capacity_bytes
        ):
            raise ValueError("invalid machine capacity")
        capacity_bytes = tuple(capacity_bytes)

    n = s.query_extent
    m = s.key_extent
    d = s.score_extent
    dv = s.value_extent

    # The machine-coordinate order follows the verified Omega schedule.
    from moa_machine import iota

    coordinates = iota(tuple(s.machine_shape))

    ranks = []

    for coordinate, (begin, end), rank in zip(
        coordinates, intervals, range(p)
    ):
        q = end - begin

        q_components = n * d
        k_components = m * d
        v_components = m * dv
        output_components = n * dv

        local_components = max(q * dv, 1)

        online_vector = dv if q > 0 else 0
        online_scalars = 2 if q > 0 else 0

        modeled = (
            q_components
            + k_components
            + v_components
            + output_components
            + local_components
            + online_vector
            + online_scalars
        )

        required_bytes = modeled * element_bytes

        capacity = (
            None if capacity_bytes is None
            else capacity_bytes[rank]
        )

        feasible = (
            None if capacity is None
            else required_bytes <= capacity
        )

        ranks.append(
            RankStorage(
                coordinate=coordinate,
                query_begin=begin,
                query_end=end,
                query_count=q,
                replicated_q_components=q_components,
                replicated_k_components=k_components,
                replicated_v_components=v_components,
                complete_output_components=output_components,
                local_output_allocated_components=local_components,
                online_vector_components=online_vector,
                online_scalar_components=online_scalars,
                modeled_components=modeled,
                modeled_bytes=required_bytes,
                capacity_bytes=capacity,
                feasible=feasible,
            )
        )

    ranks = tuple(ranks)

    statuses = tuple(r.feasible for r in ranks)

    all_feasible = (
        None if capacity_bytes is None
        else all(statuses)
    )

    cert = AttentionStorageCertificate(
        machine_shape=tuple(s.machine_shape),
        output_shape=(n, dv),
        element_bytes=element_bytes,
        ranks=ranks,
        aggregate_modeled_bytes=sum(r.modeled_bytes for r in ranks),
        maximum_rank_modeled_bytes=max(r.modeled_bytes for r in ranks),
        all_feasible=all_feasible,
    )

    cert.verify()
    return cert
