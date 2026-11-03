
"""Resource-constrained explicit Omega planning for MoA attention.

Additive module. Does not modify canonical scheduling or backends.

Derives query ownership using the existing modeled MPI storage
formula and independently verifies explicit Omega ownership.

Scope:
  - heterogeneous per-rank memory capacities
  - contiguous, disjoint query ownership
  - idle machine components
  - exact modeled storage feasibility
  - deterministic greedy allocation

Does not predict execution time or guarantee optimal scheduling.
"""

from dataclasses import dataclass

from moa_machine import Machine, Work

from moa_attention_explicit_omega import (
    ExplicitOmegaCertificate,
    schedule_omega_from_counts,
)


def modeled_attention_bytes(work, queries, element_bytes=8):
    """Modeled per-rank MPI attention storage in bytes."""
    if not isinstance(work, Work):
        raise TypeError("expected Work")

    if type(queries) is not int or not 0 <= queries <= work.n:
        raise ValueError("invalid query count")

    if type(element_bytes) is not int or element_bytes <= 0:
        raise ValueError("invalid element size")

    replicated_q = work.n * work.d
    replicated_k = work.m * work.d
    replicated_v = work.m * work.dv
    complete_output = work.n * work.dv

    local_output = max(queries * work.dv, 1)

    online_vector = work.dv if queries > 0 else 0
    online_scalars = 2 if queries > 0 else 0

    components = (
        replicated_q
        + replicated_k
        + replicated_v
        + complete_output
        + local_output
        + online_vector
        + online_scalars
    )

    return element_bytes * components


@dataclass(frozen=True)
class ResourceOmegaPlan:
    """Verified explicit Omega ownership with storage certificates."""

    certificate: ExplicitOmegaCertificate
    capacities: tuple[int, ...]
    modeled_bytes: tuple[int, ...]

    def verify(self):
        self.certificate.verify()

        counts = self.certificate.query_counts
        work_shape = self.certificate.scheduled.schedule

        work = Work(
            work_shape.query_extent,
            work_shape.key_extent,
            work_shape.score_extent,
            work_shape.value_extent,
        )

        if len(self.capacities) != len(counts):
            raise ValueError("capacity count mismatch")

        if len(self.modeled_bytes) != len(counts):
            raise ValueError("storage count mismatch")

        for q, cap, used in zip(
            counts,
            self.capacities,
            self.modeled_bytes,
        ):
            if type(cap) is not int or cap <= 0:
                raise ValueError("invalid capacity")

            expected = modeled_attention_bytes(work, q)

            if used != expected:
                raise ValueError("corrupted storage certificate")

            if expected > cap:
                raise ValueError("rank memory capacity exceeded")

        return True

    @property
    def query_counts(self):
        self.verify()
        return self.certificate.query_counts

    @property
    def intervals(self):
        self.verify()
        return self.certificate.scheduled.schedule.intervals


def plan_attention_omega(
    expr,
    work,
    machine,
    capacities,
):
    """Choose a deterministic feasible explicit Omega partition.

    Policy: allocate queries to earlier machine coordinates first,
    up to each component's modeled storage capacity.

    This is a feasible-first policy, not a minimax-time optimizer.
    """
    if not isinstance(work, Work):
        raise TypeError("expected Work")

    if not isinstance(machine, Machine):
        raise TypeError("expected Machine")

    capacities = tuple(capacities)

    if len(capacities) != machine.tau:
        raise ValueError("capacity count must equal tau(machine)")

    if any(type(c) is not int or c <= 0 for c in capacities):
        raise ValueError("invalid machine capacity")

    # Every machine component exists, including idle components.
    idle_bytes = modeled_attention_bytes(work, 0)

    if any(c < idle_bytes for c in capacities):
        raise ValueError(
            "machine cannot hold replicated attention storage"
        )

    counts = []
    remaining = work.n

    for capacity in capacities:
        if remaining == 0:
            counts.append(0)
            continue

        # The first active query introduces online state.
        one_query_bytes = modeled_attention_bytes(work, 1)

        if capacity < one_query_bytes:
            counts.append(0)
            continue

        # Additional queries increase local output storage
        # by work.dv elements per query.
        per_query_bytes = 8 * work.dv

        maximum = 1 + (
            capacity - one_query_bytes
        ) // per_query_bytes

        assigned = min(remaining, maximum)

        counts.append(assigned)
        remaining -= assigned

    if remaining != 0:
        raise ValueError(
            "no feasible Omega partition for given capacities"
        )

    certificate = schedule_omega_from_counts(
        expr,
        work,
        machine,
        tuple(counts),
    )

    modeled = tuple(
        modeled_attention_bytes(work, q)
        for q in counts
    )

    result = ResourceOmegaPlan(
        certificate=certificate,
        capacities=capacities,
        modeled_bytes=modeled,
    )

    result.verify()
    return result
