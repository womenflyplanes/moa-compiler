
"""Explicit MoA Omega scheduling with independent ownership verification.

Additive extension to the restricted typed attention IR.

Preserves the existing canonical scheduler and verifier unchanged.
Explicit schedules are verified using the existing MoA verify_lift
ownership invariants.

This module does not claim resource feasibility, performance
optimality, or compatibility with existing code-generation backends.
"""

from dataclasses import dataclass

from moa_machine import (
    Machine,
    Work,
    Owner,
    iota,
    verify_lift,
)

from moa_typed_ir import (
    LoweredAttention,
    OmegaSchedule,
    ScheduledAttention,
    verify,
)


@dataclass(frozen=True)
class ExplicitOmegaCertificate:
    """Independently verified explicit Omega ownership."""

    scheduled: ScheduledAttention
    owners: tuple[Owner, ...]

    def verify(self):
        scheduled = self.scheduled

        if not isinstance(scheduled, ScheduledAttention):
            raise ValueError("expected ScheduledAttention")

        if not isinstance(scheduled.schedule, OmegaSchedule):
            raise ValueError("invalid OmegaSchedule")

        verify(scheduled.lowered)

        s = scheduled.schedule

        machine = Machine(s.machine_shape)

        work = Work(
            s.query_extent,
            s.key_extent,
            s.score_extent,
            s.value_extent,
        )

        if len(s.intervals) != machine.tau:
            raise ValueError("Omega interval count mismatch")

        for interval in s.intervals:
            if (
                not isinstance(interval, tuple)
                or len(interval) != 2
                or any(type(x) is not int for x in interval)
            ):
                raise ValueError("invalid Omega interval")

        expected = tuple(
            Owner(coordinate, begin, end)
            for coordinate, (begin, end) in zip(
                iota(machine.shape),
                s.intervals,
            )
        )

        if self.owners != expected:
            raise ValueError("Omega owner certificate mismatch")

        verify_lift(work, machine, self.owners)

        return True

    @property
    def query_counts(self):
        self.verify()
        return tuple(owner.count() for owner in self.owners)

    @property
    def output_components(self):
        self.verify()
        dv = self.scheduled.schedule.value_extent
        return tuple(owner.count() * dv for owner in self.owners)


def schedule_omega_explicit(expr, work, machine, intervals):
    """Build an explicit Omega schedule without changing canonical IR.

    Intervals follow the machine coordinates in iota(rho(machine))
    order. Unequal partitions and idle components are permitted.
    """
    if not isinstance(expr, LoweredAttention):
        raise TypeError("expected LoweredAttention")

    if not isinstance(work, Work):
        raise TypeError("expected Work")

    if not isinstance(machine, Machine):
        raise TypeError("expected Machine")

    verify(expr)

    intervals = tuple(intervals)

    if len(intervals) != machine.tau:
        raise ValueError("Omega interval count mismatch")

    normalized = []

    for interval in intervals:
        if (
            not isinstance(interval, (tuple, list))
            or len(interval) != 2
            or any(type(x) is not int for x in interval)
        ):
            raise ValueError("invalid explicit Omega interval")

        normalized.append(tuple(interval))

    normalized = tuple(normalized)

    owners = tuple(
        Owner(coordinate, begin, end)
        for coordinate, (begin, end) in zip(
            iota(machine.shape),
            normalized,
        )
    )

    verify_lift(work, machine, owners)

    scheduled = ScheduledAttention(
        expr,
        OmegaSchedule(
            machine.shape,
            work.n,
            work.m,
            work.d,
            work.dv,
            normalized,
        ),
    )

    result = ExplicitOmegaCertificate(
        scheduled=scheduled,
        owners=owners,
    )

    result.verify()
    return result


def schedule_omega_from_counts(expr, work, machine, counts):
    """Construct contiguous Omega intervals from query counts."""
    counts = tuple(counts)

    if len(counts) != machine.tau:
        raise ValueError("Omega query-count mismatch")

    if any(type(q) is not int or q < 0 for q in counts):
        raise ValueError("invalid Omega query count")

    if sum(counts) != work.n:
        raise ValueError("Omega query counts do not preserve extent")

    intervals = []
    begin = 0

    for q in counts:
        end = begin + q
        intervals.append((begin, end))
        begin = end

    return schedule_omega_explicit(
        expr,
        work,
        machine,
        tuple(intervals),
    )
