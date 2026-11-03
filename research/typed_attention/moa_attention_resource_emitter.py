
"""Resource-aware MoA Omega C/OpenMP emission.

Additive adapter for the restricted typed forward-attention compiler.

The existing canonical C emitter generates the computational kernel.
This adapter substitutes only the two Omega ownership declarations
after verifying the explicit ownership and resource certificates.

No existing compiler source, verifier, or backend is modified.

This is C/OpenMP source generation, not MPI rank placement or
proof of physical resource allocation.
"""

from dataclasses import dataclass

from moa_machine import Machine, Work

from moa_typed_ir import (
    schedule_omega,
    emit_c,
)

from moa_attention_resource_planner import ResourceOmegaPlan


@dataclass(frozen=True)
class ResourceEmissionCertificate:
    """Auditable C emission and resource-derived Omega ownership."""

    source: str
    intervals: tuple[tuple[int, int], ...]
    capacities: tuple[int, ...]
    modeled_bytes: tuple[int, ...]

    def verify(self):
        if not isinstance(self.source, str):
            raise ValueError("invalid generated source")

        if len(self.intervals) != len(self.capacities):
            raise ValueError("ownership/capacity count mismatch")

        if len(self.modeled_bytes) != len(self.capacities):
            raise ValueError("storage/capacity count mismatch")

        if any(
            type(b) is not int or b < 0
            for b in self.modeled_bytes
        ):
            raise ValueError("invalid modeled storage")

        if any(
            type(c) is not int or c <= 0
            for c in self.capacities
        ):
            raise ValueError("invalid capacity")

        if any(
            b > c
            for b, c in zip(
                self.modeled_bytes,
                self.capacities,
            )
        ):
            raise ValueError("resource capacity exceeded")

        begin_line, end_line = ownership_declarations(
            self.intervals
        )

        if self.source.count(begin_line) != 1:
            raise ValueError("missing Omega begin declaration")

        if self.source.count(end_line) != 1:
            raise ValueError("missing Omega end declaration")

        if self.source.count(
            "void generated_attention("
        ) != 1:
            raise ValueError("invalid attention function")

        if "#pragma omp parallel for" not in self.source:
            raise ValueError("missing OpenMP ownership loop")

        return True


def ownership_declarations(intervals):
    """Construct the exact declarations used by the C emitter."""

    intervals = tuple(intervals)

    if not intervals:
        raise ValueError("empty Omega ownership")

    for interval in intervals:
        if (
            not isinstance(interval, tuple)
            or len(interval) != 2
            or any(type(x) is not int for x in interval)
        ):
            raise ValueError("invalid Omega interval")

    begin = (
        " const size_t owner_begin[] = {"
        + ",".join(str(x[0]) for x in intervals)
        + "};"
    )

    end = (
        " const size_t owner_end[] = {"
        + ",".join(str(x[1]) for x in intervals)
        + "};"
    )

    return begin, end


def emit_resource_owned_openmp(plan):
    """Emit C/OpenMP using independently verified resource ownership.

    The canonical backend is invoked normally and verifies its own
    schedule. Only its two ownership declarations are replaced.
    """

    if not isinstance(plan, ResourceOmegaPlan):
        raise TypeError("expected ResourceOmegaPlan")

    plan.verify()

    explicit = plan.certificate.scheduled
    s = explicit.schedule

    work = Work(
        s.query_extent,
        s.key_extent,
        s.score_extent,
        s.value_extent,
    )

    machine = Machine(s.machine_shape)

    # The original scheduler and C emitter remain unchanged.
    canonical = schedule_omega(
        explicit.lowered,
        work,
        machine,
    )

    original = emit_c(canonical)

    old_begin, old_end = ownership_declarations(
        canonical.schedule.intervals
    )

    new_begin, new_end = ownership_declarations(
        s.intervals
    )

    # Fail closed if the original emitter changes its format.
    if original.count(old_begin) != 1:
        raise ValueError(
            "canonical begin declaration changed"
        )

    if original.count(old_end) != 1:
        raise ValueError(
            "canonical end declaration changed"
        )

    # Replace only the verified Omega ownership declarations.
    generated = original.replace(
        old_begin,
        new_begin,
        1,
    ).replace(
        old_end,
        new_end,
        1,
    )

    # Independently check the complete expected transformation.
    expected = original.replace(
        old_begin,
        new_begin,
        1,
    ).replace(
        old_end,
        new_end,
        1,
    )

    if generated != expected:
        raise ValueError("unexpected C source transformation")

    result = ResourceEmissionCertificate(
        source=generated,
        intervals=s.intervals,
        capacities=plan.capacities,
        modeled_bytes=plan.modeled_bytes,
    )

    result.verify()
    return result
