
"""Resource-derived MoA Omega MPI C emission.

Additive adapter for the restricted typed forward-attention compiler.

Reuses the canonical MPI emitter's computation and collective call.
Changes only verified Omega ownership and MPI collective metadata.

No existing compiler module or verifier is modified.

The resource certificate is a modeled feasibility claim, not a
measurement of actual MPI memory use or network performance.
"""

from dataclasses import dataclass

from moa_machine import Machine, Work
from moa_typed_ir import schedule_omega
from moa_mpi import emit_mpi
from moa_mpi_collective import INT_MAX

from moa_attention_resource_planner import ResourceOmegaPlan


def collective_from_intervals(intervals, n, dv):
    """Derive MPI_Allgatherv metadata from explicit Omega ownership."""

    if type(n) is not int or n <= 0:
        raise ValueError("invalid query extent")

    if type(dv) is not int or dv <= 0:
        raise ValueError("invalid value extent")

    intervals = tuple(intervals)

    if not intervals:
        raise ValueError("empty ownership")

    if n * dv > INT_MAX:
        raise ValueError("MPI integer range exceeded")

    counts = []
    displacements = []
    cursor = 0

    for interval in intervals:
        if (
            not isinstance(interval, tuple)
            or len(interval) != 2
            or any(type(x) is not int for x in interval)
        ):
            raise ValueError("invalid Omega interval")

        begin, end = interval

        if begin != cursor:
            raise ValueError("gap or overlap in Omega ownership")

        if end < begin or end > n:
            raise ValueError("invalid Omega interval bounds")

        count = (end - begin) * dv
        displacement = begin * dv

        if count > INT_MAX or displacement > INT_MAX:
            raise ValueError("MPI count/displacement overflow")

        counts.append(count)
        displacements.append(displacement)
        cursor = end

    if cursor != n:
        raise ValueError("incomplete Omega ownership")

    if sum(counts) != n * dv:
        raise ValueError("output component count mismatch")

    if any(
        displacements[r] != sum(counts[:r])
        for r in range(len(counts))
    ):
        raise ValueError("noncontiguous gamma layout")

    return tuple(counts), tuple(displacements)


def c_array_declaration(name, values, ctype="int"):
    """Produce a single exact C array declaration."""

    return (
        f" const {ctype} {name}[] = {{"
        + ",".join(str(x) for x in values)
        + "};"
    )


@dataclass(frozen=True)
class ResourceMPIEmission:
    source: str
    intervals: tuple[tuple[int, int], ...]
    counts: tuple[int, ...]
    displacements: tuple[int, ...]
    capacities: tuple[int, ...]
    modeled_bytes: tuple[int, ...]
    output_components: int

    def verify(self):
        if not isinstance(self.source, str):
            raise ValueError("invalid MPI source")

        p = len(self.intervals)

        if not (
            p == len(self.counts)
            == len(self.displacements)
            == len(self.capacities)
            == len(self.modeled_bytes)
        ):
            raise ValueError("rank metadata length mismatch")

        if type(self.output_components) is not int:
            raise ValueError("invalid output component count")

        if not 0 < self.output_components <= INT_MAX:
            raise ValueError("invalid MPI output extent")

        if sum(self.counts) != self.output_components:
            raise ValueError("MPI output count mismatch")

        if any(
            type(c) is not int or c < 0 or c > INT_MAX
            for c in self.counts
        ):
            raise ValueError("invalid MPI count")

        if any(
            type(d) is not int or d < 0 or d > INT_MAX
            for d in self.displacements
        ):
            raise ValueError("invalid MPI displacement")

        if any(
            self.displacements[i]
            != sum(self.counts[:i])
            for i in range(p)
        ):
            raise ValueError("invalid MPI displacement sequence")

        if any(
            type(c) is not int or c <= 0
            for c in self.capacities
        ):
            raise ValueError("invalid resource capacity")

        if any(
            type(b) is not int or b < 0
            for b in self.modeled_bytes
        ):
            raise ValueError("invalid modeled bytes")

        if any(
            b > c
            for b, c in zip(
                self.modeled_bytes,
                self.capacities,
            )
        ):
            raise ValueError("resource capacity exceeded")

        expected_lines = (
            c_array_declaration("counts", self.counts),
            c_array_declaration(
                "displs", self.displacements
            ),
            c_array_declaration(
                "owner_begin",
                (b for b, e in self.intervals),
                "size_t",
            ),
            c_array_declaration(
                "owner_end",
                (e for b, e in self.intervals),
                "size_t",
            ),
        )

        for line in expected_lines:
            if self.source.count(line) != 1:
                raise ValueError(
                    "missing or duplicated MPI declaration"
                )

        if self.source.count("MPI_Allgatherv(") != 1:
            raise ValueError("MPI collective call missing")

        if self.source.count(
            "int generated_attention_mpi("
        ) != 1:
            raise ValueError("MPI function missing")

        return True


def emit_resource_owned_mpi(plan):
    """Generate MPI C with independently verified Omega ownership."""

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

    # Canonical schedule passes the existing verifier unchanged.
    canonical = schedule_omega(
        explicit.lowered,
        work,
        machine,
    )

    # Reuse the existing MPI computation and communication backend.
    original = emit_mpi(canonical)

    old_intervals = canonical.schedule.intervals
    new_intervals = s.intervals

    old_counts, old_displs = collective_from_intervals(
        old_intervals,
        work.n,
        work.dv,
    )

    new_counts, new_displs = collective_from_intervals(
        new_intervals,
        work.n,
        work.dv,
    )

    substitutions = (
        (
            c_array_declaration("counts", old_counts),
            c_array_declaration("counts", new_counts),
        ),
        (
            c_array_declaration("displs", old_displs),
            c_array_declaration("displs", new_displs),
        ),
        (
            c_array_declaration(
                "owner_begin",
                (b for b, e in old_intervals),
                "size_t",
            ),
            c_array_declaration(
                "owner_begin",
                (b for b, e in new_intervals),
                "size_t",
            ),
        ),
        (
            c_array_declaration(
                "owner_end",
                (e for b, e in old_intervals),
                "size_t",
            ),
            c_array_declaration(
                "owner_end",
                (e for b, e in new_intervals),
                "size_t",
            ),
        ),
    )

    generated = original

    for old, new in substitutions:
        if generated.count(old) != 1:
            raise ValueError(
                "canonical MPI emitter structure changed"
            )
        generated = generated.replace(old, new, 1)

    certificate = ResourceMPIEmission(
        source=generated,
        intervals=new_intervals,
        counts=new_counts,
        displacements=new_displs,
        capacities=plan.capacities,
        modeled_bytes=plan.modeled_bytes,
        output_components=work.n * work.dv,
    )

    certificate.verify()
    return certificate
