
"""Bridge Gate 11 typed MoA attention to Universal IR v3.3.

Preserves the original typed ψ derivation and Ω schedule.
Verifies output ownership, sequential γ addresses, and output-storage
resource limits. Does not account for replicated inputs, scratch space,
communication buffers, or physical execution time.
"""

from dataclasses import dataclass
from math import prod

from moa_typed_ir import ScheduledAttention, verify_schedule
from moa_v3 import DNF, gamma_sequential
from moa_multidimensional_lift import (
    ResourceArray,
    Rectangle,
    rectangular_lift,
)


@dataclass(frozen=True)
class AttentionMultidimensionalBridge:
    typed: ScheduledAttention
    lifted: object

    def verify(self):
        verify_schedule(self.typed)

        s = self.typed.schedule
        out = self.lifted

        if out.resources.shape != s.machine_shape:
            raise ValueError("machine shape mismatch")

        if out.sequential.dnf.shape != (
            s.query_extent, s.value_extent
        ):
            raise ValueError("output shape mismatch")

        if len(out.rectangles) != len(s.intervals):
            raise ValueError("ownership rectangle count mismatch")

        out.verify()

        for coord, (begin, end), rect in zip(
            out.resources.indices(),
            s.intervals,
            out.rectangles,
        ):
            expected = Rectangle(
                (begin, 0),
                (end, s.value_extent),
                coord,
            )
            if rect != expected:
                raise ValueError("Omega ownership mismatch")

        for (i, v), coord, address in out.mapped_indices():
            expected_owner = next(
                owner
                for owner, (begin, end) in zip(
                    out.resources.indices(), s.intervals
                )
                if begin <= i < end
            )

            if coord != expected_owner:
                raise ValueError("mapped ownership mismatch")

            if address != i * s.value_extent + v:
                raise ValueError("gamma address mismatch")

        return True


def bridge_attention(scheduled, resources, dtype_bytes=8):
    if not isinstance(scheduled, ScheduledAttention):
        raise TypeError("expected ScheduledAttention")

    verify_schedule(scheduled)

    if not isinstance(resources, ResourceArray):
        raise TypeError("expected ResourceArray")

    if type(dtype_bytes) is not int or dtype_bytes <= 0:
        raise ValueError("dtype_bytes must be positive")

    s = scheduled.schedule

    if resources.shape != s.machine_shape:
        raise ValueError("machine shape mismatch")

    if len(s.intervals) != prod(resources.shape):
        raise ValueError("ownership cardinality mismatch")

    sequential = gamma_sequential(
        DNF((s.query_extent, s.value_extent), dtype_bytes)
    )

    rectangles = tuple(
        Rectangle(
            (begin, 0),
            (end, s.value_extent),
            coord,
        )
        for coord, (begin, end) in zip(
            resources.indices(),
            s.intervals,
        )
    )

    lifted = rectangular_lift(
        sequential,
        resources,
        rectangles,
    )

    result = AttentionMultidimensionalBridge(
        scheduled,
        lifted,
    )

    result.verify()
    return result
