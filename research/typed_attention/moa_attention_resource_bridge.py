
"""Unified MoA attention resource certificate for the Gate 11 MPI backend.

Connects verified typed attention Omega ownership, Universal IR v3.3
multidimensional lifting, and modeled MPI application storage.

Does not predict physical peak memory, MPI-internal buffering,
network traffic, execution time, power, or energy.
"""

from dataclasses import dataclass

from moa_attention_multidim_bridge import (
    AttentionMultidimensionalBridge,
)
from moa_attention_storage import (
    AttentionStorageCertificate,
    derive_attention_storage,
)


@dataclass(frozen=True)
class AttentionResourceBridge:
    """Combined ownership and modeled-storage feasibility certificate."""

    attention: AttentionMultidimensionalBridge
    storage: AttentionStorageCertificate

    def verify(self):
        self.attention.verify()
        self.storage.verify()

        lifted = self.attention.lifted
        schedule = self.attention.typed.schedule
        resources = lifted.resources

        if self.storage.machine_shape != resources.shape:
            raise ValueError("resource machine shape mismatch")

        if self.storage.output_shape != (
            schedule.query_extent,
            schedule.value_extent,
        ):
            raise ValueError("resource output shape mismatch")

        if self.storage.element_bytes != 8:
            raise ValueError("MPI backend requires 8-byte components")

        if lifted.sequential.dnf.dtype_bytes != 8:
            raise ValueError("output DNF component size mismatch")

        coordinates = resources.indices()

        if len(self.storage.ranks) != len(coordinates):
            raise ValueError("resource rank count mismatch")

        for coordinate, (begin, end), rank in zip(
            coordinates,
            schedule.intervals,
            self.storage.ranks,
        ):
            machine = resources.psi(coordinate)

            if rank.coordinate != coordinate:
                raise ValueError("resource coordinate mismatch")

            if (rank.query_begin, rank.query_end) != (begin, end):
                raise ValueError("resource ownership mismatch")

            if rank.query_count != end - begin:
                raise ValueError("resource query count mismatch")

            if rank.capacity_bytes != machine.capacity_bytes:
                raise ValueError("resource capacity mismatch")

            if rank.feasible != (
                rank.modeled_bytes <= machine.capacity_bytes
            ):
                raise ValueError("resource feasibility mismatch")

        expected_all = all(r.feasible for r in self.storage.ranks)

        if self.storage.all_feasible != expected_all:
            raise ValueError("aggregate resource feasibility mismatch")

        return True

    @property
    def all_feasible(self):
        self.verify()
        return self.storage.all_feasible

    @property
    def maximum_rank_modeled_bytes(self):
        self.verify()
        return self.storage.maximum_rank_modeled_bytes


def bridge_attention_resources(attention):
    """Combine existing Omega and storage certificates.

    A mapping may be valid but infeasible for the MPI backend.
    In that case verification succeeds and all_feasible is False.

    The multidimensional bridge must already satisfy output
    ownership and output-partition resource constraints.
    """
    if not isinstance(attention, AttentionMultidimensionalBridge):
        raise TypeError(
            "expected AttentionMultidimensionalBridge"
        )

    attention.verify()

    if attention.lifted.sequential.dnf.dtype_bytes != 8:
        raise ValueError(
            "MPI backend requires 8-byte components"
        )

    resources = attention.lifted.resources

    capacities = tuple(
        resources.psi(coordinate).capacity_bytes
        for coordinate in resources.indices()
    )

    storage = derive_attention_storage(
        attention.typed,
        element_bytes=8,
        capacity_bytes=capacities,
    )

    result = AttentionResourceBridge(
        attention=attention,
        storage=storage,
    )

    result.verify()
    return result
