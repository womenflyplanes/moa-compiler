"""Step 109: audited MoA Omega schedule -> staged MPI C.

Restricted forward-attention compiler.

Preserves the canonical stable online attention computation.
Replaces MPI_Allgatherv with audited point-to-point transfers.

Predicted communication time belongs to the logical MoA model,
not to measured MPI execution.
"""

from dataclasses import dataclass

from moa_machine import Machine, gamma
from moa_attention_integrated_108 import IntegratedAttention108
from moa_attention_resource_mpi import (
    emit_resource_owned_mpi,
    collective_from_intervals,
)

INT_MAX = 2**31 - 1


@dataclass(frozen=True)
class ScheduledMPI109:
    source: str
    schedule: object
    counts: tuple
    displacements: tuple
    transfers: tuple

    def verify(self):
        if "MPI_Allgatherv(" in self.source:
            raise ValueError("collective remains in scheduled backend")

        if self.source.count("MPI_Irecv(") != 1:
            raise ValueError("missing staged receive operation")

        if self.source.count("MPI_Isend(") != 1:
            raise ValueError("missing staged send operation")

        if self.source.count("MPI_Waitall(") != 1:
            raise ValueError("missing stage completion operation")

        if sum(self.counts) <= 0:
            raise ValueError("empty output")

        expected = tuple(
            (stage_index, t.owner, t.parent, t.child, t.payload_bytes)
            for stage_index, stage in enumerate(self.schedule.stages)
            for t in stage
        )

        if self.transfers != expected:
            raise ValueError("generated transfer certificate mismatch")

        return True


def emit_scheduled_mpi_109(integrated):
    """Emit C from an independently audited Step 108 result."""

    if not isinstance(integrated, IntegratedAttention108):
        raise TypeError("IntegratedAttention108 required")

    integrated.verify()

    plan = integrated.plan
    machine = Machine(integrated.topology.machine_shape)

    if integrated.mode == "heuristic":
        schedule = integrated.scheduling.selected
    elif integrated.mode == "exact":
        schedule = integrated.scheduling.optimal
    else:
        raise ValueError("invalid scheduling mode")

    canonical = emit_resource_owned_mpi(plan)
    source = canonical.source

    s = plan.certificate.scheduled.schedule
    intervals = tuple(s.intervals)

    from moa_machine import Work
    work = Work(
        s.query_extent,
        s.key_extent,
        s.score_extent,
        s.value_extent,
    )

    counts, displacements = collective_from_intervals(
        intervals, work.n, work.dv
    )

    if len(counts) != machine.tau:
        raise ValueError("machine rank count mismatch")

    transfers = []
    stage_offsets = [0]

    for stage_index, stage in enumerate(schedule.stages):
        for t in stage:
            owner = gamma(t.owner, machine.shape)
            parent = gamma(t.parent, machine.shape)
            child = gamma(t.child, machine.shape)

            if t.payload_bytes % 8:
                raise ValueError("non-double payload")

            count = t.payload_bytes // 8

            if count != counts[owner]:
                raise ValueError("owner payload count mismatch")

            if count <= 0 or count > INT_MAX:
                raise ValueError("invalid MPI payload count")

            transfers.append(
                (
                    stage_index,
                    t.owner,
                    t.parent,
                    t.child,
                    t.payload_bytes,
                )
            )

        stage_offsets.append(len(transfers))

    if machine.tau - 1 > 32767:
        raise ValueError("portable minimum MPI tag bound exceeded")

    transfer_rows = []

    for _, owner_coord, parent_coord, child_coord, payload in transfers:
        transfer_rows.append(
            (
                gamma(owner_coord, machine.shape),
                gamma(parent_coord, machine.shape),
                gamma(child_coord, machine.shape),
                payload // 8,
            )
        )

    if not transfer_rows:
        transfer_rows = [(0, 0, 0, 0)]

    table = ",\n".join(
        "  {%d,%d,%d,%d}" % row
        for row in transfer_rows
    )

    offsets = ",".join(map(str, stage_offsets))

    replacement = f"""
 /* Copy the locally owned gamma interval into final output storage. */
 memcpy(O+displs[rank],local,(size_t)counts[rank]*sizeof(double));
 free(local);

 /* Each row is {{owner_rank,source_rank,destination_rank,count}}. */
 static const int transfers[][4] = {{
{table}
 }};
 static const int stage_offsets[] = {{{offsets}}};

 for(int stage=0;stage<{schedule.stage_count};stage++){{
   int first=stage_offsets[stage];
   int last=stage_offsets[stage+1];
   int request_capacity=2*(last-first);
   MPI_Request *requests=(MPI_Request*)malloc(
       (size_t)(request_capacity?request_capacity:1)*sizeof(MPI_Request));
   if(!requests)return 7;
   int used=0;

   /* Post receives before sends. */
   for(int k=first;k<last;k++){{
     int owner=transfers[k][0];
     int sender=transfers[k][1];
     int receiver=transfers[k][2];
     int count=transfers[k][3];
     if(rank==receiver){{
       if(MPI_Irecv(O+displs[owner],count,MPI_DOUBLE,
                    sender,owner,comm,&requests[used++])!=MPI_SUCCESS){{
         free(requests);return 8;
       }}
     }}
   }}

   for(int k=first;k<last;k++){{
     int owner=transfers[k][0];
     int sender=transfers[k][1];
     int receiver=transfers[k][2];
     int count=transfers[k][3];
     if(rank==sender){{
       if(MPI_Isend(O+displs[owner],count,MPI_DOUBLE,
                    receiver,owner,comm,&requests[used++])!=MPI_SUCCESS){{
         free(requests);return 9;
       }}
     }}
   }}

   if(MPI_Waitall(used,requests,MPI_STATUSES_IGNORE)!=MPI_SUCCESS){{
     free(requests);return 10;
   }}
   free(requests);
 }}

 return 0;
"""

    old = (
        " int status=MPI_Allgatherv("
        "local,counts[rank],MPI_DOUBLE,O,counts,displs,MPI_DOUBLE,comm);\n"
        " free(local);\n"
        " return status==MPI_SUCCESS?0:6;"
    )

    if source.count(old) != 1:
        raise ValueError("canonical MPI communication structure changed")

    source = source.replace(old, replacement.rstrip(), 1)

    source = source.replace(
        "#include <stdlib.h>",
        "#include <stdlib.h>\n#include <string.h>",
        1,
    )

    source = source.replace(
        "/* All ranks receive complete O via MPI_Allgatherv; no network model inferred. */",
        "/* Complete O via audited staged point-to-point Omega transfers. */",
        1,
    )

    result = ScheduledMPI109(
        source=source,
        schedule=schedule,
        counts=counts,
        displacements=displacements,
        transfers=tuple(transfers),
    )

    result.verify()
    return result
