"""Exact logical MPI_Allgatherv metadata and volume bounds for replicated inputs.

These are *not* topology-specific network bytes or timing predictions.
"""
from dataclasses import dataclass
from moa_typed_ir import ScheduledAttention,verify_schedule

INT_MAX=2**31-1

@dataclass(frozen=True)
class Collective:
    counts:tuple[int,...]
    displacements:tuple[int,...]
    output_components:int
    element_bytes:int
    per_rank_receive_components:int
    aggregate_logical_receive_components:int
    aggregate_nonlocal_required_components:int
    contract:str='replicated Q,K,V; every rank receives complete O'

def derive_collective(ir,element_bytes=8):
    if not isinstance(ir,ScheduledAttention):raise ValueError('typed scheduled IR required')
    verify_schedule(ir)
    if type(element_bytes)!=int or element_bytes<=0:raise ValueError('invalid component size')
    s=ir.schedule
    counts=tuple((end-begin)*s.value_extent for begin,end in s.intervals)
    displs=tuple(begin*s.value_extent for begin,end in s.intervals)
    total=s.query_extent*s.value_extent
    if total>INT_MAX:raise ValueError('MPI int count/displacement overflow')
    if sum(counts)!=total:raise ValueError('τ output mismatch')
    if any(displs[i]+counts[i]>total for i in range(len(counts))):raise ValueError('invalid displacement')
    if tuple(displs)!=tuple(sum(counts[:i]) for i in range(len(counts))):raise ValueError('noncontiguous γ layout')
    p=len(counts)
    return Collective(counts,displs,total,element_bytes,total,p*total,(p-1)*total)
