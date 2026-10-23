"""Additive v2 SelectionDNF -> v3.1 resource planner bridge.

No psi rewriting; DNF is accepted as a completed immutable input.
The planner uses a documented simplified linear makespan model.
"""
from dataclasses import dataclass
from typing import Tuple
from moa_universal_ir_v2 import SelectionDNF, lower_dnf_row_major, iota
from moa_v3 import DNF, gamma_sequential, LiftedONF
from moa_partition_planner import MeasuredResource, plan_lift

@dataclass(frozen=True)
class MappedSelection:
    dnf: SelectionDNF
    source_offsets: Tuple[int, ...]
    lifted: LiftedONF
    chunks: Tuple[int, ...]
    predicted_seconds: float
    def verify(self):
        assert self.lifted.verify()
        assert len(self.source_offsets) == len(self.dnf.source_indices)
        assert sum(self.chunks) == len(self.source_offsets)
        assert self.source_offsets == lower_dnf_row_major(self.dnf)
        # Every original selection position belongs to exactly one partition.
        positions = tuple(i for p in self.lifted.partitions for i in range(p.start,p.stop))
        assert positions == tuple(range(len(self.source_offsets)))
        return True
    def resource_offsets(self):
        return tuple((p.machine,self.source_offsets[p.start:p.stop]) for p in self.lifted.partitions)

def plan_selection(dnf: SelectionDNF, dtype_bytes: int, resources: Tuple[MeasuredResource,...]):
    """Partition flattened DNF output indices; source addressing stays gamma-derived.

    Does not implement communication, execution, general DNF rewrite, or true
    multidimensional tiling. It preserves the existing SelectionDNF object.
    """
    if not isinstance(dnf,SelectionDNF):
        raise TypeError('Expected an already-reduced v2 SelectionDNF')
    if type(dtype_bytes) is not int or dtype_bytes<=0:
        raise ValueError('dtype_bytes must be a positive integer')
    seq=gamma_sequential(DNF(dnf.shape,dtype_bytes))
    plan=plan_lift(seq,tuple(resources))
    result=MappedSelection(dnf,lower_dnf_row_major(dnf),plan.lifted,
                           plan.chunks,plan.predicted_makespan_seconds)
    result.verify()
    return result
