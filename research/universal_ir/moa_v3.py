"""Universal MoA v3 prototype: immutable DNF, explicit sequential gamma, partitioned ONF.
This is a shape-partition validation prototype, NOT a full MoA compiler or cost optimizer.
"""
from dataclasses import dataclass
from math import prod
from typing import Tuple

@dataclass(frozen=True)
class DNF:
    shape: Tuple[int, ...]
    dtype_bytes: int
    def __post_init__(self):
        if any(x < 0 for x in self.shape) or self.dtype_bytes <= 0:
            raise ValueError('Invalid DNF shape or element size')
    @property
    def tau(self): return prod(self.shape)

@dataclass(frozen=True)
class MachineArray:
    name: str
    fields: Tuple[object, ...]  # 10 established slots; extensible for MPI and beyond
    capacity_bytes: int
    max_chunk_elements: int
    def __post_init__(self):
        if len(self.fields) < 10 or self.capacity_bytes < 0 or self.max_chunk_elements < 0:
            raise ValueError('Machine record requires 10+ fields and valid capacities')

@dataclass(frozen=True)
class SequentialONF:
    dnf: DNF
    strides: Tuple[int, ...]
    def address(self, index):
        if len(index) != len(self.dnf.shape) or any(i < 0 or i >= d for i,d in zip(index,self.dnf.shape)):
            raise IndexError(index)
        return sum(i*s for i,s in zip(index,self.strides))

def gamma_sequential(dnf):
    strides = tuple(prod(dnf.shape[i+1:]) for i in range(len(dnf.shape)))
    return SequentialONF(dnf, strides)

@dataclass(frozen=True)
class Partition:
    machine: str
    start: int
    stop: int

@dataclass(frozen=True)
class LiftedONF:
    sequential: SequentialONF
    partitions: Tuple[Partition, ...]
    def verify(self):
        cursor=0
        for p in self.partitions:
            if p.start != cursor or p.stop < p.start:
                raise AssertionError('Partitions must be ordered, disjoint, and contiguous')
            cursor=p.stop
        if cursor != self.sequential.dnf.tau:
            raise AssertionError('Partitioned extent does not equal DNF tau')
        return True

def lift_with_explicit_chunks(seq, machines, chunks):
    """Caller supplies a candidate partition; rejects unsupported sizes.
    This does not claim to select optimal chunks or model replicated inputs.
    """
    if len(machines)!=len(chunks) or any(not isinstance(c,int) or c<0 for c in chunks):
        raise ValueError('Invalid partition vector')
    if sum(chunks)!=seq.dnf.tau:
        raise ValueError('Partition sizes must sum to tau(DNF)')
    parts=[]; start=0
    for m,c in zip(machines,chunks):
        if c>m.max_chunk_elements or c*seq.dnf.dtype_bytes>m.capacity_bytes:
            raise ValueError(f'Partition exceeds modeled capacity: {m.name}')
        parts.append(Partition(m.name,start,start+c)); start+=c
    lifted=LiftedONF(seq,tuple(parts)); lifted.verify(); return lifted
