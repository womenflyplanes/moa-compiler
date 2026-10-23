"""Additive v3 planner: exact discrete minimax under an EXPLICIT linear-time model.

Not a general performance predictor, nor a proof of MoA optimality.
"""
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Tuple
from moa_v3 import MachineArray, SequentialONF, LiftedONF, lift_with_explicit_chunks

FIELD_NAMES = ('U','W','M','t','n_max','M_cap','P','X','C','R')

@dataclass(frozen=True)
class MeasuredResource:
    machine: MachineArray
    reference_elements: int
    reference_seconds: float
    min_viable_elements: int = 1
    reserved_bytes: int = 0
    replicated_bytes: int = 0
    def __post_init__(self):
        if self.reference_elements <= 0 or not isfinite(self.reference_seconds) or self.reference_seconds <= 0:
            raise ValueError('Positive reference workload and time required')
        if self.min_viable_elements < 1 or min(self.reserved_bytes, self.replicated_bytes) < 0:
            raise ValueError('Invalid resource constraints')
        t = self.machine.fields[3]
        if isinstance(t, (int, float)) and not isinstance(t, bool) and isfinite(t) and t > 0:
            if abs(t-self.reference_seconds) > 1e-9 * max(t,self.reference_seconds):
                raise ValueError('Reference time conflicts with machine field 3')
    def maximum_elements(self, dtype_bytes):
        # Independent restrictions; conservative available storage after replicated inputs.
        m = self.machine
        cap = min(m.capacity_bytes, (m.fields[5] if isinstance(m.fields[5],int) and not isinstance(m.fields[5],bool) and m.fields[5]>=0 else m.capacity_bytes))
        available = max(0,cap-self.reserved_bytes-self.replicated_bytes)
        ceiling = m.fields[4]
        if not isinstance(ceiling,int) or isinstance(ceiling,bool) or ceiling < 0:
            ceiling = m.max_chunk_elements
        return min(m.max_chunk_elements,ceiling,available//dtype_bytes)
    def predicted_seconds(self, elements):
        return elements * self.reference_seconds / self.reference_elements

@dataclass(frozen=True)
class PlannedLift:
    lifted: LiftedONF
    chunks: Tuple[int,...]
    predicted_makespan_seconds: float
    model: str = 'linear reference throughput, parallel makespan, no transfers'


def plan_lift(seq: SequentialONF, resources: Tuple[MeasuredResource,...]):
    """Minimize modeled makespan exactly over integer chunk assignments.

    A nonzero chunk must meet the minimum viable size; idle resources are allowed.
    Uses monotone feasibility and binary search over candidate completion times.
    """
    if not resources:
        raise ValueError('At least one resource required')
    names = [r.machine.name for r in resources]
    if len(set(names)) != len(names):
        raise ValueError('Machine names must be unique')
    n = seq.dnf.tau
    caps = [min(n,r.maximum_elements(seq.dnf.dtype_bytes)) for r in resources]
    mins = [r.min_viable_elements for r in resources]
    if n == 0:
        chunks = (0,)*len(resources)
    else:
        # Exact dynamic-programming feasibility over total assigned elements.
        # Each candidate makespan is one of the finite possible c*t/n0 values.
        times = sorted(set(r.predicted_seconds(c) for r,cap,lo in zip(resources,caps,mins)
                           for c in range(lo,cap+1)))
        if not times:
            raise ValueError('No feasible nonempty resource partition')
        def feasible(limit):
            reachable = {0: ()}
            for r,cap,lo in zip(resources,caps,mins):
                upper = min(cap,int((limit+1e-12)*r.reference_elements/r.reference_seconds))
                choices = [0] + list(range(lo,upper+1)) if upper>=lo else [0]
                nxt = {}
                for total,path in reachable.items():
                    for c in choices:
                        if total+c<=n and total+c not in nxt:
                            nxt[total+c] = path+(c,)
                reachable=nxt
            return reachable.get(n)
        low,high=0,len(times)-1
        if feasible(times[high]) is None:
            raise ValueError('No feasible partition under capacity/minimum-chunk constraints')
        while low<high:
            mid=(low+high)//2
            if feasible(times[mid]) is None: low=mid+1
            else: high=mid
        chunks=feasible(times[low])
    # Explicit lift validates tau and coverage. Validate additional reservations above.
    lifted=lift_with_explicit_chunks(seq,tuple(r.machine for r in resources),chunks)
    for r,c,cap in zip(resources,chunks,caps):
        if c>cap or (c and c<r.min_viable_elements):
            raise AssertionError('Planner violated resource bounds')
    return PlannedLift(lifted,tuple(chunks),max((r.predicted_seconds(c) for r,c in zip(resources,chunks)),default=0.0))
