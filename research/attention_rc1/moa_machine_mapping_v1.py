#!/usr/bin/env python3
"""Provisional 10-field CPU machine descriptor and legal attention tile mapping.

Not a calibrated performance model or a general MoA dimension-lifting engine.
The descriptor fields are a proposed interface, not canonical MoA variables.
"""
from dataclasses import dataclass
from moa_structured_codegen_v1 import AttentionIR, Domain, Mapping
from moa_attention_psi_onf_v1 import emit

@dataclass(frozen=True)
class MachineShape10:
    logical_cpus: int
    performance_cores: int
    efficiency_cores: int
    l1_data_bytes: int
    l2_bytes: int
    cache_line_bytes: int
    ram_bytes: int
    max_omp_threads: int
    simd_bytes: int
    page_bytes: int

    def __post_init__(self):
        if any(v <= 0 for v in vars(self).values()):
            raise ValueError('all machine dimensions must be positive')
        if self.performance_cores + self.efficiency_cores > self.logical_cpus:
            raise ValueError('core counts exceed logical CPUs')
        if self.max_omp_threads > self.logical_cpus:
            raise ValueError('thread limit exceeds logical CPUs')
        if self.l1_data_bytes > self.l2_bytes or self.l2_bytes > self.ram_bytes:
            raise ValueError('inconsistent memory hierarchy')

@dataclass(frozen=True)
class AttentionShape:
    batch: int
    length: int
    width: int
    value_width: int

    def __post_init__(self):
        if min(vars(self).values()) <= 0:
            raise ValueError('all array extents must be positive')

@dataclass(frozen=True)
class Plan:
    query_tile: int
    key_tile: int
    requested_threads: int
    query_blocks: int
    workspace_bytes: int
    tensor_bytes: int
    total_live_bytes: int
    per_query_score_bytes: int
    l1_tile_estimate_bytes: int
    backend: str


def map_attention(machine: MachineShape10, shape: AttentionShape, backend='openmp') -> Plan:
    if backend not in ('openmp', 'sequential'):
        raise ValueError('unsupported backend')
    b,n,d,dv = shape.batch,shape.length,shape.width,shape.value_width
    # Current attention kernel materializes P and an additional B*N*N workspace.
    workspace = 8*b*n*n
    tensor = 8*b*(2*n*d + n*dv + n*n + n*dv)
    total = workspace + tensor
    if total > machine.ram_bytes:
        raise ValueError('estimated live allocation exceeds RAM')
    # Conservative estimated tile footprint: Q block, K block, score block.
    # Not a cache residency proof: ignores associativity, runtime and other traffic.
    budget = machine.l1_data_bytes // 2
    tq = tk = 1
    for q in range(1,n+1):
        for k in range(1,n+1):
            footprint = 8*(q*d + k*d + q*k)
            if footprint <= budget and q*k > tq*tk:
                tq,tk = q,k
    # Always legal even when one pair does not fit the heuristic budget.
    threads = min(machine.max_omp_threads, machine.logical_cpus,
                  b*((n+tq-1)//tq)) if backend == 'openmp' else 1
    return Plan(tq,tk,threads,(n+tq-1)//tq,workspace,tensor,total,
                8*n,8*(tq*d+tk*d+tq*tk),backend)


def emit_mapped(machine: MachineShape10, shape: AttentionShape, backend='openmp'):
    plan = map_attention(machine,shape,backend)
    ir = AttentionIR(Domain('i',shape.length,plan.query_tile),
                     Domain('j',shape.length,plan.key_tile))
    # Thread count is an execution recommendation; set OMP_NUM_THREADS externally.
    return plan, emit(ir, Mapping(backend))


def m1_pro_example():
    """Example values: verify cache/SIMD/page and runtime limits on actual host."""
    return MachineShape10(10,8,2,131072,12582912,128,32*1024**3,8,16,16384)
