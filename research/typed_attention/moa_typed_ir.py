"""Restricted, typed MoA forward-attention IR and independently checkable lowering.

This is a first non-template derivation gate, not a general MoA compiler.
"""
from dataclasses import dataclass, asdict
import json

@dataclass(frozen=True)
class Array:
    name: str
    shape: tuple[str, ...]

@dataclass(frozen=True)
class Psi:
    array: Array
    indices: tuple[str, ...]

@dataclass(frozen=True)
class Product:
    lhs: Psi
    rhs: Psi

@dataclass(frozen=True)
class Reduce:
    index: str
    term: Product

@dataclass(frozen=True)
class Scale:
    term: Reduce
    factor: str

@dataclass(frozen=True)
class SoftmaxWeightedReduce:
    score: Scale
    values: Psi
    key_index: str
    value_index: str
    stable: bool = True

@dataclass(frozen=True)
class OnlineSummary:
    score: Scale
    values: Psi
    key_index: str
    value_index: str
    state: tuple[str, ...] = ('a', 'z', 'u')

@dataclass(frozen=True)
class GammaAddress:
    array: Array
    indices: tuple[str, ...]
    width: str

@dataclass(frozen=True)
class LoweredAttention:
    summary: OnlineSummary
    addresses: tuple[GammaAddress, ...]
    ownership: str

Q=Array('Q',('n','d')); K=Array('K',('m','d')); V=Array('V',('m','dv'))

def specification():
    score=Scale(Reduce('j',Product(Psi(Q,('i','j')),Psi(K,('k','j')))), '1/sqrt(d)')
    return SoftmaxWeightedReduce(score,Psi(V,('k','v')),'k','v')

def stable_online_fold(expr):
    # Rule applies only to stable softmax-weighted reductions with the exact
    # indexed score and value dependence; reject unsupported expressions.
    if not isinstance(expr,SoftmaxWeightedReduce) or not expr.stable: raise ValueError('unsupported reduction')
    s=expr.score
    if not isinstance(s,Scale) or s.factor!='1/sqrt(d)' or not isinstance(s.term,Reduce): raise ValueError('unsupported score')
    r=s.term
    if (r.index!='j' or r.term.lhs!=Psi(Q,('i','j')) or
        r.term.rhs!=Psi(K,('k','j')) or expr.values!=Psi(V,('k','v')) or
        expr.key_index!='k' or expr.value_index!='v'):
        raise ValueError('unsupported ψ dependencies')
    return OnlineSummary(s,expr.values,expr.key_index,expr.value_index)

def gamma_lower(expr, ownership='query-disjoint'):
    if not isinstance(expr,OnlineSummary) or ownership!='query-disjoint': raise ValueError('unsupported lowering')
    return LoweredAttention(expr,(
        GammaAddress(Q,('i','j'),'d'),GammaAddress(K,('k','j'),'d'),
        GammaAddress(V,('k','v'),'dv'),GammaAddress(Array('O',('n','dv')),('i','v'),'dv')),
        ownership)

def verify(expr):
    if not isinstance(expr,LoweredAttention) or expr.ownership!='query-disjoint': raise ValueError('invalid ownership')
    expected=(('Q',('i','j'),'d'),('K',('k','j'),'d'),('V',('k','v'),'dv'),('O',('i','v'),'dv'))
    actual=tuple((a.array.name,a.indices,a.width) for a in expr.addresses)
    if actual!=expected: raise ValueError('invalid gamma mapping')
    return True

@dataclass(frozen=True)
class OmegaSchedule:
    """Typed Ω mapping of query ψ indices onto machine components."""
    machine_shape: tuple[int, ...]
    query_extent: int
    key_extent: int
    score_extent: int
    value_extent: int
    intervals: tuple[tuple[int, int], ...]

@dataclass(frozen=True)
class ScheduledAttention:
    lowered: LoweredAttention
    schedule: OmegaSchedule

def schedule_omega(expr, work, machine):
    from moa_machine import omega_lift, verify_lift
    verify(expr)
    owners = omega_lift(work, machine)
    verify_lift(work, machine, owners)
    return ScheduledAttention(expr, OmegaSchedule(machine.shape, work.n, work.m, work.d,
                                                    work.dv, tuple((o.begin,o.end) for o in owners)))

def verify_schedule(expr):
    from moa_machine import Machine, Work, omega_lift, verify_lift
    if not isinstance(expr, ScheduledAttention): raise ValueError('not a scheduled attention IR')
    verify(expr.lowered)
    s = expr.schedule
    machine = Machine(s.machine_shape)
    work = Work(s.query_extent,s.key_extent,s.score_extent,s.value_extent)
    owners = omega_lift(work,machine)
    verify_lift(work,machine,owners)
    if s.intervals != tuple((o.begin,o.end) for o in owners):
        raise ValueError('invalid Ω intervals')
    return True

def emit_c(expr):
    scheduled = isinstance(expr, ScheduledAttention)
    if scheduled:
        verify_schedule(expr)
        schedule = expr.schedule
        expr = expr.lowered
    verify(expr)
    address={a.array.name:f'{a.array.name}[gamma2({a.indices[0]},{a.indices[1]},{a.width})]' for a in expr.addresses}
    # Emission is assembled from typed ψ dependencies, gamma addresses, and
    # the verified stable summary recurrence, not a copied attention kernel.
    lines=['/* Generated from typed MoA ψ DNF → summary rule → γ ONF. */',
      '#include <math.h>','#include <stddef.h>','#include <stdlib.h>',
      '#ifdef _OPENMP','#include <omp.h>','#endif',
      'static inline size_t gamma2(size_t i,size_t j,size_t w){return i*w+j;}',
      'void generated_attention(const double *Q,const double *K,const double *V,',
      '                         double *O,size_t n,size_t m,size_t d,size_t dv){',
      ' if(!n||!m||!d||!dv)abort();',
      ' const double scale=1.0/sqrt((double)d);',
      *([f' if(n!={schedule.query_extent} || m!={schedule.key_extent} || d!={schedule.score_extent} || dv!={schedule.value_extent})abort();',
          ' const size_t owner_begin[] = {'+','.join(str(x[0]) for x in schedule.intervals)+'};',
          ' const size_t owner_end[] = {'+','.join(str(x[1]) for x in schedule.intervals)+'};',
          ' #pragma omp parallel for schedule(static)',
          f' for(size_t owner=0;owner<{len(schedule.intervals)};owner++){{',
          ' for(size_t i=owner_begin[owner];i<owner_end[owner];i++){'] if scheduled else
          [' #pragma omp parallel for schedule(static)', ' for(size_t i=0;i<n;i++){']),
      '  double *u=(double*)calloc(dv,sizeof(double)); if(!u)abort();',
      '  double a=-INFINITY,z=0;',
      '  for(size_t k=0;k<m;k++){',
      '   double score=0;',
      f'   for(size_t j=0;j<d;j++)score+={address["Q"]}*{address["K"]};',
      '   score*=scale;',
      '   double next=fmax(a,score);',
      '   double old=(z==0)?0:exp(a-next),fresh=exp(score-next);',
      '   z=old*z+fresh;',
      f'   for(size_t v=0;v<dv;v++)u[v]=old*u[v]+fresh*{address["V"]};',
      '   a=next;',
      '  }',
      f'  for(size_t v=0;v<dv;v++){address["O"]}=u[v]/z;',
      '  free(u);',
      ' }',*([' }'] if scheduled else []),'}']
    return '\n'.join(lines)+'\n'

def emit_openacc(expr):
    """Emit OpenACC from the SAME verified typed Ω-scheduled attention IR.

    A compile-time value extent supplies the private per-query accumulator.
    This is an accelerator-target source backend, not proof of device execution.
    """
    if not isinstance(expr, ScheduledAttention):
        raise ValueError('OpenACC requires typed Ω schedule')
    verify_schedule(expr)
    sch=expr.schedule
    if sch.value_extent < 1:
        raise ValueError('invalid value extent')
    address={a.array.name:f'{a.array.name}[gamma2({a.indices[0]},{a.indices[1]},{a.width})]'
             for a in expr.lowered.addresses}
    intervals=sch.intervals
    lines=[
      '/* Generated from typed MoA ψ DNF → stable summary → γ ONF → Ω ownership. */',
      '#include <math.h>', '#include <stddef.h>', '#include <stdlib.h>',
      'static inline size_t gamma2(size_t i,size_t j,size_t w){return i*w+j;}',
      'void generated_attention(const double *Q,const double *K,const double *V,',
      '                         double *O,size_t n,size_t m,size_t d,size_t dv){',
      f' if(n!={sch.query_extent} || m!={sch.key_extent} || d!={sch.score_extent} || dv!={sch.value_extent})abort();',
      ' const double scale=1.0/sqrt((double)d);',
      ' const size_t owner_begin[] = {'+','.join(str(x[0]) for x in intervals)+'};',
      ' const size_t owner_end[] = {'+','.join(str(x[1]) for x in intervals)+'};',
      ' /* The OpenACC data region transfers inputs and outputs; ownership is disjoint. */',
      ' #pragma acc data copyin(Q[0:n*d],K[0:m*d],V[0:m*dv]) copyout(O[0:n*dv])',
      ' {',
      f' #pragma acc parallel loop gang copyin(owner_begin[0:{len(intervals)}],owner_end[0:{len(intervals)}])',
      f' for(size_t owner=0;owner<{len(intervals)};owner++){{',
      '  #pragma acc loop seq',
      '  for(size_t i=owner_begin[owner];i<owner_end[owner];i++){',
      f'   double u[{sch.value_extent}];',
      '   for(size_t v=0;v<dv;v++)u[v]=0.0;',
      '   double a=-INFINITY,z=0.0;',
      '   for(size_t k=0;k<m;k++){',
      '    double score=0.0;',
      f'    for(size_t j=0;j<d;j++)score+={address["Q"]}*{address["K"]};',
      '    score*=scale;',
      '    double next=fmax(a,score);',
      '    double old=(z==0.0)?0.0:exp(a-next),fresh=exp(score-next);',
      '    z=old*z+fresh;',
      f'    for(size_t v=0;v<dv;v++)u[v]=old*u[v]+fresh*{address["V"]};',
      '    a=next;',
      '   }',
      f'   for(size_t v=0;v<dv;v++){address["O"]}=u[v]/z;',
      '  }', ' }', ' }', '}'
    ]
    return '\n'.join(lines)+'\n'

def compile_file(folder):
    from pathlib import Path
    folder=Path(folder)
    ast=specification(); fold=stable_online_fold(ast); lowered=gamma_lower(fold)
    (folder/'generated_attention_cpu.c').write_text(emit_c(lowered))
    (folder/'typed_derivation_trace.json').write_text(json.dumps({
      'dnf':asdict(ast),'stable_summary':asdict(fold),'onf':asdict(lowered),
      'scope':'restricted forward-attention typed rewrite; not general MoA compilation'
    },indent=2)+'\n')
    return lowered

if __name__=='__main__':
    from pathlib import Path
    compile_file(Path(__file__).parent)
    print('Typed MoA rewrite and γ ONF generated')
