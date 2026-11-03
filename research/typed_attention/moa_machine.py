"""Finite MoA Ω query lifting and exact logical resource accounting.

No claim of automatic network topology or physical time prediction.
"""
from dataclasses import dataclass, asdict
from math import prod
import json

@dataclass(frozen=True)
class Machine:
    shape: tuple[int,...]
    def __post_init__(self):
        if not self.shape or any(type(x) is not int or x<=0 for x in self.shape):
            raise ValueError('machine ρ must be positive and nonempty')
    @property
    def tau(self): return prod(self.shape)

@dataclass(frozen=True)
class Work:
    n:int; m:int; d:int; dv:int
    def __post_init__(self):
        if any(type(x) is not int or x<=0 for x in (self.n,self.m,self.d,self.dv)):
            raise ValueError('positive attention shapes required')

@dataclass(frozen=True)
class Owner:
    coordinate: tuple[int,...]
    begin:int
    end:int
    def count(self):return self.end-self.begin

def iota(shape):
    """MoA ι(ρmachine) in row-major γ order, returning ψ index vectors."""
    from itertools import product
    return tuple(product(*(range(s) for s in shape)))

def gamma(coord,shape):
    if len(coord)!=len(shape) or any(c<0 or c>=s for c,s in zip(coord,shape)):
        raise ValueError('out of bounds machine ψ coordinate')
    address=0
    for c,s in zip(coord,shape):address=address*s+c
    return address

def omega_lift(work,machine):
    """Map disjoint query intervals onto every machine component, including empty ones."""
    p=machine.tau
    return tuple(Owner(c,(work.n*r)//p,(work.n*(r+1))//p) for r,c in enumerate(iota(machine.shape)))

def verify_lift(work,machine,owners):
    if len(owners)!=machine.tau or tuple(x.coordinate for x in owners)!=iota(machine.shape):
        raise ValueError('Ω machine shape mismatch')
    seen=[]
    for o in owners:
        if not (0<=o.begin<=o.end<=work.n):raise ValueError('invalid ψ interval')
        seen.extend(range(o.begin,o.end))
    if seen!=list(range(work.n)):raise ValueError('output ownership not a bijection')
    if sum(o.count()*work.dv for o in owners)!=work.n*work.dv:
        raise ValueError('τ(output) not preserved')
    return True

def resource_certificate(work,machine):
    owners=omega_lift(work,machine)
    verify_lift(work,machine,owners)
    per=[]
    for o in owners:
        q=o.count()
        per.append({'coordinate':list(o.coordinate),'gamma_machine':gamma(o.coordinate,machine.shape),
          'query_begin':o.begin,'query_end':o.end,'output_components':q*work.dv,
          'score_dot_products':q*work.m,'score_multiplications':q*work.m*work.d,
          'score_additions':q*work.m*max(work.d-1,0),
          'value_accumulations':q*work.m*work.dv,
          'online_summary_components':q*(work.dv+2)})
    assert sum(x['score_dot_products'] for x in per)==work.n*work.m
    assert sum(x['output_components'] for x in per)==work.n*work.dv
    return {'scope':'logical exact counts; not physical time, power, or memory traffic',
      'rho_machine':list(machine.shape),'tau_machine':machine.tau,
      'rho_output':[work.n,work.dv],'tau_output':work.n*work.dv,
      'ownership':'disjoint query intervals; replicated read-only K and V assumed',
      'total_score_dot_products':work.n*work.m,
      'total_score_multiplications':work.n*work.m*work.d,
      'total_score_additions':work.n*work.m*(work.d-1),
      'total_value_accumulations':work.n*work.m*work.dv,
      'components':per}

if __name__=='__main__':
    from pathlib import Path
    p=Path(__file__).parent/'machine_lift_certificate.json'
    p.write_text(json.dumps(resource_certificate(Work(7,5,4,3),Machine((2,3))),indent=2)+'\n')
    print('Ω ownership certificate generated')
