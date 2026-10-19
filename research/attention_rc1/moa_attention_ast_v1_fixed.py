#!/usr/bin/env python3
"""Executable, inspectable MoA forward-attention AST prototype.
Semantic DNF AST -> index-lifted ONF AST -> sequential instantiation.
Not a general MoA compiler or formal proof; preserves both P and O outputs.
"""
from dataclasses import dataclass, asdict
from math import prod, sqrt, exp
from itertools import product
import json, random

@dataclass(frozen=True)
class Node:
    op: str
    shape: tuple
    children: tuple = ()
    attrs: tuple = ()

    def dict(self):
        return {'op': self.op, 'shape': list(self.shape), 'attrs': dict(self.attrs),
                'children': [c.dict() for c in self.children]}

def make_ast(B,N,D,Dv):
    q=Node('input',(B,N,D),attrs=(('name','Q'),))
    k=Node('input',(B,N,D),attrs=(('name','K'),))
    v=Node('input',(B,N,Dv),attrs=(('name','V'),))
    score=Node('psi_contract',(B,N,N),(q,k), (('reduction','d'),('K_index','(b,j,d)'),('Q_index','(b,i,d)')))
    prob=Node('stable_row_softmax',(B,N,N),(score,))
    out=Node('psi_contract',(B,N,Dv),(prob,v),(('reduction','j'),('P_index','(b,i,j)'),('V_index','(b,j,t)')))
    return Node('tuple_outputs',(),(prob,out),(('outputs','P,O'),))

def factor_index(i, n, tile):
    if not (0<=i<n and tile>0): raise ValueError('invalid index/tile')
    return (i//tile,i%tile)

def lift_ast(ast, query_tile, key_tile):
    B,N,D=ast.children[0].children[0].children[0].shape
    if query_tile<1 or key_tile<1: raise ValueError('invalid tile')
    # Quotient/remainder lifting: domain mask is essential when not divisible.
    return Node('lifted_attention',(),(ast,),(
        ('query_shape',( (N+query_tile-1)//query_tile,query_tile)),
        ('key_shape',((N+key_tile-1)//key_tile,key_tile)),
        ('query_valid','qi*query_tile+qj < N'),
        ('key_valid','ki*key_tile+kj < N'),
        ('query_tile',query_tile),('key_tile',key_tile),
        ('index_correspondence','i=qi*Tq+qj; j=ki*Tk+kj'),
        ('outputs','P,O')))

def check_index_mapping(N,T):
    indices=[a*T+b for a in range((N+T-1)//T) for b in range(T) if a*T+b<N]
    assert indices==list(range(N))
    return len(indices)

def instantiate(ast,Q,K,V):
    cfg=dict(ast.attrs); Tq=cfg['query_tile']; Tk=cfg['key_tile']
    B=len(Q);N=len(Q[0]);D=len(Q[0][0]);Dv=len(V[0][0]);
    assert len(K)==B and all(len(x)==N for x in K) and all(len(x)==N for x in V)
    check_index_mapping(N,Tq);check_index_mapping(N,Tk)
    P=[[[0.0]*N for _ in range(N)] for _ in range(B)]
    O=[[[0.0]*Dv for _ in range(N)] for _ in range(B)]
    for b in range(B):
        for qi in range((N+Tq-1)//Tq):
            for qj in range(Tq):
                i=qi*Tq+qj
                if i>=N:continue
                scores=[0.0]*N
                for ki in range((N+Tk-1)//Tk):
                    for kj in range(Tk):
                        j=ki*Tk+kj
                        if j>=N:continue
                        scores[j]=sum(Q[b][i][d]*K[b][j][d] for d in range(D))/sqrt(D)
                maximum=max(scores)
                weights=[exp(x-maximum) for x in scores]
                denom=sum(weights)
                for j in range(N):P[b][i][j]=weights[j]/denom
                for t in range(Dv):O[b][i][t]=sum(P[b][i][j]*V[b][j][t] for j in range(N))
    return P,O

def reference(Q,K,V):
    B=len(Q);N=len(Q[0]);D=len(Q[0][0]);Dv=len(V[0][0]);
    P=[];O=[]
    for b in range(B):
        pb=[];ob=[]
        for i in range(N):
            s=[sum(Q[b][i][d]*K[b][j][d] for d in range(D))/sqrt(D) for j in range(N)]
            m=max(s);e=[exp(x-m) for x in s];p=[x/sum(e) for x in e]
            pb.append(p);ob.append([sum(p[j]*V[b][j][t] for j in range(N)) for t in range(Dv)])
        P.append(pb);O.append(ob)
    return P,O

def max_error(a,b):
    if isinstance(a,list):return max((max_error(x,y) for x,y in zip(a,b)),default=0.0)
    return abs(a-b)

def run():
    rng=random.Random(20261016)
    report=[]
    for B,N,D,Dv,Tq,Tk in [(2,4,8,8,2,2),(2,5,3,2,2,3),(1,7,4,5,3,4),(2,9,8,3,4,5)]:
        array=lambda last:[[[rng.uniform(-1,1) for _ in range(last)] for _ in range(N)] for _ in range(B)]
        Q=array(D);K=array(D);V=array(Dv)
        semantic=make_ast(B,N,D,Dv)
        lifted=lift_ast(semantic,Tq,Tk)
        P,O=instantiate(lifted,Q,K,V)
        Pr,Or=reference(Q,K,V)
        ep=max_error(P,Pr);eo=max_error(O,Or)
        assert ep<1e-12 and eo<1e-12
        report.append(dict(B=B,N=N,D=D,Dv=Dv,Tq=Tq,Tk=Tk,P_max_abs_error=ep,O_max_abs_error=eo,checked_query_indices=check_index_mapping(N,Tq),checked_key_indices=check_index_mapping(N,Tk)))
    with open('moa_attention_ast_trace_v1.json','w') as f:
        json.dump({'semantic_ast':semantic.dict(),'lifted_ast':lifted.dict(),'tests':report,'scope':'sequential prototype; no GPU/MPI instantiation; no general proof'},f,indent=2)
    for x in report:print('PASS',x)
    print('TRACE moa_attention_ast_trace_v1.json')
if __name__=='__main__':run()
