#!/usr/bin/env python3
"""Attention C generator with typed psi accesses lowered to row-major gamma.

A restricted attention-specific ONF lowering experiment, NOT a general DNF
reduction engine. Symbolic extents are checked at runtime by emitted C.
"""
from dataclasses import dataclass
from typing import Tuple
from moa_structured_codegen_v1 import AttentionIR, Mapping

@dataclass(frozen=True)
class PsiAccess:
    array: str
    shape: Tuple[str, ...]
    index: Tuple[str, ...]

    def __post_init__(self):
        if len(self.shape) != len(self.index) or not self.shape:
            raise ValueError('complete psi index required')
        if any(not x.isidentifier() for x in self.shape + self.index):
            raise ValueError('shape and index symbols must be identifiers')
        if self.array not in ('Q', 'K', 'V', 'P', 'O'):
            raise ValueError('unknown array')

    def offset(self):
        """Generate gamma(index,shape) using row-major Horner recurrence."""
        result = f'(size_t){self.index[0]}'
        for extent, index in zip(self.shape[1:], self.index[1:]):
            result = f'(({result})*(size_t){extent}+(size_t){index})'
        return result

    def c(self):
        return f'{self.array}[{self.offset()}]'


def emit(ir: AttentionIR, mapping: Mapping) -> str:
    i, j = ir.query.name, ir.key.name
    if {i, j} != {'i', 'j'}:
        raise ValueError('v1 supports canonical i and j domains only')
    n = ir.query.extent
    if n != ir.key.extent:
        raise ValueError('mismatched query/key extents')
    q = PsiAccess('Q', ('B','N','D'), ('b','i','d')).c()
    k = PsiAccess('K', ('B','N','D'), ('b','j','d')).c()
    v = PsiAccess('V', ('B','N','Dv'), ('b','j','v')).c()
    p = PsiAccess('P', ('B','N','N'), ('b','i','j')).c()
    o = PsiAccess('O', ('B','N','Dv'), ('b','i','v')).c()
    pragma = '#pragma omp parallel for schedule(static)\n' if mapping.backend == 'openmp' else ''
    return f'''#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
int moa_attention_generated(const double *Q,const double *K,const double *V,
                            double *P,double *O,int B,int N,int D,int Dv) {{
  if (!Q || !K || !V || !P || !O || B<1 || N!={n} || D<1 || Dv<1) return -1;
  if ((size_t)B > SIZE_MAX/(size_t)N/(size_t)N/sizeof(double)) return -2;
  double *workspace=(double*)malloc((size_t)B*N*N*sizeof(double));
  if (!workspace) return -2;
  for (int b=0;b<B;++b) {{
{pragma}    for (int ib=0;ib<{ir.query.blocks};++ib) {{
      for (int il=0;il<{ir.query.tile};++il) {{
        int i=ib*{ir.query.tile}+il;
        if (i>=N) continue;
        double *s=workspace+((size_t)b*N+i)*N;
        for (int jb=0;jb<{ir.key.blocks};++jb) {{
          for (int jl=0;jl<{ir.key.tile};++jl) {{
            int j=jb*{ir.key.tile}+jl;
            if (j>=N) continue;
            double sum=0.0;
            for (int d=0;d<D;++d) sum+={q}*{k};
            s[j]=sum/sqrt((double)D);
          }}
        }}
        double mx=-INFINITY;
        for (int j=0;j<N;++j) if(s[j]>mx) mx=s[j];
        double den=0.0;
        for (int j=0;j<N;++j) {{s[j]=exp(s[j]-mx);den+=s[j];}}
        for (int j=0;j<N;++j) {p}=s[j]/den;
        for (int v=0;v<Dv;++v) {{
          double acc=0.0;
          for (int j=0;j<N;++j) acc+={p}*{v};
          {o}=acc;
        }}
      }}
    }}
  }}
  free(workspace);
  return 0;
}}
'''
