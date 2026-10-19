#!/usr/bin/env python3
"""Typed attention loop IR -> C/OpenMP. Experimental, not general MoA DNF/ONF."""
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Domain:
    name: str
    extent: int
    tile: int
    def __post_init__(self):
        if not self.name.isidentifier() or self.extent <= 0 or self.tile <= 0:
            raise ValueError('invalid domain')
    @property
    def blocks(self):
        return (self.extent + self.tile - 1) // self.tile
    def indices(self):
        return [outer*self.tile+inner for outer in range(self.blocks)
                for inner in range(self.tile) if outer*self.tile+inner < self.extent]

@dataclass(frozen=True)
class AttentionIR:
    query: Domain
    key: Domain
    def __post_init__(self):
        if self.query.name == self.key.name or self.query.extent != self.key.extent:
            raise ValueError('query and key require distinct names and matching extents')

@dataclass(frozen=True)
class Mapping:
    backend: str = 'sequential'
    def __post_init__(self):
        if self.backend not in ('sequential', 'openmp'):
            raise ValueError('unsupported backend')


def emit(ir: AttentionIR, mapping: Mapping) -> str:
    q, k = ir.query, ir.key
    n = q.extent
    parallel = mapping.backend == 'openmp'
    pragma = '#pragma omp parallel for schedule(static)\n' if parallel else ''
    # Workspace is allocated once, outside the parallel region. Each (b,i)
    # owns its disjoint score slice, so no shared writes occur within a query.
    return f'''#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
int moa_attention_generated(const double *Q, const double *K, const double *V,
                            double *P, double *O, int B, int N, int D, int Dv) {{
  if (!Q || !K || !V || !P || !O || B < 1 || N != {n} || D < 1 || Dv < 1) return -1;
  if ((size_t)B > SIZE_MAX / (size_t)N / (size_t)N / sizeof(double)) return -2;
  double *workspace = (double *)malloc((size_t)B*N*N*sizeof(double));
  if (!workspace) return -2;
  for (int b = 0; b < B; ++b) {{
{pragma}    for (int {q.name}b = 0; {q.name}b < {q.blocks}; ++{q.name}b) {{
      for (int {q.name}l = 0; {q.name}l < {q.tile}; ++{q.name}l) {{
        int {q.name} = {q.name}b*{q.tile} + {q.name}l;
        if ({q.name} >= N) continue;
        double *s = workspace + ((size_t)b*N + {q.name})*N;
        for (int {k.name}b = 0; {k.name}b < {k.blocks}; ++{k.name}b) {{
          for (int {k.name}l = 0; {k.name}l < {k.tile}; ++{k.name}l) {{
            int {k.name} = {k.name}b*{k.tile} + {k.name}l;
            if ({k.name} >= N) continue;
            double sum = 0.0;
            for (int d = 0; d < D; ++d)
              sum += Q[((size_t)b*N+{q.name})*D+d] * K[((size_t)b*N+{k.name})*D+d];
            s[{k.name}] = sum / sqrt((double)D);
          }}
        }}
        double mx = -INFINITY;
        for (int j=0; j<N; ++j) if (s[j] > mx) mx=s[j];
        double den = 0.0;
        for (int j=0; j<N; ++j) {{ s[j]=exp(s[j]-mx); den+=s[j]; }}
        for (int j=0; j<N; ++j) P[((size_t)b*N+{q.name})*N+j]=s[j]/den;
        for (int v=0; v<Dv; ++v) {{
          double acc=0.0;
          for (int j=0; j<N; ++j) acc+=P[((size_t)b*N+{q.name})*N+j]*V[((size_t)b*N+j)*Dv+v];
          O[((size_t)b*N+{q.name})*Dv+v]=acc;
        }}
      }}
    }}
  }}
  free(workspace);
  return 0;
}}
'''
