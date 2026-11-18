"""Lowerings of the published grouped-head and preallocated-cache equations."""
from textwrap import indent
SHAPES={'Q':('H','D'),'K':('G','N','D'),'V':('G','N','E'),'Out':('H','E')}
CACHE_SHAPES={'K':('G','C','D'),'V':('G','C','E'),'key':('G','D'),'value':('G','E')}

def grouped(backend):
    pragma='omp parallel for schedule(static)' if backend=='openmp' else 'acc parallel loop gang'
    return f'''scale: scalar = f32('1.0') / sqrtf(float_cast(D))
with directive("{pragma}"):
    for h in serial(H):
        group: index = h // (H // G)
        row: scratch[MAX_N]
        m: scalar = f32('-1e38')
        for k in serial(N):
            dot: scalar = f32('0.0')
            for j in serial(D):
                dot += Q[h,j] * K[group,k,j]
            row[k] = scale * dot
            if row[k] > m:
                m = row[k]
        z: scalar = f32('0.0')
        for k in serial(N):
            row[k] = expf(row[k] - m)
            z += row[k]
        for d in serial(E):
            total: scalar = f32('0.0')
            for k in serial(N):
                total += (row[k] / z) * V[group,k,d]
            Out[h,d] = total
'''

def append(backend):
    pragma='omp parallel for schedule(static)' if backend=='openmp' else 'acc parallel loop gang present(K,V,key,value)'
    return f'''with directive("{pragma}"):
    for g in serial(G):
        for j in serial(D):
            K[g,N,j] = key[g,j]
        for d in serial(E):
            V[g,N,d] = value[g,d]
return N + 1
'''

def listing3(backend):
    assert backend=='openacc'
    return '''sc: const_scalar = f32('1.0') / sqrtf(float_cast(dk))
s: heap_reverse[n]
e: heap_reverse[n]
m: scalar = f32('-1e38')
with directive("acc data copyin(q[0:dk],K[0:n*dk],V[0:n*dv]) copyout(out[0:dv]) create(s[0:n],e[0:n])"):
    with scope():
        with directive("acc parallel loop gang reduction(max:m)"):
            for l in serial(n):
                dot: scalar = f32('0.0')
                with directive("acc loop vector reduction(+:dot)"):
                    for j in serial(dk):
                        dot += q[j] * K[l,j]
                s[l] = sc * dot
                if s[l] > m:
                    m = s[l]
        Z: scalar = f32('0.0')
        with directive("acc parallel loop gang reduction(+:Z)"):
            for l in serial(n):
                e[l] = expf(s[l] - m)
                Z += e[l]
        inv_Z: scalar = f32('1.0') / Z
        with directive("acc parallel loop gang"):
            for d in serial(dv):
                out[d] = f32('0.0')
        with directive("acc parallel loop gang"):
            for l in serial(n):
                scaled_weight: scalar = e[l] * inv_Z
                with directive("acc loop vector"):
                    for d in serial(dv):
                        with directive("acc atomic"):
                            out[d] += scaled_weight * V[l,d]
free(s)
free(e)
'''
