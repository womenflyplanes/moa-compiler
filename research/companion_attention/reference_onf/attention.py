"""Reference-schedule attention recipes built from selected-component formulas."""
from textwrap import indent

SHAPES={
    "Q":("B","N","D"),"K":("B","N","D"),"V":("B","N","D"),
    "dO":("B","N","D"),"Out":("B","N","D"),
    "GV":("B","N","D"),"GQ":("B","N","D"),"GK":("B","N","D"),
    "A":("B","N","N"),"AROW":("B","N","N"),"GS":("B","N","N"),
}


def parallel(backend, body, collapse=2):
    name="omp parallel for" if backend=="openmp" else "acc parallel loop"
    if collapse!=1:
        name+=f" collapse({collapse})"
    return f'with directive("{name}"):\n'+indent(body,"    ")


def private(name,backend):
    return f"{name}: heap[N]\n" if backend=="openmp" else f"{name}: scratch[MAX_N]\n"


def release(name,backend):
    return f"free({name})\n" if backend=="openmp" else ""


def softmax(backend,destination):
    if backend=="openmp":
        return "softmax_row(arow, N)\n"+f"""for ic in serial(N):
    {destination}[b,ir,ic] = arow[ic]
"""
    return f"""sum: scalar = 0.0
for ic in serial(N):
    arow[ic] = exp(arow[ic] - m)
    sum += arow[ic]
for ic in serial(N):
    arow[ic] /= sum
    {destination}[b,ir,ic] = arow[ic]
"""


def forward(backend,destination="A"):
    row=private("arow",backend)
    if backend=="openacc":
        row+="m: scalar = -1e300\n"
    row+="""for ic in serial(N):
    s: scalar = 0.0
    for j in serial(D):
        s += Q[b,ir,j] * K[b,ic,j]
    arow[ic] = s * scale
"""
    if backend=="openacc":
        row+="""    if arow[ic] > m:
        m = arow[ic]
"""
    row+=softmax(backend,destination)
    row+="""for id in serial(D):
    o: scalar = 0.0
    for ic in serial(N):
        o += arow[ic] * V[b,ic,id]
    Out[b,ir,id] = o
"""
    row+=release("arow",backend)
    return parallel(backend,"for b in serial(B):\n    for ir in serial(N):\n"+indent(row,"        "))


def value_adjoint(backend,weights="A"):
    return parallel(backend,f"""for b in serial(B):
    for ic in serial(N):
        for id in serial(D):
            g: scalar = 0.0
            for ir in serial(N):
                g += {weights}[b,ir,ic] * dO[b,ir,id]
            GV[b,ic,id] = g
""")


def score_adjoint(backend,weights="A",query_output=False):
    row=private("ga_row",backend)
    row+="""for ic in serial(N):
    g: scalar = 0.0
    for id in serial(D):
        g += dO[b,ir,id] * V[b,ic,id]
    ga_row[ic] = g
corr: scalar = 0.0
for ic in serial(N):
"""
    row+=f"    corr += {weights}[b,ir,ic] * ga_row[ic]\n"
    if query_output:
        row+="""for j in serial(D):
    GQ[b,ir,j] = 0.0
for ic in serial(N):
"""
        row+=f"""    gs: scalar = scale * {weights}[b,ir,ic] * (ga_row[ic] - corr)
    GS[b,ir,ic] = gs
    for j in serial(D):
        GQ[b,ir,j] += gs * K[b,ic,j]
"""
    else:
        row+=f"""for ic in serial(N):
    GS[b,ir,ic] = scale * {weights}[b,ir,ic] * (ga_row[ic] - corr)
"""
    row+=release("ga_row",backend)
    return parallel(backend,"for b in serial(B):\n    for ir in serial(N):\n"+indent(row,"        "))


def query_adjoint(backend):
    return parallel(backend,"""for b in serial(B):
    for ir in serial(N):
        for j in serial(D):
            GQ[b,ir,j] = 0.0
        for ic in serial(N):
            gs: scalar = GS[b,ir,ic]
            for j in serial(D):
                GQ[b,ir,j] += gs * K[b,ic,j]
""")


def key_adjoint(backend):
    local="sum" if backend=="openmp" else "g"
    return parallel(backend,f"""for b in serial(B):
    for ic in serial(N):
        for j in serial(D):
            {local}: scalar = 0.0
            for ir in serial(N):
                {local} += GS[b,ir,ic] * Q[b,ir,j]
            GK[b,ic,j] = {local}
""")


def backward(backend):
    parts=[value_adjoint(backend)]
    if backend=="openmp":
        parts.append("GS: heapwide[B*N*N]\n")
    parts+=[score_adjoint(backend,query_output=backend=="openacc")]
    if backend=="openmp":
        parts.append(query_adjoint(backend))
    parts.append(key_adjoint(backend))
    if backend=="openmp":
        parts.append("free(GS)\n")
    return "\n".join(parts)


def fused(backend):
    return ("AROW: heapwide[B*N*N]\nGS: heapwide[B*N*N]\n" if backend=="openmp" else "")+"\n".join((forward(backend,"AROW"),value_adjoint(backend,"AROW"),
                       score_adjoint(backend,"AROW"),query_adjoint(backend),key_adjoint(backend)))+("free(AROW)\nfree(GS)\n" if backend=="openmp" else "")


def decode(backend):
    if backend=="openmp":
        pragmas=("omp parallel for num_threads(nthreads) reduction(max:m) schedule(static)",
                 "omp parallel for num_threads(nthreads) reduction(+:Z) schedule(static)",
                 "omp parallel for num_threads(nthreads) schedule(static)")
        begin="m: scalar = f32('-1e38')\n"
        score="s[l] = scale * dot"
        exponent="expf(s[l] - m)"
        zero="f32('0.0')"
        inverse="f32('1.0') / Z"
    else:
        pragmas=("acc parallel loop reduction(max:m)","acc parallel loop reduction(+:Z)",
                 "acc parallel loop")
        begin="m: scalar = -1e300\n"
        score="s[l] = scale * dot"
        exponent="exp(s[l] - m)"
        zero="0.0"
        inverse="1.0 / Z"
    return begin+f"""with directive("{pragmas[0]}"):
    for l in serial(n):
        dot: scalar = {zero}
        for j in serial(dk):
            dot += q[j] * K[l,j]
        {score}
        if s[l] > m:
            m = s[l]
Z: scalar = {zero}
with directive("{pragmas[1]}"):
    for l in serial(n):
        e[l] = {exponent}
        Z += e[l]
inv_Z: scalar = {inverse}
with directive("{pragmas[2]}"):
    for d in serial(dv):
        acc: scalar = {zero}
        for l in serial(n):
            acc += (e[l] * inv_Z) * V[l,d]
        out[d] = acc
"""
