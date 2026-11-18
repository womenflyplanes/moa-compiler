"""Selected complete-block operational schedule, explicit phase composition."""
from textwrap import indent
from attention import parallel,private,release
from block_parts import SHAPES as BASE
SHAPES=dict(BASE)
for name in ('X','GY','Z1','R1','Z2','Y','GR1','GZ1','GZ2','Q','K','V','GQ','GK','GV'):
    SHAPES[name]=('B','N','D')
for name in ('Wq','Wk','Wv','GWq','GWk','GWv'):SHAPES[name]=('D','D')
SHAPES['A']=('B','N','N')
SHAPES['Vg']=SHAPES['H']=('NT','DFF')

def token(backend,body,reductions=''):
    if backend=='openmp' and reductions:
        return 'with directive("omp parallel for reduction(+:'+reductions+')"):\n'+indent('for t in serial(NT):\n'+indent(body,'    '),'    ')
    return parallel(backend,'for t in serial(NT):\n'+indent(body,'    '),1)

def norm(backend,x,r,z,gamma):
    return token(backend,f'''sumsq: scalar = 0.0
for d in serial(D):
    x: scalar = {x}[t,d]
    sumsq += x * x
{r}[t] = sqrt(sumsq / D + EPS)
for d in serial(D):
    {z}[t,d] = {gamma}[d] * ({x}[t,d] / {r}[t])
''')

def norm_back(backend,x,r,gz,gamma,gx,gg):
    body=f'''correction: scalar = 0.0
for d in serial(D):
    x_hat_d: scalar = {x}[t,d] / {r}[t]
    correction += {gz}[t,d] * {gamma}[d] * x_hat_d
for e in serial(D):
    x_hat_e: scalar = {x}[t,e] / {r}[t]
    {gx}[t,e] += ({gz}[t,e] * {gamma}[e] - (x_hat_e / D) * correction) / {r}[t]
'''
    if backend=='openacc':body+='    with directive("acc atomic update"):\n        '+f'{gg}[e] += {gz}[t,e] * x_hat_e\n'
    else:body+=f'    {gg}[e] += {gz}[t,e] * x_hat_e\n'
    return token(backend,body,f'{gg}[0:D]')

def atomic(backend,text):
    return 'with directive("acc atomic update"):\n'+indent(text,'    ') if backend=='openacc' else text

def block(backend):
    parts=[norm(backend,'X','r1','Z1','gamma1')]
    parts.append(token(backend,'''for d2 in serial(D):
    q: scalar = 0
    k: scalar = 0
    v: scalar = 0
    for d in serial(D):
        q += Z1[t,d] * Wq[d,d2]
        k += Z1[t,d] * Wk[d,d2]
        v += Z1[t,d] * Wv[d,d2]
    Q[t,d2] = q
    K[t,d2] = k
    V[t,d2] = v
'''))
    row=private('arow',backend)+'''m: scalar = -1e300
for ic in serial(N):
    dot: scalar = 0.0
    for j in serial(D):
        dot += Q[b,ir,j] * K[b,ic,j]
    arow[ic] = scale * dot
    if arow[ic] > m:
        m = arow[ic]
s: scalar = 0.0
for ic in serial(N):
    arow[ic] = exp(arow[ic] - m)
    s += arow[ic]
for ic in serial(N):
    arow[ic] /= s
    A[b,ir,ic] = arow[ic]
for d in serial(D):
    o: scalar = 0.0
    for ic in serial(N):
        o += arow[ic] * V[b,ic,d]
    R1[b,ir,d] = X[b,ir,d] + o
'''+release('arow',backend)
    parts.append(parallel(backend,'for b in serial(B):\n    for ir in serial(N):\n'+indent(row,'        '),2 if backend=='openmp' else 1))
    parts.append(norm(backend,'R1','r2','Z2','gamma2'))
    parts.append(token(backend,'''for f in serial(DFF):
    u: scalar = 0
    v: scalar = 0
    for d in serial(D):
        u += Z2[t,d] * Wg[d,f]
        v += Z2[t,d] * Wu[d,f]
    U[t,f] = u
    Vg[t,f] = v
    H[t,f] = silu(u) * v
for d in serial(D):
    fo: scalar = 0.0
    for f in serial(DFF):
        fo += H[t,f] * Wd[f,d]
    Y[t,d] = R1[t,d] + fo
'''))
    body=('GUrow: heap[DFF]\nGVrow: heap[DFF]\n' if backend=='openmp' else 'GUrow: scratch[512]\nGVrow: scratch[512]\n')+'''for f in serial(DFF):
    gh: scalar = 0.0
    for d in serial(D):
        gh += GY[t,d] * Wd[f,d]
    GVrow[f] = gh * silu(U[t,f])
    GUrow[f] = gh * Vg[t,f] * silu_grad(U[t,f])
for f in serial(DFF):
    for d in serial(D):
'''+indent(atomic(backend,'GWd[f,d] += H[t,f] * GY[t,d]\n'),'        ')+'''for d in serial(D):
    for f in serial(DFF):
'''+indent(atomic(backend,'GWg[d,f] += Z2[t,d] * GUrow[f]\n')+atomic(backend,'GWu[d,f] += Z2[t,d] * GVrow[f]\n'),'        ')+'''gz2: scalar
for d in serial(D):
    gz2 = 0.0
    for f in serial(DFF):
        gz2 += GUrow[f] * Wg[d,f] + GVrow[f] * Wu[d,f]
    GZ2[t,d] = gz2
for d in serial(D):
    GR1[t,d] = GY[t,d]
'''
    if backend=='openmp':body+='free(GUrow)\nfree(GVrow)\n'
    parts.append(token(backend,body,'GWg[0:D*DFF], GWu[0:D*DFF], GWd[0:DFF*D]'))
    parts.append(norm_back(backend,'R1','r2','GZ2','gamma2','GR1','Ggamma2'))
    parts.append(parallel(backend,'''for i in serial(NT*D):
    GX[i] = GR1[i]
''',1))
    body='''for ic in serial(N):
    for d in serial(D):
        gv: scalar = 0.0
        for ir in serial(N):
            gv += A[b,ir,ic] * GR1[b,ir,d]
        GV[b,ic,d] = gv
for ir in serial(N):
'''+indent(private('ga_row',backend)+'''for ic in serial(N):
    g: scalar = 0.0
    for d in serial(D):
        g += GR1[b,ir,d] * V[b,ic,d]
    ga_row[ic] = g
corr: scalar = 0.0
for ic in serial(N):
    corr += A[b,ir,ic] * ga_row[ic]
for ic in serial(N):
    gs: scalar = scale * A[b,ir,ic] * (ga_row[ic] - corr)
    for j in serial(D):
        GQ[b,ir,j] += gs * K[b,ic,j]
        GK[b,ic,j] += gs * Q[b,ir,j]
'''+release('ga_row',backend),'    ')
    parts.append(parallel(backend,'for b in serial(B):\n'+indent(body,'    '),1))
    body='for d in serial(D):\n    for d2 in serial(D):\n'
    body+=indent(''.join(atomic(backend,f'GW{key}[d,d2] += Z1[t,d] * G{key.upper()}[t,d2]\n') for key in ('q','k','v')),'        ')
    body+='''for d in serial(D):
    g: scalar = 0.0
    for d2 in serial(D):
        g += GQ[t,d2] * Wq[d,d2] + GK[t,d2] * Wk[d,d2] + GV[t,d2] * Wv[d,d2]
    GZ1[t,d] = g
'''
    parts.append(token(backend,body,'GWq[0:D*D], GWk[0:D*D], GWv[0:D*D]'))
    parts.append(norm_back(backend,'X','r1','GZ1','gamma1','GX','Ggamma1'))
    return '\n'.join(parts)
