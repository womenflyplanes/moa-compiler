"""Original serial DNF and tile-lifted ONF schedule recipes.

No hand C is read here. Tile axes use ic=it*T+tt, N=NT*T.
"""
from textwrap import indent
from attention import SHAPES

def zeros():return '''for ir in serial(N):
    for id in serial(D):
        Out[b,ir,id] = 0.0
        GV[b,ir,id] = 0.0
        GQ[b,ir,id] = 0.0
        GK[b,ir,id] = 0.0
'''
def score(tiled):
    if tiled:return '''m = -DBL_MAX
for it in serial(NT):
    ic0: index = it * T
    for tt in serial(T):
        ic: index = ic0 + tt
        score_tile[tt] = 0.0
        for j in serial(D):
            score_tile[tt] += Q[b,ir,j] * K[b,ic,j]
        score_tile[tt] *= scale
        arow[ic] = score_tile[tt]
    for tt in serial(T):
        if score_tile[tt] > m:
            m = score_tile[tt]
s = 0.0
for ic in serial(N):
    arow[ic] = exp(arow[ic] - m)
    s += arow[ic]
'''
    return '''for ic in serial(N):
    arow[ic] = 0.0
    for j in serial(D):
        arow[ic] += Q[b,ir,j] * K[b,ic,j]
    arow[ic] *= scale
softmax_row(arow, N)
'''
def corr(weights,tiled):
    if tiled:return f'''corr = 0.0
for it in serial(NT):
    ic0: index = it * T
    for tt in serial(T):
        corr += {weights}[ic0+tt] * ga_row[ic0+tt]
'''
    return f'''corr = 0.0
for ic in serial(N):
    corr += {weights}[ic] * ga_row[ic]
'''
def scatter(weight,tiled):
    if tiled:
        expr=f'{weight}[ic0+tt]' if weight=='arow' else 'A[b,ir,ic0+tt]'
        return f'''for it in serial(NT):
    ic0: index = it * T
    for tt in serial(T):
        gs_tile[tt] = scale * {expr} * (ga_row[ic0+tt] - corr)
    for tt in serial(T):
        ic: index = ic0 + tt
        gs = gs_tile[tt]
        for j in serial(D):
            GQ[b,ir,j] += gs * K[b,ic,j]
            GK[b,ic,j] += gs * Q[b,ir,j]
'''
    expr='arow[ic]' if weight=='arow' else 'A[b,ir,ic]'
    return f'''for ic in serial(N):
    gs = scale * {expr} * (ga_row[ic] - corr)
    for j in serial(D):
        GQ[b,ir,j] += gs * K[b,ic,j]
        GK[b,ic,j] += gs * Q[b,ir,j]
'''
def forward(tiled):
    head='arow: scratch[N]\n'+('score_tile: scratch[T]\nm: scalar\ns: scalar\n' if tiled else '')
    row=score(tiled)
    if tiled:row+='''for ic in serial(N):
    arow[ic] /= s
    A[b,ir,ic] = arow[ic]
for id in serial(D):
    Out[b,ir,id] = 0.0
for it in serial(NT):
    ic0: index = it * T
    for tt in serial(T):
        ic: index = ic0 + tt
        for id in serial(D):
            Out[b,ir,id] += arow[ic] * V[b,ic,id]
'''
    else:row+='''for ic in serial(N):
    A[b,ir,ic] = arow[ic]
for id in serial(D):
    Out[b,ir,id] = 0.0
    for ic in serial(N):
        Out[b,ir,id] += arow[ic] * V[b,ic,id]
'''
    return head+'for b in serial(B):\n    for ir in serial(N):\n'+indent(row,'        ')

def backward(tiled):
    head='ga_row: scratch[N]\n'+('ga_tile: scratch[T]\ngs_tile: scratch[T]\n' if tiled else '')+'corr: scalar\ngs: scalar\n'
    body='''for ic in serial(N):
    for id in serial(D):
        GV[b,ic,id] = 0.0
'''
    if tiled:body+='''for it in serial(NT):
    ic0: index = it * T
    for ir in serial(N):
        for tt in serial(T):
            ic: index = ic0 + tt
            for id in serial(D):
                GV[b,ic,id] += A[b,ir,ic] * dO[b,ir,id]
'''
    else:body='''for ic in serial(N):
    for id in serial(D):
        GV[b,ic,id] = 0.0
        for ir in serial(N):
            GV[b,ic,id] += A[b,ir,ic] * dO[b,ir,id]
'''
    body+='''for ir in serial(N):
    for j in serial(D):
        GQ[b,ir,j] = 0.0
        GK[b,ir,j] = 0.0
'''
    if tiled:row='''for it in serial(NT):
    ic0: index = it * T
    for tt in serial(T):
        ga_tile[tt] = 0.0
    for id in serial(D):
        for tt in serial(T):
            ga_tile[tt] += dO[b,ir,id] * V[b,ic0+tt,id]
    for tt in serial(T):
        ga_row[ic0+tt] = ga_tile[tt]
corr = 0.0
for it in serial(NT):
    ic0: index = it * T
    for tt in serial(T):
        corr += A[b,ir,ic0+tt] * ga_row[ic0+tt]
'''
    else:row='''for ic in serial(N):
    ga_row[ic] = 0.0
    for id in serial(D):
        ga_row[ic] += dO[b,ir,id] * V[b,ic,id]
corr = 0.0
for ic in serial(N):
    corr += A[b,ir,ic] * ga_row[ic]
'''
    row+=scatter('A',tiled);body+='for ir in serial(N):\n'+indent(row,'    ')
    return head+'for b in serial(B):\n'+indent(body,'    ')

def fused(tiled):
    head='arow: scratch[N]\nga_row: scratch[N]\n'+('score_tile: scratch[T]\nga_tile: scratch[T]\ngs_tile: scratch[T]\n' if tiled else '')+'corr: scalar\ngs: scalar\nvv: scalar\noo: scalar\n'+('m: scalar\ns: scalar\n' if tiled else '')
    row=score(tiled)
    if tiled:row+='''for ic in serial(N):
    arow[ic] /= s
for it in serial(NT):
    ic0: index = it * T
    for tt in serial(T):
        ga_tile[tt] = 0.0
    for id in serial(D):
        oo = dO[b,ir,id]
        for tt in serial(T):
            ic: index = ic0 + tt
            vv = V[b,ic,id]
            ga_tile[tt] += oo * vv
            Out[b,ir,id] += arow[ic] * vv
            GV[b,ic,id] += arow[ic] * oo
    for tt in serial(T):
        ga_row[ic0+tt] = ga_tile[tt]
'''
    else:row+='''for ic in serial(N):
    ga_row[ic] = 0.0
    for id in serial(D):
        vv = V[b,ic,id]
        oo = dO[b,ir,id]
        ga_row[ic] += oo * vv
        Out[b,ir,id] += arow[ic] * vv
        GV[b,ic,id] += arow[ic] * oo
'''
    row+=corr('arow',tiled)+scatter('arow',tiled)
    return head+'for b in serial(B):\n'+indent(zeros()+'for ir in serial(N):\n'+indent(row,'    '),'    ')

for _name in ('forward','backward','fused'):
    def make(name,tiled):
        def recipe(backend):
            assert backend=='serial'
            return globals()[name](tiled)
        return recipe
    globals()[_name+'_dnf']=make(_name,False)
    globals()[_name+'_onf']=make(_name,True)
