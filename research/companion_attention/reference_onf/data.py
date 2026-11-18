"""Explicit host/device storage profiles. Sizes are C interface expressions."""
ATTN_IN=[('Q','NT*D'),('K','NT*D'),('V','NT*D')]
DATA={
 'p4_norm_ffn_gpu': [('copyin',[('X','NT*D'),('gam','D'),('GZ','NT*D'),('Wg','D*DFF'),('Wu','D*DFF'),('Wd','DFF*D'),('GF','NT*D')]),('copyout',[('Z','NT*D'),('F','NT*D'),('GX_norm','NT*D'),('Ggamma','D'),('GX_ffn','NT*D'),('GWg','D*DFF'),('GWu','D*DFF'),('GWd','DFF*D')]),('create',[('U','NT*DFF'),('r','NT')])],
 'p1_forward_gpu': [('copyin',ATTN_IN),('copyout',[('A','(long)NT*N'),('Out','NT*D')])],
 'p2_backward_gpu': [('copyin',ATTN_IN+[('dO','NT*D'),('A','(long)NT*N')]),('copyout',[(x,'NT*D') for x in ('GV','GQ','GK')]),('create',[('GS','(long)NT*N')])],
 'p2_fused_gpu': [('copyin',ATTN_IN+[('dO','NT*D')]),('copyout',[(x,'NT*D') for x in ('Out','GV','GQ','GK')]),('create',[(x,'(long)NT*N') for x in ('AROW','GS')])],
 'p3_decode_gpu': [('copyin',[('q','dk'),('K','(long)n*dk'),('V','(long)n*dv')]),('create',[('s','n'),('e','n')]),('copyout',[('out','dv')])],
 'p4_block_gpu':[
  ('copyin',[('X','NT*D'),('gamma1','D'),('gamma2','D'),('Wq','D*D'),('Wk','D*D'),('Wv','D*D'),('Wg','D*DFF'),('Wu','D*DFF'),('Wd','DFF*D'),('GY','NT*D')]),
  ('create',[('Z1','NT*D'),('r1','NT'),('Q','NT*D'),('K','NT*D'),('V','NT*D'),('A','NN'),('R1','NT*D'),('Z2','NT*D'),('r2','NT'),('U','NT*DFF'),('Vg','NT*DFF'),('H','NT*DFF'),('GR1','NT*D'),('GZ1','NT*D'),('GZ2','NT*D'),('GQ','NT*D'),('GK','NT*D'),('GV','NT*D')]),
  ('copyout',[('Y','NT*D'),('GX','NT*D'),('Ggamma1','D'),('Ggamma2','D'),('GWq','D*D'),('GWk','D*D'),('GWv','D*D'),('GWg','D*DFF'),('GWu','D*DFF'),('GWd','DFF*D')])]
}
def directive(profile):
    if profile not in DATA:return None
    return '#pragma acc data '+ ' '.join(clause+'('+', '.join(name+'[0:'+size+']' for name,size in arrays)+')' for clause,arrays in DATA[profile])

# A separate, explicit correction profile. The archival profile above is immutable.
ACCUMULATORS={
 'p4_block_gpu':{'GQ','GK','Ggamma1','Ggamma2','GWq','GWk','GWv','GWg','GWu','GWd'},
 'p4_norm_ffn_gpu':{'Ggamma','GWg','GWu','GWd'},
}
def corrected_directive(profile):
    if profile not in ACCUMULATORS:return directive(profile)
    zero=ACCUMULATORS[profile];copy=[];clauses=[]
    for clause,entries in DATA[profile]:
        retained=[]
        for name,size in entries:
            (copy if name in zero else retained).append((name,size))
        if retained:clauses.append((clause,retained))
    clauses.append(('copy',copy))
    return '#pragma acc data '+' '.join(c+'('+', '.join(n+'[0:'+s+']' for n,s in entries)+')' for c,entries in clauses)
