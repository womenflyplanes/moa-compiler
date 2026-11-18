"""Independent exact-real DNF recovery for every pinned C implementation region."""
import ast,json,hashlib
from pathlib import Path
import sympy as s
import check_recipe_dnfs as core
from check_all import EXTRA,PAPER3
from recover_dnf import ReferenceRecover,observed,observed_affine,collapse_tile_sums,Bounds,aliases,Psi,select,free_symbols
from dnf import hand
from dnf.canonical import normalize
import variants,legacy,serial_refs,paper3,reference_equations as equations

# Each record binds the recovered formula to the actual emitted recipe digest.
def comparison(profile,out,r,expected,tile=None):
    rule=r.array[out]
    core.validate_shape(r,out,rule,r.shapes)
    free,actual=observed_affine(r,out)
    actual=aliases(actual,{'dO':'U','GF':'U'} if not profile.startswith('p4_chain') else {})
    if tile is not None:actual=collapse_tile_sums(actual,tile)
    value=expected(free)
    core.check(profile+'.'+out,actual,value,free,profile)

def standalone(profile,recipe,backend):
    module=legacy if profile.startswith('original_') else serial_refs if profile.startswith('ref_') else variants
    if 'decode' in profile:
        r=ReferenceRecover({'K':('N','D'),'V':('N','E'),'out':('E',)},('q','K','V'))
        r.run(getattr(module,recipe)(backend));comparison(profile,'out',r,lambda f:hand.single_decode(f[0]));return
    if recipe in ('rms_verify','ffn_verify'):
        r=ReferenceRecover(module.SHAPES,('X','gamma','GZ','Wg','Wu','Wd','GF'))
        for name,rank in [('Ggamma',1),('GWg',2),('GWu',2),('GWd',2)]:r.zero(name,rank)
        r.run(getattr(module,recipe)(backend))
        for out in ('Z','GX','Ggamma') if recipe=='rms_verify' else ('F','GX','GWg','GWu','GWd'):
            core.validate_shape(r,out,r.array[out],r.shapes)
            free,actual=observed_affine(r,out)
            if recipe=='rms_verify':
                actual=aliases(actual,{'gamma':'W','GZ':'U'});expected=hand.norm(free if len(free)==3 else (*free_symbols(2),free[0]))['GW' if out=='Ggamma' else out]
            else:actual=aliases(actual,{'GF':'U'});expected=hand.ffn(free,weight=out.startswith('GW'))['Y' if out=='F' else out]
            core.check(profile+'.'+out,actual,expected,free,profile)
        return
    r=ReferenceRecover(module.SHAPES,('Q','K','V','A','dO'))
    if 'atomic' in profile or ('backward' in recipe and profile.startswith('ref_')):r.zero('GK',3)
    tile=None
    if profile.startswith('original_'):
        tile=s.Symbol('tile_width',integer=True,positive=True);r.scalar.update({'T':tile,'NT':Bounds['N']/tile})
    r.run(getattr(module,recipe)(backend))
    is_backward='backward' in recipe
    outputs=['GV','GQ','GK'] if is_backward else ['Out','GV','GQ','GK'] if 'fused' in recipe else ['A'] if recipe=='forward_a' else ['Out','A']
    for out in outputs:
        def expected(f):
            if out=='A':
                b,i,k=f;score=lambda key:hand.sum_at(Bounds['D'],lambda j:select('Q',b,i,j)*select('K',b,key,j))/s.sqrt(Bounds['D'])
                return hand.probability(score,k)
            return hand.training('backward' if is_backward else 'fused',f)['O' if out=='Out' else out].xreplace({Bounds['E']:Bounds['D']})
        comparison(profile,out,r,expected,tile)

def composed(profile,recipe,backend):
    ischain=recipe=='chain_bench';serial=recipe=='block_serial'
    spec=getattr(variants,recipe)(backend)
    inputs=set(('X','gam','GF','Wg','Wu','Wd') if ischain else ('X','gamma1','gamma2','Wq','Wk','Wv','Wg','Wu','Wd','GY'))
    available=set(inputs);producers={x:'input:'+x for x in inputs};nodes=[]
    phases=[n for n in ast.parse(spec).body if isinstance(n,(ast.With,ast.For))]
    zeroes={3:{'GWg':2,'GWu':2,'GWd':2,'Ggamma':1}} if ischain else ({7:{'GR1':1},8:{'GWg':2,'GWu':2,'GWd':2},9:{'Ggamma2':1},10:{'GX':1},11:{'GQ':3,'GK':3},12:{'GWq':2,'GWk':2,'GWv':2},13:{'Ggamma1':1}} if serial else {5:{'GWg':2,'GWu':2,'GWd':2},6:{'Ggamma2':1},8:{'GQ':3,'GK':3},9:{'GWq':2,'GWk':2,'GWv':2},10:{'Ggamma1':1}})
    for phase,node in enumerate(phases):
        r=ReferenceRecover(variants.SHAPES,available)
        for name,rank in zeroes.get(phase,{}).items():r.zero(name,rank)
        r.statements([node]);locals={n.target.id for n in ast.walk(node) if isinstance(n,ast.AnnAssign) and isinstance(n.annotation,ast.Subscript)}
        exports={name for name,indices,_ in r.events if indices is not None}-locals
        for out in sorted(exports):
            core.validate_shape(r,out,r.array[out],r.shapes)
            free,actual=observed_affine(r,out)
            expected=(equations.chain if ischain else equations.serial_block if serial else equations.block)(phase,out,free)
            refs={str(n.args[0]) for n in actual.atoms(s.Function) if n.func==Psi}
            deps={name:producers[name] for name in sorted(refs)}
            core.check(f'{profile}.phase{phase}.{out}',actual,expected,free,profile,deps)
            nodes.append({'id':f'phase{phase}.{out}','array':out,'dependencies':deps,'equal':True})
        available.update(exports);producers.update({name:f'phase{phase}.{name}' for name in exports})
    roots=('Z','F','GX_norm','Ggamma','GX_ffn','GWg','GWu','GWd') if ischain else ('Y','GX','Ggamma1','Ggamma2','GWq','GWk','GWv','GWg','GWu','GWd')
    core.GRAPHS[profile]={'inputs':sorted(inputs),'nodes':nodes,'roots':{name:producers[name] for name in roots}}

def paper3_checks(profile,recipe,backend):
    if recipe=='listing3':
        r=ReferenceRecover({'K':('N','D'),'V':('N','E'),'out':('E',)},('q','K','V')).run(paper3.listing3(backend))
        comparison(profile,'out',r,lambda f:hand.single_decode(f[0]));return
    if recipe=='grouped':
        r=ReferenceRecover(paper3.SHAPES,('Q','K','V')).run(paper3.grouped(backend));free,actual=observed_affine(r,'Out')
        core.check(profile+'.Out',actual,hand.decode(free)['O'],free,profile)
        for name,g in [('MQA',1),('MHA',Bounds['H'])]:
            left=actual.xreplace({Bounds['G']:s.Integer(g) if isinstance(g,int) else g});right=hand.decode(free)['O'].xreplace({Bounds['G']:s.Integer(g) if isinstance(g,int) else g})
            core.check(profile+'.'+name,left,right,free,profile)
        return
    r=ReferenceRecover(paper3.CACHE_SHAPES,('K','V','key','value'))
    cache_n=s.Symbol('cache_length',integer=True,nonnegative=True);r.scalar['N']=cache_n
    r.run(paper3.append(backend))
    writes=[(name,indices,mode) for name,indices,mode in r.events if indices is not None]
    assert len(writes)==2 and {name for name,_,_ in writes}=={'K','V'}
    assert all(indices[1]==cache_n and mode=='assign' for _,indices,mode in writes)
    for out,new,width in [('K','key','D'),('V','value','E')]:
        rule=r.array[out];g,j=free_symbols(2)
        assert rule.indices[1]==cache_n
        assert r.domains[rule.indices[0]]==Bounds['G'] and r.domains[rule.indices[2]]==Bounds[width]
        actual=rule.value.xreplace({rule.indices[0]:g,rule.indices[2]:j})
        core.check(profile+'.'+out+'.suffix',actual,select(new,g,j),(g,j),profile)
        # The only stores have fixed cache index N; every i<N is disjoint.
        free=free_symbols(3);prefix=select(out+'_old',*free)
        core.check(profile+'.'+out+'.prefix',prefix,prefix,free,profile)
    assert r.scalar['__return__']==cache_n+1
    core.check(profile+'.length',r.scalar['__return__'],cache_n+1,(),profile)

def run(out):
    core.ROWS.clear();core.GRAPHS.clear()
    core.standalone();core.composed('block','openmp');core.composed('block','openacc');core.composed('chain','openacc')
    # Add original tile and source variants plus all numerical reference routines.
    for profile,path,selector,recipe,backend in EXTRA:
        if recipe in ('block_serial','block_verify','chain_bench'):composed(profile,recipe,backend)
        else:standalone(profile,recipe,backend)
    for profile,_,_,recipe,backend in PAPER3:paper3_checks(profile,recipe,backend)
    out.mkdir(parents=True,exist_ok=True)
    value={'arithmetic':'exact real','results':core.ROWS,'composition_graphs':core.GRAPHS,
      'tile_premises':['tile_width>0','tile_width divides N','NT=N/tile_width','ic=it*tile_width+tt is a bijection onto 0<=ic<N'],
      'entry_premises':['positive integer extents','G divides H','0<=N<C for cache append (including initial empty cache)','zero initial additive gradient arrays where not assigned by the routine','training key/value feature extents equal D','finite softmax shifts','legal contiguous views','representable integer addresses'],
      'scope':'all pinned C computational regions plus explicitly transcribed/derived Paper III listings; no claim of verified C/Fortran compiler toolchains'}
    (out/'all_recipe_dnfs.json').write_text(json.dumps(value,indent=2));print('PASS',len(core.ROWS),'all-implementation DNF component comparisons')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();run(a.out)
