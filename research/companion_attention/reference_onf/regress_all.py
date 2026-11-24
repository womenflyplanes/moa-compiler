"""Adversarial checks for tiled recovery, grouped selection and stateful append."""
import json
from pathlib import Path
import sympy as s
from recover_dnf import *
from dnf import hand
from dnf.canonical import normalize
import legacy,paper3,variants


def run(out):
    rows=[]
    for algorithm in ('forward','backward','fused'):
        spec=getattr(legacy,algorithm+'_onf')('serial');tile=s.Symbol('tile_width',integer=True,positive=True)
        def recover(text,bad_extent=False):
            r=ReferenceRecover(legacy.SHAPES,('Q','K','V','A','dO'));r.scalar.update({'T':tile,'NT':Bounds['N']/tile+(1 if bad_extent else 0)})
            r.run(text);values={}
            for output in (('Out',) if algorithm=='forward' else ('GV','GQ','GK') if algorithm=='backward' else ('Out','GV','GQ','GK')):
                f,x=observed_affine(r,output);values[output]=normalize(collapse_tile_sums(x,tile))
            return values
        baseline=recover(spec)
        for defect,text,bad_extent in [('tile_stride',spec.replace('it * T','it * (T + 1)',1),False),('tile_extent',spec,True),('adjoint_or_output_sign',spec.replace(' += ',' -= ',1),False)]:
            try:rejected=recover(text,bad_extent)!=baseline
            except (ValueError,KeyError,AssertionError):rejected=True
            assert rejected,(algorithm,defect);rows.append({'algorithm':algorithm,'defect':defect,'rejected':True})
    for defect,text in [('group_modulo',paper3.grouped('openmp').replace('h // (H // G)','h % G')),('normalizer',paper3.grouped('openmp').replace('row[k] / z','row[k] / (z + 1)'))]:
        r=ReferenceRecover(paper3.SHAPES,('Q','K','V')).run(text);f,x=observed_affine(r,'Out')
        assert normalize(x)!=normalize(hand.decode(f)['O']);rows.append({'algorithm':'grouped','defect':defect,'rejected':True})
    n=s.Symbol('cache_length',integer=True,nonnegative=True)
    for defect,text in [('prefix_overwrite',paper3.append('openmp').replace('K[g,N,j]','K[g,0,j]')),('hidden_prefix_store',paper3.append('openmp').replace('K[g,N,j] = key[g,j]','K[g,0,j] = key[g,j]\n            K[g,N,j] = key[g,j]')),('state_increment',paper3.append('openmp').replace('return N + 1','return N + 2'))]:
        r=ReferenceRecover(paper3.CACHE_SHAPES,('K','V','key','value'));r.scalar['N']=n;r.run(text)
        writes=[(name,indices,mode) for name,indices,mode in r.events if indices is not None]
        accepted=len(writes)==2 and {name for name,_,_ in writes}=={'K','V'} and all(indices[1]==n and mode=='assign' for _,indices,mode in writes) and r.scalar['__return__']==n+1
        assert not accepted;rows.append({'algorithm':'append','defect':defect,'rejected':True})
    bad=paper3.listing3('openacc').replace("inv_Z: scalar = f32('1.0') / Z","inv_Z: scalar = f32('2.0') / Z")
    r=ReferenceRecover({'out':('E',)},('q','K','V')).run(bad);f,x=observed_affine(r,'out')
    assert normalize(x)!=normalize(hand.single_decode(f[0]));rows.append({'algorithm':'paper3_listing3','defect':'inverse_normalizer','rejected':True})
    out.mkdir(parents=True,exist_ok=True);(out/'all_extra_regressions.json').write_text(json.dumps(rows,indent=2));print('PASS',len(rows),'extra mathematical defects rejected')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();run(a.out)
