"""Adversarial checks for the new recipe interpreter, shape and formula checks."""
import ast,json
from pathlib import Path
from recover_dnf import *
from check_recipe_dnfs import validate_shape
import attention,block_parts,full_block
from dnf import hand


def run(out):
    rows=[]
    for recipe in ('forward','backward','fused','decode','rmsnorm','ffn'):
        module=block_parts if recipe in ('rmsnorm','ffn') else attention
        backend='openmp';spec=getattr(module,recipe)(backend)
        for defect,changed in [('reduction_sign',spec.replace(' += ',' -= ',1)),('loop_bound',spec.replace('serial(N)','serial(N + 1)',1)),('feature_bound',spec.replace('serial(D)','serial(D + 1)',1))]:
            if changed==spec:continue
            # Fail closed on unsupported recurrence or compare the entire set of
            # recovered rules to a baseline, using bound-alpha canonicalization.
            def recover(text):
                inputs={'Q','K','V','dO','A','q','X','gamma','GZ','Wg','Wu','Wd','GF'}
                r=ReferenceRecover(module.SHAPES,inputs)
                for name,rank in [('Ggamma',1),('GWg',2),('GWu',2),('GWd',2)]:r.zero(name,rank)
                r.run(text)
                values={}
                exports={'forward':['Out'],'backward':['GV','GQ','GK'],'fused':['Out','GV','GQ','GK'],'decode':['out'],'rmsnorm':['Z','GX','Ggamma'],'ffn':['F','GX','GWg','GWu','GWd']}[recipe]
                for name in exports:
                    validate_shape(r,name,r.array[name],r.shapes)
                    free,value=observed(r.array[name]);values[name]=normalize(value)
                return values
            baseline=recover(spec)
            try:rejected=recover(changed)!=baseline
            except (ValueError,KeyError,AssertionError):rejected=True
            assert rejected,(recipe,defect)
            rows.append({'recipe':recipe,'defect':defect,'rejected':True})
    for malformed in ('if arow[ic] < m:\n    m = arow[ic]','if arow[ic] > m:\n    m = 0.0'):
        r=ReferenceRecover({},('arow',));r.scalar.update({'m':s.S.Zero,'ic':s.S.Zero})
        try:r.run(malformed)
        except ValueError:rows.append({'defect':'incorrect_maximum_recurrence','rejected':True})
        else:raise AssertionError('Bad maximum accepted')
    # Missing additive identity must fail rather than silently assuming zero.
    try:ReferenceRecover({},('X',)).run('for d in serial(D):\n    Ggamma[d] += X[d]')
    except ValueError:rows.append({'defect':'missing_gradient_initialization','rejected':True})
    else:raise AssertionError('Uninitialized gradient accepted')
    r=ReferenceRecover({'Q':('B','N','D'),'K':('B','N','D'),'V':('B','N','D'),'Out':('B','N','D')},('Q','K','V')).run(attention.forward('openmp').replace('serial(B)','serial(B+1)',1))
    try:validate_shape(r,'Out',r.array['Out'],r.shapes)
    except AssertionError:rows.append({'defect':'free_output_domain','rejected':True})
    else:raise AssertionError('Wrong output domain accepted')
    out.mkdir(parents=True,exist_ok=True);(out/'recipe_regressions.json').write_text(json.dumps(rows,indent=2));print('PASS',len(rows),'recipe defects rejected')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();run(a.out)
