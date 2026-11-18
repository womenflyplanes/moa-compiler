"""Untimed build/correctness replay of the entire C implementation inventory."""
import ast,ctypes,json,re,subprocess
from pathlib import Path
import numpy as np
from check_all import run,EXTRA,PAPER3,PRIOR
from data import DATA,ACCUMULATORS,directive,corrected_directive
from compare import function,compare
from validate import HELPERS,arrays,length

DIMS=['B','N','D','DFF','n','dk','dv','nthreads','H','G','E','C','T']
REGISTRY={x[0]:x for x in PRIOR+EXTRA+PAPER3}
FLOAT={'p3_decode_cpu','ref_decode_float','paper3_listing3','paper3_grouped_openmp','paper3_grouped_openacc','paper3_append_openmp','paper3_append_openacc'}

def make_translation(body,names,profile,backend,helper_text):
    precision='float' if profile in FLOAT else 'double'
    params=[precision+' *'+name for name in names]+['int '+('HEADS' if name=='H' and 'H' in names else name) for name in DIMS]
    prefix='int NT='+('N/T;' if profile.startswith('original_') else 'B*N;')+'\nlong NN=(long)B*N*N;const double EPS=1e-6;\n'
    if not re.search(r'\b(?:double|float)\s+scale\s*=',body):
        prefix+='double scale=1.0/sqrt((double)'+('dk' if 'decode' in profile or profile=='paper3_listing3' else 'D')+');\n'
    data=directive(profile)
    if data:body=data+'\n{\n'+body+'\n}\n'
    return '#include <stdlib.h>\n#include <math.h>\n#include <float.h>\n#define MAX_N 8192\n#define QKV(b,i,d) ((b)*N*D+(i)*D+(d))\n#define AA(b,i,j) ((b)*N*N+(i)*N+(j))\n'+helper_text+'\n'+('int' if profile.startswith('paper3_append') else 'void')+' kernel('+', '.join(params)+'){\n'+prefix+body+'\n}\n'

def component_count(name,profile,case):
    B,N,D,F,H,G,E,C,T=case
    if profile.startswith('paper3_grouped'):
        return {'Q':H*D,'K':G*N*D,'V':G*N*E,'Out':H*E}[name]
    if profile.startswith('paper3_append'):
        return {'K':G*C*D,'V':G*C*E,'key':G*D,'value':G*E}[name]
    if 'decode' in profile or profile=='paper3_listing3':
        return {'q':D,'K':N*D,'V':N*E,'out':E,'s':N,'e':N}[name]
    return length(name,(B,N,D,F))

def numerical_oracle(profile,initial,case):
    B,N,D,F,H,G,E,C,T=case
    if profile.startswith('paper3_grouped'):
        q=initial['Q'].reshape(H,D);k=initial['K'].reshape(G,N,D);v=initial['V'].reshape(G,N,E);o=np.empty((H,E))
        for h in range(H):
            scores=k[h//(H//G)]@q[h]/np.sqrt(D);p=np.exp(scores-scores.max());p/=p.sum();o[h]=p@v[h//(H//G)]
        return {'Out':o.ravel()}
    if profile.startswith('paper3_append'):
        k=initial['K'].reshape(G,C,D).copy();v=initial['V'].reshape(G,C,E).copy()
        k[:,N,:]=initial['key'].reshape(G,D);v[:,N,:]=initial['value'].reshape(G,E)
        return {'K':k.ravel(),'V':v.ravel()}
    if 'decode' in profile or profile=='paper3_listing3':
        q=initial['q'];k=initial['K'].reshape(N,D);v=initial['V'].reshape(N,E)
        scores=k@q/np.sqrt(D);p=np.exp(scores-scores.max());p/=p.sum();return {'out':p@v}
    return {}

def validate(hand,transcriptions,out):
    rows=run(hand,out,transcriptions);evidence={'profile_comparisons':rows,'helper_checks':[],'mutation_rejections':[],'finite_checks':[],'corrected_profiles':[],
      'scope':'untimed GCC CPU/OpenACC host fallback only; no GPU/device performance or UA run'}
    cases=[(2,4,8,16,4,2,8,5,4),(2,4,8,16,8,2,5,7,2),(1,8,3,5,4,1,4,10,2)]
    for row in rows:
        profile=row['profile'];_,path,_,recipe,backend=REGISTRY[profile]
        source=((transcriptions if profile.startswith('paper3_') else hand)/path).read_text()
        helpers=HELPERS.replace('-DBL_MAX','-1e300') if 'gpu' in path else HELPERS
        generated=(out/(profile+'.compiled.c')).read_text();original=(out/(profile+'.hand.c')).read_text()
        for helper in ('softmax_row','silu','silu_grad'):
            if helper+'(' in generated:
                assert compare(function(source,helper),function(helpers,helper)),(profile,helper)
                evidence['helper_checks'].append({'profile':profile,'helper':helper,'equal':True})
        for label,bad in [('operation',generated.replace(' += ',' -= ',1)),('bound',re.sub(r'< (\w+);',r'< (\1 + 1);',generated,count=1)),('directive',generated.replace('#pragma','#pragma wrong',1))]:
            if bad!=generated:
                assert not compare(original,bad),(profile,label)
                evidence['mutation_rejections'].append({'profile':profile,'defect':label,'rejected':True})
        names=arrays(generated);assert names==arrays(original)
        names=sorted(set(names)|{name for _,entries in DATA.get(profile,[]) for name,_ in entries})
        ctype=ctypes.c_float if profile in FLOAT else ctypes.c_double
        funcs=[]
        for kind,body in [('hand',original),('compiled',generated)]:
            text=make_translation(body,names,profile,backend,helpers);file=out/(profile+'.'+kind+'.standalone.c');file.write_text(text);so=file.with_suffix('.so')
            flags=['-fopenacc','-foffload=disable'] if backend=='openacc' else ['-fopenmp']
            result=subprocess.run(['gcc','-O0','-shared','-fPIC',*flags,str(file),'-lm','-o',str(so)],text=True,capture_output=True)
            (out/(profile+'.'+kind+'.build.log')).write_text(result.stdout+result.stderr)
            if result.returncode:raise RuntimeError(profile+' '+kind+' build failed: '+result.stderr)
            lib=ctypes.CDLL(str(so.resolve()));f=lib.kernel;f.argtypes=[ctypes.POINTER(ctype)]*len(names)+[ctypes.c_int]*len(DIMS);f.restype=ctypes.c_int if profile.startswith('paper3_append') else None
            funcs.append((lib,f))
        profile_cases=cases+([(1,3,4,6,4,4,2,5,1),(1,5,3,5,1,1,4,7,1)] if profile.startswith('paper3_grouped') else [(1,0,4,6,1,1,3,2,1)] if profile.startswith('paper3_append') else [])
        for case in profile_cases:
            B,N,D,F,H,G,E,C,T=case;rng=np.random.default_rng(901)
            initial={name:rng.normal(0,.15,component_count(name,profile,case)).astype('float32' if ctype==ctypes.c_float else 'float64') for name in names}
            for name in names:
                if name.startswith('G') and name not in ('GY','GF','GZ'):initial[name].fill(0)
            if 'A' in initial:initial['A']=np.abs(initial['A'])
            states=[];returns=[]
            for _,f in funcs:
                values={name:array.copy() for name,array in initial.items()}
                returns.append(f(*[values[name].ctypes.data_as(ctypes.POINTER(ctype)) for name in names],B,N,D,F,N,D,E,2,H,G,E,C,T));states.append(values)
            error=max(float(np.max(np.abs(states[0][name]-states[1][name]))) for name in names)
            assert all(np.isfinite(v).all() for state in states for v in state.values()),profile
            assert error<(2e-6 if ctype==ctypes.c_float else 1e-12),(profile,error)
            oracle=numerical_oracle(profile,initial,case)
            oracle_error=max((float(np.max(np.abs(states[1][name]-expected))) for name,expected in oracle.items()),default=0)
            assert oracle_error<(2e-6 if ctype==ctypes.c_float else 1e-12),(profile,oracle_error)
            if profile.startswith('paper3_append'):assert returns==[N+1,N+1]
            evidence['finite_checks'].append({'profile':profile,'shape':case,'max_abs_pair_difference':error,'max_abs_independent_oracle_difference':oracle_error,'return_values':returns})
        if profile in ACCUMULATORS:
            corrected_sources={}
            for kind in ('hand','compiled'):
                text=(out/(profile+'.'+kind+'.standalone.c')).read_text().replace(directive(profile),corrected_directive(profile))
                file=out/(profile+'.'+kind+'.corrected.standalone.c');file.write_text(text)
                subprocess.run(['gcc','-fsyntax-only','-fopenacc','-foffload=disable',str(file)],check=True,capture_output=True)
                corrected_sources[kind]=text
            assert compare(function(corrected_sources['hand'],'kernel'),function(corrected_sources['compiled'],'kernel'))
            evidence['corrected_profiles'].append({'profile':profile,'corrected_hand_compiled_AST_equal':True,'compute_body_unchanged':True,'corrected_data_contract':corrected_directive(profile),'device_validation':'not executed'})
        print('VALIDATED',profile,flush=True)
    (out/'all_validation.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps({'profiles':len(rows),'finite_checks':len(evidence['finite_checks']),'helpers':len(evidence['helper_checks']),'defects_rejected':len(evidence['mutation_rejections']),'corrected_entry_contracts':len(evidence['corrected_profiles'])}),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--hand',type=Path,required=True);p.add_argument('--transcriptions',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();validate(a.hand,a.transcriptions,a.out)
