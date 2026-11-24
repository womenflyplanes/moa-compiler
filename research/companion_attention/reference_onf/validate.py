"""Untimed correctness and adversarial checks, never a performance benchmark."""
import ast,ctypes,hashlib,json,re,subprocess
from pathlib import Path
import numpy as np
from pycparser import c_ast
from compare import parse,compare,function
from data import directive, DATA, corrected_directive, ACCUMULATORS
from check import PROFILES,run

HELPERS='''
static void softmax_row(double *row,int n) {
 double m=-DBL_MAX;
 for(int i=0;i<n;i++) if(row[i]>m) m=row[i];
 double s=0.0;
 for(int i=0;i<n;i++){row[i]=exp(row[i]-m);s+=row[i];}
 for(int i=0;i<n;i++) row[i]/=s;
}
static double silu(double u){return u/(1.0+exp(-u));}
static double silu_grad(double u){double s=1.0/(1.0+exp(-u));return s*(1.0+u*(1.0-s));}
'''
DIMS=['B','N','D','DFF','n','dk','dv','nthreads']

def arrays(body):
    tree=parse(body); selected=set();declared=set()
    class Visitor(c_ast.NodeVisitor):
        def visit_ArrayRef(self,node):selected.add(node.name.name);self.generic_visit(node)
        def visit_Decl(self,node):declared.add(node.name);self.generic_visit(node)
    Visitor().visit(tree)
    return sorted(selected-declared)

def translation(body,names,precision,profile,hand=False):
    data=directive(profile)
    if data:body=data+'\n{\n'+body+'\n}\n'
    params=[precision+' *'+x for x in names]+['int '+x for x in DIMS]
    pre='int NT=B*N; long NN=(long)B*N*N; const double EPS=1e-6;\n'
    if profile!='p3_decode_cpu':pre+='double scale=1.0/sqrt((double)'+('dk' if 'decode' in profile else 'D')+');\n'
    return '#include <math.h>\n#include <stdlib.h>\n#include <float.h>\n#define MAX_N 8192\n#define QKV(b,i,d) ((b)*N*D+(i)*D+(d))\n#define AA(b,i,j) ((b)*N*N+(i)*N+(j))\n'+HELPERS+'\nvoid kernel('+', '.join(params)+'){\n'+pre+body+'\n}\n'

def length(name,shape):
    B,N,D,F=shape;T=B*N
    if name in ('q',):return D
    if name in ('gamma','gam','gamma1','gamma2','Ggamma','Ggamma1','Ggamma2'):return D
    if name in ('r','r1','r2'):return T
    if name in ('Wg','Wu','GWg','GWu'):return D*F
    if name in ('Wd','GWd'):return F*D
    if name in ('Wq','Wk','Wv','GWq','GWk','GWv'):return D*D
    if name in ('U','H','Vg'):return T*F
    if name in ('A','AROW','GS'):return B*N*N
    if name in ('s','e'):return N
    return T*D

def validate(hand_root,out):
    rows=run(hand_root,out)
    evidence={'profiles':rows,'helpers':[],'mutations':[],'finite_checks':[],
              'scope':'untimed local CPU/OpenACC host fallback only; no device validation'}
    for name,path,fn,recipe,backend in PROFILES:
        source=(hand_root/path).read_text()
        for helper in ('softmax_row','silu','silu_grad'):
            if helper+'(' in (out/(name+'.compiled.c')).read_text():
                # Whitespace before parentheses is allowed in original definitions.
                good=compare(function(source,helper),function(HELPERS,helper))
                evidence['helpers'].append({'profile':name,'helper':helper,'equal':good})
                assert good,(name,helper)
        hand=(out/(name+'.hand.c')).read_text();generated=(out/(name+'.compiled.c')).read_text()
        for label,bad in [('bound',generated.replace('< N;','< (N + 1);',1)),
                          ('directive',generated.replace('#pragma','#pragma changed',1)),
                          ('arithmetic',generated.replace(' += ',' -= ',1)),
                          ('address',re.sub(r'\[([^\[\]\n]+)\]',r'[(\1) + 1]',generated,count=1))]:
            if bad==generated:continue
            accepted=compare(hand,bad)
            evidence['mutations'].append({'profile':name,'mutation':label,'rejected':not accepted})
            assert not accepted,(name,label)
        names=arrays(generated);assert names==arrays(hand),(name,names,arrays(hand))
        names=sorted(set(names)|{a for _,entries in DATA.get(name,[]) for a,_ in entries})
        precision='float' if name=='p3_decode_cpu' else 'double'
        libs=[]
        for kind,body in [('hand',hand),('compiled',generated)]:
            c=out/(name+'.'+kind+'.standalone.c');c.write_text(translation(body,names,precision,name))
            so=c.with_suffix('.so')
            flags=['-fopenmp'] if backend=='openmp' else ['-fopenacc','-foffload=disable']
            subprocess.run(['gcc','-O0','-shared','-fPIC',*flags,str(c),'-lm','-o',str(so)],check=True,capture_output=True)
            lib=ctypes.CDLL(str(so.resolve()));f=lib.kernel
            ctype=ctypes.c_float if precision=='float' else ctypes.c_double
            f.argtypes=[ctypes.POINTER(ctype)]*len(names)+[ctypes.c_int]*len(DIMS);f.restype=None
            libs.append((lib,f))
        for shape in [(1,1,2,3),(2,3,4,5),(1,5,3,7)]:
            B,N,D,F=shape;rng=np.random.default_rng(901)
            values={x:(rng.normal(0,.15,length(x,shape))).astype(precision) for x in names}
            # Equal entry state and initialized additive targets; A stays positive.
            for x in names:
                if x.startswith('G') and x not in ('GY','GF','GZ'):values[x].fill(0)
            if 'A' in values:values['A']=np.abs(values['A'])
            states=[{x:v.copy() for x,v in values.items()} for _ in libs]
            for (_,f),state in zip(libs,states):
                f(*[state[x].ctypes.data_as(ctypes.POINTER(ctype)) for x in names],B,N,D,F,N,D,D,2)
            error=max(float(np.max(np.abs(states[0][x]-states[1][x]))) for x in names)
            assert all(np.isfinite(v).all() for state in states for v in state.values()),name
            assert error<1e-12 if precision=='double' else error<1e-6,(name,error)
            evidence['finite_checks'].append({'profile':name,'shape':shape,'max_abs_difference':error})
    evidence['corrected_profiles']=[]
    for profile in ACCUMULATORS:
        original=(out/(profile+'.compiled.standalone.c')).read_text()
        corrected=original.replace(directive(profile),corrected_directive(profile))
        file=out/(profile+'.corrected.standalone.c');file.write_text(corrected)
        subprocess.run(['gcc','-fsyntax-only','-fopenacc','-foffload=disable',str(file)],check=True,capture_output=True)
        evidence['corrected_profiles'].append({'profile':profile,'device_initialization':'copy initialized host accumulators into separate device storage','compute_body_unchanged':True,'hardware_validation':'pending','corrected_data_contract':corrected_directive(profile)})
    assert all(r['ordered_operational_ast_equal'] and r['data_contract_equal'] for r in rows)
    (out/'validation.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps({'profiles':len(rows),'helper_checks':len(evidence['helpers']),'mutation_rejections':len(evidence['mutations']),'finite_checks':len(evidence['finite_checks']),'max_error':max(x['max_abs_difference'] for x in evidence['finite_checks'])}))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--hand',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();validate(a.hand,a.out)
