"""Untimed instrumented hand/emitted placement comparisons on the local CPU.

The observed host is recorded. It is never labeled as UA or GPU evidence.
"""
import os,ctypes,json,subprocess
from pathlib import Path
import numpy as np
from pycparser import c_ast,c_generator
from compare import parse
from check import PROFILES
from validate import arrays,translation,DIMS,length
GEN=c_generator.CGenerator();STRIDE=1024

def instrument(body):
    tree=parse(body);phases=[];items=tree.block_items
    for k,node in enumerate(items):
        if not isinstance(node,c_ast.Pragma) or not node.string.startswith('omp parallel for'):continue
        import re
        m=re.search(r'collapse\((\d+)\)',node.string);rank=int(m.group(1)) if m else 1
        loop=items[k+1];axes=[]
        for depth in range(rank):
            assert isinstance(loop,c_ast.For)
            name=loop.init.decls[0].name;bound=GEN.visit(loop.cond.right);axes.append((name,bound))
            if depth<rank-1:loop=loop.stmt.block_items[0] if isinstance(loop.stmt,c_ast.Compound) else loop.stmt
        ordinal=axes[0][0]
        for name,bound in axes[1:]:ordinal=f'(({ordinal})*({bound})+{name})'
        call=parse(f'observe({len(phases)}, {ordinal});').block_items[0]
        if not isinstance(loop.stmt,c_ast.Compound):loop.stmt=c_ast.Compound([loop.stmt])
        loop.stmt.block_items.insert(0,call);phases.append(axes)
    rendered=GEN.visit(tree)
    return rendered[rendered.index('{')+1:rendered.rfind('}')],phases

HEADER='''
#define _GNU_SOURCE
#include <sched.h>
#include <omp.h>
static int *owner_threads,*owner_cpus;
void bind_observations(int *t,int*c){owner_threads=t;owner_cpus=c;}
static void observe(int phase,int ordinal){owner_threads[phase*1024+ordinal]=omp_get_thread_num();owner_cpus[phase*1024+ordinal]=sched_getcpu();}
'''

def run(programs,out):
    allowed=sorted(os.sched_getaffinity(0))
    if len(allowed)<2:raise RuntimeError('Two allowed logical CPU IDs required')
    chosen=allowed[:2]
    os.environ.update({'OMP_NUM_THREADS':'2','OMP_DYNAMIC':'FALSE','OMP_PROC_BIND':'close',
                       'OMP_PLACES':','.join('{'+str(c)+'}' for c in chosen)})
    out.mkdir(parents=True,exist_ok=True);observations=[]
    for profile,_,_,recipe,backend in PROFILES:
        if backend!='openmp':continue
        original={kind:(programs/(profile+'.'+kind+'.c')).read_text() for kind in ('hand','compiled')}
        names=arrays(original['compiled']);precision='float' if recipe=='decode' else 'double';ctype=ctypes.c_float if precision=='float' else ctypes.c_double
        funcs=[]
        for kind in ('hand','compiled'):
            body,phases=instrument(original[kind]);text=HEADER+translation(body,names,precision,profile)
            file=out/(profile+'.'+kind+'.instrumented.c');file.write_text(text);so=file.with_suffix('.so')
            subprocess.run(['gcc','-O0','-shared','-fPIC','-fopenmp',str(file),'-lm','-o',str(so)],check=True,capture_output=True)
            lib=ctypes.CDLL(str(so.resolve()));lib.bind_observations.argtypes=[ctypes.POINTER(ctypes.c_int)]*2
            lib.kernel.argtypes=[ctypes.POINTER(ctype)]*len(names)+[ctypes.c_int]*len(DIMS);lib.kernel.restype=None
            funcs.append(lib)
        for shape in [(1,1,2,3),(2,3,4,5),(1,5,3,7)]:
            B,N,D,F=shape;NT=B*N;n=N;dk=D;dv=D;DFF=F;nthreads=2
            sizes=[int(eval('*'.join('('+bound+')' for _,bound in axes),{'__builtins__':{}},locals())) for axes in phases]
            assert max(sizes)<STRIDE
            states=[]
            for lib in funcs:
                threads=np.full((len(phases),STRIDE),-1,dtype=np.int32);cpus=threads.copy()
                lib.bind_observations(threads.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),cpus.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
                rng=np.random.default_rng(901);buffers={name:rng.normal(0,.15,length(name,shape)).astype(precision) for name in names}
                for name in names:
                    if name.startswith('G') and name not in ('GY','GF','GZ'):buffers[name].fill(0)
                lib.kernel(*[buffers[name].ctypes.data_as(ctypes.POINTER(ctype)) for name in names],B,N,D,F,N,D,D,2)
                for phase,size in enumerate(sizes):
                    assert (threads[phase,:size]>=0).all() and (threads[phase,size:]==-1).all()
                    assert set(cpus[phase,:size])<=set(chosen)
                    assert all(cpus[phase,i]==chosen[threads[phase,i]] for i in range(size))
                states.append((threads,cpus))
            assert np.array_equal(states[0][0],states[1][0]) and np.array_equal(states[0][1],states[1][1]),profile
            observations.append({'profile':profile,'shape':shape,'equal_observed_placement':True,
              'phases':[{'axes':axes,'thread_by_owner':states[0][0][phase,:sizes[phase]].tolist(),'cpu_by_owner':states[0][1][phase,:sizes[phase]].tolist()} for phase,axes in enumerate(phases)]})
    value={'host':os.uname().nodename,'allowed_cpu_ids':allowed,'bound_cpu_ids':chosen,'openmp_environment':{x:os.environ[x] for x in ('OMP_PLACES','OMP_NUM_THREADS','OMP_PROC_BIND','OMP_DYNAMIC')},'observations':observations,'scope':'instrumented local CPU placement only; no UA or A100 execution; no performance timing'}
    (out/'local_cpu_placement.json').write_text(json.dumps(value,indent=2));print('PASS',len(observations),'local pinned CPU placement comparisons')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('programs',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.programs,a.out)
