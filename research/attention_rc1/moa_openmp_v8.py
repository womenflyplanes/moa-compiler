#!/usr/bin/env python3
"""MoA SC27 V8: AST-derived query partition -> OpenMP C, pre-execution record, validation.
Demonstrates real OpenMP parallel execution, not CPU affinity or performance prediction.
"""
from pathlib import Path
import argparse, ctypes, hashlib, json, math, os, platform, random, shutil, statistics, subprocess, tempfile, time
from moa_processor_lift_v7 import make_lift,generate
from moa_codegen_v1 import Loop,lift_loop,reference

CASES=[(2,4,8,8,2),(2,5,3,2,3),(1,7,4,5,4),(2,9,8,3,5)]
def sha(b):return hashlib.sha256(b).hexdigest()
def compiler_flags():
    # macOS Apple clang requires Homebrew libomp; GNU gcc on Linux supports -fopenmp.
    if platform.system()=='Darwin':
        base='/opt/homebrew/opt/libomp'
        if not Path(base+'/include/omp.h').exists():
            base='/usr/local/opt/libomp'
        if not Path(base+'/include/omp.h').exists():
            raise RuntimeError('Install OpenMP runtime: brew install libomp')
        return ['clang','-O2','-std=c11','-fPIC','-shared','-Xpreprocessor','-fopenmp',
                '-I'+base+'/include','-L'+base+'/lib','-Wl,-rpath,'+base+'/lib','-lomp','-lm']
    return ['gcc','-O2','-std=c11','-fPIC','-shared','-fopenmp','-lm']
def compile_c(src,output,flags):
    # Keep source between compiler and linker flags so both Apple clang and gcc work.
    subprocess.run(flags[:1]+flags[1:-2]+[str(src),'-o',str(output)]+flags[-2:],check=True)
def parallelize(source):
    needle='  for(int ib=0;ib<'
    assert source.count(needle)==1
    return source.replace(needle,'  #pragma omp parallel for schedule(static)\n'+needle,1)
def run():
    p=argparse.ArgumentParser()
    p.add_argument('--processors',type=int,default=4)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--reps',type=int,default=15)
    p.add_argument('--out',default='.')
    a=p.parse_args()
    if min(a.processors,a.threads,a.reps)<1:p.error('all counts must be positive')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    prefix=f'moa_v8_p{a.processors}_t{a.threads}'
    flags=compiler_flags()
    entries=[]
    for k,(B,N,D,Dv,Tk) in enumerate(CASES,1):
        lift=make_lift(N,a.processors)
        sequential=generate(lift,lift_loop(Loop('j',N),Tk))
        # Each query has thread-private scratch; avoid illegal return inside omp region.
        sequential=sequential.replace('double *s=(double*)malloc((size_t)N*sizeof(double));\n    if(!s)return -2;', 'double s[N];').replace('    free(s);', '    /* VLA scratch released on scope exit */')
        parallel=parallelize(sequential)
        assert sorted(lift.coverage())==list(range(N))
        files={}
        for mode,source in [('sequential',sequential),('openmp',parallel)]:
            fn=f'{prefix}_case{k}_{mode}.c';(out/fn).write_text(source)
            files[mode]={'file':fn,'sha256':sha((out/fn).read_bytes())}
        entries.append({'case':k,'shape':{'B':B,'N':N,'D':D,'Dv':Dv},'key_tile':Tk,
                        'query_lift':{'processors':a.processors,'local_extent':lift.local_extent,
                                      'formula':'i=p*local_extent+u','guard':'i<N','coverage':lift.coverage()},
                        'prediction':{'score_multiplications':B*N*N*D,'output_multiplications':B*N*N*Dv,
                                      'array_bytes':8*(2*B*N*D+2*B*N*Dv+B*N*N),
                                      'scratch_bytes_per_active_query':8*N,
                                      'max_concurrent_scratch_upper_bound_bytes':8*N*min(N,a.processors,a.threads)},
                        'source':files})
    certificate={'schema':'moa-v8-pre-execution','machine':platform.platform(),
        'processors_logical':a.processors,'threads_requested':a.threads,'compiler_flags':flags,
        'assumptions':['OpenMP partitions are not pinned to physical cores',
                       'No time or speedup prediction is made',
                       'Scratch upper bound assumes at most min(N,partitions,threads) simultaneous query rows',
                       'Allocation counts exclude allocator overhead and OpenMP runtime memory'],
        'cases':entries}
    cp=out/f'{prefix}_prediction.json';cp.write_text(json.dumps(certificate,indent=2)+'\n')
    frozen=sha(cp.read_bytes());print('PRE-EXECUTION',cp.name,'sha256',frozen,flush=True)
    rng=random.Random(20261019)
    results=[]
    with tempfile.TemporaryDirectory() as tmp:
      for e in entries:
        d=e['shape'];B,N,D,Dv=[d[k] for k in ('B','N','D','Dv')]
        Q=[rng.uniform(-1,1) for _ in range(B*N*D)]
        K=[rng.uniform(-1,1) for _ in range(B*N*D)]
        V=[rng.uniform(-1,1) for _ in range(B*N*Dv)]
        expectedP,expectedO=reference(Q,K,V,B,N,D,Dv)
        arr=lambda x:(ctypes.c_double*len(x))(*x)
        row={'case':e['case'],'results':{}}
        for mode in ('sequential','openmp'):
            src=out/e['source'][mode]['file'];assert sha(src.read_bytes())==e['source'][mode]['sha256']
            so=Path(tmp)/(mode+str(e['case'])+'.so')
            compile_c(src,so,flags)
            fn=ctypes.CDLL(str(so)).moa_attention_generated
            fn.argtypes=[ctypes.POINTER(ctypes.c_double)]*5+[ctypes.c_int]*4
            fn.restype=ctypes.c_int
            os.environ['OMP_NUM_THREADS']=str(a.threads)
            # omp_set_num_threads must be called through the same OpenMP runtime as the library.
            if mode=='openmp':
                lib=ctypes.CDLL(str(so))
                # exported omp_set_num_threads resolved through linked runtime
                try:lib.omp_set_num_threads(ctypes.c_int(a.threads))
                except AttributeError:pass
            timings=[];errors=[]
            for _ in range(a.reps):
                P=(ctypes.c_double*(B*N*N))();O=(ctypes.c_double*(B*N*Dv))()
                start=time.perf_counter_ns()
                rc=fn(arr(Q),arr(K),arr(V),P,O,B,N,D,Dv)
                timings.append(time.perf_counter_ns()-start)
                errors.append((max(abs(x-y) for x,y in zip(P,expectedP)),
                               max(abs(x-y) for x,y in zip(O,expectedO))))
                if rc!=0:raise RuntimeError(f'generated C returned {rc}')
            ep=max(x for x,y in errors);eo=max(y for x,y in errors)
            if max(ep,eo)>1e-12:raise RuntimeError(f'case {e["case"]} {mode} incorrect')
            row['results'][mode]={'P_max_abs_error':ep,'O_max_abs_error':eo,
                                  'median_elapsed_ns':statistics.median(timings),
                                  'min_elapsed_ns':min(timings),'repetitions':a.reps}
        s=row['results']['sequential']['median_elapsed_ns'];t=row['results']['openmp']['median_elapsed_ns']
        row['observed_median_ratio_sequential_over_openmp']=s/t if t else None
        print(f'PASS case={e["case"]} N={N} P={row["results"]["openmp"]["P_max_abs_error"]:.3e} '
              f'O={row["results"]["openmp"]["O_max_abs_error"]:.3e} '
              f'seq_med_ns={s} omp_med_ns={t} ratio={s/t:.3f}',flush=True)
        results.append(row)
    report={'schema':'moa-v8-post-execution','certificate_sha256':frozen,
            'certificate_file':cp.name,'status':'PASS','results':results,
            'warning':'Small tests are dominated by call and thread overhead; observed ratios are not predictive validation.'}
    rp=out/f'{prefix}_validation.json';rp.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS report',rp.name)
if __name__=='__main__':run()
