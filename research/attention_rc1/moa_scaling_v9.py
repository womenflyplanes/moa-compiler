#!/usr/bin/env python3
"""SC27 V9: disjoint calibration/holdout timing prediction for AST-generated attention C.
Calibrate and validate are separate invocations. No physical-core pinning or formal time guarantee.
"""
import argparse, ctypes, hashlib, json, math, os, platform, random, statistics, subprocess, tempfile, time
from pathlib import Path
from moa_processor_lift_v7 import make_lift, generate
from moa_codegen_v1 import Loop, lift_loop, reference
from moa_openmp_v8 import compiler_flags, compile_c, parallelize

CAL=(16,32,64)
HOLD=(96,128,192)
B=1;D=32;DV=32

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def generated(n,partitions):
    lift=make_lift(n,partitions)
    src=generate(lift,lift_loop(Loop('j',n),min(32,n)))
    src=src.replace('double *s=(double*)malloc((size_t)N*sizeof(double));\n    if(!s)return -2;', 'double s[N];').replace('    free(s);','    /* local scratch */')
    assert sorted(lift.coverage())==list(range(n))
    return {'seq':src,'omp':parallelize(src)},lift

def bench(n,partitions,threads,reps,warmups,out,runid):
    sources,lift=generated(n,partitions)
    result={'N':n,'B':B,'D':D,'Dv':DV,'partitions':partitions,'threads':threads,
      'index_coverage':len(lift.coverage()),'work_multiplications':B*n*n*(D+DV),
      'storage_bytes':8*(2*B*n*D+2*B*n*DV+B*n*n), 'source':{},'modes':{}}
    rng=random.Random(20261019+n)
    Q=[rng.uniform(-0.5,0.5) for _ in range(B*n*D)]
    K=[rng.uniform(-0.5,0.5) for _ in range(B*n*D)]
    V=[rng.uniform(-0.5,0.5) for _ in range(B*n*DV)]
    ptr=ctypes.POINTER(ctypes.c_double)
    arr=lambda v:(ctypes.c_double*len(v))(*v)
    q,k,v=arr(Q),arr(K),arr(V)
    expectedP,expectedO=reference(Q,K,V,B,n,D,DV)
    with tempfile.TemporaryDirectory() as tmp:
      for mode in ('seq','omp'):
        c=out/f'{runid}_n{n}_{mode}.c';c.write_text(sources[mode]);result['source'][mode]={'name':c.name,'sha256':digest(c)}
        so=Path(tmp)/(mode+'.so')
        compile_c(c,so,compiler_flags())
        lib=ctypes.CDLL(str(so));fn=lib.moa_attention_generated
        fn.argtypes=[ptr]*5+[ctypes.c_int]*4;fn.restype=ctypes.c_int
        if mode=='omp':
          try:lib.omp_set_num_threads(ctypes.c_int(threads))
          except AttributeError:pass
        p=(ctypes.c_double*(B*n*n))();o=(ctypes.c_double*(B*n*DV))()
        def call():
          start=time.perf_counter_ns();rc=fn(q,k,v,p,o,B,n,D,DV)
          dt=time.perf_counter_ns()-start
          if rc!=0:raise RuntimeError(f'C returned {rc}')
          return dt
        for _ in range(warmups):call()
        timings=[call() for _ in range(reps)]
        ep=max(abs(x-y) for x,y in zip(p,expectedP));eo=max(abs(x-y) for x,y in zip(o,expectedO))
        if max(ep,eo)>1e-11:raise RuntimeError(f'wrong result N={n} {mode}: {ep} {eo}')
        result['modes'][mode]={'median_ns':statistics.median(timings),'samples_ns':timings,'P_error':ep,'O_error':eo}
    return result

def fit(points,mode):
    xs=[r['work_multiplications'] for r in points];ys=[r['modes'][mode]['median_ns'] for r in points]
    xm=statistics.mean(xs);ym=statistics.mean(ys)
    slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
    intercept=ym-slope*xm
    return {'intercept_ns':intercept,'ns_per_multiplication':slope,'method':'least squares, 3 disjoint calibration sizes',
      'caution':'empirical extrapolation; not a theory-derived predictive model'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',required=True,choices=['calibrate','validate'])
    ap.add_argument('--partitions',type=int,default=4);ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--reps',type=int,default=15);ap.add_argument('--warmups',type=int,default=4)
    ap.add_argument('--out',default='.')
    a=ap.parse_args()
    if min(a.partitions,a.threads,a.reps,a.warmups)<1:ap.error('counts must be positive')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    runid=f'moa_v9_p{a.partitions}_t{a.threads}'
    frozen=out/f'{runid}_frozen_predictions.json'
    if a.phase=='calibrate':
      if frozen.exists():raise SystemExit(f'Frozen predictions already exist: {frozen}; use --phase validate or choose a fresh --out directory')
      records=[bench(n,a.partitions,a.threads,a.reps,a.warmups,out,runid+'_cal') for n in CAL]
      calibration=out/f'{runid}_calibration.json'
      write(calibration,{'schema':'moa-v9-calibration','records':records,'platform':platform.platform(),
                         'flags':compiler_flags(),'threads':a.threads,'partitions':a.partitions})
      models={mode:fit(records,mode) for mode in ('seq','omp')}
      preds=[]
      for n in HOLD:
        work=B*n*n*(D+DV)
        src,_=generated(n,a.partitions)
        pred={'N':n,'work_multiplications':work,'predicted_ns':{mode:models[mode]['intercept_ns']+models[mode]['ns_per_multiplication']*work for mode in models},
          'source_sha256':{mode:hashlib.sha256(s.encode()).hexdigest() for mode,s in src.items()}}
        preds.append(pred)
      write(frozen,{'schema':'moa-v9-pre-validation','calibration_file':calibration.name,
        'calibration_sha256':digest(calibration),'calibration_sizes':CAL,'holdout_sizes':HOLD,
        'models':models,'predictions':preds,'threads':a.threads,'partitions':a.partitions,
        'reps':a.reps,'warmups':a.warmups,'limitations':['Calibration runs precede frozen holdout predictions',
        'Predictions are empirical extrapolations, NOT derived from 15D machine shape',
        'No affinity; macOS scheduling and thermal effects uncontrolled',
        'Timing includes ctypes call overhead; input allocation excluded',
        'OpenMP version has no physical processor affinity']})
      print('FROZEN',frozen.name,'sha256',digest(frozen))
      for p in preds:print('PREDICT N=%d seq_ns=%.1f omp_ns=%.1f'%(p['N'],p['predicted_ns']['seq'],p['predicted_ns']['omp']))
      print('Now run --phase validate. Do not edit frozen predictions.')
    else:
      if not frozen.exists():raise SystemExit('Run --phase calibrate first')
      frozen_hash=digest(frozen);cert=json.loads(frozen.read_text())
      if (cert['threads'],cert['partitions'])!=(a.threads,a.partitions):raise SystemExit('Configuration differs from frozen prediction')
      calibration=out/cert['calibration_file']
      if digest(calibration)!=cert['calibration_sha256']:raise SystemExit('Calibration hash mismatch')
      if (a.reps,a.warmups)!=(cert['reps'],cert['warmups']):raise SystemExit('Repetition settings differ from frozen prediction')
      records=[]
      for pred in cert['predictions']:
        n=pred['N'];src,_=generated(n,a.partitions)
        for mode in ('seq','omp'):
          if hashlib.sha256(src[mode].encode()).hexdigest()!=pred['source_sha256'][mode]:raise SystemExit('Source identity mismatch')
        r=bench(n,a.partitions,a.threads,a.reps,a.warmups,out,runid+'_holdout')
        r['comparison']={mode:{'predicted_ns':pred['predicted_ns'][mode],
          'measured_median_ns':r['modes'][mode]['median_ns'],
          'relative_error':(r['modes'][mode]['median_ns']-pred['predicted_ns'][mode])/r['modes'][mode]['median_ns']} for mode in ('seq','omp')}
        records.append(r)
        print('PASS N=%d seq predicted %.0f measured %.0f | omp predicted %.0f measured %.0f'%(n,pred['predicted_ns']['seq'],r['modes']['seq']['median_ns'],pred['predicted_ns']['omp'],r['modes']['omp']['median_ns']))
      report=out/f'{runid}_holdout_validation.json'
      write(report,{'schema':'moa-v9-post-validation','frozen_sha256':frozen_hash,'status':'PASS_CORRECTNESS',
                    'note':'Prediction discrepancies are retained, not silently refitted','results':records})
      print('REPORT',report.name,'frozen_sha256',frozen_hash)
if __name__=='__main__':main()
