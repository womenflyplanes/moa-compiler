#!/usr/bin/env python3
"""SC27 v13: measured machine-specific cost calibration, frozen unseen-shape predictions.

This is an empirical *candidate* cost model; NOT a derivation of time from 15D machine shape.
Keeps V12 results untouched. Exact work/coverage is derived from lifted MoA query domains.
"""
import argparse, ctypes, hashlib, json, math, os, platform, random, statistics, subprocess, tempfile, time
from pathlib import Path
from moa_scaling_v9 import generated, compile_c, compiler_flags, reference, B, D, DV

CAL=(96,160,256)
HOLD=(320,448)
CANDIDATES=((4,4),(8,4),(8,8),(16,8))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
def sourcehash(s):return hashlib.sha256(s.encode()).hexdigest()
def domains(n,p,t):
    src,lift=generated(n,p)
    sizes=[sum(bool(lift.valid(j,u)) for u in range(lift.local_extent)) for j in range(p)]
    assert sum(sizes)==n and sorted(lift.coverage())==list(range(n))
    perpart=[B*z*n*(D+DV) for z in sizes]
    # OpenMP schedule(static) default distributes adjacent iterations in nearly equal contiguous chunks.
    # This is an estimate of the work assignment, not an observed affinity mapping.
    loads=[]
    for tid in range(t):
        lo=(tid*p)//t
        hi=((tid+1)*p)//t
        loads.append(sum(perpart[lo:hi]))
    # For p<t, integer boundaries can be misleading; max useful work remains bounded by a partition.
    assert sum(loads)==sum(perpart)
    return {'N':n,'partitions':p,'threads':t,'query_counts':sizes,
            'partition_multiplications':perpart,'thread_work_estimate':loads,
            'critical_work_estimate':max(loads),'total_multiplications':sum(perpart),
            'active_task_upper_bound':min(n,p,t),
            'persistent_bytes':8*(2*B*n*D+2*B*n*DV+B*n*n),
            'source_sha256':sourcehash(src['omp'])}

def machine():
    facts={'platform':platform.platform(),'python':platform.python_version(),'compiler_flags':compiler_flags(),
           'OMP_PROC_BIND':os.environ.get('OMP_PROC_BIND'), 'OMP_PLACES':os.environ.get('OMP_PLACES')}
    if platform.system()=='Darwin':
        for key in ('hw.ncpu','hw.physicalcpu','hw.memsize','machdep.cpu.brand_string'):
            try:facts[key]=subprocess.check_output(['sysctl','-n',key],stderr=subprocess.DEVNULL,text=True).strip()
            except (subprocess.CalledProcessError,FileNotFoundError):facts[key]='UNAVAILABLE'
    try:facts['compiler_version']=subprocess.check_output([compiler_flags()[0],'--version'],text=True).splitlines()[0]
    except Exception:facts['compiler_version']='UNAVAILABLE'
    return facts

def run_case(n,p,t,rounds,warmups,out,label):
    src,lift=generated(n,p)
    path=out/f'{label}_n{n}_p{p}_t{t}.c';path.write_text(src['omp'])
    expected_hash=sourcehash(src['omp']);assert sha(path)==expected_hash
    rng=random.Random(20261019+n)
    qv=[rng.uniform(-.5,.5) for _ in range(B*n*D)]
    kv=[rng.uniform(-.5,.5) for _ in range(B*n*D)]
    vv=[rng.uniform(-.5,.5) for _ in range(B*n*DV)]
    arr=lambda x:(ctypes.c_double*len(x))(*x)
    q,k,v=map(arr,(qv,kv,vv));refp,refo=reference(qv,kv,vv,B,n,D,DV)
    with tempfile.TemporaryDirectory() as td:
        so=Path(td)/'compiled.so';compile_c(path,so,compiler_flags());lib=ctypes.CDLL(str(so))
        fn=lib.moa_attention_generated;fn.argtypes=[ctypes.POINTER(ctypes.c_double)]*5+[ctypes.c_int]*4;fn.restype=ctypes.c_int
        lib.omp_set_num_threads.argtypes=[ctypes.c_int];lib.omp_set_num_threads(t)
        pp=(ctypes.c_double*(B*n*n))();oo=(ctypes.c_double*(B*n*DV))()
        def call():
            begin=time.perf_counter_ns();rc=fn(q,k,v,pp,oo,B,n,D,DV);duration=time.perf_counter_ns()-begin
            if rc:raise RuntimeError(f'C failed {rc}')
            return duration
        for _ in range(warmups):call()
        samples=[call() for _ in range(rounds)]
        ep=max(abs(x-y) for x,y in zip(pp,refp));eo=max(abs(x-y) for x,y in zip(oo,refo))
        if max(ep,eo)>1e-11:raise RuntimeError(f'Numerical mismatch N={n} p={p} t={t}: {ep} {eo}')
    return {'N':n,'partitions':p,'threads':t,'samples_ns':samples,'median_ns':statistics.median(samples),
            'P_error':ep,'O_error':eo,'source_sha256':expected_hash}

def fit(records,p,t):
    xs=[domains(r['N'],p,t)['critical_work_estimate'] for r in records]
    ys=[r['median_ns'] for r in records]
    xm=statistics.mean(xs);ym=statistics.mean(ys)
    denom=sum((x-xm)**2 for x in xs)
    if denom==0:raise RuntimeError('Insufficient work variation')
    slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/denom
    intercept=ym-slope*xm
    return {'intercept_ns':intercept,'ns_per_critical_multiply':slope,
            'method':'least-squares empirical fit to AST-derived maximum static thread work',
            'calibration_sizes':list(CAL),
            'note':'Intercept aggregates launch, synchronization, memory and other unmodeled costs; not physically isolated'}

def main():
    aparser=argparse.ArgumentParser()
    aparser.add_argument('--phase',choices=('calibrate','validate'),required=True)
    aparser.add_argument('--out',default='moa_v13_cost_model')
    aparser.add_argument('--rounds',type=int,default=9)
    aparser.add_argument('--warmups',type=int,default=4)
    a=aparser.parse_args()
    if a.rounds<3 or a.warmups<1:aparser.error('rounds >=3 and warmups >=1 required')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    frozen=out/'frozen_v13.json';calpath=out/'calibration_v13.json'
    if a.phase=='calibrate':
        if frozen.exists() or calpath.exists():raise SystemExit('Calibration/frozen files already exist. Do not overwrite; validate or choose a new --out.')
        # Persist a planned experiment BEFORE calibration begins.
        plan={'schema':'SC27-v13-calibration-plan','calibration_sizes':CAL,'holdout_sizes':HOLD,
              'candidates':CANDIDATES,'rounds':a.rounds,'warmups':a.warmups,'machine':machine(),
              'method':'Empirical affine fit versus maximum static thread work; model frozen before holdout',
              'warning':'No pinning; OpenMP scheduling and cache effects may invalidate model'}
        save(out/'plan_v13.json',plan)
        records=[]
        for n in CAL:
            for p,t in CANDIDATES:
                r=run_case(n,p,t,a.rounds,a.warmups,out,'cal');records.append(r)
                print(f'CAL N={n} P={p} T={t} median_ns={r["median_ns"]:.0f} P_err={r["P_error"]:.2e} O_err={r["O_error"]:.2e}',flush=True)
        save(calpath,{'schema':'SC27-v13-calibration','plan_sha256':sha(out/'plan_v13.json'),'records':records})
        models={f'{p}_{t}':fit([r for r in records if (r['partitions'],r['threads'])==(p,t)],p,t) for p,t in CANDIDATES}
        cases=[]
        for n in HOLD:
            options=[]
            for p,t in CANDIDATES:
                d=domains(n,p,t);m=models[f'{p}_{t}'];pred=m['intercept_ns']+m['ns_per_critical_multiply']*d['critical_work_estimate']
                options.append({**d,'predicted_ns':pred})
            chosen=min(options,key=lambda x:x['predicted_ns'])
            cases.append({'N':n,'candidates':options,'selected':[chosen['partitions'],chosen['threads']]})
        cert={'schema':'SC27-v13-frozen-predictions','calibration_sha256':sha(calpath),
              'plan_sha256':sha(out/'plan_v13.json'),'rounds':a.rounds,'warmups':a.warmups,
              'machine':plan['machine'],'models':models,'cases':cases,
              'limitations':['Empirical model, not exact 15D time prediction',
                 'Critical-work static scheduling approximation; no core affinity',
                 'Memory and synchronization costs not separately identified',
                 'Candidates limited to four mappings; cannot assert global optimality',
                 'Held-out sizes are 320 and 448; V12 sizes 384 and 512 remain untouched']}
        save(frozen,cert)
        print('FROZEN',frozen,'sha256',sha(frozen),flush=True)
        for c in cases:
            chosen=next(x for x in c['candidates'] if [x['partitions'],x['threads']]==c['selected'])
            print(f'SELECT N={c["N"]} P={chosen["partitions"]} T={chosen["threads"]} predicted_ns={chosen["predicted_ns"]:.0f}')
        print('Now run --phase validate; do not edit frozen_v13.json')
        return
    if not frozen.exists() or not calpath.exists():raise SystemExit('Run --phase calibrate first')
    cert=json.loads(frozen.read_text());digest=sha(frozen)
    if sha(calpath)!=cert['calibration_sha256']:raise SystemExit('Calibration file changed')
    if (a.rounds,a.warmups)!=(cert['rounds'],cert['warmups']):raise SystemExit('Round/warmup mismatch')
    rows=[]
    for case in cert['cases']:
        for c in case['candidates']:
            n,p,t=c['N'],c['partitions'],c['threads']
            expected=domains(n,p,t)
            if expected['source_sha256']!=c['source_sha256'] or expected['partition_multiplications']!=c['partition_multiplications']:
                raise RuntimeError('Frozen AST/source identity mismatch')
            r=run_case(n,p,t,a.rounds,a.warmups,out,'holdout')
            r['predicted_ns']=c['predicted_ns'];r['relative_error']=(r['predicted_ns']-r['median_ns'])/r['median_ns']
            r['selected']=[p,t]==case['selected'];rows.append(r)
            print(f'PASS N={n} P={p} T={t} predicted_ns={r["predicted_ns"]:.0f} measured_ns={r["median_ns"]:.0f} error={100*r["relative_error"]:+.1f}% selected={r["selected"]}',flush=True)
    if sha(frozen)!=digest:raise RuntimeError('Frozen certificate changed during validation')
    summary=[]
    for case in cert['cases']:
        rs=[r for r in rows if r['N']==case['N']];best=min(rs,key=lambda x:x['median_ns']);selected=next(r for r in rs if r['selected'])
        regret=selected['median_ns']/best['median_ns']-1
        summary.append({'N':case['N'],'selected':case['selected'],'observed_best':[best['partitions'],best['threads']],
                        'regret_fraction':regret,'selected_time_relative_error':selected['relative_error']})
        print(f'ALLOCATION N={case["N"]} selected=P{case["selected"][0]}/T{case["selected"][1]} observed_best=P{best["partitions"]}/T{best["threads"]} regret={100*regret:.2f}%')
    save(out/'holdout_v13.json',{'schema':'SC27-v13-heldout','frozen_sha256':digest,'status':'PASS_NUMERICAL',
                                  'results':rows,'summary':summary})
    print('REPORT',out/'holdout_v13.json','frozen_sha256',digest)
if __name__=='__main__':main()
