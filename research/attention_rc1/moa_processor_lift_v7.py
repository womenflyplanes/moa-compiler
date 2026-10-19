#!/usr/bin/env python3
"""SC27 V7: AST query-processor lift -> C -> verification. Sequential emulation, not parallel speedup."""
from dataclasses import dataclass,asdict
from pathlib import Path
import argparse,ctypes,hashlib,json,math,random,subprocess,tempfile,time,platform
from moa_codegen_v1 import Loop,lift_loop,emit_c,reference

@dataclass(frozen=True)
class ProcessorLift:
    extent:int
    processors:int
    local_extent:int
    def index(self,p,u):return p*self.local_extent+u
    def valid(self,p,u):return self.index(p,u)<self.extent
    def coverage(self):return [self.index(p,u) for p in range(self.processors) for u in range(self.local_extent) if self.valid(p,u)]

def make_lift(n,p):
    if n<1 or p<1:raise ValueError('positive dimensions required')
    lift=ProcessorLift(n,p,(n+p-1)//p)
    assert sorted(lift.coverage())==list(range(n))
    assert len(set(lift.coverage()))==n
    return lift

def generate(lift,key):
    # Construct query loop AST with processor and local index nodes, then
    # instantiate sequentially. Key lift remains driven by existing AST object.
    q=lift_loop(Loop('i',lift.extent),lift.local_extent)
    source=emit_c(q,key)
    source=source.replace('int ib=0;ib<'+str(q.outer_extent), 'int ib=0;ib<'+str(lift.processors))
    # q.outer_extent may be < processor count when p>n: empty partitions still valid.
    assert f'int ib=0;ib<{lift.processors}' in source
    return source

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--processors',type=int,default=4,help='logical partitions, NOT actual CPU affinity')
    ap.add_argument('--out',default='.',help='output directory')
    args=ap.parse_args()
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=True)
    cases=[(2,4,8,8,2),(2,5,3,2,3),(1,7,4,5,4),(2,9,8,3,5)]
    entries=[]
    for idx,(B,N,D,Dv,Tk) in enumerate(cases,1):
        pl=make_lift(N,args.processors);kl=lift_loop(Loop('j',N),Tk)
        source=generate(pl,kl)
        source_path=root/f'moa_v7_generated_case{idx}.c'
        source_path.write_text(source)
        entries.append({'case':idx,'dimensions':dict(B=B,N=N,D=D,Dv=Dv),
          'query_ast':{'op':'psi-index-lift','original':'i','processor_index':'p',
                       'local_index':'u','equation':'i=p*local_extent+u',**asdict(pl),
                       'guard':'i<N','coverage':pl.coverage()},
          'key_ast':{'op':'psi-index-lift','extent':N,'tile':Tk,'guard':'j<N'},
          'predicted':{'Q_bytes':8*B*N*D,'K_bytes':8*B*N*D,'V_bytes':8*B*N*Dv,
                       'P_bytes':8*B*N*N,'O_bytes':8*B*N*Dv,'scratch_bytes':8*N,
                       'score_multiplies':B*N*N*D,'output_multiplies':B*N*N*Dv},
          'source_file':source_path.name,'source_sha256':hashlib.sha256(source.encode()).hexdigest()})
    cert={'schema':'moa-v7-pre-execution','limitations':['Logical processor partitions are executed sequentially; no actual core binding or speedup claim','Predicted byte counts are allocated capacities, not measured traffic','No time prediction','Attention operation template is not yet a general MoA rewrite engine'],
          'processor_partitions':args.processors,'cases':entries}
    cp=root/'moa_v7_prediction.json';cp.write_text(json.dumps(cert,indent=2)+'\n')
    frozen=hashlib.sha256(cp.read_bytes()).hexdigest()
    print('PRE-EXECUTION certificate_sha256',frozen,flush=True)
    rng=random.Random(20261019);results=[]
    with tempfile.TemporaryDirectory() as td:
        for e in entries:
            d=e['dimensions'];B,N,D,Dv=(d[x] for x in ('B','N','D','Dv'))
            src=root/e['source_file'];so=Path(td)/f'case{e["case"]}.so'
            subprocess.run(['cc','-O2','-fPIC','-shared',str(src),'-o',str(so),'-lm'],check=True)
            fn=ctypes.CDLL(str(so)).moa_attention_generated
            ptr=ctypes.POINTER(ctypes.c_double)
            fn.argtypes=[ptr]*5+[ctypes.c_int]*4;fn.restype=ctypes.c_int
            Q=[rng.uniform(-1,1) for _ in range(B*N*D)]
            K=[rng.uniform(-1,1) for _ in range(B*N*D)]
            V=[rng.uniform(-1,1) for _ in range(B*N*Dv)]
            a=lambda v:(ctypes.c_double*len(v))(*v)
            P=(ctypes.c_double*(B*N*N))();O=(ctypes.c_double*(B*N*Dv))()
            start=time.perf_counter_ns();rc=fn(a(Q),a(K),a(V),P,O,B,N,D,Dv)
            elapsed=time.perf_counter_ns()-start
            pr,orr=reference(Q,K,V,B,N,D,Dv)
            ep=max(abs(x-y) for x,y in zip(P,pr));eo=max(abs(x-y) for x,y in zip(O,orr))
            passed=rc==0 and ep<1e-12 and eo<1e-12
            results.append({'case':e['case'],'pass':passed,'P_max_abs_error':ep,'O_max_abs_error':eo,
                            'observed_elapsed_ns_single_run_not_prediction':elapsed,
                            'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
            print(f'{"PASS" if passed else "FAIL"} case={e["case"]} P={ep:.3e} O={eo:.3e} query_coverage={N}/{N}')
    report={'schema':'moa-v7-post-execution','prediction_certificate_sha256':frozen,
            'platform':platform.platform(),'compiler':'cc -O2 -fPIC -shared -lm',
            'results':results,'status':'PASS' if all(r['pass'] for r in results) else 'FAIL'}
    (root/'moa_v7_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('REPORT moa_v7_validation.json',report['status'])
    if report['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
