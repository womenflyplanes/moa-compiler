"""Fail-closed untimed NVIDIA correctness gate. It never accepts host fallback."""
import argparse,ctypes,json,os,shutil,subprocess,hashlib
from pathlib import Path
import numpy as np
from check import PROFILES
from compare import function,timed,compute_region,clean
from data import directive,corrected_directive,ACCUMULATORS,DATA
from validate import arrays,translation,length,DIMS
PROBE='''
#include <openacc.h>
int moa_device_probe(void) {
 int on_gpu=0;
 #pragma acc parallel copy(on_gpu)
 {on_gpu=acc_on_device(acc_device_nvidia);}
 return on_gpu;
}
'''

def run(programs,out):
    out.mkdir(parents=True,exist_ok=False)
    cc=shutil.which('nvc');gpu=shutil.which('nvidia-smi')
    if not cc or not gpu:
        (out/'status.json').write_text(json.dumps({'status':'BLOCKED','reason':'nvc and nvidia-smi required; host fallback forbidden'}))
        raise RuntimeError('Real NVIDIA device and nvc required')
    environment={}
    for name,cmd in [('nvidia',['nvidia-smi','-q']),('compiler',[cc,'--version']),('host',['uname','-a'])]:
        r=subprocess.run(cmd,text=True,capture_output=True);environment[name]={'command':cmd,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
        if r.returncode:raise RuntimeError(name+' query failed')
    environment['affinity']=sorted(os.sched_getaffinity(0));environment['slurm_job_id']=os.environ.get('SLURM_JOB_ID')
    environment['cuda_visible_devices']=os.environ.get('CUDA_VISIBLE_DEVICES')
    (out/'environment.json').write_text(json.dumps(environment,indent=2))
    observations=[];frozen=[]
    os.environ['NVCOMPILER_ACC_NOTIFY']='3'
    for profile,_,_,recipe,backend in PROFILES:
        if backend!='openacc':continue
        bodies={kind:(programs/(profile+'.'+kind+'.c')).read_text() for kind in ('hand','compiled')}
        names=sorted(set(arrays(bodies['compiled']))|{a for _,entries in DATA[profile] for a,_ in entries})
        funcs=[]
        for kind in ('cpu_oracle','hand_gpu','compiled_gpu'):
            body=bodies['compiled' if kind=='compiled_gpu' else 'hand']
            text=translation(body,names,'double',profile)
            if profile in ACCUMULATORS:text=text.replace(directive(profile),corrected_directive(profile))
            if kind!='cpu_oracle':
                text=text.replace('static double silu(', '#pragma acc routine seq\nstatic double silu(').replace('static double silu_grad(', '#pragma acc routine seq\nstatic double silu_grad(')+PROBE
            file=out/(profile+'.'+kind+'.c');file.write_text(text);so=file.with_suffix('.so')
            command=([shutil.which('gcc') or 'gcc','-O0','-shared','-fPIC',str(file),'-lm','-o',str(so)] if kind=='cpu_oracle' else [cc,'-O2','-shared','-fPIC','-acc=gpu','-gpu=cc80','-Minfo=accel',str(file),'-lm','-o',str(so)])
            result=subprocess.run(command,text=True,capture_output=True);(out/(profile+'.'+kind+'.build.log')).write_text(result.stdout+result.stderr)
            if result.returncode:raise RuntimeError('Build failed '+profile+' '+kind)
            lib=ctypes.CDLL(str(so.resolve()))
            if kind!='cpu_oracle':
                lib.moa_device_probe.restype=ctypes.c_int
                if lib.moa_device_probe()!=1:raise RuntimeError('GPU probe rejected host fallback')
            func=lib.kernel;func.argtypes=[ctypes.POINTER(ctypes.c_double)]*len(names)+[ctypes.c_int]*len(DIMS);func.restype=None
            funcs.append((kind,lib,func))
            frozen.append({'source':file.name,'source_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'binary':so.name,'binary_sha256':hashlib.sha256(so.read_bytes()).hexdigest(),'build':command})
        for shape in [(1,1,2,3),(2,3,4,5),(1,5,3,7)]:
            B,N,D,F=shape;rng=np.random.default_rng(901)
            initial={name:rng.normal(0,.15,length(name,shape)).astype('double') for name in names}
            for name in names:
                if name.startswith('G') and name not in ('GY','GF','GZ'):initial[name].fill(0)
            if 'A' in initial:initial['A']=np.abs(initial['A'])
            outputs=[]
            for kind,lib,func in funcs:
                state={name:value.copy() for name,value in initial.items()}
                func(*[state[name].ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for name in names],B,N,D,F,N,D,D,2)
                outputs.append(state)
            for index,kind in [(1,'hand_gpu'),(2,'compiled_gpu')]:
                error=max(float(np.max(np.abs(outputs[0][name]-outputs[index][name]))) for name in names)
                finite=all(np.isfinite(x).all() for x in outputs[index].values())
                if not finite or error>1e-11:raise RuntimeError(f'Correctness failure {profile} {kind} {shape} {error}')
                observations.append({'profile':profile,'realization':kind,'shape':shape,'max_abs_error_vs_serial_hand':error,'device_probe':1,'timing_measured':False})
    (out/'frozen_builds.json').write_text(json.dumps(frozen,indent=2));(out/'correctness.json').write_text(json.dumps(observations,indent=2))
    (out/'status.json').write_text(json.dumps({'status':'PASS','comparisons':len(observations),'performance':'not measured','physical_launches':'runtime notification log; device probe is mandatory'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--programs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.programs,a.out)
