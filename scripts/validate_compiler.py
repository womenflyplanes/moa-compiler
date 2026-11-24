#!/usr/bin/env python3
"""Portable, fail-closed host correctness entry point. No HPC submission/timing."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--out',default='validation_results');pa.add_argument('--with-fortran',action='store_true');a=pa.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True);records=[]
    def run(label,cmd,extra=None):
        env=dict(os.environ,OMP_NUM_THREADS='4',ACC_DEVICE_TYPE='host');env.update(extra or {})
        p=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=300)
        (out/(label+'.log')).write_text(p.stdout+p.stderr)
        records.append({'check':label,'status':'PASS' if p.returncode==0 else 'FAIL','returncode':p.returncode})
        (out/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
        print(records[-1],flush=True)
        if p.returncode:raise SystemExit(p.returncode)
    run('python_syntax',[sys.executable,'-m','compileall','-q','research','fortran_frontend'])
    run('core_and_semantic_tests',[sys.executable,'-m','pytest','-q'])
    run('attention_numerical',[sys.executable,'research/attention_rc1/moa_codegen_v1.py'])
    for stem in ('test_index_dnf_onf','test_structured','test_machine_mapping','test_psi_onf_attention'):
        run(stem,[sys.executable,'tests/attention_rc1/'+stem+'.py'])
    if a.with_fortran:
        for name in ('gfortran','mpifort','mpirun'):
            if not shutil.which(name):raise SystemExit('Required tool missing: '+name)
        run('full_fields',[sys.executable,'fortran_frontend/validate_full_fields.py','--out',str(out/'full_fields.json')])
    else:
        records.append({'check':'Fortran/OpenMP/OpenACC-host/MPI full fields','status':'NOT_RUN: supply --with-fortran'})
    records.extend([{'check':'MATLAB execution','status':'NOT_RUN: requires MATLAB/Octave'}, {'check':'physical GPU execution','status':'NOT_RUN: this entry point is host-only'}])
    (out/'summary.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
