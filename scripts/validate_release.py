"""Replay integration without reclassifying completed research as pending work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--out',default='validation_results/unified');pa.add_argument('--with-fortran',action='store_true');a=pa.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='2',ACC_DEVICE_TYPE='host')
    def run(label,cmd):
        with (out/(label+'.log')).open('w') as f:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('PASS',label,flush=True)
    cmd=[sys.executable,'scripts/validate_compiler.py','--out',str(out/'compiler')]
    if a.with_fortran:cmd+=['--with-fortran']
    run('compiler',cmd)
    mod=ROOT/'research/companion_attention/reference_onf';programs=out/'attention/programs';evidence=out/'attention/evidence'
    programs.mkdir(parents=True,exist_ok=True);evidence.mkdir(parents=True,exist_ok=True)
    hand=ROOT/'evidence/attention_hand/original_hand';trans=ROOT/'research/companion_attention/hand_transcriptions'
    steps=[('validate_all.py',['--hand',str(hand),'--transcriptions',str(trans),'--out',str(programs)]),('check_all_dnfs.py',[str(evidence)]),('regress_all.py',[str(evidence)]),('recipe_regressions.py',[str(evidence)]),('check_all_lifts.py',[str(programs),str(evidence)]),('check_initialization.py',[str(programs),str(evidence)]),('close_inventory.py',['--hand',str(hand),'--programs',str(programs),'--transcriptions',str(trans),'--out',str(evidence)]),('seal_all.py',[str(programs),str(evidence)])]
    for name,args in steps:run(name[:-3],[sys.executable,str(mod/name),*args])
    run('historical_bindings',[sys.executable,str(ROOT/'scripts/audit_historical_bindings.py'),'--out',str(out/'historical')])
    from importlib import import_module
    sys.path.insert(0,str(ROOT/'tools'));export=import_module('export_attention_case')
    for case in export.REGISTRY:export.compile_case(case,out/'exports'/case)
    run('compile_exported_attention',[sys.executable,str(ROOT/'scripts/check_attention_exports.py'),str(out/'exports')])
    seal=json.loads((evidence/'FINAL_CORRESPONDENCE_CERTIFICATE.json').read_text())
    summary={'status':'PASS','purpose':'integration replay of completed scopes, not new derivations','attention_pairs':seal['paired_cases'],'DNF_components':seal['DNF_component_comparisons'],'attention_exports':44,'new_hardware_timing':False,'fortran_host_full_fields_requested':a.with_fortran,'compiler_summary':json.loads((out/'compiler/summary.json').read_text())}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
