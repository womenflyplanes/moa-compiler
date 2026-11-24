"""Validate archived freeze bindings, keeping missing historical bindings explicit."""
import json,hashlib,subprocess
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(archive,hand,out):
    root=archive/'results/job_137615_openacc_gpu';freeze=json.loads((root/'freeze.json').read_text());checks=[]
    for path,wanted in freeze['run_files'].items():
        file=root/path;actual=digest(file) if file.exists() else None
        checks.append({'path':path,'expected_sha256':wanted,'actual_sha256':actual,'match':actual==wanted})
    for field,path in [('runner_sha256',archive/'gate.py'),('compiler_sha256',archive/'compiler/moa_full_contract.py')]:
        actual=digest(path);checks.append({'path':str(path.relative_to(archive)),'expected_sha256':freeze[field],'actual_sha256':actual,'match':actual==freeze[field]})
    assert all(x['match'] for x in checks)
    observations=json.loads((root/'holdout_samples.json').read_text());assessment=json.loads((root/'assessment.json').read_text());complete=json.loads((root/'COMPLETE.json').read_text())
    inventory=[{'path':str(p.relative_to(hand)),'sha256':digest(p)} for p in sorted(hand.rglob('*.c'))]
    status={'paper1_job_137615':{'archive_consistency':'PASS','freeze_sha256':digest(root/'freeze.json'),'verified_frozen_files':len(checks),'source_binding':'frozen generated source, binary, build commands, runner and compiler verified against archival freeze','measurement_records_sha256':digest(root/'holdout_samples.json'),'complete':complete,'scope':'internal archive consistency; not independent signed execution attestation; historical source differs from current reference profiles'},
            'paper5_jobs':[
                {'job':21250395,'record_binding':'two eight-row summary tables agree with archived raw CSV','executed_source_sha256':None,'source_binding':'UNRESOLVED','reason':'no executed-source hash/snapshot recovered; v3/v4 revision history prevents substituting current filename'},
                {'job':21294299,'executed_source_sha256':None,'source_binding':'UNRESOLVED','reason':'records mention later atomic-free revision, but no immutable job source/build binding recovered'},
                {'job':21299223,'executed_source_sha256':None,'source_binding':'UNRESOLVED','reason':'mechanism/profile records identify comparison variants, not an executed-source hash'}],
            'current_hand_source_inventory':inventory,'freeze_checks':checks,
            'quantitative_transfer_policy':'reuse Paper I observations only with their frozen source/configuration; Paper V rows remain historical observations, excluded from unconditional timing predictions for new reference profiles'}
    out.mkdir(parents=True,exist_ok=True);(out/'historical_source_bindings.json').write_text(json.dumps(status,indent=2));print('PASS',len(checks),'Paper I frozen file bindings;',len(inventory),'hand source hashes; missing Paper V bindings explicit')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('hand',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.archive,a.hand,a.out)
