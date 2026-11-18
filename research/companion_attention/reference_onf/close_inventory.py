"""Fail-closed coverage of every numerical C routine in the pinned hand archive."""
import re,json,hashlib
from pathlib import Path
from check_all import PRIOR,EXTRA,PAPER3
from compare import clean

SUPPORT={'alloc_rand','load_bin','save_bin','fill_random','max_abs_diff','maxdiff','print_vec','wall_time','now','silu','silu_grad','softmax_row','fill_rand','parse_int','has_flag','fill_inputs'}

def definitions(source):
    text=clean(source);result=[]
    pattern=r'\b(?:(?:static|const)\s+)*(?:void|double|float|int)\s*\*?\s+(\w+)\s*\([^;{}]*\)\s*\{'
    for m in re.finditer(pattern,text):
        start=m.end();depth=1;stop=start
        while depth:
            depth+=(text[stop]=='{')-(text[stop]=='}');stop+=1
        result.append((m.group(1),m.start(),stop))
    return text,result

def run(hand,programs,transcriptions,out):
    rows=json.loads((programs/'all_comparisons.json').read_text());by_profile={r['profile']:r for r in rows};mapping={}
    for profile,path,selector,recipe,backend in PRIOR+EXTRA:
        text,defs=definitions((hand/path).read_text())
        if selector in ('LOOPS','DATA_REGION'):fn='main'
        elif selector is not None:fn=selector
        else:
            timer=re.search(r'double t0\s*=',text).start()
            fn=next(name for name,start,end in defs if start<timer<end)
        mapping.setdefault((path,fn),[]).append(profile)
    inventory=[];uncovered=[]
    for p in sorted(hand.rglob('*.c')):
        path=str(p.relative_to(hand));text,defs=definitions(p.read_text())
        for fn,_,_ in defs:
            profiles=mapping.get((path,fn),[])
            if profiles:
                assert all(by_profile[x]['ordered_operational_ast_equal'] and by_profile[x]['data_contract_equal'] for x in profiles)
                status='CERTIFIED_NUMERICAL_REGION'
            elif fn in SUPPORT:status='SUPPORT_OR_MATCHED_MATH_HELPER'
            elif fn=='main':status='ORCHESTRATION_AND_VERIFICATION_DRIVER'
            else:status='UNCOVERED';uncovered.append((path,fn))
            inventory.append({'source':path,'function':fn,'status':status,'profiles':profiles,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    assert not uncovered,uncovered
    targets=[x for x in inventory if x['status']=='CERTIFIED_NUMERICAL_REGION']
    assert len(targets)==len(PRIOR+EXTRA)==39
    assert len({x['source'] for x in targets})==20
    provenance=[]
    for profile,path,fn,recipe,backend in PAPER3:
        provenance.append({'profile':profile,'reference_file':path,'function':fn,'source_kind':'manual transcription of published C Listing 3' if recipe=='listing3' else 'new hand C ONF of published Python Listing 7' if recipe=='grouped' else 'new hand C ONF of published Python Listing 5 lifted over cache groups','archived_C_file':False})
    value={'scope':'all numerical C computational implementations in the pinned 20-file hand C archive; mathematical helpers matched; test/IO/timing drivers classified separately','files':20,'certified_numerical_regions':len(targets),'all_profile_cases':len(rows),'uncovered_numerical_routines':uncovered,'inventory':inventory,'paper3_listing_and_derived_references':provenance,
      'language_boundary':'Fortran comparison programs and PyTorch autograd oracles remain historical/reference material; this is a C-emission correspondence study, not a verified Fortran frontend or PyTorch implementation proof.'}
    out.mkdir(parents=True,exist_ok=True);(out/'closed_coverage_inventory.json').write_text(json.dumps(value,indent=2));print('CLOSED',len(targets),'numerical regions from 20 archived C files;',len(rows),'total paired cases; zero uncovered numerical routines')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--hand',type=Path,required=True);p.add_argument('--programs',type=Path,required=True);p.add_argument('--transcriptions',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.hand,a.programs,a.transcriptions,a.out)
