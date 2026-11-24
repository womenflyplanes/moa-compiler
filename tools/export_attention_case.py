"""Expose the completed reference-schedule study through the release interface.

This adapter reuses the existing emitter and frozen correspondence certificate.
It does not introduce a new attention derivation or claim a generic frontend.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
MOD=ROOT/'research/companion_attention/reference_onf'
sys.path.insert(0,str(MOD));sys.path.insert(1,str(MOD.parent))
import attention,block_parts,full_block,variants,legacy,serial_refs,paper3
from check_all import PRIOR,EXTRA,PAPER3
from emit import Emitter
from compare import parse,normalize
from validate_all import make_translation,FLOAT
from validate import arrays,HELPERS
from data import DATA,ACCUMULATORS,directive,corrected_directive
from seal_all import ALIASES

REGISTRY={r[0]:r for r in PRIOR+EXTRA+PAPER3}
EVIDENCE=ROOT/'research/companion_attention/evidence/final_correspondence'
PROOF_BINDINGS_SHA256='12b9a42c330813189b06001f26df8919bbd2057ca1397257c218bb266312a02a'


def compile_case(profile,out,archival=False):
    bindings_path=ROOT/'continuity/ATTENTION_PROOF_BINDINGS.json'
    if hashlib.sha256(bindings_path.read_bytes()).hexdigest()!=PROOF_BINDINGS_SHA256:raise ValueError('Proof manifest changed: integration revalidation required')
    bindings=json.loads(bindings_path.read_text())
    for path,want in bindings['sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=want:raise ValueError('Completed proof/generator dependency changed: '+path)
    if profile not in REGISTRY:raise ValueError('Unknown attention case '+profile)
    _,source_path,_,recipe,backend=REGISTRY[profile]
    module=paper3 if profile.startswith('paper3_') else legacy if profile.startswith('original_') else serial_refs if profile.startswith('ref_') else variants if profile in {r[0] for r in EXTRA} else full_block if recipe=='block' else block_parts if recipe in ('rmsnorm','ffn','chain_gpu') else attention
    spec=getattr(module,recipe)(backend)
    shapes=paper3.CACHE_SHAPES if profile.startswith('paper3_append') else {'K':('n','dk'),'V':('n','dv')} if 'decode' in profile or profile=='paper3_listing3' else module.SHAPES
    precision='float' if profile in FLOAT else 'double'
    body=Emitter(shapes,precision).body(spec)
    if profile in ('p3_decode_cpu','ref_decode_float'):
        body='const float scale = 1.0f / sqrtf((float)dk);\nfloat *s = (float *)malloc(n * sizeof(float));\nfloat *e = (float *)malloc(n * sizeof(float));\n'+body+'free(s); free(e);\n'
    digest=lambda data:hashlib.sha256(data).hexdigest()
    seal=json.loads((EVIDENCE/'FINAL_CORRESPONDENCE_CERTIFICATE.json').read_text())
    case=next(r for r in seal['certificates'] if r['profile']==profile)
    # Fail closed if the recipe or addressed output has drifted from its proof.
    if digest(spec.encode())!=case['recipe_sha256']:raise ValueError('Recipe has changed: correspondence replay required')
    tree=normalize(parse(body));tree_text=json.dumps(tree,indent=2)
    if digest(tree_text.encode())!=case['compiled_operational_AST_sha256']:raise ValueError('Emitted operational AST differs from the checked case')
    components=json.loads((EVIDENCE/'all_recipe_dnfs.json').read_text())
    rows=[r for r in components['results'] if r['profile']==ALIASES.get(profile,profile)]
    if len(rows)!=case['checked_DNF_components'] or not all(r['equal'] for r in rows):raise ValueError('Missing component certificate')
    lifted=next(r for r in json.loads((EVIDENCE/'all_lift_certificates.json').read_text())['profiles'] if r['profile']==profile)
    names=sorted(set(arrays(body))|{n for _,entries in DATA.get(profile,[]) for n,_ in entries})
    helpers=HELPERS.replace('-DBL_MAX','-1e300') if 'gpu' in source_path else HELPERS
    source=make_translation(body,names,profile,backend,helpers)
    repaired=profile in ACCUMULATORS and not archival
    if repaired:source=source.replace(directive(profile),corrected_directive(profile))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    stages={
      '01_dnf.json':{'schema':'moa.attention.checked_dnf.v1','profile':profile,'arithmetic':'exact real','components':rows,'proof_origin':'completed independent hand/recipe comparison; not a new derivation'},
      '02_onf.json':{'schema':'moa.attention.checked_onf.v1','profile':profile,'backend':backend,'precision':precision,'ordered_operational_C_AST':tree,'gamma':'declared tuple shapes lowered by existing row-major Horner emitter','computational_scope':'ordered kernel region; entry clauses specified separately','entry_contract':corrected_directive(profile) if repaired else directive(profile)},
      '03_lifted_onf.json':{'schema':'moa.attention.checked_lift.v1','profile':profile,'proof':lifted,'physical_execution_certified':False,'claim':'equality under common admissible machine shape/binding/workspace/reduction/runtime choices'},
    }
    for name,value in stages.items():(out/name).write_text(json.dumps(value,indent=2)+'\n')
    (out/'generated.c').write_text(source);(out/'recipe.py').write_text(spec)
    record={'schema':'moa.release.case_certificate.v1','profile':profile,'backend':backend,'historical_case_certificate':case,'entry_profile':'corrected fresh-device entry' if repaired else 'archival entry contract; enforce documented zero-state premises',
            'proof_artifact_bindings':bindings,'proof_status':'existing correspondence reused after recipe and operational-AST identity checks','new_hardware_execution':False,
            'files_sha256':{p.name:digest(p.read_bytes()) for p in sorted(out.iterdir()) if p.name in (*stages,'generated.c','recipe.py')},
            'premises':seal['premises']}
    (out/'certificate.json').write_text(json.dumps(record,indent=2)+'\n')
    return record
