"""All published C computational implementations, including original tiled ONFs."""
import ast,hashlib,json,re
from pathlib import Path
from pycparser import c_ast,c_generator
from check import run as run_prior,PROFILES as PRIOR
from compare import function,timed,compute_region,normalize,parse,clean
from emit import Emitter
from data import DATA,directive,ACCUMULATORS
import variants,legacy,serial_refs,paper3

PAPER3=[
 ("paper3_listing3","moa_decode_acc_listing3.c","moa_decode_acc","listing3","openacc"),
]+[("paper3_"+kind+"_"+backend,"grouped_reference.c" if kind=="grouped" else "cache_reference.c","hand_"+("grouped" if kind=="grouped" else "append")+"_"+backend,kind,backend) for kind in ("grouped","append") for backend in ("openmp","openacc")]

EXTRA=[
 ('p4_rmsnorm_verify','verification/rmsnorm_verify.c','LOOPS','rms_verify','openmp'),
 ('p4_ffn_verify','verification/mlp_verify.c','LOOPS','ffn_verify','openmp'),
 ('p4_block_verify_cpu','verification/full_block_verify.c','LOOPS','block_serial','serial'),
 ('p4_block_verify_gpu','verification/full_block_gpu_kernel.c','DATA_REGION','block_verify','openacc'),
 ('p4_chain_bench_gpu','benchmarking/fused_gpu_bench.c',None,'chain_bench','openacc'),
 ('p2_fused_atomic_gpu','experiments/fused_attn_gpu_bench_v3atomic.c',None,'fused_atomic','openacc'),
]+[(f'original_{algorithm}_{form}','attention/moa_attention.c',f'moa_{algorithm}_{form}',algorithm+'_'+form,'serial') for algorithm in ('forward','backward','fused') for form in ('dnf','onf')]

SERIAL_REFERENCE_TARGETS=[
 ('ref_forward_cpu','experiments/forward_bench.c','forward_dnf_serial','forward_ref'),
 ('ref_forward_gpu','experiments/forward_gpu_bench.c','forward_dnf_serial','forward_ref'),
 ('ref_backward_forward','experiments/backward_bench.c','forward_dnf','forward_ref'),
 ('ref_backward_serial','experiments/backward_bench.c','backward_dnf_serial','backward_ref'),
 ('ref_backward_gpu_forward','experiments/backward_gpu_bench.c','forward_dnf_serial','forward_a'),
 ('ref_backward_gpu_serial','experiments/backward_gpu_bench.c','backward_dnf_serial','backward_ref'),
 ('ref_fused_forward','experiments/fused_bench.c','forward_dnf','forward_ref'),
 ('ref_fused_backward','experiments/fused_bench.c','backward_dnf','backward_grouped_init'),
 ('ref_fused_gpu_forward','experiments/fused_attn_gpu_bench.c','forward_dnf_serial','forward_ref'),
 ('ref_fused_gpu_backward','experiments/fused_attn_gpu_bench.c','backward_dnf_serial','backward_ref'),
 ('ref_fused_v3_forward','experiments/fused_attn_gpu_bench_v3atomic.c','forward_dnf_serial','forward_ref'),
 ('ref_fused_v3_backward','experiments/fused_attn_gpu_bench_v3atomic.c','backward_dnf_serial','backward_ref'),
 ('ref_decode_double','experiments/decode_gpu_bench.c','decode_serial','decode_ref'),
 ('ref_decode_float','attention/moa_decode_bench.c','moa_decode_seq','decode_float_ref'),
]
EXTRA += [(name,path,fn,recipe,'serial') for name,path,fn,recipe in SERIAL_REFERENCE_TARGETS]

# Independent data profiles for the additional entry contracts.
import copy
DATA['p4_block_verify_gpu']=copy.deepcopy(DATA['p4_block_gpu'])
DATA['p4_block_verify_gpu'][1][1][5]=('A','B*N*N')
ACCUMULATORS['p4_block_verify_gpu']=set(ACCUMULATORS['p4_block_gpu'])
DATA['p4_chain_bench_gpu']=copy.deepcopy(DATA['p4_norm_ffn_gpu'])
DATA['p4_chain_bench_gpu'][0]=('copyin',[('X','NT*D'),('gam','D'),('GF','NT*D'),('Wg','D*DFF'),('Wu','D*DFF'),('Wd','DFF*D')])
ACCUMULATORS['p4_chain_bench_gpu']=set(ACCUMULATORS['p4_norm_ffn_gpu'])
DATA['p2_fused_atomic_gpu']=copy.deepcopy(DATA['p2_fused_gpu'])
ACCUMULATORS['p2_fused_atomic_gpu']={'GK'}

def extract(original,selector):
    if selector=='LOOPS':
        body=parse(function(original,'main'));items=[]
        for node in body.block_items:
            if isinstance(node,(c_ast.For,c_ast.Pragma)):items.append(node)
        generated=c_generator.CGenerator().visit(c_ast.Compound(items));return generated[generated.index('{')+1:generated.rfind('}')]
    if selector=='DATA_REGION':
        text=clean(original);start=text.index('#pragma acc data');stop=text.index('save_bin',start);return text[start:stop]
    return function(original,selector) if selector else timed(original)

def first_difference(a,b,path=''):
    if type(a)!=type(b):return path,a,b
    if isinstance(a,list):
        if len(a)!=len(b):return path+' length',len(a),len(b)
        for i,(x,y) in enumerate(zip(a,b)):
            result=first_difference(x,y,path+'/'+str(i))
            if result:return result
    elif a!=b:return path,a,b

def run(hand,out,transcriptions):
    rows=run_prior(hand,out)
    for profile,path,selector,recipe,backend in EXTRA+PAPER3:
        original=((transcriptions if profile.startswith('paper3_') else hand)/path).read_text();data,body=compute_region(extract(original,selector))
        module=paper3 if profile.startswith('paper3_') else legacy if profile.startswith('original_') else serial_refs if profile.startswith('ref_') else variants
        spec=getattr(module,recipe)(backend)
        shapes=paper3.CACHE_SHAPES if profile.startswith('paper3_append') else {'K':('n','dk'),'V':('n','dv')} if 'decode' in profile or profile=='paper3_listing3' else module.SHAPES
        emitted=Emitter(shapes,"float" if (profile=="ref_decode_float" or profile.startswith("paper3_")) else "double").body(spec)
        if profile=="ref_decode_float":
            emitted="const float scale = 1.0f / sqrtf((float)dk);\nfloat *s = (float *)malloc(n * sizeof(float));\nfloat *e = (float *)malloc(n * sizeof(float));\n"+emitted+"free(s); free(e);\n"
        a,b=normalize(parse(body)),normalize(parse(emitted));compiled_data=directive(profile)
        for suffix,value in [('hand.c',body),('compiled.c',emitted),('recipe.py',spec),('hand_ast.json',json.dumps(a,indent=2)),('compiled_ast.json',json.dumps(b,indent=2)),('psi_schedule_ast.txt',ast.dump(ast.parse(spec),indent=2))]:
            (out/(profile+'.'+suffix)).write_text(value)
        equal=a==b;data_equal=re.sub(r'\s+','',data or '')==re.sub(r'\s+','',compiled_data or '')
        rows.append({'profile':profile,'hand_source':path,'selector':selector,'backend':backend,'recipe':recipe,'ordered_operational_ast_equal':equal,'data_contract_equal':data_equal,'hand_sha256':hashlib.sha256(original.encode()).hexdigest(),'hand_data_contract':data,'compiled_data_contract':compiled_data})
        print(profile,equal,data_equal,first_difference(a,b) if not equal else '',flush=True)
    (out/'all_comparisons.json').write_text(json.dumps(rows,indent=2))
    assert all(r['ordered_operational_ast_equal'] and r['data_contract_equal'] for r in rows),'Inspect saved paired ASTs'
    return rows
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--hand',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--transcriptions',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);run(a.hand,a.out,a.transcriptions)
