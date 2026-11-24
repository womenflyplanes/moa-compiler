"""Replay selected reference-schedule lowerings; preserve comparison evidence."""
import argparse, json, hashlib
from pathlib import Path
from emit import Emitter
import attention, block_parts, full_block
from data import directive
import re
from compare import function,timed,compute_region,normalize,parse

PROFILES=[
 ('p4_norm_ffn_gpu','verification/fused_gpu_kernel.c','DATA_REGION','chain_gpu','openacc'),
 ('p3_decode_cpu','attention/moa_decode_bench.c','moa_decode_omp','decode','openmp'),
 ('p4_block_cpu','benchmarking/full_block_bench.c',None,'block','openmp'),
 ('p4_block_gpu','benchmarking/full_block_gpu_bench.c',None,'block','openacc'),
 ('p4_rmsnorm_cpu','benchmarking/rmsnorm_bench.c',None,'rmsnorm','openmp'),
 ('p4_ffn_cpu','benchmarking/mlp_bench.c',None,'ffn','openmp'),
 ('p1_forward_cpu','experiments/forward_bench.c','forward_dnf_omp','forward','openmp'),
 ('p1_forward_gpu','experiments/forward_gpu_bench.c',None,'forward','openacc'),
 ('p2_backward_cpu','experiments/backward_bench.c','backward_dnf_omp','backward','openmp'),
 ('p2_backward_gpu','experiments/backward_gpu_bench.c',None,'backward','openacc'),
 ('p2_fused_cpu','experiments/fused_bench.c','fused_dnf_omp','fused','openmp'),
 ('p2_fused_gpu','experiments/fused_attn_gpu_bench.c',None,'fused','openacc'),
 ('p3_decode_gpu','experiments/decode_gpu_bench.c',None,'decode','openacc'),
]

def run(hand_root,out):
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,path,fn,recipe,backend in PROFILES:
        original=(hand_root/path).read_text()
        if fn=="DATA_REGION":
            from compare import clean
            text=clean(original);start=text.index("#pragma acc data");stop=text.index("save_bin",start)
            source=text[start:stop]
        else:
            source=function(original,fn) if fn else timed(original)
        data,source=compute_region(source)
        module=full_block if recipe=="block" else block_parts if recipe in ("rmsnorm","ffn","chain_gpu") else attention
        spec=getattr(module,recipe)(backend)
        shapes=module.SHAPES if recipe!='decode' else {'K':('n','dk'),'V':('n','dv')}
        emitted=Emitter(shapes,"float" if name=="p3_decode_cpu" else "double").body(spec)
        if name=="p3_decode_cpu":
            emitted="const float scale = 1.0f / sqrtf((float)dk);\nfloat *s = (float *)malloc(n * sizeof(float));\nfloat *e = (float *)malloc(n * sizeof(float));\n"+emitted+"free(s); free(e);\n"
        compiled_data=directive(name)
        left,right=normalize(parse(source)),normalize(parse(emitted))
        for suffix,text in [('hand.c',source),('compiled.c',emitted),('recipe.py',spec)]:
            (out/(name+'.'+suffix)).write_text(text)
        (out/(name+'.hand_ast.json')).write_text(json.dumps(left,indent=2))
        (out/(name+'.compiled_ast.json')).write_text(json.dumps(right,indent=2))
        import ast
        (out/(name+'.psi_schedule_ast.txt')).write_text(ast.dump(ast.parse(spec),indent=2))
        row={'profile':name,'hand_source':path,'hand_sha256':hashlib.sha256(original.encode()).hexdigest(),
             'ordered_operational_ast_equal':left==right,'hand_data_contract':data,'compiled_data_contract':compiled_data,
             'data_contract_equal':re.sub(r'\s+','',data or '')==re.sub(r'\s+','',compiled_data or ''),
             'parallel_phases':len(re.findall(r'#pragma (?:omp parallel for|acc parallel loop)',emitted)),'scope':'computational region; equal entry state; bounded integer addresses'}
        rows.append(row)
    (out/'comparison.json').write_text(json.dumps(rows,indent=2))
    assert all(row["ordered_operational_ast_equal"] and row["data_contract_equal"] for row in rows), "Operational comparison failed; inspect saved ASTs"
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--hand',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    for row in run(args.hand,args.out):print(row['profile'],row['ordered_operational_ast_equal'])
