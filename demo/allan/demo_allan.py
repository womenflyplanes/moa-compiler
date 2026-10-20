#!/usr/bin/env python3
"""Inspect real MoA attention prototype stages. Does not claim a unified pass pipeline."""
import argparse, dataclasses, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research' / 'attention_rc1'))
from moa_structured_codegen_v1 import AttentionIR, Domain, Mapping
from moa_attention_psi_onf_v1 import PsiAccess, emit
from moa_machine_mapping_v1 import AttentionShape, map_attention, m1_pro_example

def main():
    p = argparse.ArgumentParser(description='Allan compiler architecture review demo')
    p.add_argument('--out', default='allan_demo_output')
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--length', type=int, default=9)
    p.add_argument('--width', type=int, default=8)
    p.add_argument('--value-width', type=int, default=3)
    args = p.parse_args()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    shape = AttentionShape(args.batch, args.length, args.width, args.value_width)
    machine = m1_pro_example()
    print('MoA COMPILER ARCHITECTURE REVIEW — ACTUAL PROTOTYPE OUTPUT')
    print('Source baseline: commit 7e17409; descriptor values are provisional')
    print('\n[1] SEMANTIC ARRAY SHAPES (attention-specific description; not a general MoA AST)')
    arrays = {'Q': [shape.batch,shape.length,shape.width], 'K':[shape.batch,shape.length,shape.width],
              'V':[shape.batch,shape.length,shape.value_width], 'P':[shape.batch,shape.length,shape.length],
              'O':[shape.batch,shape.length,shape.value_width]}
    print(json.dumps(arrays, indent=2))
    print('\n[2] TYPED PSI ACCESS -> GAMMA ROW-MAJOR ADDRESS')
    accesses = [PsiAccess('Q',('B','N','D'),('b','i','d')),
                PsiAccess('K',('B','N','D'),('b','j','d')),
                PsiAccess('V',('B','N','Dv'),('b','j','v')),
                PsiAccess('P',('B','N','N'),('b','i','j')),
                PsiAccess('O',('B','N','Dv'),('b','i','v'))]
    address_dump = [{'array':x.array,'shape':list(x.shape),'psi_index':list(x.index),
                     'gamma_offset_C':x.offset(),'C_access':x.c()} for x in accesses]
    print(json.dumps(address_dump, indent=2))
    print('\n[3] MACHINE DESCRIPTOR + PLANS (HEURISTIC, NOT CALIBRATED)')
    print(json.dumps(dataclasses.asdict(machine), indent=2))
    plans = {}
    for backend in ('sequential','openmp'):
        plan = map_attention(machine,shape,backend)
        ir = AttentionIR(Domain('i',shape.length,plan.query_tile),Domain('j',shape.length,plan.key_tile))
        code = emit(ir,Mapping(backend))
        (out / f'attention_{backend}.c').write_text(code)
        plans[backend] = {'plan':dataclasses.asdict(plan), 'loop_ir':dataclasses.asdict(ir),
                          'mapping':dataclasses.asdict(Mapping(backend))}
        print(backend, json.dumps(plans[backend], indent=2))
    report = {'scope':'restricted attention prototype, not a complete AST->DNF->ONF compiler',
              'arrays':arrays,'psi_gamma':address_dump,'machine_shape_example':dataclasses.asdict(machine),
              'plans':plans,'generated_files':['attention_sequential.c','attention_openmp.c']}
    (out/'compiler_trace.json').write_text(json.dumps(report,indent=2)+'\n')
    print('\n[4] GENERATED CODE:',out/'attention_sequential.c','and',out/'attention_openmp.c')
    print('[5] TRACE:',out/'compiler_trace.json')
    print('NOTE: semantic DNF rewrite engine, general AST pass manager, and measured performance are NOT demonstrated.')
    print('For compiled numerical verification run: python3 tests/attention_rc1/test_machine_mapping.py')

if __name__ == '__main__':
    main()
