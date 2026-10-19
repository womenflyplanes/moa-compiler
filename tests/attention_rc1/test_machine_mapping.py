#!/usr/bin/env python3
"""Check legal resource estimates and compiled mapped attention kernels."""
from pathlib import Path
import random
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'/'attention_rc1'))
sys.path.insert(0,str(ROOT/'tests'/'attention_rc1'))
from moa_machine_mapping_v1 import MachineShape10,AttentionShape,map_attention,emit_mapped,m1_pro_example
from moa_codegen_v1 import reference
from test_structured import run_kernel


def main():
    machine=m1_pro_example()
    for B,N,D,Dv in ((2,4,8,8),(2,5,3,2),(1,7,4,5),(2,9,8,3)):
        shape=AttentionShape(B,N,D,Dv)
        rng=random.Random(20261019+N)
        values=[[rng.uniform(-1,1) for _ in range(size)] for size in (B*N*D,B*N*D,B*N*Dv)]
        expected=reference(*values,B,N,D,Dv)
        for backend in ('sequential','openmp'):
            plan,source=emit_mapped(machine,shape,backend)
            assert 1<=plan.query_tile<=N and 1<=plan.key_tile<=N
            assert plan.query_blocks*plan.query_tile>=N
            assert plan.workspace_bytes==8*B*N*N
            assert plan.total_live_bytes==plan.tensor_bytes+plan.workspace_bytes
            assert 1<=plan.requested_threads<=machine.max_omp_threads
            with tempfile.TemporaryDirectory() as tmp:
                got=run_kernel(source,values,(B,N,D,Dv),backend=='openmp',Path(tmp),f'mapped_{N}_{backend}')
            error=max(abs(x-y) for a,b in zip(got,expected) for x,y in zip(a,b))
            assert error<1e-12,(backend,N,error)
            print(f'PASS N={N} backend={backend} tile={plan.query_tile}x{plan.key_tile} '
                  f'threads_recommended={plan.requested_threads} live_bytes={plan.total_live_bytes} err={error:.3g}',flush=True)
    try:
        map_attention(machine,AttentionShape(1,1000000,8,8))
    except ValueError:
        pass
    else:
        raise AssertionError('RAM rejection failed')
    try:
        MachineShape10(2,8,2,1024,2048,64,8192,2,16,4096)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid machine accepted')
    print('PASS resource rejection and machine-shape validation')

if __name__=='__main__':
    main()
