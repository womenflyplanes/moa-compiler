#!/usr/bin/env python3
"""Check psi-derived ONF C, sequential/OpenMP, against reference and RC1."""
import os
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research' / 'attention_rc1'))
sys.path.insert(0, str(ROOT / 'tests' / 'attention_rc1'))
from moa_attention_psi_onf_v1 import PsiAccess, emit
from moa_structured_codegen_v1 import AttentionIR, Domain, Mapping, emit as prior_emit
from moa_codegen_v1 import reference
from test_structured import CASES, run_kernel
from moa_index_dnf_onf_v1 import gamma


def check_gamma():
    for shape, names in [((2, 5, 3), ('B','N','D')), ((2, 5, 2), ('B','N','Dv'))]:
        for b in range(shape[0]):
            for i in range(shape[1]):
                for d in range(shape[2]):
                    access = PsiAccess('Q', names, ('b','i','d'))
                    assert eval(access.offset().replace('(size_t)', ''), {}, {'b': b, 'i': i, 'd': d,
                           'B': shape[0], 'N': shape[1], names[-1]: shape[2]}) == gamma((b,i,d),shape)
    try:
        PsiAccess('Q', ('B','N','D'), ('b','i'))
    except ValueError:
        pass
    else:
        raise AssertionError('incomplete psi access accepted')


def main():
    check_gamma()
    rng = random.Random(20261019)
    with tempfile.TemporaryDirectory() as folder:
        temp = Path(folder)
        for case_id, (B,N,D,Dv,Tq,Tk) in enumerate(CASES):
            ir = AttentionIR(Domain('i',N,Tq),Domain('j',N,Tk))
            vals = [[rng.uniform(-1,1) for _ in range(length)]
                    for length in (B*N*D,B*N*D,B*N*Dv)]
            expected = reference(*vals,B,N,D,Dv)
            for backend in ('sequential','openmp'):
                new = emit(ir,Mapping(backend))
                assert 'Q[' in new and 'K[' in new and 'V[' in new
                got = run_kernel(new,vals,(B,N,D,Dv),backend=='openmp',temp,f'psi_{backend}_{case_id}')
                old = run_kernel(prior_emit(ir,Mapping(backend)),vals,(B,N,D,Dv),
                                 backend=='openmp',temp,f'prior_{backend}_{case_id}')
                err = max(abs(a-b) for x,y in ((got,expected),(got,old))
                          for xx,yy in zip(x,y) for a,b in zip(xx,yy))
                assert err < 1e-12, (backend,case_id,err)
            print(f'PASS threads={os.getenv("OMP_NUM_THREADS","1")} N={N} psi-ONF seq/openmp/reference/prior',flush=True)

if __name__ == '__main__':
    main()
