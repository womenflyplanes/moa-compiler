#!/usr/bin/env python3
"""Four shapes x two backends; run once per OMP_NUM_THREADS setting."""
import ctypes
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research' / 'attention_rc1'))
from moa_structured_codegen_v1 import Domain, AttentionIR, Mapping, emit
from moa_codegen_v1 import reference

CASES = [(2,4,8,8,2,2),(2,5,3,2,2,3),(1,7,4,5,3,4),(2,9,8,3,4,5)]
THREADS = os.environ.get('OMP_NUM_THREADS', '1')

def run_kernel(source, values, shape, parallel, temp, name):
    c = temp / (name+'.c')
    lib = temp / (name+'.dylib')
    c.write_text(source)
    cmd = ['clang', '-O2', '-fPIC', '-shared', str(c), '-o', str(lib), '-lm']
    if parallel:
        cmd[1:1] = ['-Xpreprocessor', '-fopenmp', '-I/opt/homebrew/opt/libomp/include',
                    '-L/opt/homebrew/opt/libomp/lib', '-Wl,-rpath,/opt/homebrew/opt/libomp/lib']
        cmd.append('-lomp')
    subprocess.run(cmd, check=True)
    fn = ctypes.CDLL(str(lib)).moa_attention_generated
    ptr = ctypes.POINTER(ctypes.c_double)
    fn.argtypes = [ptr]*5 + [ctypes.c_int]*4
    fn.restype = ctypes.c_int
    B,N,D,Dv = shape
    make = lambda v: (ctypes.c_double*len(v))(*v)
    Q,K,V = map(make, values)
    P = (ctypes.c_double*(B*N*N))()
    O = (ctypes.c_double*(B*N*Dv))()
    assert fn(Q,K,V,P,O,B,N,D,Dv) == 0
    return list(P), list(O)

def main():
    rng = random.Random(20261019)
    with tempfile.TemporaryDirectory() as folder:
        temp = Path(folder)
        for idx,(B,N,D,Dv,Tq,Tk) in enumerate(CASES):
            ir = AttentionIR(Domain('i',N,Tq),Domain('j',N,Tk))
            assert ir.query.indices() == list(range(N))
            assert ir.key.indices() == list(range(N))
            values = [[rng.uniform(-1,1) for _ in range(size)]
                      for size in (B*N*D,B*N*D,B*N*Dv)]
            expected = reference(*values,B,N,D,Dv)
            outputs = []
            for backend in ('sequential','openmp'):
                actual = run_kernel(emit(ir,Mapping(backend)), values,
                                    (B,N,D,Dv),backend=='openmp',temp,f'{backend}_{idx}')
                err = max(abs(a-b) for got,want in zip(actual,expected)
                          for a,b in zip(got,want))
                assert err < 1e-12, (backend,idx,err)
                outputs.append(actual)
            difference = max(abs(a-b) for x,y in zip(*outputs) for a,b in zip(x,y))
            assert difference < 1e-12
            print(f'PASS threads={THREADS} N={N} tiles={Tq},{Tk} seq/openmp/reference',flush=True)

if __name__ == '__main__':
    main()
