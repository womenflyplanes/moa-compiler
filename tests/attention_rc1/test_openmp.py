#!/usr/bin/env python3
"""Numerical validation of generated OpenMP attention."""

import ctypes
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "attention_rc1"))

from moa_codegen_v1 import reference

LIB = "/tmp/moa_attention_openmp.dylib"

fn = ctypes.CDLL(LIB).moa_attention_generated
ptr = ctypes.POINTER(ctypes.c_double)
fn.argtypes = [ptr] * 5 + [ctypes.c_int] * 4
fn.restype = ctypes.c_int

B, N, D, Dv = 2, 5, 3, 2
rng = random.Random(20261019)

Q = [rng.uniform(-1, 1) for _ in range(B*N*D)]
K = [rng.uniform(-1, 1) for _ in range(B*N*D)]
V = [rng.uniform(-1, 1) for _ in range(B*N*Dv)]

def array(values):
    return (ctypes.c_double * len(values))(*values)

pr, expected = reference(Q, K, V, B, N, D, Dv)

threads = int(os.environ["OMP_NUM_THREADS"])

p = (ctypes.c_double * (B*N*N))()
o = (ctypes.c_double * (B*N*Dv))()

rc = fn(array(Q), array(K), array(V),
        p, o, B, N, D, Dv)

ep = max(abs(x-y) for x, y in zip(p, pr))
eo = max(abs(x-y) for x, y in zip(o, expected))

assert rc == 0
assert ep < 1e-12
assert eo < 1e-12

print(
    f"PASS threads={threads} "
    f"P={ep:.3e} O={eo:.3e}"
)
