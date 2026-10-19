#!/usr/bin/env python3
"""Validate sequential and OpenMP generated C against Python reference."""

import ctypes
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "attention_rc1"))

from moa_codegen_v1 import Loop, lift_loop, emit_c, reference
from moa_openmp_codegen_v1 import emit_openmp

CASES = [
    (2, 4, 8, 8, 2, 2),
    (2, 5, 3, 2, 2, 3),
    (1, 7, 4, 5, 3, 4),
    (2, 9, 8, 3, 4, 5),
]

THREADS = (1, 2, 4, 8)
PTR = ctypes.POINTER(ctypes.c_double)


def compile_kernel(source, output, parallel):
    src = output.with_suffix(".c")
    src.write_text(source)

    cmd = ["clang", "-O2", "-fPIC", "-shared"]

    if parallel:
        cmd += [
            "-Xpreprocessor", "-fopenmp",
            "-I/opt/homebrew/opt/libomp/include",
            "-L/opt/homebrew/opt/libomp/lib",
            "-Wl,-rpath,/opt/homebrew/opt/libomp/lib",
        ]

    cmd += [str(src), "-lm"]

    if parallel:
        cmd += ["-lomp"]

    cmd += ["-o", str(output)]
    subprocess.run(cmd, check=True)


def execute(library, Q, K, V, B, N, D, Dv):
    fn = ctypes.CDLL(str(library)).moa_attention_generated
    fn.argtypes = [PTR] * 5 + [ctypes.c_int] * 4
    fn.restype = ctypes.c_int

    def array(values):
        return (ctypes.c_double * len(values))(*values)

    p = (ctypes.c_double * (B*N*N))()
    o = (ctypes.c_double * (B*N*Dv))()

    rc = fn(array(Q), array(K), array(V),
            p, o, B, N, D, Dv)

    if rc != 0:
        raise RuntimeError(f"Generated kernel returned {rc}")

    return list(p), list(o)


def main():
    threads = int(os.environ["OMP_NUM_THREADS"])
    assert threads in THREADS

    rng = random.Random(20261019)

    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)

        for index, (B, N, D, Dv, Tq, Tk) in enumerate(CASES):
            query = lift_loop(Loop("i", N), Tq)
            key = lift_loop(Loop("j", N), Tk)

            seq = temp / f"sequential_{index}.dylib"
            omp = temp / f"openmp_{index}.dylib"

            compile_kernel(emit_c(query, key), seq, False)
            compile_kernel(emit_openmp(query, key), omp, True)

            Q = [rng.uniform(-1, 1) for _ in range(B*N*D)]
            K = [rng.uniform(-1, 1) for _ in range(B*N*D)]
            V = [rng.uniform(-1, 1) for _ in range(B*N*Dv)]

            expected_p, expected_o = reference(
                Q, K, V, B, N, D, Dv
            )

            seq_p, seq_o = execute(seq, Q, K, V, B, N, D, Dv)
            omp_p, omp_o = execute(omp, Q, K, V, B, N, D, Dv)

            errors = [
                max(abs(a-b) for a, b in zip(seq_p, expected_p)),
                max(abs(a-b) for a, b in zip(seq_o, expected_o)),
                max(abs(a-b) for a, b in zip(omp_p, expected_p)),
                max(abs(a-b) for a, b in zip(omp_o, expected_o)),
                max(abs(a-b) for a, b in zip(seq_p, omp_p)),
                max(abs(a-b) for a, b in zip(seq_o, omp_o)),
            ]

            worst = max(errors)
            assert worst < 1e-12, (index, errors)

            print(
                f"PASS threads={threads} B={B} N={N} "
                f"D={D} Dv={Dv} tiles={Tq},{Tk} "
                f"max_error={worst:.3e}",
                flush=True
            )


if __name__ == "__main__":
    main()
