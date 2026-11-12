"""End-to-end finite tests for the bounded typed-trace -> C compiler."""
import ctypes
import copy
import importlib.util
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("trace_emitter", ROOT / "tools/emit_paper1_trace_c.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import json


class TraceEmitterTests(unittest.TestCase):
    def trace(self):
        return json.loads((ROOT / "results/paper1_reproduction/typed_derivation_trace.json").read_text())

    def test_mutated_dnf_rejected(self):
        t = self.trace()
        t["dnf"]["score"]["term"]["term"]["rhs"]["indices"] = ["j", "k"]
        with self.assertRaises(mod.ContractError):
            mod.emit(t)

    def test_mutated_onf_rejected(self):
        t = self.trace()
        t["onf"]["addresses"][1]["indices"] = ["j", "k"]
        with self.assertRaises(mod.ContractError):
            mod.emit(t)

    def test_generated_c_against_independent_finite_oracle(self):
        if not shutil.which("cc"):
            self.skipTest("C compiler not installed")
        source = mod.emit(self.trace())
        with tempfile.TemporaryDirectory() as folder:
            c, so = Path(folder)/"generated.c", Path(folder)/"generated.so"
            c.write_text(source)
            subprocess.run(["cc", "-std=c11", "-O2", "-fPIC", "-shared",
                            str(c), "-lm", "-o", str(so)], check=True, capture_output=True)
            fun = ctypes.CDLL(str(so)).moa_trace_attention
            ptr = ctypes.POINTER(ctypes.c_double)
            fun.argtypes = [ptr, ptr, ptr, ptr] + [ctypes.c_size_t]*4
            fun.restype = ctypes.c_int
            for n, m, d, dv in [(1,1,1,1), (2,3,4,2), (3,2,2,5), (4,7,3,6)]:
                Q = [math.sin(i*.13) for i in range(n*d)]
                K = [math.cos(i*.17) for i in range(m*d)]
                V = [math.sin(i*.31) for i in range(m*dv)]
                arr = lambda x: (ctypes.c_double*len(x))(*x)
                O = (ctypes.c_double*(n*dv))()
                self.assertEqual(fun(arr(Q), arr(K), arr(V), O, n, m, d, dv), 0)
                for i in range(n):
                    scores = [sum(Q[i*d+j]*K[k*d+j] for j in range(d))/math.sqrt(d)
                              for k in range(m)]
                    mx = max(scores)
                    weights = [math.exp(x-mx) for x in scores]
                    den = sum(weights)
                    for v in range(dv):
                        expected = sum(weights[k]*V[k*dv+v] for k in range(m))/den
                        self.assertAlmostEqual(O[i*dv+v], expected, places=12)


if __name__ == "__main__":
    unittest.main()
