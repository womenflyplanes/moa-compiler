import subprocess, re, sys
import numpy as np
NX, NY, NITER = 64, 64, 100
H = 1.0/63.0
MATLAB = "/Applications/MATLAB_R2024a.app/bin/matlab"

def ref_jacobi():
    u = np.zeros((NX+2, NY+2)); u_new = np.zeros_like(u); f = np.ones_like(u)
    for _ in range(NITER):
        u_new[1:-1,1:-1] = 0.25*(u[:-2,1:-1] + u[2:,1:-1] + u[1:-1,:-2] + u[1:-1,2:] + H*H*f[1:-1,1:-1])
        tmp = u.copy(); u = u_new; u_new = tmp
    return u[32,32], u[1,1]
def ref_island():
    u = np.zeros((NX+2, NY+2)); f = np.ones_like(u)
    for _ in range(NITER):
        for iy in range(NY//16):
            for ix in range(NX//16):
                i0 = ix*16+1; i1 = (ix+1)*16+1; j0 = iy*16+1; j1 = (iy+1)*16+1
                for j in range(j0, j1):
                    for i in range(i0, i1):
                        u[i,j] = 0.25*(u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] + H*H*f[i,j])
    return u[32,32], u[1,1]
def ref_rbgs():
    u = np.zeros((NX+2, NY+2)); f = np.ones_like(u)
    for _ in range(NITER):
        for color in (0,1):
            for j in range(1, NY+1):
                for i in range(1, NX+1):
                    if (i+j) % 2 == color:
                        u[i,j] = 0.25*(u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] + H*H*f[i,j])
    return u[32,32], u[1,1]
def ref_matmul():
    return 64.0, 64.0

def run_matlab(expr):
    out = subprocess.run([MATLAB, "-batch", expr], capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        print(f"MATLAB error: {out.stderr[:1000]} {out.stdout[:1000]}")
        return ""
    return out.stdout

def parse(out):
    nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", out)
    if len(nums) < 2:
        return None
    return float(nums[-2]), float(nums[-1])

print("=== MATLAB gate (real 【entity-MATLAB¦canonical_name=MATLAB】 R2024a): .m from DNF vs Python reference ===")
all_ok = True
cases = [
    ("JACOBI", "addpath('moa_compiler/generated'); [u,un]=poisson_jacobi_out(64,64,100); fprintf('%.10e %.10e\\n', u(33,33), u(2,2));", ref_jacobi, ["u(32,32)","u(1,1)"]),
    ("ISLAND", "addpath('moa_compiler/generated'); u=poisson_island_gauss_out(64,64,100); fprintf('%.10e %.10e\\n', u(33,33), u(2,2));", ref_island, ["u(32,32)","u(1,1)"]),
    ("RBGS", "addpath('moa_compiler/generated'); u=poisson_rbgs_out(64,64,100); fprintf('%.10e %.10e\\n', u(33,33), u(2,2));", ref_rbgs, ["u(32,32)","u(1,1)"]),
    ("MATMUL", "addpath('moa_compiler/generated'); C=matmul_out(64); fprintf('%.10e %.10e\\n', C(32,32), C(1,1));", ref_matmul, ["C(32,32)","C(1,1)"]),
]
for name, expr, ref_fn, labels in cases:
    print(f"{name}: running MATLAB...")
    out = run_matlab(expr)
    vals = parse(out)
    rv = ref_fn()
    if vals is None:
        print(f"  FAIL: could not parse MATLAB output: {out!r}")
        all_ok = False
        continue
    for label, mv, rvv in zip(labels, vals, rv):
        rel = abs(mv-rvv)/max(abs(rvv),1e-12)
        status = "PASS" if rel < 1e-5 else "FAIL"
        if status=="FAIL": all_ok=False
        print(f"  {name} {label}: MATLAB={mv:.8e} Python_ref={rvv:.8e} rel_err={rel:.2e} -> {status}")
print()
print("MATLAB GATE:", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
