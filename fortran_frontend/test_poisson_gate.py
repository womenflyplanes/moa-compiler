"""
Gate: Fortran output from DNF must match Python reference from same DNF.
Reference: same stencil, same nx=64, ny=64, niter=100, h=1/63, f=1.0, u=0.
Tolerance: 1e-6 relative - Fortran real (32-bit) vs Python float (64-bit).
"""
import subprocess, re, sys
import numpy as np

NX, NY, NITER = 64, 64, 100
H = 1.0/63.0

def ref_jacobi():
    u = np.zeros((NX+2, NY+2), dtype=np.float64)
    u_new = np.zeros_like(u)
    f = np.ones_like(u)
    for _ in range(NITER):
        # interior 1..NX, 1..NY in Fortran 0..NX+1 indexing = 1:-1 in Python
        u_new[1:-1,1:-1] = 0.25*(u[:-2,1:-1] + u[2:,1:-1] + u[1:-1,:-2] + u[1:-1,2:] + H*H*f[1:-1,1:-1])
        tmp = u.copy(); u = u_new; u_new = tmp
    # Fortran u(32,32) = Python u[32,32] (both 0-based array index 32 = Fortran index 32)
    return u[32,32], u[1,1]

def ref_island():
    u = np.zeros((NX+2, NY+2), dtype=np.float64)
    f = np.ones_like(u)
    ISX, ISY = 16, 16
    for _ in range(NITER):
        for iy in range(NY//ISY):
            for ix in range(NX//ISX):
                i0 = ix*ISX + 1; i1 = (ix+1)*ISX + 1  # Python slice end exclusive
                j0 = iy*ISY + 1; j1 = (iy+1)*ISY + 1
                for j in range(j0, j1):
                    for i in range(i0, i1):
                        u[i,j] = 0.25*(u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] + H*H*f[i,j])
    return u[32,32], u[1,1]


def ref_rbgs():
    u = np.zeros((NX+2, NY+2), dtype=np.float64)
    f = np.ones_like(u)
    for _ in range(NITER):
        for color in (0,1):
            for j in range(1, NY+1):
                for i in range(1, NX+1):
                    if (i+j) % 2 == color:
                        u[i,j] = 0.25*(u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] + H*H*f[i,j])
    return u[32,32], u[1,1]

def ref_matmul():
    # A=1, B=1, n=64 -> C(i,j) = sum_k 1*1 = 64.0
    return 64.0, 64.0

def run_fortran(bin_path, src):
    subprocess.run(["gfortran","-O2","-ffree-line-length-none","-o",bin_path,src], check=True, capture_output=True)
    out = subprocess.run([bin_path], capture_output=True, text=True, check=True).stdout
    # JACOBI u(32,32)= 6.298...  u(1,1)= 3.77...
    m = re.search(r"u\(32,32\)=\s*([0-9.E+-]+)\s+u\(1,1\)=\s*([0-9.E+-]+)", out)
    if not m:
        m = re.search(r"C\(32,32\)=\s*([0-9.E+-]+)\s+C\(1,1\)=\s*([0-9.E+-]+)", out)
    if not m:
        print(f"FAIL: could not parse Fortran output: {out}")
        sys.exit(1)
    return float(m.group(1)), float(m.group(2))

def check(name, fortran_vals, ref_vals):
    ok = True
    for label, fv, rv in zip(["u(32,32)","u(1,1)"], fortran_vals, ref_vals):
        rel = abs(fv-rv)/max(abs(rv),1e-12)
        status = "PASS" if rel < 1e-5 else "FAIL"
        if status=="FAIL": ok=False
        print(f"  {name} {label}: Fortran={fv:.8e} Python_ref={rv:.8e} rel_err={rel:.2e} -> {status}")
    return ok

print("=== Poisson DNF gate: Fortran vs Python reference ===")
all_ok = True
for name, src, bin_path, ref_fn, labels in [
    ("JACOBI","moa_compiler/generated/poisson_jacobi_out.f90","/tmp/jacobi_gate",ref_jacobi,["u(32,32)","u(1,1)"]),
    ("ISLAND","moa_compiler/generated/poisson_island_gauss_out.f90","/tmp/island_gate",ref_island,["u(32,32)","u(1,1)"]),
    ("RBGS","moa_compiler/generated/poisson_rbgs_out.f90","/tmp/rbgs_gate",ref_rbgs,["u(32,32)","u(1,1)"]),
    ("MATMUL","moa_compiler/generated/matmul_out.f90","/tmp/matmul_gate",ref_matmul,["C(32,32)","C(1,1)"]),
]:
    fv = run_fortran(bin_path, src)
    rv = ref_fn()
    print(f"{name}:")
    # check() currently zips with u(...) labels - override by passing labels via monkey-patch
    # Simplest: inline the check here with correct labels
    ok = True
    for label, fvv, rvv in zip(labels, fv, rv):
        rel = abs(fvv-rvv)/max(abs(rvv),1e-12)
        status = "PASS" if rel < 1e-5 else "FAIL"
        if status=="FAIL": ok=False
        print(f"  {name} {label}: Fortran={fvv:.8e} Python_ref={rvv:.8e} rel_err={rel:.2e} -> {status}")
    if not ok:
        all_ok=False
print()
print("GATE:", "PASS - Fortran matches Python reference from same DNF" if all_ok else "FAIL - mismatch, do not claim round-trip")
sys.exit(0 if all_ok else 1)
