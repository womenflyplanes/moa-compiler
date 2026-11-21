import subprocess, re, sys
import numpy as np
NX, NY, NITER = 64, 64, 100
H = 1.0/63.0
def ref_jacobi():
    u = np.zeros((NX+2, NY+2)); u_new = np.zeros_like(u); f = np.ones_like(u)
    for _ in range(NITER):
        u_new[1:-1,1:-1] = 0.25*(u[:-2,1:-1] + u[2:,1:-1] + u[1:-1,:-2] + u[1:-1,2:] + H*H*f[1:-1,1:-1])
        tmp = u.copy(); u = u_new; u_new = tmp
    return u[32,32], u[1,1]
def ref_island_parallel():
    u = np.zeros((NX+2, NY+2)); f = np.ones_like(u)
    for _ in range(NITER):
        u_old = u.copy()
        for iy in range(NY//16):
            for ix in range(NX//16):
                i0 = ix*16+1; i1 = (ix+1)*16+1; j0 = iy*16+1; j1 = (iy+1)*16+1
                for j in range(j0, j1):
                    for i in range(i0, i1):
                        ul = u_old[i-1,j] if i-1 < i0 else u[i-1,j]
                        ur = u_old[i+1,j] if i+1 > i1 else u[i+1,j]
                        ud = u_old[i,j-1] if j-1 < j0 else u[i,j-1]
                        uu = u_old[i,j+1] if j+1 > j1 else u[i,j+1]
                        u[i,j] = 0.25*(ul+ur+ud+uu + H*H*f[i,j])
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
def run(bin_path, src, ranks):
    subprocess.run(["mpifort","-O2","-o",bin_path,src], check=True, capture_output=True)
    out = subprocess.run(["mpirun","-np",str(ranks),bin_path], capture_output=True, text=True, check=True).stdout
    m = re.search(r"u\(32,32\)=\s*([0-9.E+-]+)\s+u\(1,1\)=\s*([0-9.E+-]+)", out)
    if not m:
        m = re.search(r"C\(32,32\)=\s*([0-9.E+-]+)\s+C\(1,1\)=\s*([0-9.E+-]+)", out)
    if not m:
        print(f"FAIL parse ranks={ranks}: {out}"); sys.exit(1)
    tm = re.search(r"time=\s*([0-9.E+-]+)", out)
    t = float(tm.group(1)) if tm else float("nan")
    return float(m.group(1)), float(m.group(2)), t
print("=== MPI gate: 1, 2, 4 ranks vs Python reference ===")
print("Island uses halo-safe parallel reference, others use serial reference")
all_ok = True
for name, src, bin_path, ref_fn, labels in [
    ("JACOBI-MPI","moa_compiler/generated/poisson_jacobi_mpi.f90","/tmp/jacobi_mpi_gate",ref_jacobi,["u(32,32)","u(1,1)"]),
    ("ISLAND-MPI","moa_compiler/generated/poisson_island_gauss_mpi.f90","/tmp/island_mpi_gate",ref_island_parallel,["u(32,32)","u(1,1)"]),
    ("RBGS-MPI","moa_compiler/generated/poisson_rbgs_mpi.f90","/tmp/rbgs_mpi_gate",ref_rbgs,["u(32,32)","u(1,1)"]),
    ("MATMUL-MPI","moa_compiler/generated/matmul_mpi.f90","/tmp/matmul_mpi_gate",ref_matmul,["C(32,32)","C(1,1)"]),
]:
    rv = ref_fn()
    for ranks in (1,2,4):
        fv1, fv2, t = run(bin_path, src, ranks)
        print(f"{name} ranks={ranks} time={t:.3e}:")
        for label, fvv, rvv in zip(labels, (fv1,fv2), rv):
            rel = abs(fvv-rvv)/max(abs(rvv),1e-12)
            status = "PASS" if rel < 1e-5 else "FAIL"
            if status=="FAIL": all_ok=False
            print(f"  {label}: MPI={fvv:.8e} Python_ref={rvv:.8e} rel_err={rel:.2e} -> {status}")
print()
print("MPI GATE:", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
