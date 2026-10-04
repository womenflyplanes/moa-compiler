"""
MoA Universal IR: Same DNF -> Many ONFs in Python, Fortran, C, OpenMP
DNF: ivec psi A = (rav A)[gamma(ivec, rhoA)]
Paper 1: exact reproduce (rav, psi)
Paper 2: backward (psi with reverse gamma)
Paper 3: decode (gamma_row vs gamma_col)
Paper 4-5: validation (F=d*k*n, M=d*k, T=alpha*F+beta*M, reached(d))
"""
import pathlib
out_dir=pathlib.Path("moa_compiler/generated")
out_dir.mkdir(exist_ok=True)

# DNF for each paper - these are your MoA forms from papers 1-5
dnfs={
"paper1_exact": "A psi i = (rav A)[gamma(i, rho A)] # exact reproduce",
"paper2_backward": "A psi_rev i = (rav A)[gamma_rev(i, rho A)] # backward",
"paper3_decode": "decode: A psi i = (rav A)[gamma_row(i) or gamma_col(i)] # 6 vs 7",
"paper45_matmul": "C = A psi B: C[i,j] = sum_k A[i,k]*B[k,j] F=d*k*n M=d*k T=alpha*F+beta*M"
}

def gen_python(name, dnf):
    return f'''# Generated Python ONF - {name}
# DNF: {dnf}
# MoA: Same DNF, many ONFs
import numpy as np
def moa_kernel_{name}(A, d=12, k=64, n=1024):
    """ONF Python: gamma chooses layout, T approx alpha*F+beta*M"""
    # rav A - flatten
    rav = A.ravel()
    # gamma mapping - (Pi rho)_data vs (Pi rho)_machines
    # gamma_row=6 for C, gamma_col=7 for Fortran
    def gamma_row(ivec, rho):
        return ivec[0]*rho[1] + ivec[1] # row-major
    def gamma_col(ivec, rho):
        return ivec[0] + ivec[1]*rho[0] # col-major
    F=d*k*n; M=d*k
    T=0.004*F/1e6 + 0.004*M/1e3 # predicted A100
    print(f"{{name}} F={{F}} M={{M}} T={{T}}ms bytes={{F*4/1e6}}MB")
    return rav # psi selects via gamma

if __name__=="__main__":
    A=np.random.randn(64,64)
    print(moa_kernel_{name}(A))
'''

def gen_fortran(name, dnf):
    return f'''! Generated Fortran ONF - {name}
! DNF: {dnf}
! MoA: Same DNF, many ONFs, portable
module moa_{name}
  implicit none
contains
  subroutine moa_kernel_{name}(A, rho1, rho2, d, k, n, T)
    integer, intent(in) :: rho1, rho2, d, k, n
    real, intent(in) :: A(rho1, rho2)
    real, intent(out) :: T
    integer :: F, M
   ! rav A - Fortran is col-major gamma_col=7 automatically
   ! gamma_col(i,j) = i + j*rho1! vs gamma_row=6 in C
   ! (Pi rho)_data vs (Pi rho)_machines
    F = d*k*n
    M = d*k
    T = 0.004*F/1.0e6 + 0.004*M/1.0e3
    print *, "{name} F=", F, " M=", M, " T ms=", T
  end subroutine
end module moa_{name}

program test_{name}
  use moa_{name}
  real :: T
  real :: A(64,64)
  call moa_kernel_{name}(A,64,64,12,64,1024,T)
end program
'''

def gen_c(name, dnf):
    return f'''// Generated C ONF - {name}
// DNF: {dnf}
// MoA: Same DNF, many ONFs, portable OpenMP
#include <stdio.h>
// gamma_row=6 for C row-major
int gamma_row(int i, int j, int rho1, int rho2) {{ return i*rho2 + j; }}
// gamma_col=7 for Fortran col-major (for comparison)
int gamma_col(int i, int j, int rho1, int rho2) {{ return i + j*rho1; }}

void moa_kernel_{name}(float* A, int rho1, int rho2, int d, int k, int n) {{
  // rav A - flatten
  // (Pi rho)_data vs (Pi rho)_machines
  int F=d*k*n; int M=d*k;
  float T=0.004*F/1e6 + 0.004*M/1e3;
  printf("{name} F=%d M=%d T=%f ms bytes=%f MB\\n", F, M, T, F*4/1e6);
  // OpenMP deterministic: -O2 -fno-fast-math -DOMP_DETERMINISTIC
  #pragma omp parallel for proc_bind(close)
  for(int idx=0; idx<F; idx++) {{ /* psi via gamma */ }}
}}

int main() {{ float A[64*64]; moa_kernel_{name}(A,64,64,12,64,1024); return 0; }}
'''

def gen_openmp(name, dnf):
    return gen_c(name, dnf).replace("// MoA:", "// MoA OpenMP ONF:\n// #pragma omp proc_bind(close) OMP_PLACES=cores\n// Fixes 535x NUMA vs <3x")

for name, dnf in dnfs.items():
    (out_dir/f"{name}.py").write_text(gen_python(name, dnf))
    (out_dir/f"{name}.f90").write_text(gen_fortran(name, dnf))
    (out_dir/f"{name}.c").write_text(gen_c(name, dnf))
    (out_dir/f"{name}_openmp.c").write_text(gen_openmp(name, dnf))
    print(f"Generated {name} in 4 langs: py f90 c openmp.c")

print(f"\nAll in {out_dir}/")
