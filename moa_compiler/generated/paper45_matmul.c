// Generated C ONF - paper45_matmul
// DNF: C = A psi B: C[i,j] = sum_k A[i,k]*B[k,j] F=d*k*n M=d*k T=alpha*F+beta*M
// MoA: Same DNF, many ONFs, portable OpenMP
#include <stdio.h>
// gamma_row=6 for C row-major
int gamma_row(int i, int j, int rho1, int rho2) { return i*rho2 + j; }
// gamma_col=7 for Fortran col-major (for comparison)
int gamma_col(int i, int j, int rho1, int rho2) { return i + j*rho1; }

void moa_kernel_paper45_matmul(float* A, int rho1, int rho2, int d, int k, int n) {
  // rav A - flatten
  // (Pi rho)_data vs (Pi rho)_machines
  int F=d*k*n; int M=d*k;
  float T=0.004*F/1e6 + 0.004*M/1e3;
  printf("paper45_matmul F=%d M=%d T=%f ms bytes=%f MB\n", F, M, T, F*4/1e6);
  // OpenMP deterministic: -O2 -fno-fast-math -DOMP_DETERMINISTIC
  #pragma omp parallel for proc_bind(close)
  for(int idx=0; idx<F; idx++) { /* psi via gamma */ }
}

int main() { float A[64*64]; moa_kernel_paper45_matmul(A,64,64,12,64,1024); return 0; }
