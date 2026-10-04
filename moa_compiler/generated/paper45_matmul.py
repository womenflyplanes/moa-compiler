# Generated Python ONF - paper45_matmul
# DNF: C = A psi B: C[i,j] = sum_k A[i,k]*B[k,j] F=d*k*n M=d*k T=alpha*F+beta*M
# MoA: Same DNF, many ONFs
import numpy as np
def moa_kernel_paper45_matmul(A, d=12, k=64, n=1024):
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
    print(f"paper45_matmul F={F} M={M} T={T}ms bytes={F*4/1e6}MB")
    return rav # psi selects via gamma

if __name__=="__main__":
    A=np.random.randn(64,64)
    print(moa_kernel_paper45_matmul(A))
