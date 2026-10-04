! Generated Fortran ONF - paper45_matmul
! DNF: C = A psi B: C[i,j] = sum_k A[i,k]*B[k,j] F=d*k*n M=d*k T=alpha*F+beta*M
! MoA: Same DNF, many ONFs, portable
module moa_paper45_matmul
  implicit none
contains
  subroutine moa_kernel_paper45_matmul(A, rho1, rho2, d, k, n, T)
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
    print *, "paper45_matmul F=", F, " M=", M, " T ms=", T
  end subroutine
end module moa_paper45_matmul

program test_paper45_matmul
  use moa_paper45_matmul
  real :: T
  real :: A(64,64)
  call moa_kernel_paper45_matmul(A,64,64,12,64,1024,T)
end program
