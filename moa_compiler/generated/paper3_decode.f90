! Generated Fortran ONF - paper3_decode
! DNF: decode: A psi i = (rav A)[gamma_row(i) or gamma_col(i)] # 6 vs 7
! MoA: Same DNF, many ONFs, portable
module moa_paper3_decode
  implicit none
contains
  subroutine moa_kernel_paper3_decode(A, rho1, rho2, d, k, n, T)
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
    print *, "paper3_decode F=", F, " M=", M, " T ms=", T
  end subroutine
end module moa_paper3_decode

program test_paper3_decode
  use moa_paper3_decode
  real :: T
  real :: A(64,64)
  call moa_kernel_paper3_decode(A,64,64,12,64,1024,T)
end program
