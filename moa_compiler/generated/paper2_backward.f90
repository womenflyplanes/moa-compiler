! Generated Fortran ONF - paper2_backward
! DNF: A psi_rev i = (rav A)[gamma_rev(i, rho A)] # backward
! MoA: Same DNF, many ONFs, portable
module moa_paper2_backward
  implicit none
contains
  subroutine moa_kernel_paper2_backward(A, rho1, rho2, d, k, n, T)
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
    print *, "paper2_backward F=", F, " M=", M, " T ms=", T
  end subroutine
end module moa_paper2_backward

program test_paper2_backward
  use moa_paper2_backward
  real :: T
  real :: A(64,64)
  call moa_kernel_paper2_backward(A,64,64,12,64,1024,T)
end program
