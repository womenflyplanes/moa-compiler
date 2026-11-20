! Generated Fortran ONF - matmul - from DNF
! DNF: C psi (i,j) = sum_k ( (i,k) psi A ) * ( (k,j) psi B ) # matmul DNF
! ONF matmul: loop j, loop i, loop k, gamma_col
program matmul_out
  implicit none
  integer, parameter :: n = 64
  real :: A(n,n), B(n,n), C(n,n)
  integer :: i, j, k
  A = 1.0
  B = 1.0
  C = 0.0
  do j = 1, n
    do i = 1, n
      C(i,j) = 0.0
      do k = 1, n
        C(i,j) = C(i,j) + A(i,k) * B(k,j)
      end do
    end do
  end do
  print *, "MATMUL C(32,32)=", C(32,32), " C(1,1)=", C(1,1)
end program matmul_out
