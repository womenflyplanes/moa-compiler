! Generated GPU Fortran ONF - matmul - from DNF
! DNF: ⟨i,j⟩ ψ c = (⟨i,j⟩ ψ c+(⟨i,k⟩ ψ a*⟨k,j⟩ ψ b)); state=ordered_k_reduction
! ONF matmul GPU: i,j parallel, k sequential
! OpenACC: parallel loop over i,j is SAFE - each C(i,j) is independent
program matmul_gpu
  implicit none
  integer, parameter :: n = 64
  real :: A(n,n), B(n,n), C(n,n)
  integer :: i, j, k
  A = 1.0
  B = 1.0
  C = 0.0
  !$ACC DATA COPYIN(A, B) COPYOUT(C)
  !$ACC PARALLEL LOOP COLLAPSE(2) PRESENT(A, B, C) PRIVATE(i,j,k)
  do j = 1, n
    do i = 1, n
      C(i,j) = 0.0
      do k = 1, n
        C(i,j) = (c(i,j)+(a(i,k)*b(k,j)))
      end do
    end do
  end do
  !$ACC END PARALLEL LOOP
  !$ACC END DATA
  print *, "MATMUL-GPU C(32,32)=", C(32,32), " C(1,1)=", C(1,1)
end program matmul_gpu
