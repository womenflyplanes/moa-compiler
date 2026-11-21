! Generated MPI Fortran ONF - matmul - from DNF
! DNF: C psi (i,j) = sum_k ( (i,k) psi A ) * ( (k,j) psi B ) # matmul DNF
! ONF matmul MPI: row slab of C, broadcast B, no halo, no per-iteration comm
program matmul_mpi
  use mpi
  implicit none
  integer, parameter :: n = 64
  real, allocatable :: A(:,:), B(:,:), C(:,:)
  integer :: ierr, rank, nprocs, n_local, i0_global
  integer :: i, j, k
  real :: t0, t1, c32, c11
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
  n_local = n / nprocs
  i0_global = rank*n_local + 1
  allocate(A(n_local,n), B(n,n), C(n_local,n))
  A = 1.0; B = 1.0; C = 0.0
  ! B is same on all ranks in this simple version - broadcast is no-op, kept for ONF clarity
  call MPI_Bcast(B, n*n, MPI_REAL, 0, MPI_COMM_WORLD, ierr)
  t0 = MPI_Wtime()
  do j = 1, n
    do i = 1, n_local
      C(i,j) = 0.0
      do k = 1, n
        C(i,j) = C(i,j) + A(i,k) * B(k,j)
      end do
    end do
  end do
  t1 = MPI_Wtime()
  c32 = 0.0; c11 = 0.0
  if (32 >= i0_global .and. 32 < i0_global+n_local) c32 = C(32-i0_global+1, 32)
  if (1 >= i0_global .and. 1 < i0_global+n_local) c11 = C(1-i0_global+1, 1)
  call MPI_Allreduce(MPI_IN_PLACE, c32, 1, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(MPI_IN_PLACE, c11, 1, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  if (rank == 0) then
    print *, "MATMUL-MPI ranks=", nprocs, " time=", t1-t0, " C(32,32)=", c32, " C(1,1)=", c11
  end if
  call MPI_Finalize(ierr)
end program matmul_mpi
