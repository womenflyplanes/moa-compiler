! Generated MPI Fortran ONF - poisson_jacobi - from DNF
! DNF: ⟨i,j⟩ ψ u_next = (0.25*((((⟨i-1,j⟩ ψ u+⟨i+1,j⟩ ψ u)+⟨i,j-1⟩ ψ u)+⟨i,j+1⟩ ψ u)+((h*h)*⟨i,j⟩ ψ f))); state=previous_sweep
! ONF Jacobi MPI: 1D slab over j, halo 1 row top/bottom via MPI_Sendrecv per sweep
! Each rank: owns ny_local rows, exchanges halo, computes its slab, no other comm
program poisson_jacobi_mpi
  use mpi
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  real, parameter :: h = 0.01587301678955555
  real, allocatable :: u(:,:), u_new(:,:), f(:,:)
  integer :: ierr, rank, nprocs, ny_local, j0_global, j1_global
  integer :: i, j, it, jg, up, down
  integer :: req(4)
  real :: t0, t1
  real :: u32, u11
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
  if (nprocs > ny) then
    if (rank == 0) print *, 'Unsupported: more MPI ranks than owned rows'
    call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
  end if
  ! ny must divide nprocs for this simple slab - 64 / 1,2,4 = OK
  ny_local = ny / nprocs
  if (rank < mod(ny,nprocs)) ny_local = ny_local + 1
  j0_global = rank*(ny/nprocs) + min(rank,mod(ny,nprocs)) + 1
  j1_global = j0_global + ny_local - 1
  allocate(u(0:nx+1, 0:ny_local+1), u_new(0:nx+1, 0:ny_local+1), f(0:nx+1, 0:ny_local+1))
  f = 1.0; u = 0.0; u_new = 0.0
  up = rank - 1; if (up < 0) up = MPI_PROC_NULL
  down = rank + 1; if (down >= nprocs) down = MPI_PROC_NULL
  t0 = MPI_Wtime()
  do it = 1, niter
    ! halo exchange: send bottom row to down, top row to up, recv halo rows
    call MPI_Sendrecv(u(1,1), nx, MPI_REAL, up, 0, u(1,ny_local+1), nx, MPI_REAL, down, 0, MPI_COMM_WORLD, &
    & MPI_STATUS_IGNORE, ierr)
    call MPI_Sendrecv(u(1,ny_local), nx, MPI_REAL, down, 1, u(1,0), nx, MPI_REAL, up, 1, MPI_COMM_WORLD, &
    & MPI_STATUS_IGNORE, ierr)
    do j = 1, ny_local
      do i = 1, nx
        u_new(i,j) = (0.25*((((u(i-1,j)+u(i+1,j))+u(i,j-1))+u(i,j+1))+((h*h)*f(i,j))))
      end do
    end do
    ! swap u / u_new by copy - simple, correct, not the perf bottleneck at nx=64
    u(0:nx+1, 0:ny_local+1) = u_new(0:nx+1, 0:ny_local+1)
  end do
  t1 = MPI_Wtime()
  ! gather u(32,32) and u(1,1) - rank that owns them sends to rank 0
  u32 = 0.0; u11 = 0.0
  if (32 >= j0_global .and. 32 <= j1_global) u32 = u(32, 32-j0_global+1)
  if (1 >= j0_global .and. 1 <= j1_global) u11 = u(1, 1-j0_global+1)
  call MPI_Allreduce(MPI_IN_PLACE, u32, 1, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(MPI_IN_PLACE, u11, 1, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  if (rank == 0) then
    print *, "JACOBI-MPI ranks=", nprocs, " time=", t1-t0, " u(32,32)=", u32, " u(1,1)=", u11
  end if

  call MPI_Finalize(ierr)
end program poisson_jacobi_mpi
