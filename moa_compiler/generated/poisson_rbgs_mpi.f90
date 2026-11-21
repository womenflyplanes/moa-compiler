! Generated MPI Fortran ONF - poisson_rbgs - from DNF
! DNF: u_new psi ivec = 0.25 * sum_{ (di,dj) in {(-1,0),(1,0),(0,-1),(0,1)} } (ivec+(di,dj)) psi u + h^2 * (ivec psi f) # Poisson stencil DNF, rav + gamma_col
! ONF RBGS MPI: slab over j, halo exchange per color phase = barrier
! Red phase: halo exchange, compute red, halo exchange, compute black
program poisson_rbgs_mpi
  use mpi
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  real, parameter :: h = 1.0 / 63.0
  real, allocatable :: u(:,:), f(:,:)
  integer :: ierr, rank, nprocs, ny_local, j0_global, j1_global
  integer :: i, j, it, color, up, down
  real :: t0, t1, u32, u11
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
  ny_local = ny / nprocs
  j0_global = rank*ny_local + 1
  j1_global = (rank+1)*ny_local
  allocate(u(0:nx+1, 0:ny_local+1), f(0:nx+1, 0:ny_local+1))
  f = 1.0; u = 0.0
  up = rank - 1; if (up < 0) up = MPI_PROC_NULL
  down = rank + 1; if (down >= nprocs) down = MPI_PROC_NULL
  t0 = MPI_Wtime()
  do it = 1, niter
    do color = 0, 1
      call MPI_Sendrecv(u(1,1), nx, MPI_REAL, up, 0, u(1,ny_local+1), nx, MPI_REAL, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
      call MPI_Sendrecv(u(1,ny_local), nx, MPI_REAL, down, 1, u(1,0), nx, MPI_REAL, up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
      do j = 1, ny_local
        do i = 1, nx
          if (mod(i + (j + j0_global - 1), 2) == color) then
            u(i,j) = 0.25 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h*h*f(i,j))
          end if
        end do
      end do
    end do
  end do
  t1 = MPI_Wtime()
  u32 = 0.0; u11 = 0.0
  if (32 >= j0_global .and. 32 <= j1_global) u32 = u(32, 32-j0_global+1)
  if (1 >= j0_global .and. 1 <= j1_global) u11 = u(1, 1-j0_global+1)
  call MPI_Allreduce(MPI_IN_PLACE, u32, 1, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(MPI_IN_PLACE, u11, 1, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  if (rank == 0) then
    print *, "RBGS-MPI ranks=", nprocs, " time=", t1-t0, " u(32,32)=", u32, " u(1,1)=", u11
  end if
  call MPI_Finalize(ierr)
end program poisson_rbgs_mpi
