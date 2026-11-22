! Generated MPI Fortran ONF - poisson_jacobi - from DNF
! DNF: u_new psi ivec = 0.25 * sum_{ (di,dj) in {(-1,0),(1,0),(0,-1),(0,1)} } (ivec+(di,dj)) psi u + h^2 * (ivec psi f) # Poisson stencil DNF, rav + gamma_col
! ONF Jacobi MPI: 1D slab over j, halo 1 row top/bottom via MPI_Sendrecv per sweep
! Burkardt problem: 1..nx grid, boundaries = u_exact, f = +pi^2(x^2+y^2)sin(pi x y) interior
program poisson_jacobi_mpi
  use mpi
  implicit none
  integer :: nx, ny, it, it_max, i, j, jg
  real(kind=8) :: dx, dy, x, y, diff, diff_sum, diff_glob, err_exact, err_sum, err_glob, u_mid, pi
  real(kind=8), allocatable :: u(:,:), u_new(:,:), f(:,:)
  character(len=32) :: arg
  logical :: converged
  integer :: ierr, rank, nprocs, ny_local, j0_global, j1_global
  integer :: up, down
  real(kind=8) :: u_mid_glob
  pi = 3.141592653589793d0
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
  if (rank == 0) then
    call get_command_argument(1, arg)
    read(arg, *) nx
  end if
  call MPI_Bcast(nx, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
  ny = nx; it_max = 20000
  dx = 1.0d0 / dble(nx-1); dy = dx
  ny_local = ny / nprocs
  if (rank < mod(ny, nprocs)) ny_local = ny_local + 1
  j0_global = rank*(ny/nprocs) + min(rank, mod(ny, nprocs)) + 1
  j1_global = j0_global + ny_local - 1
  allocate(u(nx,0:ny_local+1), u_new(nx,0:ny_local+1), f(nx,0:ny_local+1))
  u = 0.0d0; u_new = 0.0d0; f = 0.0d0
  do j = 1, ny_local
    jg = j0_global + j - 1
    y = dble(jg-1)*dy
    do i = 1, nx
      x = dble(i-1)*dx
      if (i==1 .or. i==nx .or. jg==1 .or. jg==ny) then
        f(i,j) = sin(pi*x*y); u(i,j) = f(i,j); u_new(i,j) = f(i,j)
      else
        f(i,j) = pi*pi*(x*x+y*y)*sin(pi*x*y)
      end if
    end do
  end do
  if (rank == 0) then; u(:,0) = u(:,1); u_new(:,0) = u(:,1); end if
  if (rank == nprocs-1) then; u(:,ny_local+1) = u(:,ny_local); u_new(:,ny_local+1) = u(:,ny_local); end if
  up = rank - 1; if (up < 0) up = MPI_PROC_NULL
  down = rank + 1; if (down >= nprocs) down = MPI_PROC_NULL
  converged = .false.
  do it = 1, it_max
    call MPI_Sendrecv(u(1,1), nx, MPI_DOUBLE_PRECISION, up, 0, u(1,ny_local+1), nx, MPI_DOUBLE_PRECISION, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
    call MPI_Sendrecv(u(1,ny_local), nx, MPI_DOUBLE_PRECISION, down, 1, u(1,0), nx, MPI_DOUBLE_PRECISION, up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
    if (rank == 0) then; u(:,0) = u(:,1); end if
    if (rank == nprocs-1) then; u(:,ny_local+1) = u(:,ny_local); end if
    diff_sum = 0.0d0
    do j = 1, ny_local
      jg = j0_global + j - 1
      do i = 1, nx
        if (i==1 .or. i==nx .or. jg==1 .or. jg==ny) then
          u_new(i,j) = f(i,j)
        else
          u_new(i,j) = 0.25d0 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + dx*dy*f(i,j))
        end if
        diff_sum = diff_sum + (u_new(i,j)-u(i,j))**2
      end do
    end do
    u(:,1:ny_local) = u_new(:,1:ny_local)
    call MPI_Allreduce(diff_sum, diff_glob, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    diff = sqrt(diff_glob / dble(nx*ny))
    if (diff <= 1.0d-6) then; converged = .true.; end if
    call MPI_Bcast(converged, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, ierr)
    if (converged) exit
  end do
  if (.not. converged) it = it_max
  err_sum = 0.0d0
  do j = 1, ny_local
    jg = j0_global + j - 1
    y = dble(jg-1)*dy
    do i = 1, nx
      x = dble(i-1)*dx
      err_sum = err_sum + (u(i,j)-sin(pi*x*y))**2
    end do
  end do
  call MPI_Allreduce(err_sum, err_glob, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
  err_exact = sqrt(err_glob / dble(nx*ny))
  u_mid = 0.0d0
  if ((nx+1)/2 >= j0_global .and. (nx+1)/2 <= j1_global) u_mid = u((nx+1)/2, (nx+1)/2 - j0_global + 1)
  call MPI_Allreduce(u_mid, u_mid_glob, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
  if (rank == 0) print *, "GATE nx=", nx, " it=", it, " diff=", diff, " err_exact=", err_exact, " u_mid=", u_mid_glob, " converged=", converged
  call MPI_Finalize(ierr)
end program poisson_jacobi_mpi
