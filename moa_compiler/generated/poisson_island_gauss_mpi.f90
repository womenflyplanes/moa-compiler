! Generated MPI Fortran ONF - poisson_island_gauss - from DNF
! DNF: ⟨i,j⟩ ψ u_next = (0.25*((((⟨i-1,j⟩ ψ u+⟨i+1,j⟩ ψ u)+⟨i,j-1⟩ ψ u)+⟨i,j+1⟩ ψ u)+((h*h)*⟨i,j⟩ ψ f))); state=snapshot_islands
! ONF Island MPI: slab over j, halo 1 row via MPI_Sendrecv per sweep
! Inside slab: islands, Gauss-Seidel inside island, halo from u_old for island edges AND slab edges
! Halo exchange is per sweep, NOT per island - that is the point of halo-safe ONF
program poisson_island_gauss_mpi
  use mpi
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  integer, parameter :: isx = 16, isy = 16
  integer, parameter :: nix = nx / isx, niy = ny / isy
  real, parameter :: h = 0.01587301678955555
  real, allocatable :: u(:,:), u_old(:,:), f(:,:)
  real :: ul, ur, ud, uu
  integer :: ierr, rank, nprocs, ny_local, j0_global, j1_global
  integer :: i, j, it, ix, iy, i0, j0, i1, j1, j_local
  integer :: up, down
  real :: t0, t1, u32, u11
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
  if (nprocs > ny) then
    if (rank == 0) print *, 'Unsupported: more MPI ranks than owned rows'
    call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
  end if
  if (mod(ny,nprocs) /= 0 .or. mod(ny/nprocs,isy) /= 0) then
    if (rank == 0) print *, 'Unsupported: MPI slabs must align with island rows'
    call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
  end if
  ny_local = ny / nprocs
  if (rank < mod(ny, nprocs)) ny_local = ny_local + 1
  j0_global = rank*(ny/nprocs) + min(rank, mod(ny, nprocs)) + 1
  j1_global = j0_global + ny_local - 1
  allocate(u(0:nx+1, 0:ny_local+1), u_old(0:nx+1, 0:ny_local+1), f(0:nx+1, 0:ny_local+1))
  f = 1.0; u = 0.0; u_old = 0.0
  up = rank - 1; if (up < 0) up = MPI_PROC_NULL
  down = rank + 1; if (down >= nprocs) down = MPI_PROC_NULL
  t0 = MPI_Wtime()
  do it = 1, niter
    u_old = u
    ! MPI halo exchange: exchange u_old halo rows - other ranks need previous sweep halo
    call MPI_Sendrecv(u_old(1,1), nx, MPI_REAL, up, 0, u_old(1,ny_local+1), nx, MPI_REAL, down, 0, &
    & MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
    call MPI_Sendrecv(u_old(1,ny_local), nx, MPI_REAL, down, 1, u_old(1,0), nx, MPI_REAL, up, 1, &
    & MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
    ! Islands inside this slab - slab must align to island rows for this simple version
    ! ny_local=64,32,16 for nprocs=1,2,4 - all multiples of isy=16, so alignment holds
    do iy = 1, niy
      ! skip islands not in this slab
      j0 = (iy-1)*isy + 1; j1 = iy*isy
      if (j1 < j0_global .or. j0 > j1_global) cycle
      do ix = 1, nix
        i0 = (ix-1)*isx + 1; i1 = ix*isx
        do j = max(j0, j0_global), min(j1, j1_global)
          j_local = j - j0_global + 1
          do i = i0, i1
            if (i-1 < i0) then
              ul = u_old(i-1, j_local)
            else
              ul = u(i-1, j_local)
            end if
            if (i+1 > i1) then
              ur = u_old(i+1, j_local)
            else
              ur = u(i+1, j_local)
            end if
            if (j-1 < j0) then
              ! j-1 is previous island row - if in same slab, it is in u_old halo, else MPI halo in u_old
              ud = u_old(i, j_local-1)
            else
              ud = u(i, j_local-1)
            end if
            if (j+1 > j1) then
              uu = u_old(i, j_local+1)
            else
              uu = u(i, j_local+1)
            end if
            u(i, j_local) = (0.25*((((ul+ur)+ud)+uu)+((h*h)*f(i,j_local))))
          end do
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
    print *, "ISLAND-MPI ranks=", nprocs, " time=", t1-t0, " u(32,32)=", u32, " u(1,1)=", u11
  end if
  call MPI_Finalize(ierr)
end program poisson_island_gauss_mpi
