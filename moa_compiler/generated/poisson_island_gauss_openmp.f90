! Generated OpenMP Fortran ONF - poisson_island_gauss - from DNF
! DNF: ⟨i,j⟩ ψ u_next = (0.25*((((⟨i-1,j⟩ ψ u+⟨i+1,j⟩ ψ u)+⟨i,j-1⟩ ψ u)+⟨i,j+1⟩ ψ u)+((h*h)*⟨i,j⟩ ψ f))); state=snapshot_islands
! ONF Island Gauss PARALLEL: parallel over islands, sequential inside island
! HALO-SAFE: u_old = halo snapshot at start of sweep
! Neighbour inside island -> read u (newest, Gauss-Seidel)
! Neighbour outside island -> read u_old (halo, previous sweep)
! This is deterministic. Naive parallel islands without halo is racy.
program poisson_island_gauss_omp
  use omp_lib
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  integer, parameter :: isx = 16, isy = 16
  integer, parameter :: nix = nx / isx, niy = ny / isy
  real, parameter :: h = 0.01587301678955555
  real :: u(0:nx+1, 0:ny+1), u_old(0:nx+1, 0:ny+1), f(0:nx+1, 0:ny+1)
  real :: ul, ur, ud, uu
  integer :: i, j, it, ix, iy, i0, j0, i1, j1
  f = 1.0
  u = 0.0
  do it = 1, niter
    u_old = u
    !$OMP PARALLEL DO COLLAPSE(2) PRIVATE(ix,iy,i0,j0,i1,j1,i,j,ul,ur,ud,uu) SHARED(u, u_old, f)
    do iy = 1, niy
      do ix = 1, nix
        i0 = (ix-1)*isx + 1
        i1 = ix*isx
        j0 = (iy-1)*isy + 1
        j1 = iy*isy
        do j = j0, j1
          do i = i0, i1
            ! left: outside island -> halo, inside -> newest
            if (i-1 < i0) then
              ul = u_old(i-1,j)
            else
              ul = u(i-1,j)
            end if
            if (i+1 > i1) then
              ur = u_old(i+1,j)
            else
              ur = u(i+1,j)
            end if
            if (j-1 < j0) then
              ud = u_old(i,j-1)
            else
              ud = u(i,j-1)
            end if
            if (j+1 > j1) then
              uu = u_old(i,j+1)
            else
              uu = u(i,j+1)
            end if
            u(i,j) = (0.25*((((ul+ur)+ud)+uu)+((h*h)*f(i,j))))
          end do
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end do
  print *, "ISLAND-OMP u(32,32)=", u(32,32), " u(1,1)=", u(1,1)
end program poisson_island_gauss_omp
