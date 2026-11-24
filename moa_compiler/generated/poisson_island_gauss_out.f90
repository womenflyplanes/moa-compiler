! Generated Fortran ONF - poisson_island_gauss - from DNF
! DNF: ⟨i,j⟩ ψ u_next = (0.25*((((⟨i-1,j⟩ ψ u+⟨i+1,j⟩ ψ u)+⟨i,j-1⟩ ψ u)+⟨i,j+1⟩ ψ u)+((h*h)*⟨i,j⟩ ψ f))); state=lexicographic_islands
! MoA: Same DNF, many ONFs - Fortran backend, gamma_col=7
! ONF Island Gauss: parallel over islands (ix,iy), sequential inside island, in-place, halo
program poisson_island_gauss_out
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  integer, parameter :: isx = 16, isy = 16
  integer, parameter :: nix = nx / isx, niy = ny / isy
  real, parameter :: h = 0.01587301678955555
  real :: u(0:nx+1, 0:ny+1), f(0:nx+1, 0:ny+1)
  integer :: i, j, it, ix, iy, i0, j0, i1, j1
  f = 1.0
  u = 0.0
  do it = 1, niter
    ! Lexicographic island order is part of this DNF state contract
    do iy = 1, niy
      do ix = 1, nix
        i0 = (ix-1)*isx + 1
        i1 = ix*isx
        j0 = (iy-1)*isy + 1
        j1 = iy*isy
        ! inside island: sequential Gauss-Seidel - use newest u
        do j = j0, j1
          do i = i0, i1
            u(i,j) = (0.25*((((u(i-1,j)+u(i+1,j))+u(i,j-1))+u(i,j+1))+((h*h)*f(i,j))))
          end do
        end do
      end do
    end do
  end do
  print *, "ISLAND u(32,32)=", u(32,32), " u(1,1)=", u(1,1)
end program poisson_island_gauss_out
