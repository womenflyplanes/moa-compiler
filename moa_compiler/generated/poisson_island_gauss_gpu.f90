! Generated GPU Fortran ONF - poisson_island_gauss - from DNF
! DNF: u_new psi ivec = 0.25 * sum_{ (di,dj) in {(-1,0),(1,0),(0,-1),(0,1)} } (ivec+(di,dj)) psi u + h^2 * (ivec psi f) # Poisson stencil DNF, rav + gamma_col
! ONF Island GPU: parallel over islands, sequential inside island, halo-safe
! Halo: u_old snapshot at start of sweep. Inside island -> u, outside -> u_old
! OpenACC: parallel loop over ix,iy islands. Do NOT parallelize i,j inside island.
program poisson_island_gauss_gpu
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  integer, parameter :: isx = 16, isy = 16
  integer, parameter :: nix = nx / isx, niy = ny / isy
  real, parameter :: h = 1.0 / 63.0
  real :: u(0:nx+1, 0:ny+1), u_old(0:nx+1, 0:ny+1), f(0:nx+1, 0:ny+1)
  real :: ul, ur, ud, uu
  integer :: i, j, it, ix, iy, i0, j0, i1, j1
  f = 1.0
  u = 0.0
  !$ACC DATA COPY(u, f) CREATE(u_old)
  do it = 1, niter
    !$ACC KERNELS PRESENT(u, u_old)
    u_old = u
    !$ACC END KERNELS
    !$ACC PARALLEL LOOP COLLAPSE(2) PRESENT(u, u_old, f) PRIVATE(ix,iy,i0,j0,i1,j1,i,j,ul,ur,ud,uu)
    do iy = 1, niy
      do ix = 1, nix
        i0 = (ix-1)*isx + 1
        i1 = ix*isx
        j0 = (iy-1)*isy + 1
        j1 = iy*isy
        do j = j0, j1
          do i = i0, i1
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
            u(i,j) = 0.25 * (ul + ur + ud + uu + h*h*f(i,j))
          end do
        end do
      end do
    end do
    !$ACC END PARALLEL LOOP
  end do
  !$ACC END DATA
  print *, "ISLAND-GPU u(32,32)=", u(32,32), " u(1,1)=", u(1,1)
end program poisson_island_gauss_gpu
