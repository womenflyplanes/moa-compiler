! Generated Fortran ONF - poisson_jacobi - from DNF
! DNF: u_new psi ivec = 0.25 * sum_{ (di,dj) in {(-1,0),(1,0),(0,-1),(0,1)} } (ivec+(di,dj)) psi u + h^2 * (ivec psi f) # Poisson stencil DNF, rav + gamma_col
! MoA: Same DNF, many ONFs - Fortran backend, gamma_col=7
! ONF Jacobi: one phase, all ivec parallel, two arrays, variable swap only
program poisson_jacobi_out
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  real, parameter :: h = 1.0 / 63.0
  real :: u(0:nx+1, 0:ny+1), u_new(0:nx+1, 0:ny+1), f(0:nx+1, 0:ny+1)
  real :: tmp(0:nx+1, 0:ny+1)
  integer :: i, j, it
  ! test init: f = 1.0, u = 0.0 - deterministic, for gate check only
  f = 1.0
  u = 0.0
  u_new = 0.0
  do it = 1, niter
    ! psi reduction: u_new psi ivec = 0.25*sum neighbours psi u + h^2 * ivec psi f
    do j = 1, ny
      do i = 1, nx
        u_new(i,j) = 0.25 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h*h*f(i,j))
      end do
    end do
    ! variable swap only - no pointer swap in Fortran here, copy via temp
    tmp = u
    u = u_new
    u_new = tmp
  end do
  print *, "JACOBI u(32,32)=", u(32,32), " u(1,1)=", u(1,1)
end program poisson_jacobi_out
