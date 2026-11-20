! Generated OpenMP Fortran ONF - poisson_jacobi - from DNF
! DNF: u_new psi ivec = 0.25 * sum_{ (di,dj) in {(-1,0),(1,0),(0,-1),(0,1)} } (ivec+(di,dj)) psi u + h^2 * (ivec psi f) # Poisson stencil DNF, rav + gamma_col
! ONF Jacobi: one phase, all ivec parallel, two arrays
! OpenMP: parallel do over i,j is SAFE - reads u, writes u_new, no dependency
program poisson_jacobi_omp
  use omp_lib
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  real, parameter :: h = 1.0 / 63.0
  real :: u(0:nx+1, 0:ny+1), u_new(0:nx+1, 0:ny+1), f(0:nx+1, 0:ny+1)
  real :: tmp(0:nx+1, 0:ny+1)
  integer :: i, j, it
  f = 1.0
  u = 0.0
  u_new = 0.0
  do it = 1, niter
    !$OMP PARALLEL DO COLLAPSE(2) PRIVATE(i,j) SHARED(u, u_new, f)
    do j = 1, ny
      do i = 1, nx
        u_new(i,j) = 0.25 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h*h*f(i,j))
      end do
    end do
    !$OMP END PARALLEL DO
    tmp = u
    u = u_new
    u_new = tmp
  end do
  print *, "JACOBI-OMP u(32,32)=", u(32,32), " u(1,1)=", u(1,1)
end program poisson_jacobi_omp
