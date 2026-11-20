! Frontend input: Poisson Jacobi stencil
program poisson_jacobi
  implicit none
  integer, parameter :: nx = 64, ny = 64
  integer, parameter :: niter = 100
  real, parameter :: h = 1.0 / 63.0
  real :: u(0:nx+1, 0:ny+1), u_new(0:nx+1, 0:ny+1), f(0:nx+1, 0:ny+1)
  integer :: i, j, it
  do it = 1, niter
    do j = 1, ny
      do i = 1, nx
        u_new(i,j) = 0.25 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h*h*f(i,j))
      end do
    end do
    u = u_new
  end do
end program poisson_jacobi
