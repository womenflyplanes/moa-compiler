! Generated Fortran ONF - poisson_jacobi - from DNF
! DNF: ⟨i,j⟩ ψ u_next = (0.25*((((⟨i-1,j⟩ ψ u+⟨i+1,j⟩ ψ u)+⟨i,j-1⟩ ψ u)+⟨i,j+1⟩ ψ u)+((h*h)*⟨i,j⟩ ψ f))); state=previous_sweep
! MoA: Same DNF, many ONFs - Fortran backend, gamma_col=7
! ONF Jacobi: one phase, all ivec parallel, two arrays, variable swap only
program poisson_jacobi_out
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  real, parameter :: h = 0.01587301678955555
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
        u_new(i,j) = (0.25*((((u(i-1,j)+u(i+1,j))+u(i,j-1))+u(i,j+1))+((h*h)*f(i,j))))
      end do
    end do
    ! variable swap only - no pointer swap in Fortran here, copy via temp
    tmp = u
    u = u_new
    u_new = tmp
  end do
  print *, "JACOBI u(32,32)=", u(32,32), " u(1,1)=", u(1,1)
end program poisson_jacobi_out
