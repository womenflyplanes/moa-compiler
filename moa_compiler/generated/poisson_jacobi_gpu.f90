! Generated GPU Fortran ONF - poisson_jacobi - from DNF
! DNF: ⟨i,j⟩ ψ u_next = (0.25*((((⟨i-1,j⟩ ψ u+⟨i+1,j⟩ ψ u)+⟨i,j-1⟩ ψ u)+⟨i,j+1⟩ ψ u)+((h*h)*⟨i,j⟩ ψ f))); state=previous_sweep
! ONF Jacobi GPU: one kernel per sweep, all ivec parallel, two arrays
! OpenACC: parallel loop collapse(2) is SAFE - reads u, writes u_new
program poisson_jacobi_gpu
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  real, parameter :: h = 0.01587301678955555
  real :: u(0:nx+1, 0:ny+1), u_new(0:nx+1, 0:ny+1), f(0:nx+1, 0:ny+1)
  real :: tmp(0:nx+1, 0:ny+1)
  integer :: i, j, it
  f = 1.0
  u = 0.0
  u_new = 0.0
  !$ACC DATA COPY(u, u_new, f) CREATE(tmp)
  do it = 1, niter
    !$ACC PARALLEL LOOP COLLAPSE(2) PRESENT(u, u_new, f)
    do j = 1, ny
      do i = 1, nx
        u_new(i,j) = (0.25*((((u(i-1,j)+u(i+1,j))+u(i,j-1))+u(i,j+1))+((h*h)*f(i,j))))
      end do
    end do
    !$ACC END PARALLEL LOOP
    !$ACC KERNELS PRESENT(u, u_new, tmp)
    tmp = u
    u = u_new
    u_new = tmp
    !$ACC END KERNELS
  end do
  !$ACC END DATA
  print *, "JACOBI-GPU u(32,32)=", u(32,32), " u(1,1)=", u(1,1)
end program poisson_jacobi_gpu
