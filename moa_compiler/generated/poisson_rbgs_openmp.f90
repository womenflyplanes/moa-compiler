! Generated OpenMP Fortran ONF - poisson_rbgs - from DNF
! DNF: ⟨i,j⟩ ψ u_next = (0.25*((((⟨i-1,j⟩ ψ u+⟨i+1,j⟩ ψ u)+⟨i,j-1⟩ ψ u)+⟨i,j+1⟩ ψ u)+((h*h)*⟨i,j⟩ ψ f))); state=red_then_black
! ONF RBGS: red phase parallel, barrier, black phase parallel
! OpenMP: parallel do inside one color is SAFE - same color points do not neighbour each other
program poisson_rbgs_omp
  use omp_lib
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  real, parameter :: h = 0.01587301678955555
  real :: u(0:nx+1, 0:ny+1), f(0:nx+1, 0:ny+1)
  integer :: i, j, it, color
  f = 1.0
  u = 0.0
  do it = 1, niter
    do color = 0, 1
      !$OMP PARALLEL DO COLLAPSE(2) PRIVATE(i,j) SHARED(u, f) IF(color==color)
      do j = 1, ny
        do i = 1, nx
          if (mod(i+j, 2) == color) then
            u(i,j) = (0.25*((((u(i-1,j)+u(i+1,j))+u(i,j-1))+u(i,j+1))+((h*h)*f(i,j))))
          end if
        end do
      end do
      !$OMP END PARALLEL DO
      !$OMP BARRIER
    end do
  end do
  print *, "RBGS-OMP u(32,32)=", u(32,32), " u(1,1)=", u(1,1)
end program poisson_rbgs_omp
