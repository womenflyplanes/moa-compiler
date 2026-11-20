! Generated GPU Fortran ONF - poisson_rbgs - from DNF
! DNF: u_new psi ivec = 0.25 * sum_{ (di,dj) in {(-1,0),(1,0),(0,-1),(0,1)} } (ivec+(di,dj)) psi u + h^2 * (ivec psi f) # Poisson stencil DNF, rav + gamma_col
! ONF RBGS GPU: red kernel, barrier, black kernel. No barrier -> wrong answer.
! OpenACC: parallel loop inside one color is SAFE - same color points do not neighbour each other
program poisson_rbgs_gpu
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  real, parameter :: h = 1.0 / 63.0
  real :: u(0:nx+1, 0:ny+1), f(0:nx+1, 0:ny+1)
  integer :: i, j, it, color
  f = 1.0
  u = 0.0
  !$ACC DATA COPY(u, f)
  do it = 1, niter
    do color = 0, 1
      !$ACC PARALLEL LOOP COLLAPSE(2) PRESENT(u, f) PRIVATE(i,j)
      do j = 1, ny
        do i = 1, nx
          if (mod(i+j, 2) == color) then
            u(i,j) = 0.25 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h*h*f(i,j))
          end if
        end do
      end do
      !$ACC END PARALLEL LOOP
      !$ACC WAIT
    end do
  end do
  !$ACC END DATA
  print *, "RBGS-GPU u(32,32)=", u(32,32), " u(1,1)=", u(1,1)
end program poisson_rbgs_gpu
