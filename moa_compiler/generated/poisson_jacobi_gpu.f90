! Generated GPU Fortran ONF - poisson_jacobi - from DNF
! DNF: u_new psi ivec = 0.25 * sum_{ (di,dj) in {(-1,0),(1,0),(0,-1),(0,1)} } (ivec+(di,dj)) psi u + h^2 * (ivec psi f) # Poisson stencil DNF, rav + gamma_col
! ONF Jacobi GPU: one kernel per sweep, all ivec parallel, two arrays
! OpenACC: parallel loop collapse(2) is SAFE - reads u, writes u_new
! Burkardt problem: 1..nx grid, boundaries = u_exact, f = +pi^2(x^2+y^2)sin(pi x y) interior
program poisson_jacobi_gpu
  implicit none
  integer :: nx, ny, it, it_max, i, j
  real(kind=8) :: dx, dy, x, y, diff, diff_sum, err_exact, err_sum, u_mid, pi
  real(kind=8), allocatable :: u(:,:), u_new(:,:), f(:,:)
  character(len=32) :: arg
  logical :: converged
  pi = 3.141592653589793d0
  call get_command_argument(1, arg)
  read(arg, *) nx
  ny = nx; it_max = 20000
  dx = 1.0d0 / dble(nx-1); dy = dx
  allocate(u(nx,ny), u_new(nx,ny), f(nx,ny))
  u = 0.0d0; u_new = 0.0d0; f = 0.0d0
  do j = 1, ny
    do i = 1, nx
      x = dble(i-1)*dx; y = dble(j-1)*dy
      if (i==1 .or. i==nx .or. j==1 .or. j==ny) then
        f(i,j) = sin(pi*x*y); u(i,j) = f(i,j); u_new(i,j) = f(i,j)
      else
        f(i,j) = pi*pi*(x*x+y*y)*sin(pi*x*y)
      end if
    end do
  end do
  converged = .false.
  !$ACC DATA COPY(u, u_new, f)
  do it = 1, it_max
    diff_sum = 0.0d0
    !$ACC PARALLEL LOOP COLLAPSE(2) PRESENT(u, u_new, f) PRIVATE(i,j) REDUCTION(+:diff_sum)
    do j = 1, ny
      do i = 1, nx
        if (i==1 .or. i==nx .or. j==1 .or. j==ny) then
          u_new(i,j) = f(i,j)
        else
          u_new(i,j) = 0.25d0 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + dx*dy*f(i,j))
        end if
        diff_sum = diff_sum + (u_new(i,j)-u(i,j))**2
      end do
    end do
    !$ACC END PARALLEL LOOP
    diff = sqrt(diff_sum / dble(nx*ny))
    !$ACC KERNELS PRESENT(u, u_new)
    u = u_new
    !$ACC END KERNELS
    if (diff <= 1.0d-6) then; converged = .true.; exit; end if
  end do
  !$ACC END DATA
  if (.not. converged) it = it_max
  err_sum = 0.0d0
  do j = 1, ny
    do i = 1, nx
      x = dble(i-1)*dx; y = dble(j-1)*dy
      err_sum = err_sum + (u(i,j)-sin(pi*x*y))**2
    end do
  end do
  err_exact = sqrt(err_sum / dble(nx*ny))
  u_mid = u((nx+1)/2,(ny+1)/2)
  print *, "GATE nx=", nx, " it=", it, " diff=", diff, " err_exact=", err_exact, " u_mid=", u_mid, " converged=", converged
end program poisson_jacobi_gpu
