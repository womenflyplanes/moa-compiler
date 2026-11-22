! Generated Fortran ONF - poisson_jacobi - from DNF
! DNF: u_new psi ivec = 0.25 * sum_{ (di,dj) in {(-1,0),(1,0),(0,-1),(0,1)} } (ivec+(di,dj)) psi u + h^2 * (ivec psi f) # Poisson stencil DNF, rav + gamma_col
! MoA: Same DNF, many ONFs - Fortran backend, gamma_col=7
! ONF Jacobi: one phase, all ivec parallel, two arrays, variable swap only
! Burkardt problem: 1..nx grid, boundaries = u_exact, f = -pi^2(x^2+y^2)sin(pi x y) interior, u_exact on boundary
program poisson_jacobi_out
  implicit none
  integer :: nx, ny, it, it_max, i, j
  real(kind=8) :: dx, dy, x, y, diff, err_exact, u_mid, pi
  real(kind=8), allocatable :: u(:,:), u_new(:,:), f(:,:)
  character(len=32) :: arg
  logical :: converged
  pi = 3.141592653589793d0
  call get_command_argument(1, arg)
  read(arg, *) nx
  ny = nx
  it_max = 20000
  dx = 1.0d0 / dble(nx-1)
  dy = dx
  allocate(u(nx,ny), u_new(nx,ny), f(nx,ny))
  u = 0.0d0; u_new = 0.0d0; f = 0.0d0
  do j = 1, ny
    do i = 1, nx
      x = dble(i-1)*dx; y = dble(j-1)*dy
      if (i==1 .or. i==nx .or. j==1 .or. j==ny) then
        f(i,j) = sin(pi*x*y)
        u(i,j) = f(i,j); u_new(i,j) = f(i,j)
      else
        f(i,j) = pi*pi*(x*x+y*y)*sin(pi*x*y)
      end if
    end do
  end do
  converged = .false.
  do it = 1, it_max
    diff = 0.0d0
    do j = 1, ny
      do i = 1, nx
        if (i==1 .or. i==nx .or. j==1 .or. j==ny) then
          u_new(i,j) = f(i,j)
        else
          ! psi reduction: u_new psi ivec = 0.25*sum neighbours psi u + h^2 * ivec psi f
          u_new(i,j) = 0.25d0 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + dx*dy*f(i,j))
        end if
        diff = diff + (u_new(i,j)-u(i,j))**2
      end do
    end do
    diff = sqrt(diff / dble(nx*ny))
    u = u_new
    if (diff <= 1.0d-6) then; converged = .true.; exit; end if
  end do
  if (.not. converged) it = it_max
  err_exact = 0.0d0
  do j = 1, ny
    do i = 1, nx
      x = dble(i-1)*dx; y = dble(j-1)*dy
      err_exact = err_exact + (u(i,j)-sin(pi*x*y))**2
    end do
  end do
  err_exact = sqrt(err_exact / dble(nx*ny))
  u_mid = u((nx+1)/2,(ny+1)/2)
  print *, "GATE nx=", nx, " it=", it, " diff=", diff, " err_exact=", err_exact, " u_mid=", u_mid, " converged=", converged
end program poisson_jacobi_out
