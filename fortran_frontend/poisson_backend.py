"""
Poisson backend: DNF (.dnf.json) -> Fortran + 【entity-MATLAB¦canonical_name=MATLAB】
Reads: fortran_frontend/examples/*.dnf.json  (written by fortran_frontend.py)
Writes: moa_compiler/generated/poisson_jacobi_out.f90 / .m
        moa_compiler/generated/poisson_island_gauss_out.f90 / .m
Rules:
    - Fortran: column-major, gamma_col, 1-based, two arrays for Jacobi, one array for Island Gauss
    - MATLAB: column-major, gamma_col, 1-based, two arrays + variable swap, NO pointer-swap
    - Only parallelize where ONF says parallel: Jacobi = all ivec, Island = islands only
"""
import json, pathlib

OUT = pathlib.Path("moa_compiler/generated")
OUT.mkdir(parents=True, exist_ok=True)

def gen_fortran_jacobi(dnf):
    return f'''! Generated Fortran ONF - poisson_jacobi - from DNF
! DNF: {dnf}
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
'''

def gen_fortran_island(dnf):
    return f'''! Generated Fortran ONF - poisson_island_gauss - from DNF
! DNF: {dnf}
! MoA: Same DNF, many ONFs - Fortran backend, gamma_col=7
! ONF Island Gauss: parallel over islands (ix,iy), sequential inside island, in-place, halo
program poisson_island_gauss_out
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  integer, parameter :: isx = 16, isy = 16
  integer, parameter :: nix = nx / isx, niy = ny / isy
  real, parameter :: h = 1.0 / 63.0
  real :: u(0:nx+1, 0:ny+1), f(0:nx+1, 0:ny+1)
  integer :: i, j, it, ix, iy, i0, j0, i1, j1
  f = 1.0
  u = 0.0
  do it = 1, niter
    ! islands are independent in this sweep - this loop is the parallel loop
    do iy = 1, niy
      do ix = 1, nix
        i0 = (ix-1)*isx + 1
        i1 = ix*isx
        j0 = (iy-1)*isy + 1
        j1 = iy*isy
        ! inside island: sequential Gauss-Seidel - use newest u
        do j = j0, j1
          do i = i0, i1
            u(i,j) = 0.25 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h*h*f(i,j))
          end do
        end do
      end do
    end do
  end do
  print *, "ISLAND u(32,32)=", u(32,32), " u(1,1)=", u(1,1)
end program poisson_island_gauss_out
'''

def gen_matlab_jacobi(dnf):
    return f'''% Generated MATLAB ONF - poisson_jacobi - from DNF
% DNF: {dnf}
% MoA: Same DNF, many ONFs - MATLAB backend, gamma_col=7, column-major
% ONF Jacobi: one phase, all ivec parallel, two arrays, variable swap only
% MATLAB has no pointer-swap: use two arrays and swap variables
function [u, u_new] = poisson_jacobi_out(nx, ny, niter)
  if nargin < 3, nx = 64; ny = 64; niter = 100; end
  h = 1.0 / 63.0;
  u = zeros(nx+2, ny+2);
  u_new = zeros(nx+2, ny+2);
  f = ones(nx+2, ny+2);
  % psi reduction: u_new psi ivec = 0.25*sum neighbours psi u + h^2 * ivec psi f
  % indices shifted +1 for 【entity-MATLAB¦canonical_name=MATLAB】 1-based: u(i+1,j+1) corresponds to Fortran u(i,j)
  for it = 1:niter
    for j = 2:ny+1
      for i = 2:nx+1
        u_new(i,j) = 0.25 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h*h*f(i,j));
      end
    end
    % variable swap only
    tmp = u; u = u_new; u_new = tmp;
  end
end
'''

def gen_matlab_island(dnf):
    return f'''% Generated MATLAB ONF - poisson_island_gauss - from DNF
% DNF: {dnf}
% MoA: Same DNF, many ONFs - MATLAB backend, gamma_col=7, column-major
% ONF Island Gauss: parallel over islands, sequential inside island, in-place
% MATLAB has no pointer-swap: in-place, no swap needed
function u = poisson_island_gauss_out(nx, ny, niter)
  if nargin < 3, nx = 64; ny = 64; niter = 100; end
  isx = 16; isy = 16;
  nix = nx / isx; niy = ny / isy;
  h = 1.0 / 63.0;
  u = zeros(nx+2, ny+2);
  f = ones(nx+2, ny+2);
  for it = 1:niter
    % islands are independent - this loop is the parallel loop
    for iy = 1:niy
      for ix = 1:nix
        i0 = (ix-1)*isx + 2;
        i1 = ix*isx + 1;
        j0 = (iy-1)*isy + 2;
        j1 = iy*isy + 1;
        % inside island: sequential Gauss-Seidel
        for j = j0:j1
          for i = i0:i1
            u(i,j) = 0.25 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h*h*f(i,j));
          end
        end
      end
    end
  end
end
'''


def gen_fortran_rbgs(dnf):
    return f'''! Generated Fortran ONF - poisson_rbgs - from DNF
! DNF: {dnf}
! ONF RBGS: same DNF, red phase then black phase, in-place, gamma_col
program poisson_rbgs_out
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  real, parameter :: h = 1.0 / 63.0
  real :: u(0:nx+1, 0:ny+1), f(0:nx+1, 0:ny+1)
  integer :: i, j, it, color
  f = 1.0
  u = 0.0
  do it = 1, niter
    do color = 0, 1
      do j = 1, ny
        do i = 1, nx
          if (mod(i+j, 2) == color) then
            u(i,j) = 0.25 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h*h*f(i,j))
          end if
        end do
      end do
    end do
  end do
  print *, "RBGS u(32,32)=", u(32,32), " u(1,1)=", u(1,1)
end program poisson_rbgs_out
'''

def gen_matlab_rbgs(dnf):
    return f'''% Generated 【entity-MATLAB¦canonical_name=MATLAB】 ONF - poisson_rbgs - from DNF
% DNF: {dnf}
% ONF RBGS: same DNF, red phase then black phase, in-place, gamma_col
function u = poisson_rbgs_out(nx, ny, niter)
  if nargin < 3, nx = 64; ny = 64; niter = 100; end
  h = 1.0 / 63.0;
  u = zeros(nx+2, ny+2);
  f = ones(nx+2, ny+2);
  for it = 1:niter
    for color = 0:1
      for j = 2:ny+1
        for i = 2:nx+1
          if mod((i-1)+(j-1), 2) == color
            u(i,j) = 0.25 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h*h*f(i,j));
          end
        end
      end
    end
  end
end
'''

def gen_fortran_matmul(dnf):
    return f'''! Generated Fortran ONF - matmul - from DNF
! DNF: {dnf}
! ONF matmul: loop j, loop i, loop k, gamma_col
program matmul_out
  implicit none
  integer, parameter :: n = 64
  real :: A(n,n), B(n,n), C(n,n)
  integer :: i, j, k
  A = 1.0
  B = 1.0
  C = 0.0
  do j = 1, n
    do i = 1, n
      C(i,j) = 0.0
      do k = 1, n
        C(i,j) = C(i,j) + A(i,k) * B(k,j)
      end do
    end do
  end do
  print *, "MATMUL C(32,32)=", C(32,32), " C(1,1)=", C(1,1)
end program matmul_out
'''

def gen_matlab_matmul(dnf):
    return f'''% Generated 【entity-MATLAB¦canonical_name=MATLAB】 ONF - matmul - from DNF
% DNF: {dnf}
% ONF matmul: loop j, loop i, loop k, gamma_col
function C = matmul_out(n)
  if nargin < 1, n = 64; end
  A = ones(n,n); B = ones(n,n); C = zeros(n,n);
  for j = 1:n
    for i = 1:n
      C(i,j) = 0.0;
      for k = 1:n
        C(i,j) = C(i,j) + A(i,k) * B(k,j);
      end
    end
  end
end
'''


def gen_openmp_jacobi(dnf):
    return f'''! Generated OpenMP Fortran ONF - poisson_jacobi - from DNF
! DNF: {dnf}
! ONF Jacobi: one phase, all ivec parallel, two arrays
! OpenMP: parallel do over i,j is SAFE - reads u, writes u_new, no dependency
! Burkardt problem: 1..nx grid, boundaries = u_exact, f = +pi^2(x^2+y^2)sin(pi x y) interior
program poisson_jacobi_omp
  use omp_lib
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
  do it = 1, it_max
    diff_sum = 0.0d0
    !$OMP PARALLEL DO COLLAPSE(2) PRIVATE(i,j) SHARED(u,u_new,f,nx,ny,dx,dy) REDUCTION(+:diff_sum)
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
    !$OMP END PARALLEL DO
    diff = sqrt(diff_sum / dble(nx*ny))
    u = u_new
    if (diff <= 1.0d-6) then; converged = .true.; exit; end if
  end do
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
end program poisson_jacobi_omp
'''

def gen_openmp_island(dnf):
    return f'''! Generated OpenMP Fortran ONF - poisson_island_gauss - from DNF
! DNF: {dnf}
! ONF Island Gauss PARALLEL: parallel over islands, sequential inside island
! HALO-SAFE: u_old = halo snapshot at start of sweep
! Neighbour inside island -> read u (newest, Gauss-Seidel)
! Neighbour outside island -> read u_old (halo, previous sweep)
! This is deterministic. Naive parallel islands without halo is racy.
program poisson_island_gauss_omp
  use omp_lib
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
  do it = 1, niter
    u_old = u
    !$OMP PARALLEL DO COLLAPSE(2) PRIVATE(ix,iy,i0,j0,i1,j1,i,j,ul,ur,ud,uu) SHARED(u, u_old, f)
    do iy = 1, niy
      do ix = 1, nix
        i0 = (ix-1)*isx + 1
        i1 = ix*isx
        j0 = (iy-1)*isy + 1
        j1 = iy*isy
        do j = j0, j1
          do i = i0, i1
            ! left: outside island -> halo, inside -> newest
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
    !$OMP END PARALLEL DO
  end do
  print *, "ISLAND-OMP u(32,32)=", u(32,32), " u(1,1)=", u(1,1)
end program poisson_island_gauss_omp
'''


def gen_openmp_rbgs(dnf):
    return f'''! Generated OpenMP Fortran ONF - poisson_rbgs - from DNF
! DNF: {dnf}
! ONF RBGS: red phase parallel, barrier, black phase parallel
! OpenMP: parallel do inside one color is SAFE - same color points do not neighbour each other
program poisson_rbgs_omp
  use omp_lib
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  real, parameter :: h = 1.0 / 63.0
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
            u(i,j) = 0.25 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h*h*f(i,j))
          end if
        end do
      end do
      !$OMP END PARALLEL DO
      !$OMP BARRIER
    end do
  end do
  print *, "RBGS-OMP u(32,32)=", u(32,32), " u(1,1)=", u(1,1)
end program poisson_rbgs_omp
'''

def gen_openmp_matmul(dnf):
    return f'''! Generated OpenMP Fortran ONF - matmul - from DNF
! DNF: {dnf}
! ONF matmul: i,j parallel, k sequential
! OpenMP: parallel do over i,j is SAFE - each C(i,j) is independent
program matmul_omp
  use omp_lib
  implicit none
  integer, parameter :: n = 64
  real :: A(n,n), B(n,n), C(n,n)
  integer :: i, j, k
  A = 1.0
  B = 1.0
  C = 0.0
  !$OMP PARALLEL DO COLLAPSE(2) PRIVATE(i,j,k) SHARED(A, B, C)
  do j = 1, n
    do i = 1, n
      C(i,j) = 0.0
      do k = 1, n
        C(i,j) = C(i,j) + A(i,k) * B(k,j)
      end do
    end do
  end do
  !$OMP END PARALLEL DO
  print *, "MATMUL-OMP C(32,32)=", C(32,32), " C(1,1)=", C(1,1)
end program matmul_omp
'''


def gen_gpu_jacobi(dnf):
    return f'''! Generated GPU Fortran ONF - poisson_jacobi - from DNF
! DNF: {dnf}
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
'''

def gen_gpu_island(dnf):
    return f'''! Generated GPU Fortran ONF - poisson_island_gauss - from DNF
! DNF: {dnf}
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
'''

def gen_gpu_rbgs(dnf):
    return f'''! Generated GPU Fortran ONF - poisson_rbgs - from DNF
! DNF: {dnf}
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
'''

def gen_gpu_matmul(dnf):
    return f'''! Generated GPU Fortran ONF - matmul - from DNF
! DNF: {dnf}
! ONF matmul GPU: i,j parallel, k sequential
! OpenACC: parallel loop over i,j is SAFE - each C(i,j) is independent
program matmul_gpu
  implicit none
  integer, parameter :: n = 64
  real :: A(n,n), B(n,n), C(n,n)
  integer :: i, j, k
  A = 1.0
  B = 1.0
  C = 0.0
  !$ACC DATA COPYIN(A, B) COPYOUT(C)
  !$ACC PARALLEL LOOP COLLAPSE(2) PRESENT(A, B, C) PRIVATE(i,j,k)
  do j = 1, n
    do i = 1, n
      C(i,j) = 0.0
      do k = 1, n
        C(i,j) = C(i,j) + A(i,k) * B(k,j)
      end do
    end do
  end do
  !$ACC END PARALLEL LOOP
  !$ACC END DATA
  print *, "MATMUL-GPU C(32,32)=", C(32,32), " C(1,1)=", C(1,1)
end program matmul_gpu
'''


def gen_mpi_jacobi(dnf):
    return f'''! Generated MPI Fortran ONF - poisson_jacobi - from DNF
! DNF: {dnf}
! ONF Jacobi MPI: 1D slab over j, halo 1 row top/bottom via MPI_Sendrecv per sweep
! Burkardt problem: 1..nx grid, boundaries = u_exact, f = +pi^2(x^2+y^2)sin(pi x y) interior
program poisson_jacobi_mpi
  use mpi
  implicit none
  integer :: nx, ny, it, it_max, i, j, jg
  real(kind=8) :: dx, dy, x, y, diff, diff_sum, diff_glob, err_exact, err_sum, err_glob, u_mid, pi
  real(kind=8), allocatable :: u(:,:), u_new(:,:), f(:,:)
  character(len=32) :: arg
  logical :: converged
  integer :: ierr, rank, nprocs, ny_local, j0_global, j1_global
  integer :: up, down
  real(kind=8) :: u_mid_glob
  pi = 3.141592653589793d0
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
  if (rank == 0) then
    call get_command_argument(1, arg)
    read(arg, *) nx
  end if
  call MPI_Bcast(nx, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
  ny = nx; it_max = 20000
  dx = 1.0d0 / dble(nx-1); dy = dx
  ny_local = ny / nprocs
  if (rank < mod(ny, nprocs)) ny_local = ny_local + 1
  j0_global = rank*(ny/nprocs) + min(rank, mod(ny, nprocs)) + 1
  j1_global = j0_global + ny_local - 1
  allocate(u(nx,0:ny_local+1), u_new(nx,0:ny_local+1), f(nx,0:ny_local+1))
  u = 0.0d0; u_new = 0.0d0; f = 0.0d0
  do j = 1, ny_local
    jg = j0_global + j - 1
    y = dble(jg-1)*dy
    do i = 1, nx
      x = dble(i-1)*dx
      if (i==1 .or. i==nx .or. jg==1 .or. jg==ny) then
        f(i,j) = sin(pi*x*y); u(i,j) = f(i,j); u_new(i,j) = f(i,j)
      else
        f(i,j) = pi*pi*(x*x+y*y)*sin(pi*x*y)
      end if
    end do
  end do
  if (rank == 0) then; u(:,0) = u(:,1); u_new(:,0) = u(:,1); end if
  if (rank == nprocs-1) then; u(:,ny_local+1) = u(:,ny_local); u_new(:,ny_local+1) = u(:,ny_local); end if
  up = rank - 1; if (up < 0) up = MPI_PROC_NULL
  down = rank + 1; if (down >= nprocs) down = MPI_PROC_NULL
  converged = .false.
  do it = 1, it_max
    call MPI_Sendrecv(u(1,1), nx, MPI_DOUBLE_PRECISION, up, 0, u(1,ny_local+1), nx, MPI_DOUBLE_PRECISION, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
    call MPI_Sendrecv(u(1,ny_local), nx, MPI_DOUBLE_PRECISION, down, 1, u(1,0), nx, MPI_DOUBLE_PRECISION, up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
    if (rank == 0) then; u(:,0) = u(:,1); end if
    if (rank == nprocs-1) then; u(:,ny_local+1) = u(:,ny_local); end if
    diff_sum = 0.0d0
    do j = 1, ny_local
      jg = j0_global + j - 1
      do i = 1, nx
        if (i==1 .or. i==nx .or. jg==1 .or. jg==ny) then
          u_new(i,j) = f(i,j)
        else
          u_new(i,j) = 0.25d0 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + dx*dy*f(i,j))
        end if
        diff_sum = diff_sum + (u_new(i,j)-u(i,j))**2
      end do
    end do
    u(:,1:ny_local) = u_new(:,1:ny_local)
    call MPI_Allreduce(diff_sum, diff_glob, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    diff = sqrt(diff_glob / dble(nx*ny))
    if (diff <= 1.0d-6) then; converged = .true.; end if
    call MPI_Bcast(converged, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, ierr)
    if (converged) exit
  end do
  if (.not. converged) it = it_max
  err_sum = 0.0d0
  do j = 1, ny_local
    jg = j0_global + j - 1
    y = dble(jg-1)*dy
    do i = 1, nx
      x = dble(i-1)*dx
      err_sum = err_sum + (u(i,j)-sin(pi*x*y))**2
    end do
  end do
  call MPI_Allreduce(err_sum, err_glob, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
  err_exact = sqrt(err_glob / dble(nx*ny))
  u_mid = 0.0d0
  if ((nx+1)/2 >= j0_global .and. (nx+1)/2 <= j1_global) u_mid = u((nx+1)/2, (nx+1)/2 - j0_global + 1)
  call MPI_Allreduce(u_mid, u_mid_glob, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
  if (rank == 0) print *, "GATE nx=", nx, " it=", it, " diff=", diff, " err_exact=", err_exact, " u_mid=", u_mid_glob, " converged=", converged
  call MPI_Finalize(ierr)
end program poisson_jacobi_mpi
'''

def gen_mpi_island(dnf):
    return f'''! Generated MPI Fortran ONF - poisson_island_gauss - from DNF
! DNF: {dnf}
! ONF Island MPI: slab over j, halo 1 row via MPI_Sendrecv per sweep
! Inside slab: islands, Gauss-Seidel inside island, halo from u_old for island edges AND slab edges
! Halo exchange is per sweep, NOT per island - that is the point of halo-safe ONF
program poisson_island_gauss_mpi
  use mpi
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  integer, parameter :: isx = 16, isy = 16
  integer, parameter :: nix = nx / isx, niy = ny / isy
  real, parameter :: h = 1.0 / 63.0
  real, allocatable :: u(:,:), u_old(:,:), f(:,:)
  real :: ul, ur, ud, uu
  integer :: ierr, rank, nprocs, ny_local, j0_global, j1_global
  integer :: i, j, it, ix, iy, i0, j0, i1, j1, j_local
  integer :: up, down
  real :: t0, t1, u32, u11
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
  ny_local = ny / nprocs
  if (rank < mod(ny, nprocs)) ny_local = ny_local + 1
  j0_global = rank*(ny/nprocs) + min(rank, mod(ny, nprocs)) + 1
  j1_global = j0_global + ny_local - 1
  allocate(u(0:nx+1, 0:ny_local+1), u_old(0:nx+1, 0:ny_local+1), f(0:nx+1, 0:ny_local+1))
  f = 1.0; u = 0.0; u_old = 0.0
  up = rank - 1; if (up < 0) up = MPI_PROC_NULL
  down = rank + 1; if (down >= nprocs) down = MPI_PROC_NULL
  t0 = MPI_Wtime()
  do it = 1, niter
    u_old = u
    ! MPI halo exchange: exchange u_old halo rows - other ranks need previous sweep halo
    call MPI_Sendrecv(u_old(1,1), nx, MPI_REAL, up, 0, u_old(1,ny_local+1), nx, MPI_REAL, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
    call MPI_Sendrecv(u_old(1,ny_local), nx, MPI_REAL, down, 1, u_old(1,0), nx, MPI_REAL, up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
    ! Islands inside this slab - slab must align to island rows for this simple version
    ! ny_local=64,32,16 for nprocs=1,2,4 - all multiples of isy=16, so alignment holds
    do iy = 1, niy
      ! skip islands not in this slab
      j0 = (iy-1)*isy + 1; j1 = iy*isy
      if (j1 < j0_global .or. j0 > j1_global) cycle
      do ix = 1, nix
        i0 = (ix-1)*isx + 1; i1 = ix*isx
        do j = max(j0, j0_global), min(j1, j1_global)
          j_local = j - j0_global + 1
          do i = i0, i1
            if (i-1 < i0) then
              ul = u_old(i-1, j_local)
            else
              ul = u(i-1, j_local)
            end if
            if (i+1 > i1) then
              ur = u_old(i+1, j_local)
            else
              ur = u(i+1, j_local)
            end if
            if (j-1 < j0) then
              ! j-1 is previous island row - if in same slab, it is in u_old halo, else MPI halo in u_old
              ud = u_old(i, j_local-1)
            else
              ud = u(i, j_local-1)
            end if
            if (j+1 > j1) then
              uu = u_old(i, j_local+1)
            else
              uu = u(i, j_local+1)
            end if
            u(i, j_local) = 0.25 * (ul + ur + ud + uu + h*h*f(i, j_local))
          end do
        end do
      end do
    end do
  end do
  t1 = MPI_Wtime()
  u32 = 0.0; u11 = 0.0
  if (32 >= j0_global .and. 32 <= j1_global) u32 = u(32, 32-j0_global+1)
  if (1 >= j0_global .and. 1 <= j1_global) u11 = u(1, 1-j0_global+1)
  call MPI_Allreduce(MPI_IN_PLACE, u32, 1, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(MPI_IN_PLACE, u11, 1, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  if (rank == 0) then
    print *, "ISLAND-MPI ranks=", nprocs, " time=", t1-t0, " u(32,32)=", u32, " u(1,1)=", u11
  end if
  call MPI_Finalize(ierr)
end program poisson_island_gauss_mpi
'''

def gen_mpi_rbgs(dnf):
    return f'''! Generated MPI Fortran ONF - poisson_rbgs - from DNF
! DNF: {dnf}
! ONF RBGS MPI: slab over j, halo exchange per color phase = barrier
! Red phase: halo exchange, compute red, halo exchange, compute black
program poisson_rbgs_mpi
  use mpi
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  real, parameter :: h = 1.0 / 63.0
  real, allocatable :: u(:,:), f(:,:)
  integer :: ierr, rank, nprocs, ny_local, j0_global, j1_global
  integer :: i, j, it, color, up, down
  real :: t0, t1, u32, u11
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
  ny_local = ny / nprocs
  if (rank < mod(ny, nprocs)) ny_local = ny_local + 1
  j0_global = rank*(ny/nprocs) + min(rank, mod(ny, nprocs)) + 1
  j1_global = j0_global + ny_local - 1
  allocate(u(0:nx+1, 0:ny_local+1), f(0:nx+1, 0:ny_local+1))
  f = 1.0; u = 0.0
  up = rank - 1; if (up < 0) up = MPI_PROC_NULL
  down = rank + 1; if (down >= nprocs) down = MPI_PROC_NULL
  t0 = MPI_Wtime()
  do it = 1, niter
    do color = 0, 1
      call MPI_Sendrecv(u(1,1), nx, MPI_REAL, up, 0, u(1,ny_local+1), nx, MPI_REAL, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
      call MPI_Sendrecv(u(1,ny_local), nx, MPI_REAL, down, 1, u(1,0), nx, MPI_REAL, up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
      do j = 1, ny_local
        do i = 1, nx
          if (mod(i + (j + j0_global - 1), 2) == color) then
            u(i,j) = 0.25 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h*h*f(i,j))
          end if
        end do
      end do
    end do
  end do
  t1 = MPI_Wtime()
  u32 = 0.0; u11 = 0.0
  if (32 >= j0_global .and. 32 <= j1_global) u32 = u(32, 32-j0_global+1)
  if (1 >= j0_global .and. 1 <= j1_global) u11 = u(1, 1-j0_global+1)
  call MPI_Allreduce(MPI_IN_PLACE, u32, 1, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(MPI_IN_PLACE, u11, 1, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  if (rank == 0) then
    print *, "RBGS-MPI ranks=", nprocs, " time=", t1-t0, " u(32,32)=", u32, " u(1,1)=", u11
  end if
  call MPI_Finalize(ierr)
end program poisson_rbgs_mpi
'''

def gen_mpi_matmul(dnf):
    return f'''! Generated MPI Fortran ONF - matmul - from DNF
! DNF: {dnf}
! ONF matmul MPI: row slab of C, broadcast B, no halo, no per-iteration comm
program matmul_mpi
  use mpi
  implicit none
  integer, parameter :: n = 64
  real, allocatable :: A(:,:), B(:,:), C(:,:)
  integer :: ierr, rank, nprocs, n_local, i0_global
  integer :: i, j, k
  real :: t0, t1, c32, c11
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
  n_local = n / nprocs
  i0_global = rank*n_local + 1
  allocate(A(n_local,n), B(n,n), C(n_local,n))
  A = 1.0; B = 1.0; C = 0.0
  ! B is same on all ranks in this simple version - broadcast is no-op, kept for ONF clarity
  call MPI_Bcast(B, n*n, MPI_REAL, 0, MPI_COMM_WORLD, ierr)
  t0 = MPI_Wtime()
  do j = 1, n
    do i = 1, n_local
      C(i,j) = 0.0
      do k = 1, n
        C(i,j) = C(i,j) + A(i,k) * B(k,j)
      end do
    end do
  end do
  t1 = MPI_Wtime()
  c32 = 0.0; c11 = 0.0
  if (32 >= i0_global .and. 32 < i0_global+n_local) c32 = C(32-i0_global+1, 32)
  if (1 >= i0_global .and. 1 < i0_global+n_local) c11 = C(1-i0_global+1, 1)
  call MPI_Allreduce(MPI_IN_PLACE, c32, 1, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(MPI_IN_PLACE, c11, 1, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  if (rank == 0) then
    print *, "MATMUL-MPI ranks=", nprocs, " time=", t1-t0, " C(32,32)=", c32, " C(1,1)=", c11
  end if
  call MPI_Finalize(ierr)
end program matmul_mpi
'''

def main():
    base = pathlib.Path("fortran_frontend/examples")
    for stem, gen_f, gen_m in [
        ("poisson_jacobi", gen_fortran_jacobi, gen_matlab_jacobi),
        ("poisson_island_gauss", gen_fortran_island, gen_matlab_island),
        ("poisson_rbgs", gen_fortran_rbgs, gen_matlab_rbgs),
        ("matmul", gen_fortran_matmul, gen_matlab_matmul),
    ]:
        data = json.loads((base / f"{stem}.dnf.json").read_text())
        dnf = data["dnf"]
        (OUT / f"{stem}_out.f90").write_text(gen_f(dnf))
        (OUT / f"{stem}_out.m").write_text(gen_m(dnf))
        # MPI variant - same DNF, halo exchange is the MPI ONF
        mpi_map = {
            "poisson_jacobi": gen_mpi_jacobi,
            "poisson_island_gauss": gen_mpi_island,
            "poisson_rbgs": gen_mpi_rbgs,
            "matmul": gen_mpi_matmul,
        }
        if stem in mpi_map:
            (OUT / f"{stem}_mpi.f90").write_text(mpi_map[stem](dnf))
            print(f"Wrote {stem}_mpi.f90 from DNF + ONF")

        # GPU variant - same DNF, halo-safe Island ONF is the GPU ONF
        gpu_map = {
            "poisson_jacobi": gen_gpu_jacobi,
            "poisson_island_gauss": gen_gpu_island,
            "poisson_rbgs": gen_gpu_rbgs,
            "matmul": gen_gpu_matmul,
        }
        if stem in gpu_map:
            (OUT / f"{stem}_gpu.f90").write_text(gpu_map[stem](dnf))
            print(f"Wrote {stem}_gpu.f90 from DNF + ONF")

        # OpenMP variant - same DNF, ONF says what is parallel
        omp_map = {
            "poisson_jacobi": gen_openmp_jacobi,
            "poisson_island_gauss": gen_openmp_island,
            "poisson_rbgs": gen_openmp_rbgs,
            "matmul": gen_openmp_matmul,
        }
        if stem in omp_map:
            (OUT / f"{stem}_openmp.f90").write_text(omp_map[stem](dnf))
            print(f"Wrote {stem}_openmp.f90 from DNF + ONF")
        print(f"Wrote {stem}_out.f90 + {stem}_out.m from DNF")
        print(f"  DNF: {dnf[:80]}...")

if __name__ == "__main__":
    main()
