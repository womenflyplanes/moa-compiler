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
program poisson_jacobi_gpu
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  real, parameter :: h = 1.0 / 63.0
  real :: u(0:nx+1, 0:ny+1), u_new(0:nx+1, 0:ny+1), f(0:nx+1, 0:ny+1)
  real :: tmp(0:nx+1, 0:ny+1)
  integer :: i, j, it
  f = 1.0
  u = 0.0
  u_new = 0.0
  !$ACC DATA COPY(u, u_new, f, tmp)
  do it = 1, niter
    !$ACC PARALLEL LOOP COLLAPSE(2) PRESENT(u, u_new, f)
    do j = 1, ny
      do i = 1, nx
        u_new(i,j) = 0.25 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h*h*f(i,j))
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
! Each rank: owns ny_local rows, exchanges halo, computes its slab, no other comm
program poisson_jacobi_mpi
  use mpi
  implicit none
  integer, parameter :: nx = 64, ny = 64, niter = 100
  real, parameter :: h = 1.0 / 63.0
  real, allocatable :: u(:,:), u_new(:,:), f(:,:)
  integer :: ierr, rank, nprocs, ny_local, j0_global, j1_global
  integer :: i, j, it, jg, up, down
  integer :: req(4)
  real :: t0, t1
  real :: u32, u11
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
  ! ny must divide nprocs for this simple slab - 64 / 1,2,4 = OK
  ny_local = ny / nprocs
  j0_global = rank*ny_local + 1
  j1_global = (rank+1)*ny_local
  allocate(u(0:nx+1, 0:ny_local+1), u_new(0:nx+1, 0:ny_local+1), f(0:nx+1, 0:ny_local+1))
  f = 1.0; u = 0.0; u_new = 0.0
  up = rank - 1; if (up < 0) up = MPI_PROC_NULL
  down = rank + 1; if (down >= nprocs) down = MPI_PROC_NULL
  t0 = MPI_Wtime()
  do it = 1, niter
    ! halo exchange: send bottom row to down, top row to up, recv halo rows
    call MPI_Sendrecv(u(1,1), nx, MPI_REAL, up, 0, u(1,ny_local+1), nx, MPI_REAL, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
    call MPI_Sendrecv(u(1,ny_local), nx, MPI_REAL, down, 1, u(1,0), nx, MPI_REAL, up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
    do j = 1, ny_local
      do i = 1, nx
        u_new(i,j) = 0.25 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h*h*f(i,j))
      end do
    end do
    ! swap u / u_new by copy - simple, correct, not the perf bottleneck at nx=64
    u(0:nx+1, 0:ny_local+1) = u_new(0:nx+1, 0:ny_local+1)
  end do
  t1 = MPI_Wtime()
  ! gather u(32,32) and u(1,1) - rank that owns them sends to rank 0
  u32 = 0.0; u11 = 0.0
  if (32 >= j0_global .and. 32 <= j1_global) u32 = u(32, 32-j0_global+1)
  if (1 >= j0_global .and. 1 <= j1_global) u11 = u(1, 1-j0_global+1)
  call MPI_Allreduce(MPI_IN_PLACE, u32, 1, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(MPI_IN_PLACE, u11, 1, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
  if (rank == 0) then
    print *, "JACOBI-MPI ranks=", nprocs, " time=", t1-t0, " u(32,32)=", u32, " u(1,1)=", u11
  end if

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

# Every public emitter validates and lowers a structured arithmetic AST.
from lower_semantic_ir import wrap_generator
from semantic_ir import snapshot_islands
for _name in tuple(globals()):
    if _name.startswith('gen_'):
        globals()[_name] = wrap_generator(_name, globals()[_name])


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot-islands', action='store_true', help='Explicitly choose a different island finite-sweep state contract for parallel variants')
    args=parser.parse_args()
    if not args.snapshot_islands:
        parser.error('Full fixture generation requires --snapshot-islands: parallel island sweeps have a distinct state contract')
    base=pathlib.Path('fortran_frontend/examples')
    for stem,family in [('poisson_jacobi','jacobi'),('poisson_island_gauss','island'),('poisson_rbgs','rbgs'),('matmul','matmul')]:
        data=json.loads((base/f'{stem}.dnf.json').read_text())
        for mode,prefix,suffix in [('serial','gen_fortran_','out.f90'),('matlab','gen_matlab_','out.m'),('openmp','gen_openmp_','openmp.f90'),('gpu','gen_gpu_','gpu.f90'),('mpi','gen_mpi_','mpi.f90')]:
            contract=data
            if family=='island' and mode in ('openmp','gpu','mpi'):
                if not args.snapshot_islands:
                    print('SKIP',stem,mode,': explicit --snapshot-islands required');continue
                contract=snapshot_islands(data)
            name=prefix+family
            target=OUT/f'{stem}_{suffix}'
            source=globals()[name](contract)
            target.write_text(source)
            import hashlib
            record=dict(contract)
            from moa_stages import stages
            record['stages']=stages(contract,'openacc' if mode=='gpu' else mode)
            record['emission']={'mode':mode,'generated_sha256':hashlib.sha256(source.encode()).hexdigest(),
                                'lowering_modules_sha256':{name:hashlib.sha256((pathlib.Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('semantic_ir.py','moa_stages.py','lower_semantic_ir.py','poisson_backend.py')}}
            (target.with_suffix(target.suffix+'.ir.json')).write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
            print('Wrote',target,'state='+contract['state_semantics'])

if __name__ == '__main__':
    main()
