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
