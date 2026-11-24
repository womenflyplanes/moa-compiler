% Generated MATLAB ONF - poisson_island_gauss - from DNF
% DNF: ⟨i,j⟩ ψ u_next = (0.25*((((⟨i-1,j⟩ ψ u+⟨i+1,j⟩ ψ u)+⟨i,j-1⟩ ψ u)+⟨i,j+1⟩ ψ u)+((h*h)*⟨i,j⟩ ψ f))); state=lexicographic_islands
% MoA: Same DNF, many ONFs - MATLAB backend, gamma_col=7, column-major
% ONF Island Gauss: parallel over islands, sequential inside island, in-place
% MATLAB has no pointer-swap: in-place, no swap needed
function u = poisson_island_gauss_out(nx, ny, niter)
  if nargin < 3, nx = 64; ny = 64; niter = 100; end
  if nx ~= 64 || ny ~= 64 || niter ~= 100, error('Shape/iteration contract changed: regenerate semantic IR'); end
  isx = 16; isy = 16;
  nix = nx / isx; niy = ny / isy;
  h = 0.01587301678955555;
  u = zeros(nx+2, ny+2);
  f = ones(nx+2, ny+2);
  u=single(u); f=single(f); h=single(h);
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
            u(i,j) = (0.25*((((u(i-1,j)+u(i+1,j))+u(i,j-1))+u(i,j+1))+((h*h)*f(i,j))));
          end
        end
      end
    end
  end
end
