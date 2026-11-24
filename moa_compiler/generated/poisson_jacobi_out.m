% Generated MATLAB ONF - poisson_jacobi - from DNF
% DNF: ⟨i,j⟩ ψ u_next = (0.25*((((⟨i-1,j⟩ ψ u+⟨i+1,j⟩ ψ u)+⟨i,j-1⟩ ψ u)+⟨i,j+1⟩ ψ u)+((h*h)*⟨i,j⟩ ψ f))); state=previous_sweep
% MoA: Same DNF, many ONFs - MATLAB backend, gamma_col=7, column-major
% ONF Jacobi: one phase, all ivec parallel, two arrays, variable swap only
% MATLAB has no pointer-swap: use two arrays and swap variables
function [u, u_new] = poisson_jacobi_out(nx, ny, niter)
  if nargin < 3, nx = 64; ny = 64; niter = 100; end
  if nx ~= 64 || ny ~= 64 || niter ~= 100, error('Shape/iteration contract changed: regenerate semantic IR'); end
  h = 0.01587301678955555;
  u = zeros(nx+2, ny+2);
  u_new = zeros(nx+2, ny+2);
  f = ones(nx+2, ny+2);
  % psi reduction: u_new psi ivec = 0.25*sum neighbours psi u + h^2 * ivec psi f
  % indices shifted +1 for MATLAB 1-based: u(i+1,j+1) corresponds to Fortran u(i,j)
  u=single(u); f=single(f); h=single(h); u_new=single(u_new);
  for it = 1:niter
    for j = 2:ny+1
      for i = 2:nx+1
        u_new(i,j) = (0.25*((((u(i-1,j)+u(i+1,j))+u(i,j-1))+u(i,j+1))+((h*h)*f(i,j))));
      end
    end
    % variable swap only
    tmp = u; u = u_new; u_new = tmp;
  end
end
