% Generated 【entity-MATLAB¦canonical_name=MATLAB】 ONF - poisson_rbgs - from DNF
% DNF: u_new psi ivec = 0.25 * sum_{ (di,dj) in {(-1,0),(1,0),(0,-1),(0,1)} } (ivec+(di,dj)) psi u + h^2 * (ivec psi f) # Poisson stencil DNF, rav + gamma_col
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
