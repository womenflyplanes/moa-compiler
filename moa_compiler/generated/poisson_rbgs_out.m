% Generated MATLAB ONF - poisson_rbgs - from DNF
% DNF: ⟨i,j⟩ ψ u_next = (0.25*((((⟨i-1,j⟩ ψ u+⟨i+1,j⟩ ψ u)+⟨i,j-1⟩ ψ u)+⟨i,j+1⟩ ψ u)+((h*h)*⟨i,j⟩ ψ f))); state=red_then_black
% ONF RBGS: same DNF, red phase then black phase, in-place, gamma_col
function u = poisson_rbgs_out(nx, ny, niter)
  if nargin < 3, nx = 64; ny = 64; niter = 100; end
  if nx ~= 64 || ny ~= 64 || niter ~= 100, error('Shape/iteration contract changed: regenerate semantic IR'); end
  h = 0.01587301678955555;
  u = zeros(nx+2, ny+2);
  f = ones(nx+2, ny+2);
  u=single(u); f=single(f); h=single(h);
  for it = 1:niter
    for color = 0:1
      for j = 2:ny+1
        for i = 2:nx+1
          if mod((i-1)+(j-1), 2) == color
            u(i,j) = (0.25*((((u(i-1,j)+u(i+1,j))+u(i,j-1))+u(i,j+1))+((h*h)*f(i,j))));
          end
        end
      end
    end
  end
end
