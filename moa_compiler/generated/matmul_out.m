% Generated MATLAB ONF - matmul - from DNF
% DNF: ⟨i,j⟩ ψ c = (⟨i,j⟩ ψ c+(⟨i,k⟩ ψ a*⟨k,j⟩ ψ b)); state=ordered_k_reduction
% ONF matmul: loop j, loop i, loop k, gamma_col
function C = matmul_out(n)
  if nargin < 1, n = 64; end
  if n ~= 64, error('Shape/iteration contract changed: regenerate semantic IR'); end
  A = ones(n,n); B = ones(n,n); C = zeros(n,n);
  A=single(A); B=single(B); C=single(C);
  for j = 1:n
    for i = 1:n
      C(i,j) = 0.0;
      for k = 1:n
        C(i,j) = (c(i,j)+(a(i,k)*b(k,j)));
      end
    end
  end
end
