% Generated 【entity-MATLAB¦canonical_name=MATLAB】 ONF - matmul - from DNF
% DNF: C psi (i,j) = sum_k ( (i,k) psi A ) * ( (k,j) psi B ) # matmul DNF
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
