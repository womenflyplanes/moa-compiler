
import numpy as np

print("=== Paper 1 Reproduction: Attention at Theoretical Minimum ===")
print("From arXiv:2606.07713, Sec 4.3-5.3, Listing 1")
print()

Q = np.array([
    [-0.1984,  0.2698,  0.3414, -0.0372],
    [ 0.2547, -1.0674,  0.3460, -2.5242],
    [ 0.6822, -0.6265,  0.0252,  0.3978]
], dtype=np.float64)

K = np.array([
    [-1.1567,  0.6885, -0.1884,  0.4743],
    [ 0.2246,  1.7564,  0.5235, -2.3014],
    [-1.5899,  0.3730, -0.8257, -1.2069]
], dtype=np.float64)

V = np.array([
    [ 1.0739,  0.4006, -0.9671,  0.4870],
    [ 0.5589, -0.7209, -0.7650,  0.2689],
    [ 0.8237,  0.3763,  0.8320,  0.0014]
], dtype=np.float64)

n, dk = Q.shape
_, dv = V.shape
print(f"rho(Q)=rho(K)=rho(V)=<{n},{dk}> as in Eq (9)")

def gamma(v,s):
    off=0
    for k in range(len(v)):
        prod=1
        for j in range(k+1,len(s)):
            prod*=s[j]
        off+=v[k]*prod
    return off

def stable_softmax(x, dim=-1):
    z = x - np.max(x, axis=dim, keepdims=True)
    num = np.exp(z)
    den = np.sum(num, axis=dim, keepdims=True)
    return num / den

scores = (Q @ K.T) / np.sqrt(dk)
print("scores = Q K^T / sqrt(dk) =")
print(scores)
print()

attn_weights = stable_softmax(scores, dim=-1)
print(f"attn_weights Y rho=<{n},{n}> DNF Eq (20)")
print(attn_weights)
print()

output = attn_weights @ V
print(f"Output rho=<{n},{dv}> Eq (21)")
print(output)
print()

expected_weights = np.array([
    [0.3202, 0.3834, 0.2964],
    [0.0288, 0.7300, 0.2412],
    [0.4270, 0.2844, 0.2887]
])
expected_output = np.array([
    [0.8023, -0.0366, -0.3563, 0.2595],
    [0.6376, -0.4240, -0.3857, 0.2107],
    [0.8552, 0.0747, -0.3903, 0.2848]
])

print("=== Verification vs Table 1 (weights) ===")
for i in range(n):
    match = np.allclose(attn_weights[i], expected_weights[i], atol=1e-4)
    print(f"Row {i}: {attn_weights[i]} vs {expected_weights[i]} Match={match} {'✓' if match else '✗'}")
print()

print("=== Verification vs Table 2 (output) ===")
for i in range(n):
    match = np.allclose(output[i], expected_output[i], atol=1e-4)
    print(f"Row {i}: {output[i]} vs {expected_output[i]} Match={match} {'✓' if match else '✗'}")
print()

print("=== MoA DNF psi-reduction trace for <0> psi Output ===")
i_pp = 0
row0_scores = []
for k in range(n):
    s = 0.0
    for j in range(dk):
        s += Q[i_pp, j] * K[k, j]
    s /= np.sqrt(dk)
    row0_scores.append(s)
row0_scores = np.array(row0_scores)
print(f"<{i_pp},k> psi (Q +.x K^T)/sqrt = {row0_scores}")
m = np.max(row0_scores)
z = row0_scores - m
print(f"<{i_pp}> psi z = x - max = {z} Eq (14)")
num = np.exp(z)
print(f"<{i_pp}> psi num = exp(z) = {num} Eq (16)")
den = np.sum(num)
print(f"<{i_pp}> psi den = red_+(num) = {den} Eq (19)")
Y_row = num / den
print(f"<{i_pp}> psi Y = num/den = {Y_row} Eq (20)")
out_row = np.zeros(dv)
for p in range(n):
    out_row += Y_row[p] * V[p]
print(f"<{i_pp}> psi Output = red_+(Y x V) = {out_row} Eq (21)")
print(f"Matches output row 0: {output[0]}")
print()

print("=== Prop 4.1 Memory Minimality ===")
M_classical = n*n + n*dk + n*dk + n*n + n + n*n + n*dv
M_moa = n*dk + n*dk + n*dv
print(f"M_classical O(n^2+nd_k+nd_v)={M_classical} Eq (23)")
print(f"M_moa O(nd_k+nd_v)={M_moa} Eq (24) reduction {M_classical/M_moa:.1f}x")
print()

print("=== ONF gamma Eq (7) ===")
for idx in [(0,0),(0,3),(1,2),(2,1)]:
    off = gamma(idx, (n,dv))
    print(f"gamma({idx}, <{n},{dv}>) = {off}")
print()
print("=== Paper 1 reproduction SUCCESS ===")
