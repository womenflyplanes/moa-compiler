# Paper I — MoA ψ-indexed online summary correctness (restricted finite domain)

**Status:** mathematical derivation for the implemented recurrence, not yet a machine-checked proof or a compiler-wide theorem. Scope: rank-2, output-only forward attention, finite real inputs, positive dimensions `n,m,d,dv`, no mask, fixed key order, query-disjoint ownership. Floating-point execution is checked by tolerances, not claimed bitwise equivalent to exact real arithmetic.

## DNF contract

Let `ρQ=⟨n,d⟩`, `ρK=⟨m,d⟩`, `ρV=⟨m,dv⟩`, `ρO=⟨n,dv⟩`. For each valid query index `i`, key index `k`, reduction index `j`, and value index `v`:

```
s(i,k) = (1/sqrt(d)) × +red_j ((⟨i,j⟩ ψ Q) × (⟨k,j⟩ ψ K))
⟨i,v⟩ ψ O =
  (+red_k (exp(s(i,k)-a_i) × (⟨k,v⟩ ψ V)))
  / (+red_k exp(s(i,k)-a_i))
a_i = maxred_k s(i,k)
```

This is the output-only observable contract. No `m×m` or `n×m` attention-weight array is required by the contract.

## Inductive summary invariant

For each query `i`, after processing the prefix of keys `0 ≤ k < t`, maintain `(a_t,z_t,u_t)`, with `u_t` a rank-1 array of shape `⟨dv⟩`:

```
a_t = maxred_{0≤k<t} s(i,k)
z_t = +red_{0≤k<t} exp(s(i,k)-a_t)
⟨v⟩ ψ u_t =
  +red_{0≤k<t} (exp(s(i,k)-a_t) × (⟨k,v⟩ ψ V))
```

Base `t=0`: `a_0=-∞`, `z_0=0`, and `u_0` is the zero rank-1 array. No exponent is evaluated at this empty prefix.

For the next score `s=s(i,t)`, define `a'=max(a_t,s)`, `w_old=0` when `t=0` and otherwise `exp(a_t-a')`, and `w_new=exp(s-a')`. The update is

```
a_{t+1} = a'
z_{t+1} = z_t × w_old + w_new
⟨v⟩ ψ u_{t+1} =
  ((⟨v⟩ ψ u_t) × w_old)
  + (w_new × (⟨t,v⟩ ψ V))
```

**Inductive step:** For each old key, the exponent identity
`exp(s(i,k)-a')=exp(s(i,k)-a_t)×exp(a_t-a')`
holds over the reals. Distributing the finite reductions gives the stated updates. The new key contributes `w_new`. Hence the invariant holds at `t+1`. At `t=m`, `z_m>0` for finite scores and `⟨i,v⟩ψO=(⟨v⟩ψu_m)/z_m`, which equals the DNF contract.

## γ/ONF and Ω implications

For contiguous row-major layout `γ(⟨i,j⟩,⟨n,d⟩)=i×d+j`, `γ(⟨k,j⟩,⟨m,d⟩)=k×d+j`, `γ(⟨k,v⟩,⟨m,dv⟩)=k×dv+v`, and `γ(⟨i,v⟩,⟨n,dv⟩)=i×dv+v`. The emitter constructs these address expressions from the validated ψ-access records.

Because different query indices write disjoint output intervals, a query-partitioned Ω lifting is *semantically legal*, provided each owner processes all keys in order and has access to the required Q, K and V components. The current emitter **does not yet construct or verify** this Ω mapping or the ten-field machine property array.

Scratch per active query is `τ(u_t)=dv` double components plus scalar summary state; this excludes inputs, output, allocation metadata, and execution/runtime overhead. The implementation allocates `u` separately for each query and does not materialize the attention score/weight array.

## Proof limits and obligations

1. This is an exact-real arithmetic proof; floating-point reassociation, overflow/underflow, infinities, and NaNs require separate treatment. The C emitter currently checks dimensions and pointers, **not finiteness of all inputs or scores**.
2. The proof establishes the online summary transformation, **not** that a general ψ-reduction engine produced the saved DNF.
3. The saved JSON trace is validated against a restricted schema; its source derivation and hand-derived canonical DNF equivalence are still open.
4. Generated C and independent reference tests must be run and logged from the exact pinned GitHub commit.
5. Ω resource feasibility, cost prediction, machine-shape choices, and empirical performance remain separate open gates.

## Reviewer-facing claim

Permissible after reproducing the tests: “For the restricted rank-2 output-only forward-attention DNF, we derive a stable online summary by induction, lower ψ-accesses to γ-addressed C, and check finite numerical executions against an independent oracle.” Not permissible: “the full compiler and all machine mappings are formally verified.”
