# Paper I MoA ψ-indexed embedding: unbatched output-only to batched full-(A,O)

**Status:** mathematical correspondence under explicit restrictions; **not** a proof of hand-derived/compiler DNF convergence. This note is a proof obligation and an algebraic derivation, not a new hardware result.

## 1. Shapes and index domains

Let the unbatched inputs be rank-2 arrays
`ρQ = ⟨n,d⟩`, `ρK = ⟨m,d⟩`, `ρV = ⟨m,dv⟩`.
For the archived batched contract, let `ρQ' = ρK' = ρV' = ⟨B,N,D⟩`.

The shared-domain embedding requires **B=1, n=m=N, d=dv=D**. Define the lift by ψ selection for all valid `i,j,k,v`:

```text
⟨0,i,j⟩ ψ Q' = ⟨i,j⟩ ψ Q
⟨0,k,j⟩ ψ K' = ⟨k,j⟩ ψ K
⟨0,k,v⟩ ψ V' = ⟨k,v⟩ ψ V
```

This is a rank-changing embedding, not an assertion that rank-2 and rank-3 arrays are identical. It also does **not** cover the general cross-attention case `n ≠ m` or distinct value width `dv ≠ d` without a more general batched contract.

## 2. ψ-indexed attention and reduction correspondence

For any legal `i,k`, define the scaled score through the same reduction in each domain:

```text
s(i,k)  = (1/sqrt(d)) × +red_j ((⟨i,j⟩ ψ Q) × (⟨k,j⟩ ψ K))
s'(0,i,k) = (1/sqrt(D)) × +red_j ((⟨0,i,j⟩ ψ Q') × (⟨0,k,j⟩ ψ K'))
```

By the embedding and `d=D`, the indexed summands coincide, hence `s'(0,i,k)=s(i,k)` in exact arithmetic with the same reduction semantics.

Let `a(i,k)` and `a'(0,i,k)` denote the normalized score components computed by the same stable row-wise normalization rule. Equality of the indexed score sequences yields `a'(0,i,k)=a(i,k)` for all valid `i,k`, assuming normalization is defined (finite valid denominator).

For the output:

```text
⟨i,v⟩ ψ O = +red_k (a(i,k) × (⟨k,v⟩ ψ V))
⟨0,i,v⟩ ψ O' = +red_k (a'(0,i,k) × (⟨0,k,v⟩ ψ V'))
```

Substitution yields **`⟨0,i,v⟩ ψ O' = ⟨i,v⟩ ψ O`**, with matching shape/index ranges and exact-arithmetic reduction semantics. Therefore the unbatched O can be obtained by the batch-zero ψ slice of O' in this restricted domain.

## 3. Crucial limits

- The full-(A,O) contract exposes A in addition to O. The output-only implementation does not produce an independently checked A observable. Equality of O does **not** establish full-contract equality.
- The proof above relates **denotations under a restricted embedding**. It is not an independent comparison of the original hand-derived DNF and compiler-generated DNF.
- The archived JSON stage formats must be parsed and their reduction conventions independently checked before applying the theorem to actual files.
- Exact arithmetic equality does not automatically imply bitwise-identical floating-point reductions; the compiler must document ordering and tolerances.
- Ω machine lifting is not part of this input-rank embedding. Ω legality and mapping are separate obligations.

## 4. Required follow-on checks

1. Recover and cite the original hand-derived Paper I ψ-reduced DNF, with source version and page/line location.
2. Extract the actual compiler-produced DNF for the **same** input and output contract.
3. Normalize bound indices and reduction conventions and compare the canonical expressions, recording theorem applications and preconditions.
4. Separately compare γ/ONF under matched layouts, and Ω-lifted ONF under the same machine shape.
5. Preserve this embedding as a *restricted correspondence theorem*, not a replacement for the primary DNF convergence certificate.

This is the MoA-first way to reconcile the two archived contracts without pretending they are already the same expression.
