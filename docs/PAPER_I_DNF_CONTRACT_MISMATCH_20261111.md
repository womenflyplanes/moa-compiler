# Paper I DNF contract mismatch discovered — 2026-11-11

**Finding: the two archived typed-DNF JSON records are NOT directly comparable as identical canonical DNFs.** This is an important review blocker, not an observed semantic failure.

The November 8 consolidated archive's `compiler_ast/01_typed_dnf.json` specifies a rank-2 unbatched attention input:
- Q shape `[n,d]`, K shape `[m,d]`, V shape `[m,dv]`.
- Score reduction over `j`; key index `k`; stable normalization.
- The accompanying independent interpreter checks the **output-only** contract.

The November 10 matched-validation archive's `local_validation/2_4_8_2_serial/01_psi_dnf.json` specifies a rank-3 batched contract:
- Q, K, V each have symbolic shape `[B,N,D]`.
- Indices include batch index `b`; score reduction `j`, normalization/output reduction `k`.
- Observables explicitly include **both A and O**.

Consequences:
1. Byte-for-byte or canonical-JSON equality across these two ASTs is not meaningful.
2. The shape relation `n = N`, `m = N`, `d = D`, `dv = D` and an explicit batch lifting/slicing relation must be established before comparison.
3. Even under those restrictions, the full-(A,O) observable contract is stronger than the output-only contract. The output-only test cannot certify A.
4. The original **hand-derived** mathematical DNF must be sourced separately; neither of these compiler JSON records is by itself that independent hand-derived DNF.

## Acceptance action

Define an explicit MoA ψ-indexed embedding/projection theorem between the two supported domains, with shape/index assumptions and output observables stated. Then compare the **same** semantic contract and canonical DNF across the hand and compiler paths. Do not silently rename shapes or drop the A output to make the records appear equal.

This finding supersedes any unqualified claim that the November 8 and November 10 records are already the same canonical DNF. Their respective internal stage-identity and finite numerical tests remain valid in their stated bounded scopes.
