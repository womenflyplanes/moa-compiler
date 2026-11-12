# CC 2027 — anonymous 10-page MoA compiler paper blueprint

**Working title:** *From ψ-Indexed Semantics to Machine-Shaped Execution: Mechanizing the Mathematics of Arrays*

**Scope:** Paper I forward attention only; matrix multiplication as foundational algebraic example. The hand-derived attention/ACCESS studies are prior independent evidence, not compiler output. **This is a writing blueprint, not a completed paper.**

| Main-paper section | Target pages | Evidence gate |
| --- | ---: | --- |
| Abstract and introduction: one algebra, multiple machines, contribution limits | 0.8 | exact verified claims |
| MoA foundations: shapes, ψ, DNF, γ/ONF, Ω and ten-field machine array | 1.5 | notation audit |
| Compiler design: source→DNF→ONF→Ω→emission, with actual AST dependencies | 2.0 | continuous executable provenance |
| Correctness: ψ-reduction and γ; online summary induction; Ω legality | 1.5 | scoped proofs and testable invariants |
| Paper I attention case and hand-derived DNF convergence | 1.0 | matched observable contract |
| Prospective prediction and measurements, including unsuccessful holdout | 1.3 | frozen manifests, machine properties, plots |
| Related work and limitations | 1.2 | checked citations |
| Conclusion | 0.4 | no overclaim |
| **Total planned** | **9.7** | 10-page maximum excluding references |

## Required reviewer figures and tables

1. **Figure 1:** shaped-array transformation chain with ψ-indexed semantic contract and provenance SHA links.
2. **Figure 2:** matrix multiplication ψ/DNF→γ/ONF, contrasting contiguous and noncontiguous accesses under stated layouts. Do not assert universal speedup.
3. **Figure 3:** query-disjoint Ω ownership of attention, each owner maintaining `ρu=⟨dv⟩`.
4. **Table 1:** hand-derived vs compiler-generated paths: DNF, ONF, Ω, implementation, prediction, validation, each with verified/open state.
5. **Table 2:** measured and predicted CPU OpenMP outcomes with prospective freeze and failed holdout explicitly shown.
6. **Table 3 (space permitting):** original published matrix multiplication/ACCESS results, precisely attributed with platform, size, comparator, and measured speedup.

## Required supplementary archive

- Complete derivation/proofs, AST JSON and generated C, immutable manifest SHA256.
- Ten-field machine-property arrays, Ω mapping certificates and predicted costs.
- Exact environment and execution commands, tests, independent oracle, output logs, negative tests.
- Hand-derived Paper I references and source/observable contract bridge.
- All relevant figures and full experiment tables.
- A `RESUME_HERE` and limitations file.

## Critical acceptance conditions

- Paper claims must not outrun actual executable tests.
- A saved typed DNF is **not** evidence that arbitrary source expressions are ψ-reduced by the compiler.
- A query-disjoint C loop is **not** evidence of automated Ω lifting.
- Numerical agreement is **not** a proof of global compiler correctness.
- An anonymous submission must not expose author identity via file metadata, repository links, embedded paths or comments.
- The full manuscript must compile as `acmart` 10pt, review, anonymous, sigplan and be checked for exactly ≤10 pages of non-reference content.
- Registration: Dec 1 AoE; paper: Dec 3 AoE. Recheck official CFP for changes.

## Next implementation order

1. Reproduce the committed trace emitter/tests in CI and capture logs.
2. Make ψ-normalization produce the saved typed DNF rather than ingesting a precomputed trace.
3. Mechanize and verify Ω ownership/resource mapping from ten-field machine array.
4. Compare independent hand and compiler DNFs under the same Paper I observable contract.
5. Add prospective prediction and execution records; then freeze paper figures and manuscript text.
