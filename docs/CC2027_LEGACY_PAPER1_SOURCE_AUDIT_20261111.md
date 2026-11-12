# Reviewer audit: legacy Paper I writeup versus executable compiler — 2026-11-11

**Inspected source:** `moa_paper1_writeup.tex` on `cc2027-artifact` (historical manuscript; preserved without edits). Also inspected `research/attention_rc1/moa_index_dnf_onf_v1.py` and `tests/attention_rc1/test_index_dnf_onf.py`.

## Findings requiring resolution before reuse in a CC 2027 submission

1. **Hand-derived provenance and compiler provenance are mixed.** The legacy text describes Paper I equations, a Python NumPy reference, and a `MoACompiler` class in the same document. The class's `psi_reduce(Q,K,V)` directly evaluates loops, NumPy max/exp/sum, and `Y @ V`; that alone is not evidence of a general AST ψ-rewrite engine or independently certified DNF normalization.
2. **The source's code excerpt is not a reliable standalone executable artifact.** The visible `emit_c_openmp` method uses `return "` followed by a raw newline and source text, which would be invalid Python if copied verbatim; likewise the visible Fortran/MPI emitter snippets. Verify actual generated files and run them rather than assuming a LaTeX listing compiles.
3. **Memory minimality is asserted more broadly than the shown code supports.** `scores = np.zeros((n,n))` and `Y = num/den` materialize quadratic intermediate arrays in the `psi_reduce` method. The mathematical streaming/ONF storage claim must be distinguished from this reference implementation.
4. **Numerical reproducibility claims need machine-backed evidence.** The manuscript includes broad hardware percentages, speedups, and predictions. Each number needs a source log, frozen input, environment, method, and hand/compiler attribution. No new hardware verification was performed in this audit.
5. **A valid limited MoA index prototype exists.** `moa_index_dnf_onf_v1.py` explicitly states it is a *minimal, testable index→DNF→sequential ONF prototype*, not a general ψ-reduction engine. It provides `iota`, `gamma`, prefix `DNFSelect`, `SequentialONF`, `psi`, and `pack`. Its test script exercises row-/column-major layouts, prefix selections, scalar and zero-extents, and invalid-index rejection. This is valuable bounded evidence but not Paper I full-attention DNF convergence.

## Required paper corrections

- Treat `moa_paper1_writeup.tex` as historical material, **not** a verified current CC manuscript.
- Provide a claim-to-code-to-test-to-log evidence matrix and remove or qualify unverified claims.
- Keep mathematical DNF/ONF/Ω claims separate from NumPy reference execution and compiler-generated implementation.
- Independently extract the hand-derived Paper I ψ-normal form from the actual Paper I source, then compare against compiler-generated expression with identical shapes/observables.
- Run generated C/OpenMP and other backends through their own correctness, reproducibility, and performance gates. Do not infer compiler verification from a Python reference.

**Verdict:** useful mathematical source material, but the visible legacy manuscript is not reviewer-ready as compiler evidence. No claims of CI passing, current generated-code compilation, or independent hand/compiler DNF equality are made here.
