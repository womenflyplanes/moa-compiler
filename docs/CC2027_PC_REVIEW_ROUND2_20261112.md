# CC 2027 — independent-minded program committee review, round 2
Date: 2026-11-12. Subject: separate anonymous paper `submissions/CC2027/moa_cc2027_10page.tex` and the `cc2027-artifact` branch.

## Recommendation today: major revision / not ready for acceptance

**Summary.** The manuscript offers a distinctive shaped-array mathematical route from ψ-indexed semantics through DNF and γ-addressed ONF to Ω-lifted machine execution. The online forward-attention reduction is mathematically motivated. However, the executable artifact currently establishes a restricted saved-trace emitter and a disconnected finite Ω ownership checker, rather than the end-to-end compilation advertised by the title. Evaluation and scholarly comparison are incomplete.

### Ratings (provisional internal assessment, not external reviews)

| Criterion | Assessment | Reviewer reason |
| --- | --- | --- |
| Significance | Promising | Unified semantic and machine-shape theory, potentially broad portability |
| Originality | Needs evidence | Must differentiate mechanized MoA laws from existing array and polyhedral lowering |
| Technical soundness | Partial | Restricted proof obligations, missing source-to-DNF and Ω-to-emitter links |
| Empirical evaluation | Insufficient | No complete matched, compiler-generated end-to-end results in this paper |
| Reproducibility | Partial | Code/tests committed; no confirmed clean-run logs, immutable public full archive |
| Clarity | Improving | The manuscript still reads like a roadmap rather than a completed compiler result |
| Submission compliance | Unverified | No compiled PDF, final bibliography, page-count or anonymity audit |

## Detailed technical objections

**R1. Actual compiler boundary.** Show a typed source MoA AST that is *independently* reduced to DNF, and prove that generated code is sensitive to AST changes. A fixed attention schema with hard-coded stable recurrence is not a general array compiler.

**R2. Normal form criterion.** Specify the supported ψ rewrite rules, their well-typed preconditions, shape preservation, and normalization strategy. Do not assert Church–Rosser or uniqueness merely because index domains are finite. Termination of a finite computation is not proof that an arbitrary rewrite system is confluent.

**R3. Exact equivalence.** For matched observable contracts, independently derive hand and compiler DNF and compare their canonical forms. The archived unbatched output-only and batched full-(A,O) contracts are not directly interchangeable.

**R4. γ proof and layout scope.** The rank-two row-major bijection proof is valid under positive extents and bounded indices, but show actual code emission from γ, layout alternatives, and the semantics of rav storage. Preserve vector rank-one and matrix rank-two distinctions.

**R5. Ω proof versus Ω compiler.** Query partition coverage is useful, but a checker not connected to emitted loops cannot certify the executable. Feed a verified Ω ownership certificate to emission and prove output ownership and required Q/K/V availability. Distinguish logical P from physical compute units.

**R6. Ten-field machine properties.** State all ten fields and their dimensional units; demonstrate actual use by legality and cost models. An appended property array must preserve earlier field indices and be auditable.

**R7. Stable summary numerical behavior.** The exact-real induction does not establish FP stability for arbitrary values. Test all finite input assumptions, empty or invalid dimensions, extreme score ranges, NaNs/infinities, allocation failure, integer overflow, and output error tolerances. Do not claim bitwise determinism.

**R8. Performance and failed predictions.** Present a prospective frozen model, candidate Ω shapes, full machine and affinity details, observed timings with dispersion, chosen vs actual fastest, and the known wrong-selection holdout. Do not extrapolate hand-derived ACCESS measurements as compiler measurements.

**R9. Comparison and attribution.** Include original MoA derivation papers, historical DGEMM evidence with exact comparator and machine, attention implementations, array compilers, polyhedral systems, and portable execution tools. Distinguish conceptual overlap and actual algorithmic differences. No invented speedups or unsupported priority claims.

**R10. Artifact preservation and anonymity.** Produce a complete, verifiable SHA256 snapshot with actual ZIP bytes; test a clean checkout; scrub PDF/ZIP metadata, repository URLs, local paths, authors and identifying acknowledgments from anonymous material. Keep unrestricted full research documents separate from the conference artifact.

## Concrete manuscript defect discovered and fixed

The current LaTeX had two occurrences of `\\mathop` before max reductions. These are malformed commands in mathematical context, not a valid `\mathop`. Corrected in the separate CC submission at commit `d9cb8185074a01f00b0ac1cb03a74ade77e1dab7`. A TeX build is still required to detect further errors.

## What would change the recommendation

A complete, reproducible source AST→ψ DNF→γ ONF→verified Ω→code generation path with at least one independently matched hand-derived case, explicit theorems and tests, prospective measured CPU OpenMP results including negative holdouts, a rigorous related-work section, and a polished compliant anonymous PDF. An excellent exposition alone cannot substitute for this evidence.

## Preservation

Do not overwrite the full research manuscript or historical logs to make the short submission look stronger. Maintain the separate CC paper and its exact evidence pointers; any claim not traceable to a pinned artifact must be marked open.
