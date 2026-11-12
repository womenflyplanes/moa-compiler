# CC 2027 submission: release and reviewer acceptance gates

The conference deliverable is a self-contained, anonymous ZIP. This checklist is **not** a claim that the gates have passed.

## Required archive contents

- `paper/moa_cc2027_10page.tex` and all included LaTeX source
- `paper/moa_cc2027_10page.pdf` generated from that exact source
- `figures/`: every PNG actually referenced, plus editable sources for diagrams
- `bibliography/`: BibTeX source and resolved citations
- `tables/`: raw and processed data, table-generating scripts, descriptions of units and sample counts
- `compiler/`: pinned MoA source AST→DNF→ONF→Ω→emitter implementation and configuration
- `tests/`: unit, metamorphic, mutation, differential and independent hand/compiler tests
- `results/`: frozen predictions, observations, hardware and machine-shape descriptors, failures and raw logs
- `README.md`: scope, system requirements, quick start, expected outputs, known limitations
- `REPRODUCE.md`: exact commands for a clean, offline-compatible run where possible
- `MANIFEST.sha256`: digests for every included file, generated after final packaging
- `REVIEW_REPORT.md`: independent reviewer objections, resolutions, residual risks and test outcomes
- `ANONYMITY_AUDIT.md`: paths, metadata, PDF properties, links, affiliations and self-citation review

## Mandatory release gates

1. **MoA semantic soundness:** shapes, ψ index domains, τ and δ, reduction axes, DNF and output contracts checked.
2. **Compiler authenticity:** typed source expression genuinely determines the DNF, γ ONF, Ω lifted ONF and emitted code; no fixed output substitution.
3. **Transformation proof:** independently specified preconditions and invariants; rewrite convergence is claimed only if proved.
4. **Machine correctness:** target shape and all ten machine-property fields actively determine legality, ownership and cost.
5. **Implementation correctness:** compiler-generated outputs checked against an independent oracle and independently hand-derived DNF for identical contracts.
6. **Numerical correctness:** finite precision, overflow, edge cases, invalid inputs and tolerance policy specified and tested.
7. **Performance honesty:** frozen predictions before measurements; successful and unsuccessful holdouts retained; comparisons matched by machine and observable contract.
8. **Reproducibility:** clean checkout; tests actually executed; logs and SHA256 digests retained; no unconfirmed CI marked passed.
9. **Paper completeness:** contribution statement, related work, limitations, citations, quantitative tables and evidence-rich figures present.
10. **ACM CC compliance:** anonymous 10pt two-column, abstract and page count checked on compiled PDF; no appendices if disallowed.
11. **Visual quality:** inspect every page, equation, caption, plot, table, margins, fonts and figure resolution.
12. **Package integrity:** build from ZIP alone, verify every referenced asset and manifest, run reproduction instructions.

## Review process

Maintain three distinct passes:
- **Author audit:** user's MoA theoretical, notation, machine and preservation standards.
- **Technical adversarial review:** challenge compiler claims, proof obligations, experiments, baselines and artifact reproducibility.
- **Publication review:** clarity, significance, contribution, narrative, figure quality, page limit, anonymity and citations.

Record each gate as `PASS`, `FAIL`, or `UNVERIFIED`, with evidence path, command, date and commit. `UNVERIFIED` is never `PASS`. Repeat the full audit after material revisions.

## Current state

The paper is a short draft. End-to-end compilation, full hardware evidence, final bibliography, PDF page count and complete anonymous ZIP are **not yet verified**. Original historical archives must be copied by bytes, not represented by hashes alone. Preserve the separate full research monograph and its historical evidence without altering it to meet conference constraints.
