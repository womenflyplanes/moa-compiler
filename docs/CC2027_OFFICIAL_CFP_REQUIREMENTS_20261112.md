# CC 2027 official submission requirements — verified 2026-11-12

Official sources:
- https://conf.researchr.org/home/CC-2027
- https://conf.researchr.org/track/CC-2027/CC-2027-research-papers
- https://cc27.hotcrp.com (announced to open 2026-11-15)

## Binding requirements

- Abstract registration deadline **2026-12-01 AoE (UTC−12)**.
- Full paper deadline **2026-12-03 AoE (UTC−12)**.
- Abstract **100–400 words**, full author/contact/affiliation information entered in submission system.
- Main paper **at most 10 pages** of text and figures, **references excluded**; no appendix.
- PDF printable on US letter paper.
- LaTeX: `\\documentclass[sigplan,10pt,review,anonymous]{acmart}`.
- Double-blind paper and supplementary materials. Self-references in third person; retain necessary citations.
- Anonymous supplementary proofs, code, datasets allowed in PDF or ZIP; reviewer inspection discretionary.
- References must list **all author names** (no “et al.” in references).
- Artifact Evaluation is optional, organized after acceptance; submission form asks whether interested.
- Accepted paper requires in-person presentation by at least one author, subject to stated exceptions.

## Paper I / MoA manuscript implementation plan

1. Create a **separate anonymous 10-page CC paper**. Keep long research manuscript, historical documents, hand-derived publications, and full proofs unchanged.
2. Main paper must include formal ψ/DNF→γ/ONF→Ω contract, compiler architecture and actual executable provenance, an attention example, correctness claims limited to proved cases, machine-shape cost model, measured results including negative holdout, and related work.
3. Supplement can include full MoA proofs, ASTs, trace manifests, machine property arrays, generated source, tests, performance tables, plots, SHA256 manifests and reproducibility scripts.
4. Anonymity audit of PDF metadata, source comments, ZIP names/paths, links and GitHub references. A public named GitHub repo must not be presented as an anonymous artifact without deliberate review.
5. Finish and verify the real compiler chain before making end-to-end or performance-portability claims. Distinguish hand-derived ACCESS evidence from compiler-generated UA evidence.
6. Before registration: prepare a 100–400 word abstract and exact author list; submit abstract by Dec 1. Final PDF due Dec 3 AoE.
7. Run a strict page-count and reference-format check on the built PDF; do not claim compliance based on source alone.

The conference website was recently launched and may be updated. Recheck before submission.
