# Paper I DNF convergence acceptance contract

Status: **proposed verification contract; not yet discharged**. Scope: forward attention, Paper I only.

## Governing MoA principle

The human hand derivation and compiler derivation may use different valid sequences of transformations. Their shared mathematical obligation is to ψ-reduce to the **same canonical DNF**, for the same source expression and compatible shape/domain assumptions. A proven theorem may abbreviate any long derivation if its applicability conditions and normal-form preservation are checked.

Data and machine are both modeled using MoA shaped arrays. Subsequent γ/ONF and Ω lifting are MoA transformations, not architecture-specific substitutes for the algebra. Ω-lifted ONFs may differ with target machine shape while preserving the source semantics.

## Required proof-carrying records

For each comparison, retain:

1. The exact hand-derived source expression, the exact compiler input expression, and their shape/domain assumptions.
2. A canonical MoA ψ-indexed DNF from each path, with explicit normalization conventions (bound index names, reduction order, scalar operations, and admissible shape constraints).
3. A checked equality certificate for the two DNFs. A matching file hash of a shared symbolic template **does not suffice**: each path must have independently established provenance.
4. For every theorem shortcut, its statement, proof reference, preconditions, substitution, and resulting DNF.
5. γ/layout assumptions and ONF correspondence; distinguish mathematical addressing from execution.
6. The ten-field machine-property array, target machine shape, Ω-lifted ONF, ownership/resource legality, and any associated proof certificate.
7. Frozen pre-execution predictions, generated code and build provenance, independently obtained measurements, validation tolerances, and residuals.

## Equivalence scope and numerical caveat

A DNF equality certificate is valid only relative to a specified MoA expression language, rewrite semantics, and shape/domain constraints. Claims about general canonical uniqueness, termination, or confluence require separate proof or clearly bounded scope. Reordering floating-point reductions can change numerical results; semantic-preservation claims must state the intended arithmetic model and any permitted tolerances.

## Reviewer gate

- [ ] Independently extract hand-derived Paper I DNF from the original research evidence.
- [ ] Independently derive compiler DNF from its actual input and transformation trace.
- [ ] Compare normalized forms, not merely stage filenames or copied JSON objects.
- [ ] Verify theorem shortcuts against preconditions.
- [ ] Verify ONF correspondence for matched γ/layout.
- [ ] Verify each Ω mapping against its own target machine shape.
- [ ] Connect generated implementations and results to frozen source and machine records.

**Until checked, DNF convergence is a research acceptance criterion, not a completed result.** The 16-case structural identity audit in [PAPER_I_STAGE_IDENTITY_AUDIT_20261111.md](PAPER_I_STAGE_IDENTITY_AUDIT_20261111.md) does not replace these steps.
