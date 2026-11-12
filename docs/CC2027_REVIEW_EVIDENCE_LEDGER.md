# CC 2027 — reviewer evidence ledger and acceptance checklist
**Status:** live audit; never treat an unchecked item as complete.

| Gate | Concrete evidence | Current assessment | Next verification |
| --- | --- | --- | --- |
| A. Correct MoA semantics | ψ, ρ, τ, γ in separate manuscript; full historical derivations | Partial | Formal supported source grammar and index laws |
| B. Source to canonical DNF | Saved trace and typed modules | Open | Independent executable source→DNF pipeline |
| C. DNF to addressed ONF | `tools/emit_paper1_trace_c.py` | Restricted | Compile and compare distinct ψ-indexed programs |
| D. Machine-shaped Ω | `tools/moa_query_omega.py` | Restricted; disconnected | Feed certificate into emitter, test distinct shapes |
| E. Ten-field machine property array | Historical compiler machine files | Unverified integration | Show fields, values, unit contracts and actual use |
| F. Correctness proof | `docs/PAPER_I_ONLINE_SUMMARY_MOA_PROOF_20261112.md` | Restricted exact-real proof | Independent check, error bounds, edge cases |
| G. Independent hand/compiler equivalence | Paper I audit and embedding notes | Open | Matched shape/domain/output contract |
| H. Prospective prediction | Frozen CPU experiment records | Mixed results | Reproduce all outcomes and wrong-choice holdout |
| I. Experimental comparisons | Historical ACCESS and UA evidence | Not consolidated | Full tables, original baselines, uncertainty |
| J. Related work | Historical bibliography scattered in masters | Not yet checked | Verify references and build bibliography |
| K. Artifact | GitHub code, SHA manifest generator | Partial | Freeze actual bytes and clean-checkout tests |
| L. Anonymous submission | Separate `submissions/CC2027/` LaTeX | Draft | Build PDF, inspect layout, page count, metadata |

## Manuscript style and editorial standards

- Lead with the actual MoA compiler problem and contributions, not the history of a single attention kernel.
- Distinguish semantic DNF invariance from γ-dependent ONF and machine-shape-dependent lifted ONF.
- Derivation steps may differ; independently reduced DNFs must agree for the same observable contract.
- Use only MoA-shaped arrays, ψ-indexed selections and reductions in foundational explanations.
- Visuals must teach actual transformations, AST dependencies, machine ownership, and evidence; no decorative figures.
- Prefer one clear mathematical theorem with explicit preconditions over broad unproved assertions.
- Avoid presenting pending work as a result. The introduction and abstract must match the executable scope at the final freeze.
- Every performance table must specify hardware, machine shape, workload, baseline, repetitions, uncertainty and frozen prediction.
- Keep a full unabridged research record; never edit it down to satisfy the 10-page submission limit.

## Review cadence

After each substantive compiler or manuscript change: check semantics, negative tests, provenance, anonymity and claim-evidence alignment. Before submission: independent cold-read, TeX build, visual page inspection, artifact rehearsal, and exact compliance audit.
