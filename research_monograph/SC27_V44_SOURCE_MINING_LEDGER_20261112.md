# Source-mining ledger: full SC27 v44 manuscript

Source: `MoA_SC27_Complete_v44/MoA_SC27_Complete_v44.tex` on branch `cc2027-artifact`, blob `98953c673bdaeb1e17ebcfa639221bb3c7199732`.
Size: 150002 characters. Figure inclusion commands: 20; table environments: 10.

This is an inventory of **source claims and locations**, not independent verification of their experimental results. The manuscript predates the November 8–12 frozen archives, which must be cross-linked without rewriting their provenance.

## Extract into the conference paper and complete monograph

### Theory and correctness
- Foundations: shape, index domain, and $\psi$ (source character offset 5299)
- MoA all the way: the constructive proof architecture (source character offset 5889)
- Cartesian-indexed DNF: the end of semantic reduction (source character offset 10955)
- DNF to sequential ONF: addressing is the realization (source character offset 12050)
- Sequential to parallel and distributed ONF by dimension lifting (source character offset 13434)
- Constructive equivalence and performance selection (source character offset 15428)
- Paper 1 mathematical path: input expression to reduced DNF (source character offset 17386)
- DNF to $\gamma$-addressed sequential ONF (source character offset 21525)
- $\psi$ reduction, DNF, and sequential ONF (source character offset 24302)
- Dimension lifting and shaped execution (source character offset 25301)

### ASTs and compiler
- A visible attention AST rewrite sequence (source character offset 8848)
- Worked attention expression and AST (source character offset 16192)
- Input array expression and typed semantic AST (source character offset 17946)
- Compiler acceptance test against the hand derivation (source character offset 23611)
- Mechanization milestones and limitations (source character offset 32919)
- MoA all the way: four-paper compiler reproduction contract (source character offset 36069)
- A traceable AST-to-instruction construction (source character offset 42690)
- AST-to-instruction evidence and the portable realization objective (source character offset 48456)
- AST~4: shape-dependent address instantiation (source character offset 49390)
- AST~5a--6a: CPU ownership and C/OpenMP instructions (source character offset 49923)
- AST~5b--6b: portable accelerator ownership with OpenACC (source character offset 50462)
- AST~5c--6c: sequential, OpenMP, OpenACC, and MPI realizations (source character offset 51320)
- Proof obligations and compiler acceptance criteria (source character offset 52079)
- AST A: the full mathematical specification (source character offset 53418)
- AST B: stable normalization as a mergeable MoA reduction (source character offset 54639)
- AST C: memory-reduced online ONF (source character offset 55831)
- AST D: shape-dependent addresses and instruction lowering (source character offset 56609)
- AST E: CPU, GPU, and coordinated heterogeneous realization (source character offset 57824)
- Validation and compiler acceptance gate (source character offset 59009)
- A reproducible AST-to-C mechanization prototype (source character offset 59765)
- Restricted rewrite passes and three executable target lowerings (v12) (source character offset 63160)
- November 2026 compiler integration: typed ASTs, verified ownership, and audited topology (source character offset 82295)

### Machine shape and predictions
- Lifted ONF, exact coverage, and resource allocation (source character offset 22337)
- Machine shape and prediction obligations (source character offset 26240)
- Audited Paper I baseline prediction evidence (source character offset 29605)
- Verified resource-owned OpenMP execution (source character offset 31089)
- MoA as a unified mathematics of computation, machines, and prediction (source character offset 33894)
- Experimental lineage and resource provenance (source character offset 39643)
- Coordinated heterogeneous resource allocation (source character offset 40985)
- AST~5a--6a: CPU ownership and C/OpenMP instructions (source character offset 49923)
- AST~5b--6b: portable accelerator ownership with OpenACC (source character offset 50462)
- MoA output ownership and execution realization (source character offset 67579)
- Implications for machine-shape prediction (source character offset 73654)
- A shaped communication extension: deterministic routing and port scheduling (Steps 68--73) (source character offset 76880)
- November 2026 compiler integration: typed ASTs, verified ownership, and audited topology (source character offset 82295)
- $\Omega$ ownership, resource constraints, and independent routing audit (source character offset 88730)
- Verified November compiler extension: $\Omega$ port scheduling and heterogeneous communication time (source character offset 94886)
- From $\Omega$ ownership to legal port-constrained stages (source character offset 95605)
- Directed-link contention and predictive $\Omega$ schedule selection (Steps 103--104) (source character offset 102496)
- From MoA Semantics to Verified Machine-Array Communication (source character offset 110250)
- $\Omega$ ownership on a shaped machine (source character offset 119318)
- Logical resource and communication certificates (source character offset 122169)
- Derived abstract topology and legal stage scheduling (source character offset 123738)
- Prediction is distinct from legality and measurement (source character offset 125321)

### Hand vs generated validation
- Compiler acceptance test against the hand derivation (source character offset 23611)
- UAlbany DGX held-out attention validation (source character offset 27668)
- Audited Paper I baseline prediction evidence (source character offset 29605)
- Verified resource-owned OpenMP execution (source character offset 31089)
- MoA all the way: four-paper compiler reproduction contract (source character offset 36069)
- Experimental lineage and resource provenance (source character offset 39643)
- AST~5a--6a: CPU ownership and C/OpenMP instructions (source character offset 49923)
- AST~5b--6b: portable accelerator ownership with OpenACC (source character offset 50462)
- AST~5c--6c: sequential, OpenMP, OpenACC, and MPI realizations (source character offset 51320)
- Proof obligations and compiler acceptance criteria (source character offset 52079)
- Validation and compiler acceptance gate (source character offset 59009)
- DGX hardware validation of generated CPU--GPU attention (v13) (source character offset 66859)
- Experimental configuration and numerical checks (source character offset 68601)
- Portable realization: OpenMP, OpenACC, and MPI (source character offset 74797)
- November 2026 compiler integration: typed ASTs, verified ownership, and audited topology (source character offset 82295)
- The actual typed semantic AST, not a hypothetical compiler tree (source character offset 82906)
- Verified November compiler extension: $\Omega$ port scheduling and heterogeneous communication time (source character offset 94886)
- DGX-head-node regression evidence (source character offset 100012)
- The MoA compilation contract: portable, scalable, verifiable (source character offset 111009)
- Experimental evidence from MoA-derived signal and image processing (source character offset 136692)

### Comparators and broader vision
- MoA as a unified mathematics of computation, machines, and prediction (source character offset 33894)
- MoA all the way: four-paper compiler reproduction contract (source character offset 36069)
- Related work and comparative positioning (source character offset 131443)
- Experimental evidence from MoA-derived signal and image processing (source character offset 136692)
- Conclusion (source character offset 141423)

## Required provenance and certification discipline

- Treat v44 as the full-length mathematical and historical source, **not** as evidence that every stated compiler stage is integrated in the current branch.
- Extract the complete Paper I DNF→γ ONF→Ω derivation and exact observable contract before summarizing it.
- Cite original hand-derived ACCESS programs and measurements separately from compiler-generated UA records.
- Cross-reference the frozen November 8 and 10 CPU archives, November 12 archived-compiler 96/96 regeneration, and six-case A100 audit; preserve failures.
- Keep speculative portability and Papers II–IV generalization as future validation rather than completed compiler demonstrations.
- Check all source figure paths, tables, ASTs, listings, bibliography entries, machine property array, and numerical tolerances against original records.
- CC paper: extract only evidence necessary for the bounded Paper I contribution and stay within ten pages of main text.
- Full monograph: retain all relevant v44 derivations, figures, tables, source listings and historical evidence, then append newer validated records. Never substitute the 20-page summary for v44.
- Independently inspect the full historical PDF page count before asserting it is over forty pages.
