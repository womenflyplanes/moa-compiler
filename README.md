# Mathematics of Arrays compiler research

**Active scope: Paper I / Program 2 (forward attention).** The mathematical pipeline is ψ-indexed specification → DNF → γ/ONF → machine-dependent Ω lifting → generated implementation → prospective prediction → independent validation. Semantic equivalence is the requirement; intermediate derivations need not be textually identical and lifted ONFs depend on machine shape.

**[Installation, checkpoint preparation, and usage](docs/PAPER_I_USAGE.md)**

**CC 2027 review:** [Artifact verification status and release gates](docs/CC2027_ARTIFACT_STATUS.md) · [Reproduction procedure](docs/CC2027_REPRODUCE.md) · [Repository map and preservation policy](docs/REPOSITORY_MAP.md). This branch is a work-in-progress audit, not a reproducible release.

The documented source/evidence checkpoints are the verified **8 November consolidated manuscript/compiler archive** and **10 November matched-validation archive**, stored separately from GitHub. The former contains `main.tex`/`main.pdf`, figures, typed ASTs, and compiler code; the latter contains the full-(A,O) contract, matched-validation scripts, and original hand-derived evidence. See the guide for exact filenames, SHA256 values, and development-checkout preparation. **The full source/evidence release is not yet synchronized into GitHub**, so this branch alone is not a complete reproduction package.

Keep the O-only online-fold and full-(A,O) interfaces distinct, as well as logical Ω ownership and physical execution. Tests provide finite evidence, not a general correctness proof. Prospective machine-shape selection and absolute-time accuracy are separate claims. Preserve historical failures and original SHA256-frozen evidence. The ten-field machine property array remains append-only extensible. Full Paper I is ongoing; Papers II–IV are out of scope.

Historical checkpoints and research records should be retained until their provenance and references have been audited; documentation cleanup must not destroy evidence.
