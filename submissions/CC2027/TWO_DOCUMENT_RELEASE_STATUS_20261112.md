# CC 2027 submission and full research monograph: exact release state

## Full historical document (source complete; PDF build in CI)
The **unabridged** research manuscript is [MoA_Compiler_Unabridged_v44_With_20261112_Evidence.tex](../../research_monograph/MoA_Compiler_Unabridged_v44_With_20261112_Evidence.tex). It retains the historical v44 mathematics, ASTs, experiments, figures and tables and appends the frozen November CPU/A100 evidence, including negative prediction results. Its 20 figure files are tracked under `MoA_SC27_Complete_v43/figures/` and the TeX `\graphicspath` points there.

[GitHub Actions PDF build](../../.github/workflows/build-moa-unabridged.yml) compiles the **full document** and uploads the PDF as an artifact. Check the run result before declaring the full PDF certified. The earlier 20-page integrated summary is **not** this full monograph.

## Conference paper (locally compiled; GitHub source sync outstanding)
The latest locally compiled ACM review candidate is `MoA_Final_Release_20261112/Conference_CC2027/main.tex` with its matching PDF, distributed in the conversation archive `MoA_CC2027_Paper_GitHub_Monograph_Build_20261112.zip`. It has 10 pages of main text, references beginning at the end of page 10, and zero overfull-box warnings on its two-pass LaTeX build.

**Do not mistake** the tracked `submissions/CC2027/moa_cc2027_10page.tex` (an older 11.6KB incomplete draft) for that compiled local paper (approximately 56KB). This is an outstanding source synchronization gate, not a passing gate.

## Scientific boundaries
Paper I forward attention only. For identical observable contracts, DNF semantics must agree; derivations can differ. γ ONF depends on layout and Ω ONF on machine shape. Historical ACCESS hand-derived measurements and compiler-generated UA measurements have distinct provenance. A100 shape selection performed well in the frozen six-case study; absolute-time C4 failed twice. Do not retroactively change the predictions.

## Open gates
- Full monograph PDF CI success and visual review
- Exact CC candidate LaTeX/PDF sync to GitHub and reviewer anonymity
- Primary-source bibliography verification and complete supplementary archive provenance
- Clean-checkout compiler replay (frozen archive replay alone is not this test)
