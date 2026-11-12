# Two-document publication architecture

**1. Complete research monograph:** `research_monograph/MoA_Compiler_Complete_Research_20261112.tex`

This is a cumulative navigational and mathematical synthesis. It is **not yet a fully merged transcription of every historical manuscript, program, plot, table, or log**. The authoritative primary records remain intact in `MoA_SC27_Complete_v44/`, `MoA_SC27_Complete_v43/`, `SC27_FINAL/`, root LaTeX variants, `research/`, `results/`, `tools/`, and `tests/`. This monograph references an existing pipeline PNG in `MoA_SC27_Complete_v44/figures/`. The compilation environment and figure availability must be checked. Never delete historical source to shorten this monograph.

**2. Separate anonymous CC 2027 paper:** `submissions/CC2027/moa_cc2027_10page.tex`

This is the conference-focused ten-page submission track. It has a separate document class, figures, editorial priorities, and anonymity constraints. It must be built and page-count verified. It must never replace the research monograph.

## Sync rule

For each claim in the conference paper, record: the corresponding monograph section, mathematical theorem or derivation, implementation revision, test and log, machine property array, source of numerical data, and SHA256-frozen evidence. Update the monograph first with verified details, then distill them into the conference paper. A historical hand-derived result must never be labeled compiler-generated.

## Next editorial integration

1. Inventory all legacy TeX, programs, ASTs, PNGs, tables and experiment logs; produce a machine-readable inclusion index.
2. Merge complete relevant derivations and figures into the monograph with clear provenance and without erasing source variants.
3. Verify the ten machine-property fields and their units from canonical code.
4. Add full tables and frozen predictions with original evidence, including failures.
5. Build and visually inspect both PDFs; record their actual page counts and compilation logs.
6. Finalize anonymous bibliography and artifact for the conference paper.

## Known limitations

The current monograph is a substantial first synthesis, **not** yet the promised all-inclusive finished volume. The original frozen external ZIPs are not yet confirmed present in this repository. The conference paper remains an incomplete draft. No PDF has been generated or verified by this GitHub commit.
