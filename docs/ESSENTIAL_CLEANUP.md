# Essential repository cleanup

Removed 206 nonessential files (60,648,525 bytes) from the current checkout: obsolete manuscript variants, unused screenshots, bundled repository ZIPs, duplicated source/evidence trees, illustrative cost-only artifacts and obsolete external launchers.

The compiler, all 20 pinned hand C sources, 44-case/246-DNF correspondence evidence, current manuscripts, tests, required historical binding archive and distinct run records remain. Five historical root logs have a single canonical location in `evidence/historical_legacy/`. Author/anonymous Overleaf sources remain self-contained.

Every removed file has a recovery commit, byte hash and Git blob identity in `continuity/ESSENTIAL_CLEANUP.json`. No history was rewritten or scientific experiment rerun.

Validation after cleanup: 303 tests and 84 subtests passed (one documented skip); all 44 proof-bound attention exports passed; all 403 Paper I historical bindings and 20 hand-source hashes passed; current entry-point links were checked.

`main` is the canonical current release. Historical branches remain checkpoints, not competing current implementations.
