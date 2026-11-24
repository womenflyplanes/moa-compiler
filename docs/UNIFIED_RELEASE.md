# Consolidation release: completed science, one reproducible entry point

This release merges the existing 44-pair attention correspondence branch with
the November 24 bounded compiler-audit branch. It preserves both parents and
completed evidence. No algorithm derivations are reopened.

`python moa.py attention --case PROFILE --out DIRECTORY` selects one existing
case, verifies its frozen proof/generator dependencies, recipe hash and addressed
operational AST, and exports DNF, ONF, lifted ONF, source and a bound certificate.
Fresh-device initialization corrections are selected by default where required.
`--archival-entry` explicitly requests the historical entry contract instead.

`python moa.py fortran SOURCE --target TARGET --out DIRECTORY` dispatches to the
bounded Fortran adapter; MPI also requires `--owners`. Snapshot-island algorithms
require explicit `--snapshot-islands` and retain a different versioned DNF.
These are adapters with distinct proven scopes, not a claim of a universal parser.

`python moa.py validate --with-fortran --out DIRECTORY` runs the compiler host
suite, the independent complete attention replay and seal, historical bindings,
all 44 exports and C syntax checks. No GPU or cluster job is submitted.

The release adds a persistent completed/pending ledger, evidence index and
manuscript status. Subsequent work should extend missing language, target and
prediction capabilities, rather than repeat the completed attention study.
