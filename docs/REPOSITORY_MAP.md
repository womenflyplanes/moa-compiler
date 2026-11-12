# Repository map and preservation policy

This repository contains active compiler development alongside historical manuscripts, validation logs, experiments, generated files, and source archives. **Do not assume every root-level item is part of the installable compiler.**

## Entry points

- [README](../README.md): current Paper I research scope.
- [Paper I usage](PAPER_I_USAGE.md): current documented development-checkout procedure and archive hashes.
- [CC 2027 artifact status](CC2027_ARTIFACT_STATUS.md): verified, pending, and unconfirmed claims.
- `research/`: typed-attention and Paper I research paths (inspect specific checkpoint before use).
- `tests/`: tests and audit programs (not all necessarily independent or runnable without optional toolchains).
- `papers/`, `*.tex`, `*.pdf`: manuscripts and historical drafts.
- `results/`, `*.log`, `*.zip`: evidence and archival records; their existence alone does not establish provenance.

## Cleanup rules

1. Preserve original experimental records, manifests, and SHA256-frozen artifacts.
2. Inventory before moving; moving a file may break imports, scripts, citations, or archived relative paths.
3. Treat compiled PDFs, TeX auxiliaries, caches, and research ZIPs differently: some are reproducible build outputs; others are historical evidence.
4. Make structural changes in reviewable commits and verify all documented paths after each move.
5. Do not conflate a semantic machine shape with the ten-field machine-property record.
6. Keep Papers II–IV as compositional motivation, not as completed compiler reproduction.
7. Never claim a clean-clone run, GPU execution, or performance validation without its logs and source hashes.

## Suggested future layout (proposal, not yet implemented)

`compiler/`, `examples/paper1_forward_attention/`, `tests/`, `docs/`, `artifacts/paper1/`, and `papers/`.

Reorganize only after dependency analysis and a reproducibility gate.
