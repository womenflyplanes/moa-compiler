# CC 2027 Paper I artifact: verification status

Checkpoint: 2026-11-11. Branch: `cc2027-artifact`. This is a cleanup and audit branch, **not a release**.

## Scope and evidence classes

The active case study is Paper I forward attention. The target method is MoA specification (rho, psi, iota, reductions) -> canonical DNF -> gamma/ONF -> machine-shape-dependent Omega lifting -> mnemonic backend realization -> prospective prediction -> independent validation.

- **Mathematical requirement:** canonical DNF and ONF preserve the specification. Hand and compiler derivation sequences need not be textually identical. Lifted ONF is target-machine dependent. A compiler trace or finite test is not a general proof.
- **Compiler-generated evidence:** the preserved November 8 CPU foundation and November 10 matched-validation materials document bounded implementations and tests. The entire two-archive reproduction procedure has **not yet been retested from a fresh clone**.
- **Hand-derived evidence:** the original ACCESS GPU work is a distinct provenance class. Its A100 results must not be labeled compiler-generated.
- **Predictions:** report frozen prospective model selection separately from absolute time accuracy. Retain unsuccessful predictions and measured residuals.
- **Pending:** end-to-end generated A100 evidence, comprehensive physical machine-shape cost modeling, cross-platform matched validations, complete evidence distribution, and clean-checkout reproduction. Do not mark these complete.

## Independently rerun local archive checks (2026-11-11)

The preserved November 8 and November 10 local archive copies passed ZIP integrity checks and three CPU/protocol test commands in a disposable Linux environment. See [dated commands, environment, and results](PAPER_I_LOCAL_REPRO_20261111.md). This is **archive-based finite verification**, not clean-clone reproducibility or GPU validation.

## Current blockers to a review-ready artifact

1. The two dated ZIP archives listed in [PAPER_I_USAGE.md](PAPER_I_USAGE.md) are not yet hosted as a verifiable GitHub release or other stable public artifact.
2. Installation instructions depend on extracting both archives into a particular working-copy layout. This has not been independently exercised from a fresh clone.
3. The default branch `main` and this development branch have different audiences and contents; the canonical entry point needs a reviewed merge.
4. The repository root includes historical logs, generated TeX/PDF auxiliary files, binary archives, and multiple manuscript versions. Preserve these until provenance and dependencies are classified; do not delete or rewrite history.
5. Existing repository description references Papers 1–5 and a five-device ensemble; do not interpret this as evidence that the current compiler has generated and validated Papers II–IV.

## Release gates

- [ ] Pin the source commit and each immutable evidence archive by SHA256.
- [ ] Publish accessible archives with integrity checks; no fabricated links.
- [ ] Fresh-clone install on documented Python/GCC/OpenMP environment.
- [ ] Run the O-only and full-(A,O) Paper I paths; record exact commands, logs, checksums, and machine properties.
- [ ] Audit each emitted DNF/ONF/Omega artifact against the mathematical specification and the hand-derived reference.
- [ ] Keep hand-derived, compiler-generated, prediction-only, and unverified rows distinct in results tables.
- [ ] Audit citation and manuscript claims against the released artifact.
- [ ] Confirm clean checkout and documented licensing before release.

No checkboxes above are asserted complete by the creation of this document.

## Archive verification runner (committed, not yet end-to-end tested)

[tools/verify_paper1_archives.py](../tools/verify_paper1_archives.py) verifies the exact two frozen archive SHA256 digests, rejects unsafe ZIP paths and symlink entries, runs three CPU/protocol commands, and writes a JSON report plus stdout/stderr logs in a separate output directory. This new runner is **not yet verified by execution from a fresh GitHub checkout**; its code has been committed for review. It requires Python, NumPy, and GCC/OpenMP and the two original archive files.

```sh
python tools/verify_paper1_archives.py \\
  --consolidated /path/to/MoA_PaperI_Consolidated_20261108.zip \\
  --matched /path/to/MoA_PaperI_Matched_Validation_20261110.zip \\
  --output /tmp/moa-paper1-check
```

Use a fresh empty output directory for each run. The runner does not fetch private archives, does not execute a GPU kernel, and does not reproduce hardware timing predictions.

## 2026-11-11 refinement checkpoint

A second local execution of the preserved-archive verifier reported matching SHA256 hashes and three successful test exit codes (0, 0, 0). The saved 16-case stage-identity audit was also independently rerun. Its DNF/ONF JSON records contain symbolic dimensions; matching serialized templates must **not** be presented as a proof of instantiated semantic equivalence. See [the qualified stage audit](PAPER_I_STAGE_IDENTITY_AUDIT_20261111.md) and [its repeatable checker](../tools/audit_moa_stages.py). These checks were local, not a passing GitHub Actions run, a clean-clone artifact reproduction, or A100 validation.

## Mathematical acceptance gate: DNF convergence

The [Paper I DNF convergence contract](PAPER_I_DNF_CONVERGENCE_CONTRACT.md) is now the governing criterion for comparing the hand-derived and compiler-generated paths. Different legal derivation sequences are allowed; independently sourced ψ-reductions must reach the same canonical DNF under documented shape/domain assumptions. The existing symbolic JSON identity check is **not** that certificate. γ/ONF and target-specific Ω lifting follow as separate verified stages. **Status: open.**
