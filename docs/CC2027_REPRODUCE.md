# CC 2027 reviewer-facing reproducibility procedure

**Status: verification in progress, not a complete release.** The Paper I research artifacts remain externally frozen. This page documents a finite local CPU check, not a prospective GPU result.

## Inputs and integrity

Obtain the two original, immutable archives listed in [Paper I usage](PAPER_I_USAGE.md). The currently documented SHA256 digests are:

```text
296b2767e77749a909039c64c4909ef7761a644c4942d42680d49f5cb4a78659  MoA_PaperI_Consolidated_20261108.zip
099b9ca79f65e51a636884bde0efa75d0fc5f985db75b0e1c6c7565a203fa151  MoA_PaperI_Matched_Validation_20261110.zip
```

The archives are **not** yet a public GitHub release. A fresh clone alone is therefore insufficient. Do not claim otherwise.

## Verification commands

```sh
git clone --branch cc2027-artifact https://github.com/womenflyplanes/moa-compiler.git
cd moa-compiler
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy
python tools/verify_paper1_archives.py \
  --consolidated /absolute/path/MoA_PaperI_Consolidated_20261108.zip \
  --matched /absolute/path/MoA_PaperI_Matched_Validation_20261110.zip \
  --output /tmp/moa-paper1-check-$(date +%Y%m%d%H%M%S)
```

Prerequisites: Python 3.10+, NumPy, GCC with OpenMP support, sufficient temporary disk space. Do not use this command on a cluster login node unless local compilation is permitted. The runner checks both archive hashes, validates ZIP members, extracts into a fresh output directory, and runs the finite CPU and protocol tests. It writes stdout, stderr, and a JSON record. It does not run Slurm, MPI, OpenACC, or A100 experiments.

A test's exit code is necessary but not sufficient to prove the full compiler semantics. Review the emitted test output and independent references. In particular, compare the **canonical** DNF and ONF for the same source specification and layout; compare Ω-lifted ONFs only with identical target machine shape and mapping constraints.

## Paper-facing claims gate

| Claim | Minimum evidence | Status |
|---|---|---|
| Frozen archive identity | Matching SHA256 | Local copies verified 2026-11-11 |
| Finite CPU compiler correctness | Original independent reference and logs | Local archive tests passed |
| Protocol prevents unfrozen reruns | Negative protocol tests | Local archive tests passed |
| Fresh clone is independently reproducible | Public source/evidence + clean environment run | **Open** |
| Canonical DNF and ONF match hand derivation | Independently inspected stage records and proof obligations | **Open** |
| Prospective selection accuracy | Pre-execution freeze + independent held-out runs | Mixed historical evidence; full audit open |
| Compiler-generated A100 case study | Generated code, host/device logs, validation, frozen predictions | **Open** |
| General semantic preservation | Formal rules and proof, beyond finite tests | **Open** |

Preserve failed holdouts and residuals, do not retrofit a model to held-out measurements, and do not conflate hand-derived ACCESS measurements with compiler-generated measurements.

## Release blockers

Publish checksum-verifiable archives, confirm source license and dependencies, run the documented steps on a clean checkout, preserve exact tool versions and compiler flags, and record the successful commit SHA. Until these are done, this is a documented development workflow, not a complete artifact evaluation package.
