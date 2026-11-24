# Mathematics of Arrays compiler: consolidated research release

MoA is being developed as a universal intermediate representation for array
languages, with portable, scalable, predictable and validated designs.

**[Start/resume here](RESUME_HERE.md)** · **[Completed results](docs/HAND_COMPILER_RESULTS.md)** · **[Compiler audit](docs/COMPILER_AUDIT_20261124.md)** · **[Architecture](docs/MOA-UNIVERSAL-IR.md)** · **[Repository guide](docs/REPOSITORY_GUIDE.md)**

## Completed attention correspondence

The Papers I–IV C-emission study is closed: all 39 numerical regions in 20 pinned
hand C files have compiler counterparts. All 44 paired cases and all 246 end-DNF
component comparisons pass; paired addressed ONFs, computational C structures and
common-admissible-machine lifted structures match. All 138 untimed pair checks
have zero difference. Source provenance and initialization premises are explicit.

## One release interface

```bash
python -m pip install -r requirements-validation.txt
python moa.py status
python moa.py attention --case p1_forward_cpu --out build/forward
python moa.py fortran fortran_frontend/examples/matmul.f90 --target mpi --owners 3 --out build/matmul
python moa.py validate --with-fortran --out validation_results/unified
```

Exports contain DNF, ONF, lifted ONF, generated code and hash-bound certificates.
The attention adapter reuses the completed study after verifying its recipe and
operational AST identities. The bounded Fortran adapter retains its own checked
schema. A unified interface does not imply a general language parser or a fully
shared cross-family AST. Existing source contracts and proof boundaries remain.

GCC is required for attention execution. Full Fortran/MPI replay additionally
requires GFortran, MPI development libraries and `mpirun`. OpenACC execution in
this replay is explicitly host-only. It submits no jobs and performs no new
hardware performance experiment.

## Evidence and continuity

- [Evidence index](continuity/EVIDENCE_INDEX.md)
- [Machine-readable completed/pending state](continuity/PROJECT_STATE.json)
- [Manuscript status](continuity/MANUSCRIPT_STATUS.md)
- [Archived material and cleanup](archive/README.md)

The five-paper attention suite remains the research foundation. Earlier Paper I
compiler hardware results and Paper V hand-realization measurements are retained
with their original identities. Missing build/source bindings limit quantitative
timing transfer; mathematical correspondence itself is completed in the stated
scope. The ten-field machine property record remains append-only extensible.

## Repository hygiene

`main` is the canonical current compiler and manuscript tree. Superseded drafts,
unused screenshots, bundled ZIPs, duplicate evidence trees and obsolete launchers
are removed from the current checkout; exact recovery hashes remain in
`continuity/ESSENTIAL_CLEANUP.json`. Required source and validation records are
preserved. See the repository guide for the essential directory layout.
