# Compiler correctness audit and bounded semantic lowering

Reviewed source: `develop/mac-compiler-integration` at
`885cdb00f3537fa39b2088296ba6271122bf6132` (23 November update).
Corrections are isolated on `codex/compiler-correctness-20261124`.
The original branch, historical measurements, CSV and plots are preserved.

## Outcome

The original new Fortran path did **not** support a general source→DNF→code
correctness claim. The older typed IR/scheduling suites passed, but independent
source mutations exposed semantic loss. This audit adds a bounded, fail-closed
source adapter, structured arithmetic/state contracts, actual expression lowering,
and full-field host verification. It is an engineering correction and extension,
not a universal compiler proof or a new device-performance study.

### Findings at the reviewed commit

| Finding | Evidence | Resolution in this branch |
|---|---|---|
| Frontend ignored numerical source changes | Changing 0.25 to 0.5 or removing the forcing term yielded the same DNF | Source arithmetic becomes a validated AST; changes affect DNF and all supported generated computations |
| Backend did not lower its DNF argument | All 20 generator functions produced identical executable text for an unrelated replacement DNF | Public emitters require structured semantic IR and render its arithmetic tree into the update |
| Unknown/case-varied Fortran was misclassified | Uppercase source lost neighbours; an empty program received a fabricated stencil | Case-insensitive bounded skeleton adapter; unsupported statements/loop skeletons reject |
| Displayed DNF misplaced source coefficient | Printed `0.25*sum + h²*f`; source executed `0.25*(sum+h²*f)` | Parenthesized AST-derived ψ expression preserves source association |
| Island finite-sweep algorithms were conflated | Serial component (32,32) = 0.0125835231; snapshot-parallel = 0.0116506834 at the original 64×64 fixture | Explicit `lexicographic_islands` versus `snapshot_islands` state contracts; no implicit equivalence claim |
| Island MPI accepted illegal decompositions | At 3 ranks it returned 0.00905112550 at (32,32), despite the 1/2/4-rank snapshot value 0.0116506834 | Unaligned slabs reject; rank counts exceeding owned rows reject |
| Matmul MPI dropped remainder rows | `n_local=n/nprocs` had no remainder ownership | Balanced quotient/remainder ownership; full 33×33 fields checked at 2 and 5 ranks |
| Jacobi gates no longer tested emitted problem | Generators changed to CLI Burkardt problem, but gates supplied no argument and parsed old output; all four gate entry points failed | Fixed-sweep canonical fixtures restored consistently across targets; Burkardt variant remains in original Git history, not falsely certified as a source round-trip |
| Parallel Python island oracle crossed an island edge | Exclusive Python end bound used `>` instead of `>=` | Reference edge test corrected |
| Source-to-code provenance missing | DNF was comment text without binding to emitted computation | Each generated file has semantic IR and source/code/lowering-module hashes |
| Historical performance not reproducible from checkout | CSV has aggregate rows; `generate_island_gauss.py`, `check_island_gauss.py`, standard sources and raw repeat logs are absent | Results remain historical/reported; no re-certification or unconditional timing transfer |

Raw original probes and actual compiled/run observations are in
[compiler_audit_20261124](compiler_audit_20261124/).
Original Fortran runs used bounds checking and, where needed, unlimited line length.
OpenACC used explicit CPU fallback. Compiler setup failures are recorded separately
from numerical failures; missing local MPI development files were supplied before
rerunning the original code.

## What is implemented now

- **Bounded input:** the four reviewed loop/declaration skeletons (Jacobi,
  lexicographic Island Gauss, RBGS and square matmul), case-insensitive, with
  checked scalar parameters and arithmetic ASTs. Unknown executable statements,
  initializations or loop structures fail rather than silently generate code.
- **Numerical expression:** ordered addition/subtraction/multiplication/division,
  real scalar literals, `h`, and the supported tuple ψ selections. Types are
  restricted to the real32 fixture contract; parameter arithmetic follows its
  real32/integer category. This is not arbitrary Fortran or an external parser.
- **State semantics:** previous-sweep Jacobi, lexicographic island sweeps,
  red-then-black updates, or ordered k accumulation. A shared local stencil is
  not a proof that complete iterative solvers have the same DNF.
- **Explicit transformation:** `snapshot_islands` opts into a different
  finite-sweep state contract for OpenMP/OpenACC/MPI. The adapter rejects silent
  substitution for lexicographic islands. Serial and MATLAB snapshot schedules
  are not implemented and reject rather than reuse the wrong schedule.
- **Output:** trusted schedule templates lower the validated arithmetic AST to
  Fortran serial/OpenMP/OpenACC/MPI and MATLAB. Shape, iteration and spacing
  parameters are propagated. Fixture diagnostics require extents ≥32; island
  tiles must divide extents, and MPI island slabs must align to tiles.
- **Honest proof boundary:** the structured object is a local-update/state IR.
  It does not yet constitute general ψ reduction of arbitrary Fortran to a fully
  eliminated end-DNF, or a mechanically verified ONF/lifting theorem. Template
  schedules and lowering code are part of the trusted base. Floating-point
  comparisons are finite evidence. Optimizing or reordering these schedules
  requires a separate legality argument.

MATLAB output uses explicit single-precision arrays and rejects run-time shapes
that contradict the compiled IR. It has not been executed here. OpenACC inner
island sweeps have explicit sequential loop directives. Physical device behavior
still requires device execution; host fallback is always labeled as host-only.

## Reproduce

Python 3.10+, NumPy and pytest; GCC C/OpenMP for attention checks. The extended
Fortran run also needs gfortran with OpenACC support, compatible MPI Fortran
modules/libraries and mpirun. MATLAB/Octave and GPUs are optional independent
validation environments, not implicitly covered by this command.

```bash
python -m pip install numpy pytest
python fortran_frontend/fortran_frontend.py fortran_frontend/examples/*.f90
python fortran_frontend/poisson_backend.py --snapshot-islands
python scripts/validate_compiler.py --with-fortran --out validation_results
```

Generation without explicit `--snapshot-islands` fails before writing outputs,
since the full fixture set includes a distinct parallel-island algorithm.
`pytest.ini` selects actual pytest suites. Older attention scripts with a `main`
entry point are executed separately by the replay command, rather than collected
as pytest fixtures or run through their old macOS-only build instructions.

The full-field harness adds diagnostic output to generated programs without
changing their numerical updates. It compares every interior output component,
not only the two printed samples. Stencils use 40×48 shapes and seven sweeps;
matmul uses 33×33, including uneven MPI ownership. Two numerical source mutations
(halved stencil coefficient and removed forcing term) are executed across targets.
The independent original fixed-example gates are also retained and repaired.

## Interpretation and remaining validation

The verified attention correspondence branch remains separate:
[44-pair Papers I–IV evidence](https://github.com/womenflyplanes/moa-compiler/tree/codex/papers-i-iv-correspondence).
That study does not automatically certify these newly added Poisson solvers.

No CUDA/AMD/NVHPC device run, MATLAB execution, cluster submission or prospective
performance experiment was performed by this audit. Historical Delta CSV speedups
cannot be independently certified without the exact executed MoA/standard sources,
compiler flags, allocation/affinity, raw repetitions and source hashes. Matching
DNFs or GATE values alone do not establish equal measured performance.

The next mathematical extension is to make the array-wide versioned state and
schedule legality explicit in a common MoA AST, prove DNF→ONF obligations for each
schedule, and replace additional template assumptions with checked transformations.
General language frontends remain future work.

## Explicit stage interface and latest verification

`compile_moa.py` exports separate `01_dnf.json`, `02_onf.json`,
`03_lifted_onf.json`, generated source and a source/stage/code-hash certificate.
For matmul the denotation is a finite sum, with the operational accumulator
eliminated. Iterative stencil DNFs retain versioned array reads and sweep/phase
state; the common local stencil is not used to claim identical solver semantics.
ONFs record phases, reduction order and γ layouts. Lifted ONFs record logical
ownership, actual local layout, communication and admissibility. MPI rank-count
contracts are also enforced in code emitted by the new CLI. All emitters consume
these checked stages before lowering scalar expressions into schedule templates.

```bash
python fortran_frontend/compile_moa.py fortran_frontend/examples/matmul.f90 \
  --target mpi --owners 3 --out build/matmul_mpi
python fortran_frontend/compile_moa.py fortran_frontend/examples/poisson_jacobi.f90 \
  --target openmp --owners 4 --out build/jacobi_openmp
python fortran_frontend/compile_moa.py fortran_frontend/examples/poisson_island_gauss.f90 \
  --target openacc --snapshot-islands --out build/island_openacc
```

OpenMP/OpenACC owner bindings remain runtime-assigned. Logical coverage and
work-count invariants are not physical placement certificates. MPI owned values
preserve τ across disjoint owner intervals; extra halo storage is explicitly
accounted separately rather than called a τ-preserving array reshape.

Latest checks: **297 pytest cases pass, one skips, and 84 subtests pass**;
**48 full-field compiled comparisons pass** with zero observed difference against
the scalar real32 reference, including actual multi-process MPI host execution.
The four repaired canonical gate scripts pass. Additional attention numerical,
shape/index, machine-mapping and ψ→ONF executable checks pass. MATLAB and physical
GPU execution were not performed. Logs and machine-readable results are in
[compiler_audit_20261124](compiler_audit_20261124/).

Cleanup moved cost-only/empty-loop “paper” examples, unsupported legacy GUI
claims, old architecture drafts and unreproducible Poisson benchmark drafts out
of active paths. It removed only regenerable build/desktop files and exact
byte-duplicate documents. Original experiments and original archived source bytes
were preserved. The full move/removal list is in `cleanup.json`.
