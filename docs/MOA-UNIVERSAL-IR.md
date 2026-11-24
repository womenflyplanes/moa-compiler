# MoA universal intermediate representation: goal and current implementation

The goal is a common mathematical representation for array languages, supporting
portable, scalable, predictable and validated designs. The five-paper attention
suite supplies the motivating algorithms, hand derivations and empirical research
basis. Language and machine coverage must be established incrementally.

## Representation contract

| Stage | Meaning | Current checked implementation |
|---|---|---|
| Source adapter | Preserve source shapes, types, indexing, arithmetic and state | Bounded Fortran fixture grammar; structured attention specifications in separate modules |
| DNF | Index-only meaning with explicit versioned array reads | `moa_stages.dnf`; matmul finite sum eliminates its accumulator; stencil state recurrences distinguish solvers |
| ONF | Address/layout, loop and reduction order, storage, phases and legal parallelism | `moa_stages.onf`; explicit phase/layout records consumed during emission; schedule templates remain trusted |
| Lifted ONF | Map work and state to target ownership and communication | `moa_stages.lift`; checked logical MPI coverage, halo contracts, shared-memory/accelerator phase obligations |
| Code and validation | Implement the declared contract and bind evidence | Semantic scalar lowering, source/stage/code hashes, full-array reference checks and negative tests |

Canonical selection is `⟨i,j⟩ ψ A`; all represented arrays here have rank two.
Column-major γ uses `(i−lower_i) + ρA[0]*(j−lower_j)`. Scalar arithmetic order is
retained. A DNF for an iterative algorithm must include the versions of the
arrays it reads. Jacobi, lexicographic Island Gauss, snapshot-island sweeps and
RBGS are not interchangeable finite-sweep computations merely because they use
the same local stencil. The explicit snapshot-island transform changes that
state contract and is never silently presented as a semantics-preserving rewrite.

MPI ownership partitions the logical interior into disjoint covered slabs. The
sum of owned component counts is preserved: τ(owned before)=τ(owned after).
Communicated halo storage is additional storage, not a τ-preserving reshape of
the original logical interior. Island slab boundaries must align with islands.
OpenMP/OpenACC lifting records independent work and phase barriers; it does not
assert physical placement from a pragma. Machine parameters and runtime binding
must be supplied and validated before making physical performance claims.

## Compile and inspect the stages

From the repository root:

```bash
python fortran_frontend/compile_moa.py fortran_frontend/examples/matmul.f90 \
  --target mpi --owners 3 --out build/matmul_mpi
python fortran_frontend/compile_moa.py fortran_frontend/examples/poisson_jacobi.f90 \
  --target openmp --out build/jacobi_openmp
python fortran_frontend/compile_moa.py fortran_frontend/examples/poisson_island_gauss.f90 \
  --target openacc --snapshot-islands --out build/island_openacc
```

Each output contains `01_dnf.json`, `02_onf.json`, `03_lifted_onf.json`, generated
source and `certificate.json`. MPI compiled with an explicit owner count rejects
a run using another rank count. Output source alone does not establish execution.

## Validate

```bash
python -m pip install numpy pytest
python scripts/validate_compiler.py --with-fortran
```

The full host replay requires GCC, gfortran, MPI Fortran development files and
mpirun. OpenACC checks explicitly disable device offload and use host execution.
MATLAB and actual accelerator execution remain separate validation tasks.
See the [audit](COMPILER_AUDIT_20261124.md) for results, prerequisites and limits.

## Relationship to the attention suite

The [Papers I–IV correspondence branch](https://github.com/womenflyplanes/moa-compiler/tree/codex/papers-i-iv-correspondence)
contains its own 44-pair evidence and source contracts. Earlier Paper I hardware
measurements and Paper V performance research supply empirical context, not an
automatic certificate for newly added frontends or kernels. Each new source
adapter and target mapping must meet the same semantic/evidence requirements.

A general array-language frontend, common cross-family AST, mechanically proved
schedule transformations, broad target coverage and machine-calibrated predictive
models remain extensions. No architecture-independent timing, power or energy
accuracy is claimed from the present normal-form records alone.
