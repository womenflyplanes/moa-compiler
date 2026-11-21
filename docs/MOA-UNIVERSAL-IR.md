# MoA Compiler: One DNF, Many Backends — A Universal Intermediate Language

## 1. The Claim

One denotational form, one operational form per target, many backends.
No hand-written side files. No per-backend re-derivation of the mathematics.

```
Fortran source
  -> Fortran frontend
  -> DNF  (Denotational Normal Form: what the computation IS)
  -> ONF  (Operational Normal Form: how it runs on a target)
  -> Gamma / Omega (shape, layout, cost)
  -> Backend: Fortran / MATLAB / OpenMP / GPU (OpenACC) / MPI / (Julia next)
```

If the DNF is right, every backend must give the same answer, up to floating-point
rounding, for the same problem. That is the test. Not speed. Correctness first,
then performance.

This document records what is done, what is gated, and what "universal IR" means
in this work. All numbers are for the canonical test case unless stated otherwise:
nx = 64, ny = 64, niter = 100, h = 1/63, f = 1.0, u = 0, island size 16x16.

Branch: develop/mac-compiler-integration, repo womenflyplanes/moa-compiler.
Working folder: moa-compiler-fresh. Do not edit moa-compiler-push (archive, main).

---

## 2. MoA in One Paragraph

Mathematics of Arrays (MoA) treats an array program as an algebra of shapes and
indexing, not as loops.

- An array has a **shape** (its extent in each dimension) and a **ravel** (its
  elements in row-major order).
- Indexing is **psi** (ψ): `A ψ ivec` is the element of A at index vector ivec.
- A program is a composition of psi-reductions, reductions, and reshapes.
- **Psi reduction** rewrites indexing into a normal form. That normal form is the DNF.
- The DNF says nothing about loops, threads, halo, or memory. It says what
  value lives at each index.
- The ONF adds the operational decision: what is parallel, what is sequential,
  what is exchanged, what is stored. Different targets get different ONFs from
  the same DNF.
- **Gamma / Omega** carry shape, layout, and cost: how big the arrays are,
  how they are laid out, and what the computation costs in time, bytes, energy,
  dollars. Cost model: T ≈ αF + βM, where F is flops and M is memory traffic.
  The compiler chooses the ONF that minimises cost for the target, not the
  programmer.

That is the whole idea. The rest is engineering.

---

## 3. DNF — Denotational Normal Form

DNF = what the computation IS. No loops. No threads. No halo. No memory.

### 3.1 Poisson — one DNF for three solvers

Jacobi, Island Gauss, and Red-Black Gauss-Seidel share one DNF. They differ
only in ONF: how the update is scheduled.

```
u_new ψ ivec = 0.25 * sum_{ (di,dj) in {(-1,0),(1,0),(0,-1),(0,1)} }
                 (ivec + (di,dj)) ψ u  +  h^2 * (ivec ψ f)
```

In words: the new value at ivec is the average of its four neighbours in the
old u, plus the source term. That is the mathematics. It does not say whether
neighbours are read from the old array or the new array, whether islands run
in order or in parallel, or whether the halo is exchanged. Those are ONF
decisions.

### 3.2 Matmul — a second DNF

```
C ψ (i,j) = sum_k ( (i,k) ψ A ) * ( (k,j) ψ B )
```

In words: C(i,j) is the dot product of row i of A and column j of B.
Again, no loops, no parallelism. The ONF decides: i,j parallel, k sequential.
Parallelising k would be a reduction, not a parallel loop. The ONF forbids it.

### 3.3 What the frontend does

`fortran_frontend.py` parses a Fortran subset — do loops, neighbour offsets,
`mod(i+j,2)`, island blocks — and produces the DNF as JSON (`.dnf.json`).
It does not emit Fortran directly. It emits DNF. The backend emits Fortran,
MATLAB, OpenMP, GPU, MPI from that DNF.

Frontend subset, as gated:
- Stencil + matmul
- Do loops
- Neighbour offsets (i±1, j±1)
- mod(i+j,2) for Red-Black colouring
- Island blocks (16x16)

Four frontend examples, each with a `.dnf.json`:
- poisson_jacobi
- poisson_island_gauss
- poisson_rbgs
- matmul

---

## 4. ONF — Operational Normal Form

ONF = how it runs on a target. Same DNF, different ONF per backend.

### 4.1 The ONF rule

**Parallelise only where ONF says parallel. Nowhere else.**

| Program | ONF parallel | ONF sequential | Why |
|---|---|---|---|
| Jacobi | all ivec | sweep (iteration) | Reads old u, writes new u_new. No dependency. |
| Island Gauss, serial | islands in order | inside island, Gauss-Seidel | Later islands see earlier islands' updates in same sweep. |
| Island Gauss, parallel / GPU / MPI | islands in parallel | inside island, Gauss-Seidel | Halo-safe: halo from u_old, inside from u. See §5. |
| RBGS | one colour at a time, all ivec of that colour | colour phases, barrier between | Same-colour points do not neighbour each other. |
| Matmul | i,j | k (reduction) | Each C(i,j) is independent. k is a sum. |

### 4.2 Why Island needed a second ONF

Serial Island and parallel Island are different ONFs, and they give different
answers. This was found by gating, not by theory alone.

- Serial Island: Gauss-Seidel across islands too. Later islands see earlier
  islands' updates in the same sweep. Result at nx=64: u(32,32) ≈ 0.0125835.
- Parallel Island, halo-safe: Gauss-Seidel inside island, Jacobi across
  islands. Halo reads from u_old (previous sweep). Result: u(32,32) ≈ 0.0116506.

Both are correct ONFs. They are not the same ONF. The parallel ONF is the
one that is correct for OpenMP, GPU, and MPI. The naive parallel ONF — parallel
islands, no halo — is racy and nondeterministic. It gave a different number
every run on OpenMP (0.012574, then 0.012583, then ...). That is not a rounding
error. That is a race on island edges: island (ix,iy) writes u(i1,j) at its
right edge, island (ix+1,iy) reads that same cell as its left halo in the same
sweep. Without a halo, the reader sometimes sees the old value, sometimes the
new value.

Fix: halo-safe ONF. At the start of each sweep, copy u -> u_old. Inside an
island:
- Neighbour inside the same island -> read from u (newest, Gauss-Seidel)
- Neighbour outside the island -> read from u_old (halo, previous sweep)

That is deterministic. It is what a GPU halo exchange and an MPI halo exchange
do. It is the ONF that makes Island correct on every parallel backend.

### 4.3 Gamma / Omega

Gamma and Omega are not yet fully wired as separate passes in this codebase.
They are the shape / layout / cost layer that the ONF must respect:

- **Shape**: nx, ny, island size, slab size, array extents including halo.
  The compiler knows the shape at compile time, not just at run time.
- **Layout**: row-major, slab over j for MPI, island blocks for GPU, halo
  rows for MPI, u_old snapshot for parallel Island.
- **Cost**: T ≈ αF + βM. F = flops per sweep, M = bytes moved (halo + arrays).
  At nx=64, M dominates for MPI: halo exchange costs more than compute saved.
  At nx=1024 / 2048, F dominates and parallel backends win. The gate measures
  correctness; the bench measures cost. Both are part of the ONF decision.

---

## 5. All Programs — What Each One Is

All four share the same test case: nx=64, ny=64, niter=100, h=1/63, f=1.0, u=0.
All numbers below are the gated reference values (Python reference, double
precision, compared to Fortran single precision in the gates with tolerance
1e-5, observed errors ~3e-7).

### 5.1 poisson_jacobi

- **Mathematics**: u_new(i,j) = 0.25 * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h^2 f(i,j))
- **ONF**: one phase per sweep, all ivec parallel, two arrays (u, u_new).
- **Reference**: u(32,32) ≈ 0.0062988, u(1,1) ≈ 0.0003770
- **Backends**: Fortran, MATLAB, OpenMP, GPU, MPI — all PASS, same number.

### 5.2 poisson_island_gauss

- **Mathematics**: same DNF as Jacobi. Different ONF.
- **Serial ONF**: islands in order, Gauss-Seidel inside island, Gauss-Seidel
  across islands. Reference: u(32,32) ≈ 0.0125835.
- **Parallel ONF (halo-safe)**: islands in parallel, Gauss-Seidel inside,
  Jacobi across islands via u_old halo. Reference: u(32,32) ≈ 0.0116506.
- **Why two numbers**: they are two different ONFs. Both are correct. The
  parallel ONF is the one used for OpenMP, GPU, MPI. Do not expect them to match.
- **Backends**: Fortran (serial ONF), MATLAB (serial ONF), OpenMP (parallel ONF),
  GPU (parallel ONF), MPI (parallel ONF) — all PASS against their own reference.

### 5.3 poisson_rbgs (Red-Black Gauss-Seidel)

- **Mathematics**: same DNF as Jacobi. Different ONF.
- **ONF**: red phase parallel, barrier, black phase parallel. Colour = mod(i+j,2).
  Same-colour points do not neighbour each other, so one colour is safe to
  parallelise. The barrier is not optional. Without it, black reads a mix of
  old and new red values and the answer is wrong and nondeterministic.
- **Reference**: u(32,32) ≈ 0.0125272, u(1,1) ≈ 0.0004318
- **Backends**: Fortran, MATLAB, OpenMP, GPU, MPI — all PASS.

### 5.4 matmul

- **Mathematics**: C(i,j) = sum_k A(i,k) * B(k,j)
- **ONF**: i,j parallel, k sequential reduction.
- **Reference**: C(32,32) = 64.0 exactly (A = 1.0, B = 1.0, n = 64)
- **Backends**: Fortran, MATLAB, OpenMP, GPU, MPI — all PASS, exact.

---

## 6. All Backends — What Each One Emits

All backends are generated from the same DNF by `fortran_frontend/poisson_backend.py`.
No hand-written side files. Regeneration is `python3 fortran_frontend/poisson_backend.py`.

Generated files live in `moa_compiler/generated/`:
- `*_out.f90` — serial Fortran
- `*_out.m` — MATLAB
- `*_openmp.f90` — OpenMP Fortran
- `*_gpu.f90` — OpenACC GPU Fortran
- `*_mpi.f90` — MPI Fortran

### 6.1 Fortran (serial) — commit d0fe20b, 6c0ec48

- Direct translation of DNF to Fortran do loops.
- Jacobi: two arrays, swap per sweep.
- Island: islands in order, Gauss-Seidel inside.
- RBGS: colour loop, if (mod(i+j,2)==color).
- Matmul: triple loop, i,j outer, k inner.
- Gate: `test_poisson_gate.py`, PASS for all four, ~1e-7 vs Python.

### 6.2 MATLAB — commit 2cab04b

- Direct translation of DNF to MATLAB `.m`, vectorised where natural.
- Gate: `test_matlab_gate.py` + `test_matlab_gate_matlab.py`, real MATLAB R2024a
  at /Applications/MATLAB_R2024a.app/bin/matlab, PASS for all four, ~1e-11.
- Octave gate is SKIP-only if Octave is absent. Real MATLAB gate is the
  authoritative one.

### 6.3 OpenMP — commit d18aa3d

- `!$OMP PARALLEL DO COLLAPSE(2)` where ONF says parallel.
- Jacobi: parallel over i,j. Safe.
- Island: parallel over ix,iy islands, sequential inside. Halo-safe via u_old.
  Do NOT parallelise i,j inside island — that breaks Gauss-Seidel.
- RBGS: parallel over i,j inside one colour, barrier between colours.
- Matmul: parallel over i,j, k sequential.
- Gate: `test_openmp_gate.py`, 16 checks (4 programs x 2 thread counts x 2 values),
  1 thread vs 4 threads vs Python reference, all PASS.
- Island halo-safe fix was found by gating: naive parallel Island failed the
  gate with rel_err 7.2e-04, nondeterministic. Halo-safe passes with 2.28e-07.

### 6.4 GPU (OpenACC) — commit 8f4f015

- `!$ACC PARALLEL LOOP COLLAPSE(2)`, `!$ACC DATA COPY`, `!$ACC WAIT` where
  ONF says barrier.
- Jacobi: one kernel per sweep, all ivec parallel.
- Island: one kernel per sweep, parallel over islands, halo-safe via u_old.
- RBGS: two kernels per sweep, red then black, `!$ACC WAIT` between.
- Matmul: one kernel, i,j parallel, k sequential.
- Gate: `test_gpu_gate.py`, 8 checks, all PASS. On Mac, `-fopenacc` runs on
  host fallback (no GPU device). Same .f90 runs on Delta with
  `nvfortran -acc -gpu=cc80` with no code change — only compiler flags change.
- Island-GPU matches Island-OpenMP halo-safe number (0.0116506), not serial
  Island (0.0125835). That is the proof the halo works on GPU too.

### 6.5 MPI — commit cbe0260

- 1D slab over j, halo 1 row top/bottom via `MPI_Sendrecv` per sweep.
- Jacobi: halo exchange, compute slab, swap. No other communication.
- Island: halo exchange per sweep, NOT per island. That is the point of
  halo-safe ONF. Inside slab: islands, Gauss-Seidel inside, halo from u_old
  for island edges AND slab edges. Slab must align to island rows: ny_local
  = 64,32,16 for nprocs=1,2,4, all multiples of isy=16, so alignment holds.
- RBGS: halo exchange per colour phase = barrier. Red: exchange, compute red.
  Black: exchange, compute black.
- Matmul: row slab of C, broadcast B, no halo, no per-iteration communication.
- Gate: `test_mpi_gate.py`, 24 checks (4 programs x 3 rank counts x 2 values),
  1 / 2 / 4 ranks vs Python reference, all PASS. Island is rank-independent:
  1, 2, and 4 ranks give the same number (0.0116506).
- Bench: `bench_mpi.py`, Mac baseline. At nx=64, MPI is slower than serial
  for Poisson (halo overhead dominates). Matmul scales (no halo). See §7.

### 6.6 Julia — next, not yet done

Same pattern as MATLAB: `.jl` from DNF, gate against Python reference.
Not claimed until gated. Do not claim Julia until the gate passes.

---

## 7. Performance — What We Know, What We Do Not Know

### 7.1 Mac baseline, nx=64, niter=100, MPI bench

```
JACOBI-MPI,1,9.999e-05
JACOBI-MPI,2,4.217e-03
JACOBI-MPI,4,5.419e-03
ISLAND-MPI,1,1.491e-03
ISLAND-MPI,2,9.550e-04
ISLAND-MPI,4,6.045e-03
RBGS-MPI,1,7.800e-04
RBGS-MPI,2,3.834e-03
RBGS-MPI,4,4.591e-03
MATMUL-MPI,1,1.370e-04
MATMUL-MPI,2,7.200e-05
MATMUL-MPI,4,3.600e-05
```

Reading: at nx=64, MPI is slower than serial for Poisson. That is expected.
Halo exchange costs more than the compute saved. Matmul scales because it has
no halo exchange per sweep. Do not quote nx=64 numbers as "MPI is slower" in
general. Quote them as "at nx=64, overhead dominates."

### 7.2 Delta experiment — in progress, not yet run

- Scaling script: `bench_delta.py`, generates scaled copies at nx=256, 1024, 2048
  by replacing nx, ny, niter, h. 36 files in /tmp/delta_bench.
- Delta: same .f90 files, no code change. Only launcher changes:
  - Serial: `./jacobi_out_nx1024`
  - OpenMP: `OMP_NUM_THREADS=4 ./jacobi_omp_nx1024`
  - MPI: `srun -n 4 ./jacobi_mpi_nx1024`
  - GPU: `srun --partition=gpuA100x4 --gpus-per-node=1 ./jacobi_gpu_nx1024`
  - Compile on Delta: `module load gcc openmpi nvhpc`,
    `nvfortran -acc -gpu=cc80` for GPU, `mpifort` for MPI.
- Original vs MoA-derived comparison: pending. Need the original Fortran
  programs (hand-written, before frontend) to add to the same harness:
  same nx, ny, niter, same sweep-only timing, same ranks / threads, Mac + Delta.
  Until the originals are identified, do not claim a speedup.

### 7.3 What "original vs MoA-derived" means

- **Original**: hand-written Fortran Poisson programs, as first written, before
  the frontend. They encode one ONF implicitly, in loops.
- **MoA-derived**: `*_out.f90`, `*_openmp.f90`, `*_gpu.f90`, `*_mpi.f90` generated
  from DNF by `poisson_backend.py`. They encode the ONF explicitly, from DNF.
- The comparison is apples-to-apples only if both are timed the same way:
  same problem size, same niter, sweep-only timing, 3 repeats, min time,
  same compiler flags, same ranks / threads.
- The hypothesis to test on Delta: at large nx, MoA-derived MPI / GPU should
  match or beat hand-written, because the ONF is the same mathematics with
  explicit halo and explicit parallelism, and the compiler does not have to
  rediscover it. That hypothesis is not yet proven. It is the Delta experiment.

---

## 8. Gates — How We Know It Is Correct

Every backend has a gate. No backend is claimed until its gate passes.

| Backend | Gate file | Checks | Result |
|---|---|---|---|
| Fortran | test_poisson_gate.py | 4 programs x 2 values | PASS, ~1e-7 |
| MATLAB | test_matlab_gate.py, test_matlab_gate_matlab.py | 4 programs | PASS, ~1e-11, real MATLAB R2024a |
| OpenMP | test_openmp_gate.py | 4 x 2 threads x 2 values = 16 | PASS |
| GPU | test_gpu_gate.py | 4 x 2 values = 8 | PASS, host fallback on Mac |
| MPI | test_mpi_gate.py | 4 x 3 ranks x 2 values = 24 | PASS |
| Julia | — | — | Not yet done, not claimed |

Gate rule: compare Fortran (single precision) to Python reference (double
precision) with tolerance 1e-5. Observed errors are ~3e-7, well inside.
Matmul is exact: 64.0.

Gate rule for Island: serial and parallel have different references. Serial
Island reference is 0.0125835. Parallel (halo-safe) reference is 0.0116506.
Do not mix them. The gate for OpenMP / GPU / MPI Island uses the parallel
reference. The gate for serial Fortran / MATLAB Island uses the serial reference.

---

## 9. MoA Goals — Universal IR

### 9.1 What "universal" means here

Universal does not mean "one backend that runs everywhere." It means:

1. **One DNF for all backends.** The mathematics is written once, as DNF.
   Every backend is a translation of that DNF, not a re-derivation.
2. **ONF per target, derived from DNF.** The operational decision — what is
   parallel, what is sequential, what is exchanged — is made by the compiler
   from DNF + shape + cost, not by the programmer rewriting loops.
3. **Correctness by construction + gating.** If the DNF is right and the ONF
   respects the DNF's dependencies, the backend must be correct. The gate
   checks it. The Island race was found by the gate, not by inspection.
4. **Cost by shape.** T ≈ αF + βM. The compiler predicts cost from shape
   before it runs. At nx=64, it should predict MPI will lose. At nx=2048,
   it should predict MPI / GPU will win. That prediction is part of the IR,
   not an afterthought.
5. **Portability by ONF, not by rewriting.** Same DNF, same ONF, different
   compiler flags: `-fopenacc` on Mac (host fallback) vs `nvfortran -acc`
   on Delta (real GPU). Same file, same ONF, different target. That is the
   promise. The Delta experiment tests it.

### 9.2 What is NOT claimed

- Julia backend: not yet gated, not claimed.
- GPU on real hardware: not yet run on Delta A100 / H100. Mac host fallback
  validates the ONF, not the GPU performance.
- MPI scaling at large nx: not yet measured. Mac nx=64 shows overhead
  dominates. Delta at 1024 / 2048 is the test.
- Original vs MoA-derived performance: not yet measured. Pending originals.
- Gamma / Omega as separate compiler passes: not yet fully wired. Shape and
  cost are currently in the backend generators and bench scripts, not as
  standalone passes.
- Fortran frontend completeness: subset only (stencil + matmul, do loops,
  neighbour offsets, mod, island blocks). Full Fortran frontend is future work.

### 9.3 Why this matters for Fortran

Fortran is not dead. It is the language of the codes that must be correct:
weather, climate, CFD, nuclear, aerospace. Those codes are large, old, and
hard to parallelise by hand. Hand-parallelising them introduces races like
the Island race — silent, nondeterministic, hard to find.

MoA offers a different path: write the mathematics once as DNF, let the
compiler derive the ONF for each target, gate every backend against the same
reference. The programmer does not rewrite loops for each new machine. The
compiler does not have to rediscover the mathematics from loops. The DNF is
the contract between mathematics and machine.

That is the goal of a universal IR: one mathematics, many machines, provable
correctness, predictable cost.

---

## 10. Commit History (this work)

```
8870ad2 Add Delta bench scaling script for nx 256 1024 2048
be75503 Add MPI bench harness for Mac baseline
cbe0260 Add MPI Fortran backend from DNF ONF with halo exchange per sweep
8f4f015 Add GPU Fortran backend from DNF ONF with halo-safe Island Gauss
d18aa3d Add OpenMP Fortran backend from DNF ONF with halo-safe Island Gauss
2cab04b Add MATLAB gate: .m from DNF matches Python reference, PASS for all four, uses real MATLAB R2024a
d0fe20b Add Fortran frontend: Jacobi, Island Gauss, RBGS share one DNF, Matmul second DNF, gate PASS for all four, backend emits Fortran + MATLAB
6c0ec48 Close Fortran round-trip: frontend DNF -> backend Fortran + MATLAB, gate PASS for Jacobi and Island Gauss
39842cc Docs: make verified two-archive Paper I setup and current limitations clear
```

All on develop/mac-compiler-integration, origin at same commit, working tree clean
after each push. Do not push to main. Do not edit moa-compiler-push.

---

## 11. Next Steps

1. **Delta experiments**: run bench_delta.py files on Delta at nx=256, 1024, 2048,
   serial / OpenMP / MPI / GPU, 3 repeats, min time. Record in one table.
2. **Original vs MoA-derived**: identify original Fortran programs, add to same
   harness, same timing, same sizes, Mac + Delta. Do not claim speedup until
   this is done.
3. **Julia backend**: same frontend-to-gate pattern, .jl from DNF, gate PASS
   before claiming.
4. **Gamma / Omega as passes**: move shape / layout / cost from backend
   generators into explicit passes, so ONF choice is driven by cost model
   T ≈ αF + βM, not by hand-coded backend branches.
5. **Paper I**: use this write-up as the basis for the compiler section —
   DNF, ONF, halo-safe Island as the motivating example of a race found by
   gating, universal IR as the claim.

---

*Written from the gated work on develop/mac-compiler-integration, Nov 2026.
Numbers are for nx=64, ny=64, niter=100 unless stated otherwise.
Do not copy commit messages from chat verbatim; type them cleanly.*
