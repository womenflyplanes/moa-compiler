# MoA Compiler — Prove by Construction

## One psi expression -> DNF -> ONF -> dimension lifting -> Serial / OpenMP / MPI / GPU

Prove by construction, not by testing alone.

- GitHub: https://github.com/womenflyplanes/moa-compiler
- Branch: develop/mac-compiler-integration
- Zenodo DOI: 10.5281/zenodo.23090349

## What we can now claim (with proof)

Same problem, same stopping rule, same GATE. Burkardt Poisson.

### Island Gauss — Delta, 12/12 CONVERGED PASS
- Serial: 1.09x-1.18x
- OpenMP: 1.16x-1.49x
- MPI: 1.02x-1.99x
- nx 51 / 101 / 201 / 401, diff <= 1e-6

### Jacobi — Delta
- Serial: 1.27-1.53x
- OpenMP: 1.51-2.29x
- MPI: 1.16-2.11x
- GPU direct-run: 1.18-1.31x

### GPU for Island Gauss
- NVIDIA A100 job 24398513, gpuA100x4, queued
- AMD MI100 job 24412991, gpuMI100x8, queued
- Same GATE when they finish. Paper does not block on GPU.

## Multi-language input — honest version
Today: Fortran + MoA psi is the source of truth. Fortran frontend (poisson_backend.py) takes Poisson Jacobi / Island Gauss / RBGS and derives DNF from psi.
Not yet: arbitrary C / Python / Fortran in. That is roadmap, not claim.
MoA psi is language-neutral, Fortran is the first frontend proven.

## Multi-language output — proven
From SAME DNF emit Fortran 90, C with OpenMP / OpenACC / OpenMPI, MATLAB .m.
One DNF -> many ONFs.

## Predictive
Shape-based cost: T ~ alpha*F + beta*M, predicts time / bytes / dollars / energy from shape before you run.
MoA fuses diff into the sweep, eliminates the separate RMS pass Standard does.

## Extrapolation: Hand-Derived Designs -> Compiler-Generated Code
Hand-derived ONF = compiler-generated ONF -> hand timings predict compiler timings.
Do NOT claim compiler was tested on numerous ACCESS supercomputers.
Claim: Hand-derived MoA designs were tested on numerous ACCESS supercomputers and documented in arXiv Papers 1-4, with predictive theory in Paper 5. The compiler now generates the same ONFs automatically. Compiler-generated code is proven on Delta (Jacobi + Island Gauss, 12/12 GATE PASS, faster at every row), and by ONF equivalence it inherits the hand-derived performance envelope on other ACCESS machines.

3-step chain:
1. Same psi -> same DNF (same psi expression, same psi reduction)
2. Same DNF -> same ONF (ONF is deterministic function of DNF + machine shape + dimension lifting; same loop order, same fusion, same halo exchange = same program modulo variable names)
3. Same ONF + same GATE -> same performance envelope

Paper 5: arXiv:2610.22769, One Communicator, Three Vendors: Extending the MoA Predictive Theory to a Real NVIDIA+AMD+CPU MPI Ensemble, Mullin & Hains.
Earlier work: From array algebra to energy efficiency on GPUs (arXiv:2306.11148), Machine Shape and Hierarchical Blocking (arXiv:2608.10119).

### AST Equivalence: Hand vs Compiled — The Proof That Closes the Loop
Chain is now 4 links: Same psi -> same DNF -> same AST -> same ONF -> same performance envelope.
GitHub docs in womenflyplanes/moa-compiler contain hand-derived vs compiler-generated ASTs, and they are equivalent. ASTs match, so DNF -> ONF -> code is the same derivation, not a similar-looking rewrite.
Check: for each example (Jacobi, Island Gauss, RBGS, Attention), hand AST and compiler AST side by side, same nodes, same shapes, same fusion / halo / dimension-lifting decisions. If ASTs match and GATE matches, hand timings on ACCESS transfer to compiler code.

| Check | Hand derivation | Compiler |
|---|---|---|
| Psi expression | Papers 1-4 | moa-compiler DNF |
| ONF loop / fusion | Hand ONF | Generated .f90 - diff, not eyeball |
| GATE it / diff / err / umid | Hand run on ACCESS | Compiler run on Delta |
| T ~ alpha F + beta M fit | Hand timings | Compiler timings, same alpha, beta per machine |

Caveat: Any mismatch is a compiler bug, not a theory failure. Example: MPI halo bug gave it=786 at nx=101, correct halo-before-sweep gives it=5988 matching Standard, GATE caught it.

## Validation
GATE: same problem, same stopping diff <= 1e-6, check it, diff, err vs sin(pi*x*y) exact, umid centre value.
Gated against Python reference + MATLAB R2024a.
GATE catches wrong-problem bugs: MPI halo bug above.

## HPC portable + scalable
Delta, ACCESS allocations CIS261396 / CIS261342, NCSA Delta, TACC Stampede3, PSC Bridges-2.
Partitions: cpu, gpuA100x4, gpuMI100x8. Same DNF scales 51 to 401, Serial to MPI.

## How to sell it
1. Keep your Fortran, get portable GPU/MPI for free — from one DNF, not four rewrites.
2. Proof by construction + GATE — not benchmarks alone.
3. AST-equivalent, not just benchmark-equivalent.
4. Open, no-fee, reproducible — scripts + CSV + tex on GitHub.

## Try it
git clone, paper/paper1_island_gauss.tex, results_island_gauss.csv, run_island_gauss_amd_gpu.sh.
SC27 paper, Fortran Discourse post.
