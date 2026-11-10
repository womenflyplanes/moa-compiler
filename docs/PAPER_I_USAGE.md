# Using the Paper I MoA compiler

Current checkpoint: 10 November 2026. Active branch: `develop/mac-compiler-integration`.

## Install

```bash
git clone --branch develop/mac-compiler-integration https://github.com/womenflyplanes/moa-compiler.git
cd moa-compiler
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy
```

Use Python 3.10 or newer. CPU execution requires GCC with OpenMP support. On macOS, the system Apple compiler alone does not provide the GCC/OpenMP toolchain used by these commands. NVIDIA device experiments require NVHPC and an allocated compute node; see the experiment README before running them.

## Prepare the dated checkpoint

The November 10 source/evidence synchronization is pending. The commands below use the dated checkpoint supplied separately as `MoA_PaperI_All_Documents_and_Evidence_20261110.zip`; it is not yet committed to this repository. Extract that ZIP outside the checkout, then copy its expanded manuscript project into the checkout:

```bash
mkdir -p research/paper1_20261110
cp -R /path/to/extracted/MoA_PaperI_All_Documents_and_Evidence_20261110/Manuscript_Project/MoA_PaperI_Algorithm1_CPU_OpenACC_Extension_20261110/. research/paper1_20261110/
```

Replace `/path/to/extracted` with the actual extraction directory. Keep the original archive unchanged. Its `START_HERE.md`, inventory and SHA256 manifest describe the packaged evidence. This manual preparation is required until the source checkpoint is synchronized.

## Generate the O-only online-fold implementation

From the repository root:

```bash
python research/paper1_20261110/evidence/cpu_frozen_sources/compiler/compile_paper1.py --n 3 --m 5 --d 4 --dv 3 --machine 2,2 --out build/paper1_online
```

The compiler writes `generated_attention.c` and `moa_certificate.json`. The inputs have shapes ρQ=⟨n,d⟩, ρK=⟨m,d⟩, ρV=⟨m,dv⟩; the output has shape ⟨n,dv⟩. Ω=⟨2,2⟩ describes logical query ownership, not a physical thread-placement certificate. This path eliminates the full attention-probability intermediate. Its certificate supplies logical counts and ownership; physical runtime is not derived by this invocation.

Compile the generated source:

```bash
gcc -std=c11 -O2 -fopenmp -shared -fPIC build/paper1_online/generated_attention.c -lm -o build/paper1_online/kernel.so
```

Run the independent finite CPU checks, which compile and call the generated implementation:

```bash
(cd research/paper1_20261110/evidence/cpu_frozen_sources/compiler && OMP_NUM_THREADS=2 python test_compiler.py)
```

## Generate the full A,O interface and AST records

This additive path materializes both outputs: ρA=⟨B,N,N⟩ and ρO=⟨B,N,D⟩. It is distinct from the O-only online fold. From the repository root:

```bash
cd research/paper1_20261110/evidence/matched_validation_20261110
python - <<'PY'
from compiler.moa_full_contract import compile_contract
compile_contract(2,4,8,(2,), 'openmp', 'build/example')
PY
gcc -O3 -std=c11 -fPIC -shared -fopenmp build/example/generated.c -lm -o build/example/kernel.so
```

The output directory contains `01_psi_dnf.json`, `02_stable_materialized_onf.json`, `03_gamma_onf.json`, `04_omega.json`, `generated.c`, and `certificate.json`. The records trace ψ/DNF → stable materialized ONF → γ addresses → Ω ownership. The certificate binds the source and stages by SHA256. Source generation does not establish GPU execution.

For fresh local host checks, run from a disposable copy of this gate directory so archived results remain intact:

```bash
python test_full_contract.py
python test_protocol.py
```

These checks cover both outputs, common-input comparison with the retained hand CPU reference, uneven ownership, singleton reductions, batch isolation, invalid semantics, and evidence-protocol guards. Host fallback for OpenACC or OpenMP target is CPU evidence only.

## Prospective hardware prediction and validation

Read the supplied checkpoint’s `evidence/matched_validation_20261110/README.md`. Its `run_matched.sh` requires a Slurm allocation and protects existing run directories. `sbatch submit_all.sbatch` requests a new four-mode experiment; it is not an installation or smoke-test command. Review the requested resources before submission.

The original hand-reference audit requires `evidence/hand_derived/original_software_archive.zip`, preserved in the downloadable matched-validation archive. This nested original ZIP and large raw-run archives are retained in the consolidated downloadable evidence package rather than duplicated in Git. The audit must fail if required original evidence is absent; do not bypass that check.

Keep the prescribed sequence: calibrate → freeze hashes/model → predict unseen cases → execute → independently validate → assess selection and timing separately. Do not refit using held-out results or replace previous evidence. The calibrated time model is not an established architecture-only ten-field predictor.

## Evidence and scope

The supplied manuscript project includes the manuscript sources, PDF, figures, tables, generated programs, ASTs, and saved audit records. The supplied `MoA_RESUME_HERE_20261110.md` records completed versus pending obligations. Original dated records remain historical; consult the current usage guide for repository paths.

The bounded CPU foundation and saved finite GPU audits do not establish universal correctness, broad predictive accuracy, or completion of Paper I. The new full-output target sweep and framework comparison remain pending in this checkpoint. Papers II–IV are outside this update. Preserve MoA notation, the ten-field machine property array, and append-only extension; logical Ω counts must not be presented as complete physical machine certification.
