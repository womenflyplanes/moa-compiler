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

## Obtain and verify the two preserved checkpoints

The source/evidence checkpoint has **not yet been synchronized to GitHub**. Two actual, verified archives are retained separately on the UAlbany DGX host, in the user's home directory:

- `MoA_PaperI_Consolidated_20261108.zip` — SHA256 `296b2767e77749a909039c64c4909ef7761a644c4942d42680d49f5cb4a78659`. Includes `main.tex`, `main.pdf`, figures, compiler source, typed DNF/ONF/Ω records, tests, and prospective evidence.
- `MoA_PaperI_Matched_Validation_20261110.zip` — SHA256 `099b9ca79f65e51a636884bde0efa75d0fc5f985db75b0e1c6c7565a203fa151`. Includes the full-(A,O) compiler contract, hand-derived original archive, matched-validation tests, and historical performance evidence.

The archive ZIP integrity checks passed on DGX on 11 November 2026. **These ZIPs are not yet downloadable from GitHub**. Do not rely on the previously named `MoA_PaperI_All_Documents_and_Evidence_20261110.zip`; its existence was not confirmed.

Copy the archives to your workstation using the SSH hostname that works for your UAlbany account, then verify the hashes:

```bash
scp USER@DGX_HOST:~/MoA_PaperI_Consolidated_20261108.zip .
scp USER@DGX_HOST:~/MoA_PaperI_Matched_Validation_20261110.zip .
shasum -a 256 MoA_PaperI_Consolidated_20261108.zip MoA_PaperI_Matched_Validation_20261110.zip
```

Extract **outside the repository**. For the commands below, install the two dated checkpoints into the expected *working-copy* layout, without modifying the ZIPs or the archived evidence:

```bash
mkdir -p /tmp/moa-paper1-archives research/paper1_20261110/evidence
unzip -q MoA_PaperI_Consolidated_20261108.zip -d /tmp/moa-paper1-archives
unzip -q MoA_PaperI_Matched_Validation_20261110.zip -d /tmp/moa-paper1-archives
cp -R /tmp/moa-paper1-archives/MoA_PaperI_Consolidated_20261108/latest_compiler \
  research/paper1_20261110/evidence/cpu_frozen_sources
cp -R /tmp/moa-paper1-archives/MoA_PaperI_Matched_Validation_20261110 \
  research/paper1_20261110/evidence/matched_validation_20261110
```

These are instructions for a **separate development checkout**, not a published reproducible release. The commands below have not yet been retested against this two-archive layout; verify the extracted paths and use a disposable working copy before executing tests. Do not run `submit_all.sbatch` as a smoke test.

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

The November 8 archive contains the candidate master manuscript (`main.tex`/`main.pdf`) and figures. The November 10 archive contains the matched-validation gate and hand-derived evidence. Each archive contains its own README/status records. The archives are distinct and neither should be presented as a fully synchronized GitHub release. Original dated records remain historical.

The bounded CPU foundation and saved finite GPU audits do not establish universal correctness, broad predictive accuracy, or completion of Paper I. The new full-output target sweep and framework comparison remain pending in this checkpoint. Papers II–IV are outside this update. Preserve MoA notation, the ten-field machine property array, and append-only extension; logical Ω counts must not be presented as complete physical machine certification.
