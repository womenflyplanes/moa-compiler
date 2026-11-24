#!/bin/bash
# run_island_gauss_amd_gpu.sh -- Island Gauss on Delta AMD MI100 (gpuMI100x8)
# Same GATE, same 4 sizes, MoA vs Standard GPU only.
# NVIDIA A100 job 24398513 stays queued -- this is a SEPARATE GPU target, do not cancel A100.
# AMD MI100 does NOT use nvfortran. Try in order: Cray ftn (OpenACC), gfortran OpenACC offload to amdgcn.
#SBATCH --job-name=island_amd_gpu
#SBATCH --account=bibg-delta-gpu
#SBATCH --partition=gpuMI100x8
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=island_amd_gpu_%j.log
set -e
mkdir -p bin results
python3 generate_island_gauss.py
echo "=== AMD GPU compiler probe ==="
which ftn || true
which gfortran || true
which rocminfo || true
# Try Cray ftn first (Delta AMD path)
if command -v ftn >/dev/null 2>&1; then
  echo "Using Cray ftn for AMD MI100"
  ftn -O2 -hacc -o bin/moa_island_gpu island_gauss_src/moa_island_gpu.f90 || echo "ftn -hacc failed, trying gfortran offload"
  ftn -O2 -hacc -o bin/std_island_gpu island_gauss_src/std_island_gpu.f90 || echo "ftn -hacc failed, trying gfortran offload"
fi
# Fallback: gfortran OpenACC offload to AMD GCN
if [ ! -f bin/moa_island_gpu ]; then
  echo "Trying gfortran -fopenacc -foffload=amdgcn-amdhsa"
  gfortran -O2 -fopenacc -foffload=amdgcn-amdhsa="-march=gfx908" -o bin/moa_island_gpu island_gauss_src/moa_island_gpu.f90 || echo "gfortran AMD offload failed -- check Delta AMD docs, do not force nvfortran"
  gfortran -O2 -fopenacc -foffload=amdgcn-amdhsa="-march=gfx908" -o bin/std_island_gpu island_gauss_src/std_island_gpu.f90 || echo "gfortran AMD offload failed"
fi
if [ ! -f bin/moa_island_gpu ]; then
  echo "AMD GPU build failed. Paste this log -- do NOT use nvfortran for AMD, it targets NVIDIA only."
  exit 1
fi
for NX in 51 101 201 401; do
  echo "--- nx=$NX backend=gpu-amd ---"
  for WHICH in moa std; do
    for REP in 1 2 3; do
      ./bin/${WHICH}_island_gpu $NX > results/tmp_${WHICH}.log 2>&1 || true
      cat results/tmp_${WHICH}.log
      cp results/tmp_${WHICH}.log results/${WHICH}_gpu_amd_${NX}_rep${REP}.log
    done
  done
done
# NOTE: checker expects results/*_gpu_*.log -- amd logs use _gpu_amd_ prefix, rename for checker or run checker with pattern update
# For now, copy amd logs to gpu naming for checker:
for f in results/*_gpu_amd_*.log; do
  # keep original, checker v4 in paper update handles _gpu_amd_ -- see README
  echo "amd log: $f"
done
python3 check_island_gauss.py || true
echo "Done AMD GPU"
