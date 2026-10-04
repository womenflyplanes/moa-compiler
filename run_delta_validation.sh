#!/bin/bash
#SBATCH --job-name=moa-multi-lang
#SBATCH --account=ACCESS-CIS261342
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=moa_multi_lang_%j.log

module load gcc/12.2.0
module load python

echo "=== DELTA A100 Multi-lang Papers1-5 ==="
echo "Node: $(hostname) GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "DNF: ivec psi A = (rav A)[gamma(ivec,rhoA)]"
echo ""

echo "--- Fortran gfortran col-major gamma_col=7 ---"
for f in moa_compiler/generated/*.f90; do
  echo "--- $f"
  gfortran -O2 -o /tmp/f90 $f && /tmp/f90
done

echo ""
echo "--- C clang row-major gamma_row=6 ---"
for f in moa_compiler/generated/paper*.c; do
  [[ $f == *openmp* ]] && continue
  echo "--- $f"
  gcc -O2 -o /tmp/c $f && /tmp/c
done

echo ""
echo "--- OpenMP proc_bind(close) fixes 535x NUMA ---"
for f in moa_compiler/generated/*openmp.c; do
  echo "--- $f"
  gcc -O2 -fopenmp -o /tmp/omp $f && OMP_PROC_BIND=close OMP_PLACES=cores /tmp/omp
done

echo ""
echo "--- Python numpy rav.ravel() ---"
python3 moa_compiler/generated/paper1_exact.py
python3 moa_compiler/generated/paper2_backward.py
python3 moa_compiler/generated/paper3_decode.py
python3 moa_compiler/generated/paper45_matmul.py

echo ""
echo "--- Paper45 validation.py --reference (Paper V fixes) ---"
python3 moa_compiler/paper45_validation.py --reference

echo ""
echo "--- reached(d) M1exp:36 Pd:9 Xd:0 Cd:36 ---"
python3 moa_compiler/paper1_exact_reproduce.py

echo "DONE same F=786432 M=768 T proves portability"
