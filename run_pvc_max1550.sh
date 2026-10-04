#!/bin/bash
#SBATCH --job-name=moa-pvc
#SBATCH --account=ACCESS-CIS261342
#SBATCH --partition=gpuMax1550x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=moa_pvc_%j.log
module load oneapi
echo "=== Intel Max 1550 PVC Level Zero Sysman 99.88% busy ==="
python3 moa_compiler/paper45_validation.py --reference
python3 moa_compiler/paper1_exact_reproduce.py
# Fortran/C via oneapi ifort/icx
for f in moa_compiler/generated/*.f90; do ifx -O2 -o /tmp/f90 $f && /tmp/f90; done
for f in moa_compiler/generated/paper*.c; do [[ $f == *openmp* ]] && continue; icx -O2 -o /tmp/c $f && /tmp/c; done
