#!/bin/bash
# Submit to all 5 devices for M1exp:36 proof

for part in "gpuA100x4" "gpuMI100x8" "gpuH100x8" "gpuV100x4" "gpuA40x4"; do
  echo "Submitting $part"
  sbatch --partition=$part --job-name=moa-$part --account=ACCESS-CIS261342 --nodes=1 --gpus-per-node=1 --time=00:30:00 --output=moa_$part_%j.log ./run_delta_validation.sh
done
