#!/usr/bin/env bash
set -euo pipefail
compiler_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
if [[ $# != 2 ]]; then
    echo "Usage: bash replay.sh EXTRACTED_HAND_ARCHIVE OUTPUT_DIRECTORY" >&2
    exit 2
fi
hand="$(realpath "$1")"
mkdir -p "$2"
out="$(realpath "$2")"
cc_flags=(-std=c11 -O2 -ffp-contract=off)
export OMP_NUM_THREADS=4 ACC_DEVICE_TYPE=host
for backend in openmp openacc; do
    for case_name in backward fused decode mqa single_decode append rmsnorm ffn block; do
        kernel="$case_name"
        shape=()
        case "$case_name" in
            mqa) kernel=decode; shape=(--g 1) ;;
            single_decode) kernel=decode; shape=(--h 1 --g 1) ;;
        esac
        destination="$out/$case_name/$backend"
        python3 "$compiler_dir/compile.py" "$kernel" "$backend" "$destination" "${shape[@]}"
        gcc "${cc_flags[@]}" -Wall -Wextra -Werror -shared -fPIC "-f$backend" \
            "$destination/kernel.c" -lm -o "$destination/kernel.so"
    done
done
gcc "${cc_flags[@]}" -shared -fPIC "$hand/attention/moa_attention.c" -lm -o "$out/hand.so"
gcc "${cc_flags[@]}" "$hand/verification/full_block_verify.c" -lm -o "$out/full_block_hand"
gcc "${cc_flags[@]}" -fopenacc "$hand/verification/full_block_gpu_kernel.c" -lm -o "$out/full_block_hand_acc"
mkdir -p "$out/hand_block/data"
python3 - "$out/hand_block/data" <<'PY'
import sys
from pathlib import Path
import numpy as np
rng = np.random.default_rng(20261117)
counts = dict(X=64, gamma1=8, gamma2=8, Wq=64, Wk=64, Wv=64, Wg=128, Wu=128, Wd=128, GY=64)
for name, count in counts.items():
    rng.uniform(-0.6, 0.6, size=count).tofile(Path(sys.argv[1]) / f"fb_{name}.bin")
PY
(cd "$out/hand_block" && "$out/full_block_hand" && "$out/full_block_hand_acc")
python3 "$compiler_dir/verify.py" "$out" "$hand" "$out/hand.so" "$out/hand_block/data"
