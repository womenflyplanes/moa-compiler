# MoA Universal IR — Interactive GUI User Guide
Papers 4-5 Validation: arXiv:2609.33916 | NSF ACCESS CIS261342 + CHI-261724 + UA SUNY

## What This Is
Same DNF, many ONFs, predictable $, bytes, times, energy, heat.
Formula: ivec ψ A ≡ (rav A)[γ(ivec,ρA)] with (Πρ)_data vs (Πρ)_machines
Proven 5-device: Delta A100, MI100, Stampede3 Max1550 pvc, Bridges-2 H100/V100, UA A40
M1exp:36 Pd:9 Xd:0 Cd:36 reached(d) ✓

## Quick Start (No Install)
open moa_compiler/gui.html
# OR
/usr/bin/python3 moa_compiler/gui_web.py

## Interface
- d depth 4-36: model depth
- k vocab 16-256: hidden dim
- n seq 128-4096: sequence length
Blue curve T≈αF+βM, Green M1exp:36, Red current d,k,n
Info: F=d*k*n, M=d*k, T, $, bytes, Energy Wh, Heat kJ

## How To Use — Seeing Maxes
1. MIN cost: d=9 k=64 n=1024 → Pd:9 first reached(d), lowest $, energy — optimal UA A40
2. MAX proven: d=36 k=64 n=1024 → screenshot for SC27, M1exp:36 validated 5-device
3. MAX cost: d=36 k=256 n=4096 → bytes/$/energy explosion, shows why γ mapping matters
4. Compare hardware: Delta A100 α=0.004 fast vs MI100 α=0.028 slower, same DNF different α

## Why
2.00x atomics →2.5x fix via private acc, 535x NUMA vs <3x oversub via proc_bind(close), 3.17x/3.35x C vs Fortran row vs col γ
Think in MoA → faster, formal, beautiful. Save human, machine, energy.
