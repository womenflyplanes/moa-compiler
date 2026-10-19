SC27 VERSION 13 — MoA AST work / machine-specific empirical cost model

Run in ~/UA-signalproc/SC27-UAresults (macOS with libomp installed):

  unzip -o moa_cost_model_v13_package.zip
  python3 moa_cost_model_v13.py --phase calibrate --rounds 9 --warmups 4
  python3 moa_cost_model_v13.py --phase validate --rounds 9 --warmups 4

DO NOT repeat calibration after frozen_v13.json exists; validation checks its SHA-256.

This package uses the same AST-driven attention code generator as V9–V12.
Calibration: N=96,160,256; four mappings P4/T4, P8/T4, P8/T8, P16/T8.
Holdout: N=320,448 (not V12's N=384,512).

Exact: lifted psi query-index coverage, per-partition multiplication counts,
source SHA-256, persistent array storage arithmetic, numerical P and O checks.

Empirical: affine timing model using the maximum estimated work assigned to
one OpenMP thread by schedule(static), with one model per candidate mapping.
This is not an exact 15-dimensional machine-shape timing derivation. The
intercept includes multiple unisolated costs; OS placement is not controlled.

Files saved in moa_v13_cost_model/: plan_v13.json, calibration_v13.json,
frozen_v13.json, holdout_v13.json and generated C files.

Do not modify the V12 frozen certificate or V12 results.

Warning: OpenMP libomp deployment target warnings may persist on macOS.
