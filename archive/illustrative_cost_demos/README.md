# Historical illustrative cost demos — not correctness or hardware evidence

These files were moved out of the active generated-kernel directory during the
24 November audit. Several emit only F/M/model-T values or empty computational
loops; identical printed model values do not validate attention algorithms,
compiler correctness, actual GPU execution, or measured runtime.

The historical `paper45_validation.py` assigns illustrative timing values and
ratios; these are not independent measured evidence. These files are retained
for traceability and must not be used as current validation entry points.

Use `scripts/validate_compiler.py`, `fortran_frontend/compile_moa.py`, and
`docs/COMPILER_AUDIT_20261124.md` for the current checked path. The separate
Papers I–IV 44-pair correspondence evidence is linked from that audit.
