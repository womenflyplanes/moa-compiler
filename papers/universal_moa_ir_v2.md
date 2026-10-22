# Universal MoA IR v2 — concrete ψ selection and layout-independent DNF

## Scope

This prototype extends v1 without modifying it. A shaped array `Array(name, shape)` supports prefix ψ indexing, including the empty index (the whole array), partial index (a residual array), and full index (a scalar, represented by residual shape `()`).

For `i` a valid prefix index of shape `ρX`, the residual shape is `drop(len(i), ρX)`. The DNF of `i ψ X` is the residual shape together with the exact source index domain `{i ++ j | j in ι(residual_shape)}`. This definition is independent of layout.

`gamma_row_major` is an explicit layout choice, used only when lowering the DNF to scalar offsets. `compose_psi` checks the concrete ψ composition identity `(j ψ (i ψ X)) = ((i ++ j) ψ X)` at the index level. `Partition1D` and `lift_psi` retain rank-one partition lifting with a valid-tail mask.

## What is not yet implemented

This is **not** general DNF normalization for arbitrary MoA expressions. It does not yet support symbolic indices, operators and reductions, array-valued index arrays, dependence analysis, general multidimensional lifting, arbitrary γ layouts, executable generated kernels, or parallel schedules. The finite test suite is validation of these bounded rules, not a general correctness proof.

## Tests

Run `python3 tests/universal_ir/test_universal_ir_v2.py`. This tests concrete empty, partial, and full ψ indices, composition, layout-independent DNF index domains, explicit row-major lowering, 40 rank-one lifts, and invalid input rejection. Preserve and rerun the v1 suite separately.
