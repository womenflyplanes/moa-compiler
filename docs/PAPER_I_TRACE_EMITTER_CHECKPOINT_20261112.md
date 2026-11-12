# Paper I typed-trace-driven C emission checkpoint — 2026-11-12

## New executable path

- [tools/emit_paper1_trace_c.py](../tools/emit_paper1_trace_c.py) consumes the existing `results/paper1_reproduction/typed_derivation_trace.json`, validates the bounded ψ-index DNF, stable-summary, and γ/ONF address records, and emits output-only C.
- [tests/test_paper1_trace_emitter.py](../tests/test_paper1_trace_emitter.py) compiles emitted C and compares finite shapes to an independent Python numerical reference; it also checks that mutated DNF and ONF records are rejected.
- The GitHub Actions Paper I workflow now requests these tests on pushes to the artifact branch.

## How to reproduce from a clone

```sh
python -m unittest discover -s tests -p 'test_paper1_trace_emitter.py' -v
python tools/emit_paper1_trace_c.py \
  results/paper1_reproduction/typed_derivation_trace.json \
  /tmp/paper1_trace_generated.c
cc -std=c11 -O2 -c /tmp/paper1_trace_generated.c -o /tmp/paper1_trace_generated.o
```

The emitter prints SHA256 digests of the trace bytes and generated C bytes. The output-only C computes stable online attention with query-disjoint ownership and `O(dv)` scratch per active query. The generated C's address expressions come from the validated ψ-indexed trace. The online recurrence is presently a supported lowering rule selected for the validated `[a,z,u]` summary; it is **not yet backed by a machine-checkable theorem certificate**.

## Evidence classification

A local prototype with equivalent validation logic passed three finite tests in this working session (four combinations of `n,m,d,dv`, C compilation, output checks, and two negative tests). **The exact committed GitHub version and GitHub Actions execution have not yet been independently rerun/confirmed.** Do not treat a CI workflow file as proof that CI passed.

## Remaining release blockers

1. General MoA ψ-reduction from a user-level source expression into this DNF is not implemented by this bounded emitter.
2. The online summary transformation requires an explicit checked derivation/certificate, rather than assuming the recurrence from its state names.
3. Ω machine lifting and ten-field machine-property-array-driven mapping/prediction are not connected to this emitter.
4. Independent hand-derived DNF convergence and full-(A,O) contract matching remain open.
5. Clean-clone CI, frozen hardware prediction validation, and archival reproducibility remain open.
6. No manuscript should claim the entire Paper I compiler pipeline is mechanized until these gates close.

This is an engineering checkpoint, not a submission-ready result.
