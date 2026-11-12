# Paper I trace-to-emission provenance audit — 2026-11-12

## Inspected files (same GitHub branch)

- `results/paper1_reproduction/typed_derivation_trace.json`
- `research/attention_rc1/moa_attention_psi_onf_v1.py`
- `tests/attention_rc1/test_psi_onf_attention.py`
- `research/attention_rc1/moa_index_dnf_onf_v1.py`

## Observations

The saved typed trace records unbatched `Q[n,d]`, `K[m,d]`, `V[m,dv]` with a score reduction index `j`, key index `k`, value index `v`, stable summary state `[a,z,u]`, and ONF address records. Its own scope says **restricted forward-attention typed rewrite; not general MoA compilation**.

The separate `moa_attention_psi_onf_v1.py` generator accepts `AttentionIR` and `Mapping`, and constructs hard-coded `PsiAccess` objects for `Q[B,N,D]`, `K[B,N,D]`, `V[B,N,Dv]`, `P[B,N,N]`, and `O[B,N,Dv]`. It emits row-major `gamma` addressing and a batched C implementation. Its `emit` function does **not** take the saved `typed_derivation_trace.json` or a DNF AST as input. Consequently, no source-to-trace-to-code provenance chain is established by these files alone.

The generated implementation allocates a `B*N*N` workspace, materializes row probabilities `P`, and returns both `P` and `O`. This may be appropriate for the **full-(A,O)** observable contract, but it must not be cited as evidence of an output-only streaming implementation with no quadratic intermediate storage.

The test `test_psi_onf_attention.py` checks `PsiAccess.offset()` against `gamma`, compiles/runs generated sequential/OpenMP C through `run_kernel`, and compares both outputs to a reference and prior generator at tolerance `1e-12`. The tests support finite implementation equivalence if executed successfully, **not** that emission was derived from the saved DNF trace. No tests were run in this audit; findings are based on source inspection.

## Acceptance criteria for the missing provenance link

1. Define a versioned, typed MoA DNF representation that records the exact source contract, shape assumptions, ψ-index expressions, and observable outputs.
2. Make γ/ONF lowering consume that representation (or a separately verified transformation of it), rather than reconstructing the attention formula from fixed names.
3. Make the emitter consume the resulting ONF and record source/IR/code digests and machine-property-array identity.
4. Test a deliberately altered valid DNF expression and verify the emitted implementation changes correspondingly, plus reject malformed shapes/indexes.
5. Verify hand-derived and compiler-generated DNFs under the **same** observable contract; preserve distinct derivation histories.
6. Run clean-clone tests, retain build/runtime logs, and separately verify resource predictions and machine-specific Ω legality.

## Reviewer disposition

**Open blocker: AST-to-emitter provenance.** The source contains credible bounded pieces, but the currently inspected trace and generator do not establish the claimed single MoA derivation chain. No claim is made that other, uninspected compiler paths lack such a chain.
