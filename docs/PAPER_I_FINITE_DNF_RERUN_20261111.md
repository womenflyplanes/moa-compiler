# Paper I typed-DNF finite-semantics rerun — 2026-11-11

**Source:** immutable November 8 consolidated archive, SHA256 `296b2767e77749a909039c64c4909ef7761a644c4942d42680d49f5cb4a78659`. The source files are `INDEPENDENT_DNF_SEMANTICS_AUDIT.py` and `compiler_ast/01_typed_dnf.json`.

**Method:** copied those two files into a disposable working directory and executed `python INDEPENDENT_DNF_SEMANTICS_AUDIT.py` there. The original archive and extracted source were not modified. No GPU, MPI, or cluster job was run.

**Observed output:**
```json
{
  "status": "PASS",
  "finite_shapes_checked": 192,
  "max_abs_error": 4.440892098500626e-16,
  "mutants_rejected": [
    "Q-index-swap",
    "K-wrong-owner",
    "V-wrong-key",
    "missing-scale"
  ],
  "scope": "typed DNF schema and finite output-only denotation; not hand-derived C equivalence, general compiler proof, or generated C verification"
}
```

**Reviewer interpretation:** This is finite numerical agreement between a Python interpreter of the saved typed DNF and a separately coded stable-attention oracle for 192 small shapes. The mutation checks validate rejection by `require_schema`, not an exhaustive test of the rewrite system. The oracle and interpreter use the same attention formulation; they are not independently sourced complete hand/compiler derivations. This test does not prove confluence or uniqueness of DNF, full-(A,O) equivalence, correctness of emitted C, or prospective performance.

**Remaining key obligation:** independently source the original hand-derived complete ψ-reduced DNF and the compiler's DNF, document shape/domain assumptions, and certify equality of canonical forms. See [DNF convergence contract](PAPER_I_DNF_CONVERGENCE_CONTRACT.md).
