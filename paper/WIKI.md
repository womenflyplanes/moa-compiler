# Mathematics of Arrays compiler

The goal is a universal intermediate representation for array languages:
portable, scalable, predictable and validated designs grounded in MoA.

- [Current architecture and compiler commands](../docs/MOA-UNIVERSAL-IR.md)
- [Correctness audit, repairs and validation status](../docs/COMPILER_AUDIT_20261124.md)
- [Attention Papers I–IV correspondence evidence](https://github.com/womenflyplanes/moa-compiler/tree/codex/papers-i-iv-correspondence)

The current source adapter is a bounded Fortran subset, not a general frontend.
DNF, ONF and lifted ONF records are explicit. Parallel island variants have an
explicitly different state contract from serial lexicographic island sweeps.
OpenACC host validation is not GPU validation. Historical CSV measurements remain
reported results pending complete executed-source and raw-run provenance.

Older wiki claims are preserved in Git history. They have been superseded by the
audit rather than reused as current proof or performance evidence.
