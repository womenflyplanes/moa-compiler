# Mathematics of Arrays compiler research

Our goal is a **universal intermediate representation for array languages**, with
portable, scalable, predictable and validated designs grounded in MoA.

- **[Architecture, scope and compiler usage](docs/MOA-UNIVERSAL-IR.md)**
- **[24 November correctness audit and verified results](docs/COMPILER_AUDIT_20261124.md)**
- **[Attention Papers I–IV correspondence study](https://github.com/womenflyplanes/moa-compiler/tree/codex/papers-i-iv-correspondence)**

The current extension has a bounded Fortran source adapter, structured versioned
DNFs, ONFs and logical lifted ONFs for serial, OpenMP, OpenACC and MPI schedules.
It rejects unsupported source or machine mappings. Numerical expressions are
lowered from their actual ASTs; schedule templates remain an explicit trusted
basis. MATLAB generation is available but was not executed in this audit.

```bash
python -m pip install numpy pytest
python scripts/validate_compiler.py --with-fortran
```

The full replay needs GCC, gfortran and MPI development tools. OpenACC is tested
on the host, not on a physical GPU. No new hardware timing was performed.

The five-paper attention suite is the research foundation. Its checked
correspondence, hardware measurements and predictive results have separate
contracts; they do not automatically certify a new frontend or solver.
The ten-field machine property array remains append-only extensible.

Historical manuscripts, raw logs and measurements remain available. Cost-only
examples have moved to `archive/illustrative_cost_demos/` and are not correctness
or hardware evidence. Generated build clutter and exact duplicate documents were
removed; the audit records the cleanup.
