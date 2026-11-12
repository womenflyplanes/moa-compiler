# CC 2027 paper and complete MoA research monograph: forward-only synchronization

**Checkpoint:** 2026-11-12. **Scope:** Paper I forward attention. This document does not supersede original experiment records or claim new experiments.

## Research claim — correct scope

The compiler mechanizes the **MoA operations defining attention**, rather than arbitrary programs. The hypothesis that the same transformation discipline applies to other computations expressible in MoA algebra is a research implication, **not an experimentally established universal theorem**. Validate Paper I first; extend to Papers II–IV in sequence; use Paper V's historical hardware validation as comparative evidence.

For an identical observable contract, independent hand and compiler derivations may take different proof paths, but their canonical ψ-DNF semantics must agree. The γ-addressed ONF depends on layout; the Ω-lifted ONF depends on the target machine shape. Semantic equivalence must be maintained, subject to explicit finite-precision tolerances in numerical execution.

## Already established in the current evidence baseline

- **Nov 8 UA CPU holdouts:** frozen prospective predictions, including an unsuccessful Ω choice and a later bounded successful selection. Both outcomes belong in the paper and monograph.
- **Nov 10 matched-validation archive:** typed ψ-DNF, stable materialized equations, γ-addressed ONF, Ω owner intervals, generated C, certificates and SHA256 checks. A restricted both-output batched contract must not be conflated with the earlier output-only contract.
- **Nov 12 independent host replay:** original archive SHA256 verified; 16 generated CPU-host variants checked; independent hand reference for both observables; batch isolation; invalid-specification rejection; prospective-protocol guard tests. Host replay does not establish GPU device execution.
- **UA A100 OpenACC evidence:** generated programs, predeclared predictions, diagnostic outputs, timings and logs preserved. Distinguish artifact availability from completed independent matched analysis.
- **ACCESS:** hand-derived programs and historical CPU/GPU experiments. Never label these compiler-generated.
- **Comparative literature:** Polly, Halide, TACO, TVM, MLIR, Triton, FlashAttention and other attention implementations; separate conceptual comparison from measured head-to-head baselines.

## Required side-by-side manuscript structure

| Claim | CC paper (concise) | Complete research monograph (unabridged) |
| --- | --- | --- |
| Semantic specification | ψ-indexed Paper I observable contract, shape and reduction equations | Every index-domain derivation, identity, algebraic transformation and original hand proof |
| Mechanization | Trace from typed attention operations to DNF, γ ONF, Ω lifting and generated C | Full ASTs, intermediate records, compiler modules, tests, certificates, command lines |
| Machine mapping | One fully worked Ω example and ten-field machine array semantics | All target machine descriptions, ownership partitions, resource equations and predictions |
| Correctness | Independent CPU-host differential tests with exact contract labels | Full logs, mutation cases, edge cases, finite-precision tolerances, negative results |
| Performance | Prospective frozen predictions and UA/ACCESS evidence with non-equivalence caveats | Raw timings, plots, calibration, host topology, original ACCESS data and all failed holdouts |
| OpenMP / OpenACC | Separate CPU-host verification from A100 device execution | All sources, job scripts, device metadata, run outputs, provenance and diagnostics |
| Related work | Primary-source citations and precise novelty boundaries | Full bibliographic metadata and detailed comparative methodology |
| Limitations | Explicit bounded compiler scope and pending validation | Complete audit ledger and future Paper II–IV extension plan |

## Publication discipline

1. No new experimental claim without its original record and an evidence path.
2. No unnecessary repetition of experiments already adequate to support a claim.
3. No blanket “general compiler” or “universal optimality” claim.
4. No promotion of a recorded result to independently verified without a documented check.
5. Do not erase or rationalize failed predictions.
6. Preserve the larger monograph independently of the anonymous ten-page CC manuscript.
7. Keep all visuals, tables, TeX, bibliographic sources, logs and programs in the final archive with a manifest.
8. Apply adversarial mathematical, compiler, performance and publication reviews before release.

## Next concrete writing pass

In the CC paper, make the **restricted compiler transformation and its independently checked Paper I evidence** the central contribution; present a complete worked example and one concise evidence matrix. In the monograph, expand the same example to every AST, γ address, Ω ownership certificate and original experimental record. Reconcile OpenACC A100 timing/contract provenance before treating it as a verified matched comparison.

**Status:** Scope and evidence alignment established in this checkpoint; final manuscript reconciliation and complete A100 independent audit are still open.
