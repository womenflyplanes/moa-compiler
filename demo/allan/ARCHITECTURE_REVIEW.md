# Allan's MoA compiler architecture review — checkpoint before UAlbany

**Baseline:** `develop/mac-compiler-integration` at `7e17409` (19 October 2026). **Status:** review prototype, not a completed general MoA compiler.

## Run on the Mac from the repository root

```bash
python3 demo/allan/demo_allan.py --out allan_demo_output
python3 tests/attention_rc1/test_machine_mapping.py
```

The first command prints the actual attention-specific typed ψ accesses, explicit row-major γ expressions, structured loop IR, ten-field machine descriptor, mapping plans, and generates `allan_demo_output/attention_{sequential,openmp}.c` and `compiler_trace.json`. The second invokes the existing compiled numerical regression (requires macOS C/OpenMP toolchain and Python dependencies already installed). No tests are modified. Do **not** commit the generated output folder.

## What is actually implemented

| Layer | Evidence in current repository | Scope and caveats |
|---|---|---|
| Semantic AST experiment | `research/attention_rc1/moa_attention_ast_v1_fixed.py` | Independent attention AST prototype; **not** wired to the production code-generation path. Inspect `make_ast`, `lift_ast`, `instantiate`. |
| Shape/ψ/γ semantics | `moa_index_dnf_onf_v1.py` | Tested index-domain, prefix selection and γ layout behavior; **not** a general DNF reducer. |
| Typed attention loop IR | `moa_structured_codegen_v1.py` | `Domain`, `AttentionIR`, `Mapping`; narrow attention-specific IR. |
| ψ-derived C addresses | `moa_attention_psi_onf_v1.py` | `PsiAccess` produces row-major γ offsets for Q/K/V/P/O; canonical `i,j` domain restrictions. |
| Machine mapping | `moa_machine_mapping_v1.py` | Proposed ten-field descriptor and heuristic tile/thread planning; uncalibrated. |
| C / OpenMP backend | `moa_attention_psi_onf_v1.py::emit` | Generates C for fixed N, parallelizes query blocks; OpenMP requested thread count is an external runtime setting. |
| Validation | `tests/attention_rc1/test_psi_onf_attention.py`, `test_machine_mapping.py` | Previously passed numerical comparisons on M1 Pro; not performance or memory-use validation. |

## Distinguish the transformations

**MoA semantic goal:** shaped array expressions with `ρ`, `ι`, `ψ` → ψ reduction and machine-independent DNF → layout-specific sequential ONF via γ → shape-directed dimension lifting → machine ONF → backend.

**Implemented today:** an attention-specific AST experiment exists *separately*; a narrow loop IR and typed ψ accesses feed C/OpenMP generation; a heuristic planner chooses tile sizes. There is **no connected general AST-to-DNF rewrite engine**, general pass manager, verified lowering proof, or GPU/MPI backend in this branch. The label “DNF” must not be attached to an invented trace stage.

## Example invariant and proof obligations

For valid complete index `i` of `A`, show `i ψ A = rav_γ(A)[γ(i,ρ A)]` for each supported layout. For lifted index `i = ib*Tq+il`, establish a bijection between valid lifted indices and `0 ≤ i < N`, including masked tail blocks. For OpenMP, establish disjoint ownership of each output `(b,i)` and its workspace row. Address integer overflow and allocation failure explicitly.

## Review questions for Allan

1. Should shaped-array semantics be encoded in a typed SSA IR, an algebraic term IR, or both? Which shape/index invariants belong in types?
2. How should a terminating, confluent ψ-reduction / DNF rewrite system be specified and tested? Which transformations require proof certificates?
3. Where is the machine-independent/machine-dependent boundary? Is sequential ONF layout-neutral until γ is chosen?
4. How should index domains, short/prefix ψ selection, bounds, and non-divisible tiles be represented explicitly?
5. Should γ be a layout-map node or a lowering pass? How should alternate layouts and memory hierarchies be modeled?
6. Should dimension lifting be an IR transformation with explicit index correspondence and masks, rather than emitter-side loops?
7. How do we define the canonical **first ten machine-shape variables**? The current descriptor is provisional; Allan should challenge it before calibration.
8. What are the pass preconditions/postconditions, verification hooks, and reproducible dumps?
9. What tests separate semantic equivalence, generated-code correctness, resource prediction, and measured speedup?
10. What is the minimum architecture Allan recommends before UAlbany CPU, MPI, and GPU backends?

## Suggested live review (45–60 minutes)

- 10 min: MoA semantic goals and exact current implementation boundaries
- 15 min: AST experiment, structured IR, ψ/γ dumps and generated code
- 10 min: run existing correctness tests and inspect machine planner estimates
- 15–25 min: Allan's critique; record pass boundaries, invariants, prioritized refactors

## Requested feedback artifact

Record **accept / change / defer** for AST design, DNF rewrite system, ONF/γ representation, dimension lifting, ten machine-shape variables, backend separation, and verification strategy. Record an owner and next test for each accepted change. Freeze UAlbany expansion until review outcomes are incorporated into a revised compiler plan.
