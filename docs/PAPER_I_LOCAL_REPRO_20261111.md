# Paper I independent local archive test — 2026-11-11

## Provenance and scope

The test runner extracted the preserved `MoA_PaperI_Consolidated_20261108(2).zip` (201 members) and `MoA_PaperI_Matched_Validation_20261110(1).zip` (172 members) into a disposable directory. Both archives passed Python `ZipFile.testzip()` with no corrupt member. These are user-provided local archive copies; they have **not** been published to GitHub. The tests below were **not** run from a clean GitHub clone.

Environment: Python 3.13.5, NumPy 2.3.5, GCC 14.2.0 (Debian), Linux; `OMP_NUM_THREADS=2` for compiler CPU checks. This is **not** the UA CPU/GPU environment.

## Commands and observed output

From the extracted `MoA_PaperI_Consolidated_20261108`:

```sh
(cd latest_compiler/compiler && OMP_NUM_THREADS=2 python test_compiler.py)
```

Exit code 0:
```text
PASS (3, 5, 4, 3) (2, 2) max_error 1.1102230246251565e-16
PASS (9, 7, 3, 5) (2, 3) max_error 2.220446049250313e-16
PASS (16, 11, 8, 4) (1,) max_error 2.220446049250313e-16
PASS (2, 3, 2, 2) (4,) max_error 0.0
```

From extracted `MoA_PaperI_Matched_Validation_20261110`:

```sh
python test_full_contract.py
python test_protocol.py
```

Both exit code 0:
```json
{"implementations_checked": 16, "batch_isolation": "PASS", "semantic_rejections": 3, "original_hand_CPU_reference": "PASS on common inputs and both outputs", "gpu_validation": "pending"}
{"protocol_guards": ["existing run rejected", "changed frozen model rejected", "second execution rejected before measurement"], "hardware_workloads_executed": 0}
```

## What this establishes and does not establish

This is finite local correctness and protocol evidence for two **archive-based** Paper I code paths. It is not a general semantic-preservation proof; not an end-to-end clean-checkout run; not a measured performance result; not a new A100 validation; and not evidence that the current GitHub tree contains the complete compiler artifact. No historical performance data were modified. The archive suffixes `(1)` and `(2)` are local filenames and do not imply altered canonical archive contents.

## Remaining release gates

- Publish the original immutable archives or a verified equivalent public release with checksums.
- Verify clean clone, install, extraction, source generation, compilation, and tests from a documented toolchain.
- Audit canonical DNF/ONF and machine-dependent Omega-lifted ONF against the hand-derived Paper I specification.
- Independently trace predictions and A100 measurements to their original source artifacts and machine properties.
