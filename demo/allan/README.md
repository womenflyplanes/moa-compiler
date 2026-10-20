# Allan compiler review demo

Install from repository root: `unzip -o ../moa_allan_compiler_review.zip` (adjust ZIP path as needed).

Run `python3 demo/allan/demo_allan.py --out allan_demo_output` from the repository root.

Review `demo/allan/ARCHITECTURE_REVIEW.md` for exact implemented-vs-planned boundaries and questions. The generated C files are for inspection, not standalone executables; compiled numerical verification uses the existing `tests/attention_rc1/test_machine_mapping.py` harness.

The package adds only `demo/allan/` and does not overwrite any existing compiler modules or tests.
