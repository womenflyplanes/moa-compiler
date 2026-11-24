# Repository guide

| Location | Purpose |
|---|---|
| `moa.py` | Current compiler/status/validation entry point |
| `research/companion_attention/` | Completed attention correspondence implementation and canonical proof records |
| `fortran_frontend/` | Bounded semantic source adapter and target-stage lowering |
| `research/typed_attention/`, `research/universal_ir/`, `research/attention_rc1/` | Typed IR, machine mapping, scheduling and supporting compiler modules |
| `tests/`, `scripts/`, `tools/` | Validation, reproducibility and release adapters |
| `manuscripts/` | Current CC27 variants and correspondence report |
| `evidence/attention_hand/` | Single pinned original hand C source set |
| `evidence/historical/` | Preserved Paper I source/run binding archive |
| `evidence/consolidated_release/` | Integration replay output |
| `evidence/historical_legacy/` | Deduplicated historical output with explicit limitations |
| `continuity/` | Completed/pending ledger, evidence index, manuscript status and recovery hashes |
| `archive/README.md` | Recover removed historical/nonessential files from Git history |

The author/anonymous Overleaf folders intentionally remain self-contained.
Source and proof obligations are not removed based merely on apparent lack of
imports: they may be independent mathematical checks or archival evidence.

For a small current checkout, use a shallow clone of `main`. Recover historical
files from the recorded pre-cleanup commit when needed; a shallow clone may need
to fetch that commit first. Historical branches remain research checkpoints.
