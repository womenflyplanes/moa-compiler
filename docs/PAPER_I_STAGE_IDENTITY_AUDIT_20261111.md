# Paper I MoA stage-identity audit — 2026-11-11

Input: preserved `MoA_PaperI_Matched_Validation_20261110` archive, verified SHA256 `099b9ca79f65e51a636884bde0efa75d0fc5f985db75b0e1c6c7565a203fa151`.

Audit: 16 complete directories in `local_validation/`. Parsed each stage JSON, serialized using sorted keys and compact separators, and compared SHA256 values across cases.

| Stage | Unique canonical JSON objects | Cases |
|---|---:|---:|
| `01_psi_dnf.json` | 1 | 16 |
| `02_stable_materialized_onf.json` | 1 | 16 |
| `03_gamma_onf.json` | 1 | 16 |
| `04_omega.json` | 4 | 16 |

Each of the four Ω records occurs in four backend-specific directories. This is consistent with machine/workload-specific Ω mapping and backend-independent canonical stages.

**Limits:** This audit establishes structural identity of saved serialized stages, not semantic equivalence, the correctness of their MoA transformations, or that the saved files were independently generated from a clean GitHub checkout. It also does not establish correspondence to the original hand-derived DNF/ONF without a separate derivation audit.

The reusable script and full per-group digests are available as local artifacts pending upload. Paper claims must retain this qualification.
