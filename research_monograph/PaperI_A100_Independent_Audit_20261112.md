# Paper I — Independent audit of UA A100 OpenACC six-case generalization (2026-11-12)

**Evidence:** Original frozen `MoA_OpenACC_A100_Generalization_136902.tar(1).gz`, `PREDECLARED_PLAN.json`, results under `results/job_136651/` and `results/job_136902/`. Audit recomputed from saved raw timing observations, output checks, Ω certificates, and per-job frozen predictions; **no new GPU experiments**.

The archive contains six new Paper I output-only attention workloads, each realized at logical Ω owner counts 32, 64, 128 and 256. Two separate job directories record 48 timed realizations and 432 retained observations. All 48 recorded device flags identify NVIDIA execution, and the numerical status fields passed. We checked the 48 ownership certificates for disjoint complete query intervals and τ(O) conservation; the recorded timed and diagnostic output hashes matched.

| Predeclared criterion | Result |
|---|---|
| C1 numerical agreement | **PASS** in archived finite checks |
| C2 Ω ownership and τ(O) conservation | **PASS** in independent certificate checks |
| C3 selected/fastest median ≤1.05 | **PASS** in both jobs for all six workloads |
| C4 selected absolute-time relative error ≤0.25 | **FAIL**: job 136651 case 0 ≈26.4%, case 1 ≈26.8% |
| C5 equal-work geometry ratio ≤1.25 | **PASS**; worst ratio ≈1.103 |
| C6 separate-allocation repeatability ratio ≤1.25 | **PASS**; worst ratio ≈1.218 |

The frozen model selected Ω⟨128⟩ for case 0 and Ω⟨256⟩ for cases 1–5. For case 0, the fastest measured Ω was ⟨64⟩, but regret was only 1.013 and 1.008 in the two jobs. All other cases selected the measured fastest candidate. **Shape selection succeeded under the predeclared near-optimal criterion even though absolute-time accuracy failed.** This is a bounded result, not universal optimality.

The original C4 threshold is preserved. No retuning or discarded timing samples. The experiment measures complete synchronous OpenACC calls including transfers; it is not isolated GPU kernel timing, power prediction, or a matched performance comparison to ACCESS, FlashAttention or PyTorch.

**Publication interpretation:** This experiment is evidence for the MoA attention-specific compilation and Ω candidate selection pipeline, not for arbitrary program compilation. It is distinct from the November 10 batched full-output (A,O) CPU-host validation and from the historical hand-derived ACCESS results. Include the negative C4 result explicitly in the CC paper and complete monograph.

**Remaining:** Clean independent replay of compiler generation from original typed source and full hardware provenance beyond internally consistent archived logs; anonymous reviewer artifact and bibliography/layout audit. Do not mark these passed merely because the experiment archive is intact.
