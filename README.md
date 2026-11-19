# Mathematics of Arrays compiler research

**Completed milestone: hand-to-compiler C correspondence across Papers I–IV.**
All 39 numerical routines/regions in 20 pinned archived hand C files have compiler
counterparts. With Paper III's printed and explicitly derived references, all
44 paired cases pass: 246 exact-real end-DNF component comparisons, matching
operational C/ONF structures and common-machine lifted structures, and 138
untimed finite comparisons with zero hand/compiler difference.

**[Results, evidence links and LinkedIn announcement](docs/HAND_COMPILER_RESULTS.md)**  
**[Methods, reproduction commands and contracts](research/companion_attention/reference_onf/README.md)**

The mathematical pipeline is ψ-indexed specification → DNF → γ/ONF →
machine-dependent Ω lifting → generated implementation → prospective prediction
→ independent validation. Final normal forms must agree per pair; intermediate
derivation steps may differ. Lifted ONFs depend on machine shape and binding.

The completed correspondence scope is C numerical kernels with documented entry
contracts. Paper III's newly derived C references are identified separately from
archived C. Five paired fresh-device initialization corrections are included.
OpenACC finite checks used host fallback; this milestone adds no new GPU timing
claims. The broader Paper I performance-prediction research remains ongoing.

## Earlier Paper I foundation

[Installation, checkpoint preparation and usage](docs/PAPER_I_USAGE.md) documents
the earlier 8 November consolidated and 10 November matched-validation archives.
Those historical checkpoint packages are distinct from the completed
correspondence source and evidence now on this branch.

Keep O-only online-fold and full-(A,O) interfaces distinct, as well as logical Ω
ownership and physical execution. Prospective machine-shape selection and
absolute-time accuracy are separate claims. Preserve historical failures and
original SHA256-frozen evidence. The ten-field machine property array remains
append-only extensible. Historical records should be retained for provenance.
