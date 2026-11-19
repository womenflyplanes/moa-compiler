# Hand-derived and compiler-generated AI kernels: results

The Papers I–IV C correspondence study is complete within the declared contracts.
Each compiler output matches its own hand-derived target; different algorithms
and schedules need not share an ONF.

| Evidence | Result |
|---|---:|
| Archived numerical routines/regions | 39/39 |
| Archived hand C files | 20 |
| Paired cases including Paper III references | 44/44 |
| Exact-real end-DNF component comparisons | 246/246 |
| Operational C/ONF and common-machine lifted structures | 44/44 |
| Untimed finite numerical comparisons | 138/138 |
| Largest hand/compiler difference in those checks | 0 |
| Deliberately corrupted operational ASTs rejected | 109 |
| Corrected paired fresh-device initialization contracts | 5/5 |

Coverage includes forward/backward/fused attention; untiled and tiled variants;
serial references; decode, grouped attention and cache append; RMSNorm, FFN,
norm–FFN chains and Transformer blocks. OpenMP/OpenACC computational structures
are checked. The emitter does not read hand C: independent comparison tools do.

## Inspect and reproduce

- [Methods, commands and explicit premises](../research/companion_attention/reference_onf/README.md)
- [Per-case certificate and source/recipe/AST hashes](../research/companion_attention/evidence/final_correspondence/FINAL_CORRESPONDENCE_CERTIFICATE.json)
- [Evidence directory](../research/companion_attention/evidence/final_correspondence/)
- [Compiler and validation source](../research/companion_attention/reference_onf/)
- [Paper III transcription and derived references](../research/companion_attention/hand_transcriptions/)
- [Original hand-source repository](https://github.com/womenflyplanes/moa-attention-verified-mullin/tree/0cf1d1fd368cb4a7e8d8ec4edfee97be758fc1bc)

The original archive covers 20 pinned hand C files. Paper III adds one manually
transcribed printed C listing and four newly derived C references for published
Python algorithms; these four are explicitly not historical archived C.

## What the result establishes

Independent exact-real end-DNF comparisons and ordered, typed operational C AST
comparisons agree per pair. Common admissible machine shape, layout, binding,
workspace, reduction and runtime choices yield equal lifted ONFs. Required
shape, precision, initialization, scratch, allocation, nonaliasing and no-overflow
premises are documented. Five fresh-device initialization corrections are paired
on both paths, while archived originals remain preserved.

This is executable translation validation with an explicit trusted basis, not a
proof-assistant theorem or a universal compiler-correctness theorem. Finite
floating-point checks supplement the symbolic and structural comparisons.

## Relationship to hardware evidence

Earlier hand-derived hardware studies and Paper I compiler measurements remain
separate evidence. This correspondence study ran no new hardware timing jobs;
OpenACC finite checks use GCC host fallback. Matching computation structures do
not alone establish identical binaries, runtime schedules or measured timing.
Some historical Paper V executed-source hashes remain unavailable, limiting
unconditional quantitative timing transfer. No Fortran/PyTorch frontend
certification is claimed.

## LinkedIn announcement draft

From hand-derived mathematics to compiler-generated AI kernels: a completed MoA correspondence milestone.

Across our four attention and Transformer studies, we have now checked that the compiler reproduces the recovered hand-derived C implementations.

• 39 numerical routines/regions across 20 archived C files: all covered.
• 44 paired implementation cases, including explicitly identified Paper III references.
• 246 end-DNF component comparisons: all match.
• Matching operational normal forms, addressed C computation trees, and lifted ONFs under the same admissible machine mapping.
• 138 untimed numerical comparisons: zero hand/compiler difference.

The coverage includes forward and backward attention, fused variants, decode and grouped-query attention, KV-cache append, RMSNorm, feed-forward networks, and Transformer-block variants, with OpenMP and OpenACC implementations.

Why does this matter? Expert hand derivations can become reproducible compiler outputs while preserving the checked mathematics and operational structure. The derivation steps may differ; the final normal forms must agree.

This establishes implementation correspondence and connects the compiler to our earlier hand-derived hardware studies. It does not claim new GPU measurements or automatic transfer of historical timings.

Code, certificates, provenance, and reproduction instructions:
https://github.com/womenflyplanes/moa-compiler/tree/codex/papers-i-iv-correspondence

AI is our first application domain. The longer-term goal is mathematically grounded, verifiable, portable, scalable, and predictable compilation of array algorithms.

#CompilerConstruction #MathematicsOfArrays #AI #HPC #OpenMP #OpenACC
