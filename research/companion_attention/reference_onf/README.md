# Complete hand-to-compiler C correspondence study

The declared C-emission study is closed: all 39 numerical routines/regions in the
20 pinned hand C files, plus Paper III's printed C listing and four explicitly
derived grouped-head/cache C references, have passing correspondence records.
There are 44 paired cases, 244 exact-real DNF component comparisons, 138 untimed
finite checks (largest hand/compiled difference zero), 27 matched helper checks,
and 109 deliberately corrupted operational ASTs rejected. Five separate-device
entry contracts have matched corrected hand and compiler variants.

The emitter never reads hand C. Independent structured recipes carry tuple ψ
selections, storage, arithmetic order, and schedule choices. The DNF interpreter
recovers their actual formulas, including mixed-radix tile writes and exact
finite-sum reindexing. Independent hand equations are compared in canonical
form. A separate parser compares the emitted and pinned hand operational C ASTs.

The table covers original untiled/tiled forward, backward and fused attention;
all benchmark serial references; CPU/GPU norm, FFN, full-block and norm–FFN chain
variants; single-query and grouped GQA/MQA/MHA decode; and cache append including
initial empty caches. Each compiler output matches its own chosen hand target.
Different hand schedules are not asserted equal to one another.

## Replay the full study

Python, NumPy, SymPy, pycparser, GCC/cpp are required. From this directory:

```bash
OMP_NUM_THREADS=2 ACC_DEVICE_TYPE=host python validate_all.py \
  --hand ORIGINAL_HAND_DIRECTORY --transcriptions ../hand_transcriptions --out results/programs
python check_all_dnfs.py results/evidence
python regress_all.py results/evidence
python check_all_lifts.py results/programs results/evidence
python check_initialization.py results/programs results/evidence
python close_inventory.py --hand ORIGINAL_HAND_DIRECTORY --programs results/programs \
  --transcriptions ../hand_transcriptions --out results/evidence
python seal_all.py results/programs results/evidence
```

Use the archive's `replay_all.sh` for these commands plus earlier regression and
archival checks. The original hand directory is `paper_V_software` in published
repository `womenflyplanes/moa-attention-verified-mullin`, pinned at commit
`0cf1d1fd368cb4a7e8d8ec4edfee97be758fc1bc`.

`check_all.py` performs source/AST comparisons without numerical execution.
`validate_all.py` generates paired standalone kernel translation units and builds
them as shared libraries. It performs no performance timing or remote submission.
OpenACC finite checks explicitly use GCC host fallback, which is never labeled as
a GPU run. Original input/output, timing and test drivers are retained separately.

## Exact claim and premises

DNFs agree in exact-real arithmetic. The matched operational trees preserve
floating type, scalar arithmetic order, loop bounds, storage, helpers, atomics and
parallel directives. Normalization removes formatting/single-statement braces,
canonicalizes exact decimal spelling, and compares integer address polynomials.
Address-cast normalization requires representable intermediates and no overflow.
This is trusted executable translation validation, not a proof-assistant theorem.

Positive extents, standard attention scaling, epsilon=1e-6, required nonaliasing,
allocation success, legal contiguous views, N divisible by tile width, H divisible
by G, and scratch bounds are explicit contracts. Cache length may be zero and
must be less than preallocated capacity. Additive gradient targets require zero
entry values where the routine does not assign them. Fixed 512-element scratch
imposes the relevant D/DFF bounds; MAX_N scratch requires N<=8192.

Equal fixed-machine lifted ONFs follow under the same admissible machine shape,
slot binding, private-workspace mapping, reduction tree and runtime choices.
All loops/directives are extracted independently from both C regions. Finite
checks cover 259 owner lifts, 280 tile bijections and 119 grouped-head maps.
Serial references remain serial; a new parallel extension must satisfy its own
admissibility/workspace premises. No physical A100/UA placement or performance is
inferred from a pragma or from the logical certificates.

## Provenance and corrected GPU entries

All 20 recovered hand C files match the pinned published source snapshot.
Paper III Listing 3 is a documented manual transcription of printed page 8 of
arXiv:2607.19456. The grouped/cache references are newly derived C ONFs of the
published Python Listings 7/5, not invented historical C files. Their numerical
checks also use an independent NumPy oracle; the cache proof checks every store,
unchanged prefix/tail and returned length. All source kinds are in the inventory. The new Paper III grouped/cache references use the published default FP32 contract; historical FP32/FP64 interfaces retain their own types. Cross-language mathematical equality is not a claim of bitwise PyTorch/BLAS operation order.

Five archival GPU profiles omit initialization of additive accumulators in fresh
separate device storage. `check_all.py` registers those variants. Corrected hand
and compiler data clauses transfer the host-zeroed targets with `copy`; arithmetic
bodies are unchanged and corrected paired ASTs match. Preserve the archival data
profiles for provenance; use corrected profiles for the fresh-device contract.
There has been no new GPU execution in this study.

Paper I job 137615's 403 freeze bindings are verified in the supplied archive.
Some Paper V executed-source hashes remain absent; this limits unconditional
quantitative timing transfer, not the completed current-source correspondence.
Fortran comparison programs and PyTorch autograd oracles are retained as
reference material; this study does not certify a Fortran or PyTorch frontend.
