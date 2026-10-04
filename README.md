# MoA Universal IR: Same DNF, Many ONFs, Predictable

**Claim:** Papers 1-5 via compiler in Python and Fortran (and C, OpenMP) same F=786432 M=768 T=0.006218ms.

**Proof:**
- `moa_compiler/generated/` 16 files Papers 1-5 in 4 langs
- `MULTI_LANG_PROOF.txt` all same output
- `./moa_compiler/run_all_langs.sh`

**5-device:** Delta A100, MI100, Max1550 pvc, V100/H100, A40 M1exp:36 Pd:9 Xd:0 Cd:36

**GUI:** Open `moa_compiler/gui.html` drag d,k,n live cost.

**Paper:** `moa_SC27_FINAL_WITH_ACK.pdf`
