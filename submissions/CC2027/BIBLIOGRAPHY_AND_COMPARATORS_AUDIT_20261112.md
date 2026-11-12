# CC 2027 bibliography and competitor audit — 2026-11-12

The expanded local reviewer draft has 12 bibliography items after this pass. This audit distinguishes verified primary sources from unresolved entries. The local PDF was rebuilt successfully to 10 pages total, **not** 10 pages of main text plus references.

## Checked primary references and comparisons

| Key | Citation | Primary record | Relevance |
| --- | --- | --- | --- |
| feautrier | Feautrier (1992), *Some efficient solutions to the affine scheduling problem: Part I. One-dimensional time*, IJPP 21(5), 313–348, DOI 10.1007/BF01407835 | https://polyhedral.info/publications.html | Polyhedral affine scheduling, dependence and legality |
| halide2012 | Ragan-Kelley et al. (2012), *Decoupling Algorithms from Schedules for Easy Optimization of Image Processing Pipelines*, ACM TOG 31(4), Article 32, DOI 10.1145/2185520.2185528 | https://dspace.mit.edu/entities/publication/121e12b2-928b-4e8f-a78a-79f44d11cc6b | Original algorithm/schedule separation |
| halide | Ragan-Kelley et al. (2013), *Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines*, PLDI, 519–530, DOI 10.1145/2491956.2462176 | https://users.cs.duke.edu/~lkw34/papers/halide-pldi2013.pdf | Explicit scheduling and compiler realization |
| halideauto | Adams et al. (2016), *Automatically Scheduling Halide Image Processing Pipelines* | https://research.google/pubs/automatically-scheduling-halide-image-processing-pipelines/ | Automated scheduling, not merely manual schedules |
| tvm | Chen et al. (2018), *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning*, OSDI, 578–594 | https://www.usenix.org/conference/osdi18/presentation/chen | Hardware-aware compilation and cost-model search |
| mlir | Lattner et al. (2021), *MLIR: Scaling Compiler Infrastructure for Domain Specific Computation*, CGO | https://research.google/pubs/mlir-scaling-compiler-infrastructure-for-domain-specific-computation/ | Multi-level compiler IR and lowering |
| flash | Dao et al. (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 35, 16344–16359 | https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract.html | IO-aware attention algorithm and GPU comparator |
| flash2 | Dao (2024), *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, ICLR | https://proceedings.iclr.cc/paper_files/paper/2024/hash/98ed250b203d1ac6b24bbcf263e3d4a7-Abstract-Conference.html | Attention partitioning and GPU baseline |
| hardware | Mullin and Hains (2026), *Validating Memory-Optimal Transformer Kernels on Real Hardware*, arXiv:2609.33916 | https://arxiv.org/abs/2609.33916 | Historical **hand-derived** validation, not compiler result |

## Still requiring final primary-record verification

- **paperI:** arXiv:2606.07713 title and author names should be reconciled with current arXiv version metadata, and self-citation must be anonymized in prose.
- **oneDerivation:** title and author names are present in project materials, but exact public archival identifier and publication status remain unresolved. Do not invent a DOI.
- **prediction:** arXiv:2610.03197 title and author names should be reconciled with current arXiv version metadata.
- **feautrier pagination:** dblp lists 313–347 while another bibliography lists 313–348. DOI is verified; check publisher PDF before final page range.

## Competitor coverage still needed

Add grounded comparison to XLA, Tensor Comprehensions, TACO, Polly/isl, affine and array-oriented compilers, Triton, PyTorch's SDPA implementation, and one or more contemporary CPU attention baselines. These are **candidate** related works; their exact papers and relevance must be verified before citation. Do not inflate bibliography by inserting uncited papers.

## Review standard

For every competitor: identify semantic IR, transformation method, target machine model, whether scheduling is explicit or searched, correctness evidence, and what experiment would make a fair comparison. Do not claim MoA is the first to separate semantics and scheduling: Halide already does so. Do not claim compiler-generated MoA beats FlashAttention, TVM, or PyTorch without matched benchmarks.
