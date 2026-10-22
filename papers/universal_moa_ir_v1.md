# The Universal Mathematics of Arrays Intermediate Representation
## Shape-Directed Transformations from DNF toward Dimension-Lifted ONF

**Working research manuscript — version 0.1, October 2026**

### Abstract
We propose a universal, typed, shape-aware abstract syntax tree (AST) as the single intermediate representation for Mathematics of Arrays (MoA) compilation. Mathematical normalization, ψ selection, index-domain transformation, shape partitioning, dimension lifting, and architecture-specific addressing are expressed as transformations of this shared representation rather than unrelated backend languages. We formalize an initial rank-one partition-lifting transformation and establish coverage, uniqueness, and indexed-element equivalence. A small reference implementation exhaustively checks these properties over finite test domains. The general DNF rewrite system, arbitrary-rank lifting, dependency-aware schedules, and machine-predictive models remain future work.

### 1. Mathematical vocabulary
For an array X, ρX denotes its shape. The valid index domain is ι(ρX). The expression i ψ X denotes selection at valid index i. The element count is τ(X) = Π(ρX). A concrete layout mapping γ maps a complete index to an address in rav(X); γ depends on the chosen layout. Mathematical selection is distinguished from physical addressing.

### 2. Universal representation
A single typed AST is intended to represent the source expression and its successive normal forms and machine-directed annotations. Each transformation has an explicit precondition, postcondition, and semantic-preservation obligation. A phase label or annotation is not itself a proof that normalization occurred. In this v1 implementation, the nodes are Array, Psi, Partition1D, and LiftedPsi. This is a restricted kernel of the intended IR, not a complete general AST or DNF normalizer.

### 3. Rank-one shape lifting
Let ρX = ⟨N⟩ and T > 0 be a tile extent. Let P = ⌈N/T⌉. Define the lifted rectangular domain of shape ⟨P,T⟩ and its valid subset

    V = {⟨p,t⟩ : 0 ≤ p < P, 0 ≤ t < T, pT+t < N}.

Define f(⟨p,t⟩) = ⟨pT+t⟩. A lifted selection of X is defined by

    ⟨p,t⟩ ψ Lift_T(X) = ⟨pT+t⟩ ψ X, for ⟨p,t⟩ ∈ V.

The final partition may contain masked positions; these are *not* valid selections. No zero-padding semantics are assumed.

**Theorem 1 (coverage and uniqueness).** The function f: V → ι(⟨N⟩) is bijective.

*Proof.* For any i with 0 ≤ i < N, Euclidean division by T gives unique p = ⌊i/T⌋ and t = i mod T, with 0 ≤ t < T and pT+t = i. Since i < N, ⟨p,t⟩ ∈ V. Hence f is surjective. If pT+t = p'T+t' and both remainders are in [0,T), uniqueness of quotient and remainder gives p=p' and t=t'. Thus f is injective. The empty N=0 case holds vacuously. ∎

**Corollary (element preservation).** For any valid i, writing p=⌊i/T⌋ and t=i mod T gives ⟨i⟩ ψ X = ⟨p,t⟩ ψ Lift_T(X). This establishes indexed-value equivalence; it does *not* by itself establish legality of arbitrary parallel schedules in the presence of dependencies or side effects.

### 4. Operational addressing
For row-major rank-one storage, γ(⟨i⟩,⟨N⟩)=i. The reference implementation maps both an original Psi node and its LiftedPsi equivalent to the same scalar offset. More general γ layouts and arbitrary-rank ONF lowering are outside v1.

### 5. Compiler obligations
The intended pipeline is MoA AST → ψ reductions/DNF → shape transformation and lifting → ONF with explicit layout/addressing → backend mapping. In v1, only the rank-one shape-lifting rewrite and row-major selection lowering are implemented. The full DNF/ONF stages remain specified goals. Machine descriptors and the existing attention-specific generator are separate, already validated prototypes; they have **not** yet been integrated into this universal IR.

### 6. Verification methodology
The implementation checks all valid indices for each test shape and tile, including empty domains and non-divisible extents. Tests verify coverage, uniqueness, ψ/address equivalence, and invalid-index rejection. These finite tests complement, but do not replace, the mathematical proof.

### 7. Research roadmap
1. General shape and typed operation AST, including scalar and prefix ψ selection.
2. Formal ψ-reduction and machine-independent DNF rewrite rules.
3. Arbitrary-rank partitions, dependence constraints, and reduction semantics.
4. Layout-parametric γ and dimension-lifted ONF.
5. Connect the existing attention generator to the universal IR.
6. Derive machine resource predictions and validate CPU, OpenMP, OpenACC, and MPI mappings on UAlbany systems.

### 8. Scope of claims
No general compiler, performance optimality, exact execution-time prediction, or complete DNF-to-ONF proof is claimed at this stage. The contribution here is a precise starting representation and one semantics-preserving shape transformation.
