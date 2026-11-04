# Step 107: Minimum-Stage versus Minimum-Time Ω Scheduling

## A Mathematics of Arrays Counterexample

**Status:** Independently verified mathematical result  
**Compiler commit:** a5a7fe5  
**Target:** SC27 MoA Compiler Paper  
**Version:** Revised Step 107

---

## 1. Research Objective

The Mathematics of Arrays (MoA) provides a mathematical
foundation for deriving algorithms and mapping them onto
shaped machines.

The compiler follows the theory-first sequence:

    MoA semantic formulation
              |
              v
       DNF derivation
              |
              v
       ONF derivation
              |
              v
       Ω dimension lifting
              |
              v
       Machine-array mapping
              |
              v
       Legal communication
              |
              v
       Predictive scheduling
              |
              v
       Independent validation

The compiler must distinguish three questions:

1. Is a communication schedule legal?
2. What is its predicted communication cost?
3. Which legal schedule minimizes that cost?

Step 107 demonstrates that minimizing the number of
communication stages does not necessarily minimize
predicted communication time.

This distinction is fundamental to predictive,
machine-independent optimization.

---

## 2. MoA Machine Array

Let M denote the target machine array.

Its shape is

    ρM = <3>

Its coordinates are

    <0>, <1>, <2>.

The logical directed communication topology is

    <0> <----> <1> <----> <2>

There is no direct logical communication edge
between <0> and <2>.

The maximum number of outgoing transfers initiated
by each coordinate in one stage is

    p = 2.

The ownership distribution is

    <1,1,1>.

Each machine coordinate initially owns one payload.

Each required transfer carries 24 bytes.

These are logical MoA machine properties, not
measurements of the DGX physical interconnect.

---

## 3. Required Ω Communication

Every owner must deliver its payload to the
other two machine coordinates.

The six required transfers are:

Owner <0>:

    <0> -> <1>
    <1> -> <2>

Owner <1>:

    <1> -> <0>
    <1> -> <2>

Owner <2>:

    <2> -> <1>
    <1> -> <0>

A transfer may execute only when its source
possesses the corresponding owner's payload.

A legal schedule respects:

- Ownership and required delivery.
- Directed topology.
- Receive-before-forward dependencies.
- Outgoing port limits.
- Uniqueness of required transfers.

Communication stages are synchronous.

A payload received during one stage cannot be
forwarded until a subsequent stage.

Different owners' transfers may use the same
directed link in one stage.

---

## 4. Heterogeneous Directed-Link Costs

Each directed link e has two parameters:

    α(e): startup latency in seconds
    β(e): bandwidth in bytes/second

The verified counterexample uses:

| Directed link | α (µs) | β (GB/s) |
|---------------|-------:|---------:|
| <0> -> <1>    | 0.1    | 50       |
| <1> -> <0>    | 0.1    | 1        |
| <1> -> <2>    | 20     | 1        |
| <2> -> <1>    | 0      | 5        |

The directed link

    <1> -> <2>

has a large startup latency.

This asymmetry allows stage grouping to reduce
predicted time even when the number of stages
increases.

---

## 5. Step 103 Communication-Time Model

Let S be a legal communication stage.

For each directed link e, define

    B(e,S)

as the sum of payload bytes assigned to e
during stage S.

For an active directed link, its predicted
stage contribution is

    α(e) + B(e,S)/β(e).

The stage completion time is

    T(S) =
      max over active links e
      [α(e) + B(e,S)/β(e)].

For a schedule

    Q = (S_0, S_1, ..., S_(k-1)),

the predicted total communication time is

    T(Q) = Σ T(S_i).

Each active link pays its startup cost once
per stage, regardless of how many logical
transfers share that link.

This is the Step 103 logical contention model.

It does not include all physical hardware
effects and does not constitute a measured
execution-time prediction for a specific DGX
configuration without further calibration.

---

## 6. Minimum-Stage Heuristic Schedule

The Step 104 heuristic selects a legal
two-stage schedule.

### Stage 0

    Owner <0>: <0> -> <1>
    Owner <1>: <1> -> <0>
    Owner <1>: <1> -> <2>
    Owner <2>: <2> -> <1>

The expensive link <1> -> <2> carries 24 bytes.

Its cost is

    20 + 24/1000
      = 20.024 µs.

Thus,

    T(S_0) = 20.024 µs.

### Stage 1

    Owner <0>: <1> -> <2>
    Owner <2>: <1> -> <0>

Again, the expensive link carries 24 bytes.

Thus,

    T(S_1) = 20.024 µs.

### Total

    T_heuristic
      = 20.024 + 20.024
      = 40.048 µs.

The heuristic schedule uses two stages.

It activates the expensive directed link
in both stages.

---

## 7. Exact Minimum-Time Schedule

The Step 105 exact scheduler searches the
legal stage space using memoized dynamic
programming.

For this finite instance, it produces
three stages.

### Stage 0

    Owner <2>: <2> -> <1>

The link carries 24 bytes at 5 GB/s
with zero startup latency.

    T(S_0)
      = 24/5000
      = 0.0048 µs.

### Stage 1

    Owner <0>: <0> -> <1>
    Owner <1>: <1> -> <0>
    Owner <2>: <1> -> <0>

The directed link <1> -> <0> carries

    24 + 24 = 48 bytes.

Its cost is

    0.1 + 48/1000
      = 0.148 µs.

Therefore,

    T(S_1) = 0.148 µs.

### Stage 2

    Owner <0>: <1> -> <2>
    Owner <1>: <1> -> <2>

The expensive link <1> -> <2> carries

    24 + 24 = 48 bytes.

Its cost is

    20 + 48/1000
      = 20.048 µs.

Therefore,

    T(S_2) = 20.048 µs.

### Total

    T_exact
      = 0.0048 + 0.148 + 20.048
      = 20.2008 µs.

The exact schedule uses three stages.

However, the expensive directed link is
activated only once.

---

## 8. Proposition 107.1

### Minimum-Stage Scheduling Does Not Imply Minimum Time

**Proposition.**

Under the Step 103 heterogeneous directed-link
contention model, a legal Ω communication
schedule with the minimum possible number
of stages need not minimize predicted
communication time.

### Proof by Counterexample

Consider the MoA machine array

    ρM = <3>

with outgoing port limit

    p = 2.

The required Ω communication consists of
the six transfers specified in Section 3.

First, establish a lower bound on stage count.

The payload initially owned by coordinate

    <0>

must reach coordinate

    <2>.

Because there is no direct logical link,
delivery requires the directed path

    <0> -> <1> -> <2>.

Under the receive-before-forward constraint,
the transfer

    <0> -> <1>

must complete before

    <1> -> <2>

can begin.

These transfers therefore cannot occur
in the same communication stage.

Consequently, every legal complete
communication schedule requires at least
two stages.

The Step 104 heuristic constructs a legal
two-stage schedule.

Therefore,

    minimum legal stage count = 2.

Its predicted communication time is

    T_2 = 40.048 µs.

The Step 105 exact scheduler constructs
a legal three-stage schedule with

    T_3 = 20.2008 µs.

Independent memoized verification confirms
that this is the minimum predicted cost
for the specified finite instance.

Hence,

    T_3 < T_2

while

    3 > 2.

Therefore, a minimum-stage legal schedule
does not necessarily minimize predicted
communication time.

**QED.**

---

## 9. Quantitative Result

The absolute predicted reduction is

    ΔT = 40.048 - 20.2008

       = 19.8472 µs.

The relative reduction is

    100 × ΔT / 40.048

       = 49.55852976428285%.

Thus, the exact scheduler achieves

    49.56% lower predicted communication time.

The two schedules perform the same six
required logical transfers.

The improvement arises from stage grouping,
directed-link startup costs, and the
dependency structure.

It is not a reduction in required
communication volume.

---

## 10. Independent Verification

The counterexample was discovered using

    moa_attention_adversarial_search.py

and verified using

    test_attention_adversarial_schedule.py.

Four targeted regression tests passed:

1. Strict improvement over the heuristic.
2. Deterministic exact scheduling.
3. Consolidation of the expensive link.
4. Independent reference-optimum agreement.

The independent reference optimizer examined

    36 states
    176 transitions.

The complete universal-IR regression suite
reported

    259 tests executed
    258 passed
    1 skipped
    0 failures.

The skipped test is an expensive brute-force
case covered separately by a memoized
reference implementation.

The result establishes optimality under
the specified model for this finite instance.

It does not establish that exact scheduling
is computationally practical for arbitrary
large machine arrays.

---

## 11. Compiler Consequences

The MoA compiler must distinguish legality,
prediction, and optimization.

### Legality

MoA-derived dependencies and machine
constraints determine which schedules
are permissible.

### Prediction

The communication model assigns a
predicted cost to each legal schedule.

Stage count alone is insufficient
when link costs are heterogeneous.

### Optimization

The compiler must compare legal schedules
using the selected predictive objective.

Exact optimization provides a reference
for tractable instances.

Heuristic scheduling remains useful for
larger instances, but its output cannot
be assumed globally optimal.

This separation is essential to the
predictive compiler architecture.

---

## 12. SC27 Research Significance

The result demonstrates that machine-aware
optimization must be derived from the
mathematical properties of communication.

The MoA pipeline is

    MoA algorithm
         |
         v
    ψ-indexed DNF
         |
         v
    Addressed ONF
         |
         v
    Ω dimension lifting
         |
         v
    Shaped machine mapping
         |
         v
    Legal Ω communication
         |
         v
    Predicted schedule costs
         |
         v
    Optimization
         |
         v
    Independent verification

The compiler's task is not simply
to generate executable code.

It must preserve MoA semantics while
deriving a legal mapping whose resource
requirements and predicted costs can
be examined before execution.

The counterexample shows why
optimization objectives must be
mathematically explicit.

---

## 13. Scope and Limitations

This result establishes a counterexample
within the Step 103 model.

It does not establish:

- A measured 49.56% DGX speedup.
- A universal speedup on all machines.
- Global optimality of the Step 104 heuristic.
- Tractable exact optimization at every scale.
- Complete modeling of physical contention.

The independent exact verification applies
to the specified finite transfer instance.

The broader research program will examine
which machine shapes, link parameters,
ownership distributions, and resource
constraints admit efficient predictive
optimization.

---

## 14. Reproducibility

Compiler branch:

    develop/mac-compiler-integration

Verified Step 106 commit:

    a5a7fe5

Research implementation:

    research/typed_attention/
      moa_attention_adversarial_search.py

Regression verification:

    tests/universal_ir/
      test_attention_adversarial_schedule.py

This revised Step 107 document records
the mathematical interpretation of
the verified counterexample.

---

## Conclusion

The Mathematics of Arrays provides a
foundation for reasoning about algorithms,
machine shape, communication, and
execution mappings within a unified
mathematical framework.

The Step 107 counterexample demonstrates:

    MINIMUM COMMUNICATION STAGES
                  DOES NOT IMPLY
    MINIMUM PREDICTED COMMUNICATION TIME.

The distinction is not merely an
implementation detail.

It follows from the mathematical
interaction of dependencies, machine
topology, heterogeneous directed-link
costs, and stage scheduling.

The MoA compiler must therefore optimize
predicted execution cost over legal
machine mappings rather than relying
on stage count as a surrogate.

**MoA all the way.**
