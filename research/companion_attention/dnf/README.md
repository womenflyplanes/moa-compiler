# Independent DNF corroboration

The checker compares manually transcribed hand equations with symbolic recovery
from the compiler's existing phase specifications. All reductions have explicit
zero-based bounds. Expressions use ψ selections before any address lowering.

Run from the companion_attention directory with Python 3.10+ and SymPy.

```bash
PYTHONPATH=. python3 dnf/check.py results
PYTHONPATH=. python3 dnf/regressions.py results
```

check.py validates input/output shapes and free-index domains, recovers bounded
reductions and substitutes scalar and array temporaries, and compares canonical
expressions. It writes both expressions, their digests, the shape contracts and
the complete block substitution graph to JSON. The graph closes over all ten
block observables and the original block inputs.

The code in hand.py is a manual transcription from the supplied paper equations
and the recovered reference routines. It imports only mathematical constructors
from recover.py, and does not import the compiler's kernel definitions or phase
strings. The compiler expressions are recovered by interpreting the actual phase
strings that compile.py uses for code emission.

The trusted basis includes this manual transcription, the restricted recovery
algorithm, SymPy's exact symbolic algebra, and the explicit shape premises.
The checker covers the pinned specifications, including the pre-norm block and
its derivatives. Other compiler inputs or hand implementation variants require
their own source binding and checks.

The exact-real comparison permits scalar algebra, finite-sum linearity, pulling
constant factors from sums, cancellation of a common finite softmax shift, and
capture-avoiding bound-index renaming. Each bound extent remains in the canonical
expression. Integer head grouping uses the premise that G divides H. The block
uses T=B*N and E=D, with an explicit token/batch-query bijection.

regressions.py checks deliberate expression, bound, interface and shape defects.
It also verifies acceptance of consistent bound renaming. It performs no timing,
OpenMP run, OpenACC run, or job submission.
