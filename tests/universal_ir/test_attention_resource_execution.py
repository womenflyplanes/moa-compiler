
"""Numerical execution tests for resource-owned MoA attention.

Additive tests: no existing compiler modules are modified.

The generated C/OpenMP kernel is compiled and executed against
an independent stable attention reference.

Tests heterogeneous ownership, idle components, numerical
correctness, source preservation, and output completeness.
"""

import math
import shutil
import subprocess
import tempfile
import unittest

from pathlib import Path

from moa_machine import Machine, Work

from moa_typed_ir import (
    specification,
    stable_online_fold,
    gamma_lower,
    schedule_omega,
    emit_c,
)

from moa_attention_resource_planner import (
    modeled_attention_bytes,
    plan_attention_omega,
)

from moa_attention_resource_emitter import (
    emit_resource_owned_openmp,
    ownership_declarations,
)


def reference_attention(Q, K, V, n, m, d, dv):
    """Independent stable attention calculation."""
    output = []

    for i in range(n):
        scores = []

        for k in range(m):
            dot = math.fsum(
                Q[i * d + j] * K[k * d + j]
                for j in range(d)
            )
            scores.append(dot / math.sqrt(d))

        maximum = max(scores)

        weights = [
            math.exp(score - maximum)
            for score in scores
        ]

        denominator = math.fsum(weights)

        for v in range(dv):
            numerator = math.fsum(
                weights[k] * V[k * dv + v]
                for k in range(m)
            )
            output.append(numerator / denominator)

    return output


def make_driver(n, m, d, dv, trial):
    """Create a standalone C driver with deterministic inputs."""
    return """
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void generated_attention(
    const double*, const double*, const double*,
    double*, size_t, size_t, size_t, size_t
);

int main(void) {
    const size_t n = %d, m = %d, d = %d, dv = %d;
    const int trial = %d;

    double Q[%d], K[%d], V[%d], O[%d];

    for (size_t i = 0; i < n*d; i++)
        Q[i] = sin((i+1)*0.17 + trial*0.013)
               * (trial %% 2 ? 1.0 : 7.0);

    for (size_t i = 0; i < m*d; i++)
        K[i] = cos((i+3)*0.13 - trial*0.019)
               * (trial %% 2 ? 1.0 : 7.0);

    for (size_t i = 0; i < m*dv; i++)
        V[i] = sin((i+5)*0.11 + trial*0.023);

    for (size_t i = 0; i < n*dv; i++)
        O[i] = NAN;

    generated_attention(Q, K, V, O, n, m, d, dv);

    for (size_t i = 0; i < n*dv; i++) {
        if (!isfinite(O[i])) return 2;
        printf("%%.17g\\n", O[i]);
    }

    return 0;
}
""" % (
        n, m, d, dv, trial,
        n * d, m * d, m * dv, n * dv,
    )


def python_inputs(n, m, d, dv, trial):
    """Reproduce the deterministic C input formulas."""
    Q = [
        math.sin((i + 1) * 0.17 + trial * 0.013)
        * (1.0 if trial % 2 else 7.0)
        for i in range(n * d)
    ]

    K = [
        math.cos((i + 3) * 0.13 - trial * 0.019)
        * (1.0 if trial % 2 else 7.0)
        for i in range(m * d)
    ]

    V = [
        math.sin((i + 5) * 0.11 + trial * 0.023)
        for i in range(m * dv)
    ]

    return Q, K, V


@unittest.skipUnless(
    shutil.which("cc"),
    "C compiler unavailable",
)
class TestResourceOwnedExecution(unittest.TestCase):

    def setUp(self):
        self.expr = gamma_lower(
            stable_online_fold(specification())
        )

    def make_plan(self, work, shape, counts):
        machine = Machine(shape)

        capacities = tuple(
            modeled_attention_bytes(work, q)
            for q in counts
        )

        return plan_attention_omega(
            self.expr,
            work,
            machine,
            capacities,
        )

    def execute_case(self, work, shape, counts):
        plan = self.make_plan(work, shape, counts)
        certificate = emit_resource_owned_openmp(plan)

        self.assertTrue(certificate.verify())
        self.assertEqual(plan.query_counts, counts)

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)

            source = folder / "generated.c"
            driver = folder / "driver.c"
            binary = folder / "run"

            source.write_text(certificate.source)

            worst = 0.0

            for trial in range(5):
                driver.write_text(
                    make_driver(
                        work.n,
                        work.m,
                        work.d,
                        work.dv,
                        trial,
                    )
                )

                subprocess.run(
                    [
                        "cc",
                        "-std=c11",
                        "-O1",
                        "-fopenmp",
                        str(source),
                        str(driver),
                        "-lm",
                        "-o",
                        str(binary),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                result = subprocess.run(
                    [str(binary)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env={"OMP_NUM_THREADS": "2"},
                )

                actual = [
                    float(line)
                    for line in result.stdout.splitlines()
                ]

                Q, K, V = python_inputs(
                    work.n,
                    work.m,
                    work.d,
                    work.dv,
                    trial,
                )

                expected = reference_attention(
                    Q,
                    K,
                    V,
                    work.n,
                    work.m,
                    work.d,
                    work.dv,
                )

                self.assertEqual(
                    len(actual),
                    work.n * work.dv,
                )

                for observed, target in zip(
                    actual,
                    expected,
                ):
                    self.assertTrue(math.isfinite(observed))

                    error = abs(observed - target)
                    worst = max(worst, error)

                    self.assertLessEqual(error, 1e-11)

            print(
                "PASS numerical resource Omega:",
                "shape=", shape,
                "counts=", counts,
                "maxerr=", worst,
            )

    def test_heterogeneous_ownership(self):
        self.execute_case(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
        )

    def test_idle_machine_components(self):
        self.execute_case(
            Work(3, 6, 2, 4),
            (2, 3),
            (3, 0, 0, 0, 0, 0),
        )

    def test_noncanonical_unequal_ownership(self):
        self.execute_case(
            Work(11, 7, 8, 4),
            (3, 2),
            (1, 2, 3, 1, 2, 2),
        )

    def test_only_ownership_declarations_change(self):
        work = Work(7, 5, 4, 3)
        machine = Machine((2, 3))

        plan = self.make_plan(
            work,
            machine.shape,
            (1, 2, 1, 1, 1, 1),
        )

        emitted = emit_resource_owned_openmp(plan)

        canonical = schedule_omega(
            self.expr,
            work,
            machine,
        )

        original = emit_c(canonical)

        old_begin, old_end = ownership_declarations(
            canonical.schedule.intervals
        )

        new_begin, new_end = ownership_declarations(
            plan.intervals
        )

        expected = original.replace(
            old_begin, new_begin, 1
        ).replace(
            old_end, new_end, 1
        )

        self.assertEqual(emitted.source, expected)


if __name__ == "__main__":
    unittest.main()
