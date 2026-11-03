
"""Numerical validation of resource-derived MoA Omega MPI emission.

Reuses the existing Gate 09 sequential fake-MPI C harness.
This is NOT a real MPI runtime, multi-process test, or network test.

All tests are additive. No existing compiler or harness is modified.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from moa_machine import Machine, Work

from moa_typed_ir import (
    specification,
    stable_online_fold,
    gamma_lower,
    schedule_omega,
)

from moa_mpi import emit_mpi

from moa_attention_resource_planner import (
    modeled_attention_bytes,
    plan_attention_omega,
)

from moa_attention_resource_mpi import (
    collective_from_intervals,
    c_array_declaration,
    emit_resource_owned_mpi,
)


ROOT = Path(__file__).resolve().parents[1] / "mpi_gate11"
HARNESS = ROOT / "test_mpi_emulated.c"
MPI_STUB = ROOT / "mpi_syntax_stub"


def make_expression():
    return gamma_lower(
        stable_online_fold(specification())
    )


def make_plan(work, shape, counts):
    capacities = tuple(
        modeled_attention_bytes(work, q)
        for q in counts
    )

    return plan_attention_omega(
        make_expression(),
        work,
        Machine(shape),
        capacities,
    )


@unittest.skipUnless(
    shutil.which("cc"),
    "C compiler unavailable",
)
class TestResourceOmegaMPI(unittest.TestCase):

    def execute_case(self, work, shape, counts):
        plan = make_plan(work, shape, counts)

        self.assertEqual(plan.query_counts, counts)
        self.assertTrue(plan.verify())

        emitted = emit_resource_owned_mpi(plan)

        self.assertTrue(emitted.verify())

        expected_counts = tuple(
            q * work.dv for q in counts
        )

        expected_displacements = tuple(
            sum(expected_counts[:r])
            for r in range(len(counts))
        )

        self.assertEqual(
            emitted.counts,
            expected_counts,
        )

        self.assertEqual(
            emitted.displacements,
            expected_displacements,
        )

        self.assertEqual(
            emitted.output_components,
            work.n * work.dv,
        )

        self.assertEqual(
            sum(emitted.counts),
            work.n * work.dv,
        )

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)

            source = folder / "resource_attention_mpi.c"
            binary = folder / "run"

            source.write_text(emitted.source)

            p = len(counts)

            command = [
                "cc",
                "-std=c11",
                "-O1",
                "-Wall",
                "-Wextra",
                "-Werror",
                f"-DN={work.n}",
                f"-DM={work.m}",
                f"-DD={work.d}",
                f"-DDV={work.dv}",
                f"-DP={p}",
                "-I",
                str(MPI_STUB),
                str(source),
                str(HARNESS),
                "-lm",
                "-o",
                str(binary),
            ]

            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )

            result = subprocess.run(
                [str(binary)],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertIn(
                "PASS emulated MPI rank ownership",
                result.stdout,
            )

            self.assertIn(
                "trials=25",
                result.stdout,
            )

            print(result.stdout.strip())

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

    def test_collective_rejects_gap(self):
        with self.assertRaises(ValueError):
            collective_from_intervals(
                ((0, 1), (2, 3)),
                3,
                4,
            )

    def test_collective_rejects_overlap(self):
        with self.assertRaises(ValueError):
            collective_from_intervals(
                ((0, 2), (1, 3)),
                3,
                4,
            )

    def test_collective_rejects_incomplete_output(self):
        with self.assertRaises(ValueError):
            collective_from_intervals(
                ((0, 1), (1, 2)),
                3,
                4,
            )

    def test_collective_rejects_mpi_integer_overflow(self):
        with self.assertRaises(ValueError):
            collective_from_intervals(
                ((0, 2**30),),
                2**30,
                8,
            )

    def test_only_four_declarations_change(self):
        work = Work(7, 5, 4, 3)
        machine = Machine((2, 3))

        plan = make_plan(
            work,
            machine.shape,
            (1, 2, 1, 1, 1, 1),
        )

        emitted = emit_resource_owned_mpi(plan)

        canonical = schedule_omega(
            make_expression(),
            work,
            machine,
        )

        original = emit_mpi(canonical)

        old_intervals = canonical.schedule.intervals
        new_intervals = plan.intervals

        old_counts, old_displs = collective_from_intervals(
            old_intervals,
            work.n,
            work.dv,
        )

        replacements = (
            (
                c_array_declaration("counts", old_counts),
                c_array_declaration(
                    "counts", emitted.counts
                ),
            ),
            (
                c_array_declaration(
                    "displs", old_displs
                ),
                c_array_declaration(
                    "displs", emitted.displacements
                ),
            ),
            (
                c_array_declaration(
                    "owner_begin",
                    (b for b, e in old_intervals),
                    "size_t",
                ),
                c_array_declaration(
                    "owner_begin",
                    (b for b, e in new_intervals),
                    "size_t",
                ),
            ),
            (
                c_array_declaration(
                    "owner_end",
                    (e for b, e in old_intervals),
                    "size_t",
                ),
                c_array_declaration(
                    "owner_end",
                    (e for b, e in new_intervals),
                    "size_t",
                ),
            ),
        )

        expected = original

        for old, new in replacements:
            self.assertEqual(expected.count(old), 1)
            expected = expected.replace(old, new, 1)

        self.assertEqual(emitted.source, expected)

    def test_collective_metadata_exact(self):
        intervals = (
            (0, 1),
            (1, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
        )

        counts, displacements = collective_from_intervals(
            intervals,
            7,
            3,
        )

        self.assertEqual(
            counts,
            (3, 6, 3, 3, 3, 3),
        )

        self.assertEqual(
            displacements,
            (0, 3, 9, 12, 15, 18),
        )


if __name__ == "__main__":
    unittest.main()
