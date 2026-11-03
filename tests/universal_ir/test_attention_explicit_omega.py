
"""Regression tests for explicit MoA Omega ownership.

Tests:
  - Unequal query ownership
  - Idle machine components
  - Preservation of tau(output)
  - Overlapping ownership rejection
  - Missing ownership rejection
  - Incorrect interval count
  - Malformed intervals
  - Invalid query counts
  - Certificate tampering rejection
  - Canonical Omega scheduling remains unchanged
"""

import unittest
from dataclasses import replace

from moa_machine import Machine, Work

from moa_typed_ir import (
    specification,
    stable_online_fold,
    gamma_lower,
    schedule_omega,
    verify_schedule,
)

from moa_attention_explicit_omega import (
    schedule_omega_explicit,
    schedule_omega_from_counts,
)


class TestExplicitOmega(unittest.TestCase):

    def setUp(self):
        self.expr = gamma_lower(
            stable_online_fold(specification())
        )
        self.work = Work(7, 5, 4, 3)
        self.machine = Machine((2, 3))

    def test_unequal_partition(self):
        cert = schedule_omega_from_counts(
            self.expr,
            self.work,
            self.machine,
            (0, 1, 2, 0, 3, 1),
        )

        self.assertTrue(cert.verify())

        self.assertEqual(
            cert.query_counts,
            (0, 1, 2, 0, 3, 1),
        )

        self.assertEqual(
            cert.scheduled.schedule.intervals,
            (
                (0, 0),
                (0, 1),
                (1, 3),
                (3, 3),
                (3, 6),
                (6, 7),
            ),
        )

    def test_idle_machine_components(self):
        cert = schedule_omega_from_counts(
            self.expr,
            self.work,
            self.machine,
            (0, 1, 2, 0, 3, 1),
        )

        self.assertEqual(
            cert.query_counts.count(0),
            2,
        )

    def test_tau_output_preserved(self):
        cert = schedule_omega_from_counts(
            self.expr,
            self.work,
            self.machine,
            (0, 1, 2, 0, 3, 1),
        )

        self.assertEqual(
            cert.output_components,
            (0, 3, 6, 0, 9, 3),
        )

        self.assertEqual(
            sum(cert.output_components),
            self.work.n * self.work.dv,
        )

    def test_overlap_rejected(self):
        intervals = (
            (0, 2),
            (1, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
        )

        with self.assertRaises(ValueError):
            schedule_omega_explicit(
                self.expr,
                self.work,
                self.machine,
                intervals,
            )

    def test_gap_rejected(self):
        intervals = (
            (0, 1),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
        )

        with self.assertRaises(ValueError):
            schedule_omega_explicit(
                self.expr,
                self.work,
                self.machine,
                intervals,
            )

    def test_wrong_interval_count_rejected(self):
        with self.assertRaises(ValueError):
            schedule_omega_explicit(
                self.expr,
                self.work,
                self.machine,
                ((0, 7),),
            )

    def test_malformed_interval_rejected(self):
        intervals = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            ("5", 7),
        )

        with self.assertRaises(ValueError):
            schedule_omega_explicit(
                self.expr,
                self.work,
                self.machine,
                intervals,
            )

    def test_negative_query_count_rejected(self):
        with self.assertRaises(ValueError):
            schedule_omega_from_counts(
                self.expr,
                self.work,
                self.machine,
                (0, 1, 2, -1, 4, 1),
            )

    def test_incorrect_query_total_rejected(self):
        with self.assertRaises(ValueError):
            schedule_omega_from_counts(
                self.expr,
                self.work,
                self.machine,
                (0, 1, 2, 0, 2, 1),
            )

    def test_certificate_tampering_rejected(self):
        cert = schedule_omega_from_counts(
            self.expr,
            self.work,
            self.machine,
            (0, 1, 2, 0, 3, 1),
        )

        altered = replace(
            cert,
            owners=cert.owners[:-1],
        )

        with self.assertRaises(ValueError):
            altered.verify()

    def test_canonical_scheduler_unchanged(self):
        canonical = schedule_omega(
            self.expr,
            self.work,
            self.machine,
        )

        self.assertTrue(
            verify_schedule(canonical)
        )

        self.assertEqual(
            canonical.schedule.intervals,
            (
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 7),
            ),
        )

    def test_canonical_verifier_rejects_noncanonical(self):
        cert = schedule_omega_from_counts(
            self.expr,
            self.work,
            self.machine,
            (0, 1, 2, 0, 3, 1),
        )

        with self.assertRaises(ValueError):
            verify_schedule(cert.scheduled)


if __name__ == "__main__":
    unittest.main()
