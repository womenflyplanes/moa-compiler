"""Finite independent coverage tests for query-disjoint MoA Omega lifting."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("omega", ROOT / "tools/moa_query_omega.py")
omega = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = omega
spec.loader.exec_module(omega)


class QueryOmegaTests(unittest.TestCase):
    def test_coverage_and_uniqueness(self):
        for n in range(1, 24):
            for shape in [(1,), (2,), (3,), (2, 2), (2, 3), (3, 2), (2, 2, 2)]:
                mapping = omega.lift_queries(n, shape)
                self.assertTrue(mapping.verify())
                owned = [i for a, b in mapping.owner_intervals for i in range(a, b)]
                self.assertEqual(owned, list(range(n)))
                for i in range(n):
                    owner = mapping.psi_owner(i)
                    self.assertIn(i, range(*mapping.owner_intervals[owner]))

    def test_corrupt_certificate_rejected(self):
        mapping = omega.lift_queries(7, (2, 3))
        intervals = list(mapping.owner_intervals)
        intervals[0] = (0, 3)
        bad = omega.QueryOmega(7, (2, 3), tuple(intervals))
        self.assertFalse(bad.verify())
        with self.assertRaises(ValueError):
            bad.psi_owner(0)

    def test_invalid_domains(self):
        for n, shape in [(0, (2,)), (3, (0,)), (3, ()), (3, (2.0,)), (True, (2,))]:
            with self.assertRaises(ValueError):
                omega.lift_queries(n, shape)


if __name__ == "__main__":
    unittest.main()
