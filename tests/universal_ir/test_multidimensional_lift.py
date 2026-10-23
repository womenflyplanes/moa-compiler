import unittest
from moa_v3 import DNF, MachineArray, gamma_sequential
from moa_multidimensional_lift import ResourceArray, Rectangle, rectangular_lift


def machine(name, capacity=1024, chunk=1024):
    return MachineArray(name, (1, 1, 0, 1.0, chunk, capacity, 0, None, None, 0), capacity, chunk)


class MultidimensionalLiftTests(unittest.TestCase):
    def test_two_by_two_machine_array(self):
        dnf = DNF((4, 6), 4)
        seq = gamma_sequential(dnf)
        resources = ResourceArray((2, 2), tuple(machine(str(i)) for i in range(4)))
        rects = tuple(Rectangle((r * 2, c * 3), ((r + 1) * 2, (c + 1) * 3), (r, c))
                      for r in range(2) for c in range(2))
        result = rectangular_lift(seq, resources, rects)
        self.assertEqual(len(result.mapped_indices()), 24)
        self.assertEqual(result.mapped_indices()[0], ((0, 0), (0, 0), 0))
        self.assertEqual(result.mapped_indices()[-1], ((3, 5), (1, 1), 23))
        self.assertIs(result.sequential.dnf, dnf)

    def test_uneven_tiles_and_rank_three(self):
        seq = gamma_sequential(DNF((3, 5, 2), 8))
        resources = ResourceArray((2,), (machine('cpu'), machine('gpu')))
        rects = (Rectangle((0, 0, 0), (1, 5, 2), (0,)),
                 Rectangle((1, 0, 0), (3, 5, 2), (1,)))
        result = rectangular_lift(seq, resources, rects)
        self.assertEqual(len(result.mapped_indices()), 30)
        self.assertEqual(result.mapped_indices()[-1][2], 29)

    def test_overlap_rejected_even_when_counts_match(self):
        seq = gamma_sequential(DNF((2, 2), 4))
        resources = ResourceArray((2,), (machine('a'), machine('b')))
        with self.assertRaisesRegex(ValueError, 'covered 2 times'):
            rectangular_lift(seq, resources, (Rectangle((0, 0), (1, 2), (0,)),
                                                    Rectangle((0, 0), (1, 2), (1,))))

    def test_hole_rejected(self):
        seq = gamma_sequential(DNF((2, 2), 4))
        resources = ResourceArray((1,), (machine('a'),))
        with self.assertRaisesRegex(ValueError, 'covered 0 times'):
            rectangular_lift(seq, resources, (Rectangle((0, 0), (1, 2), (0,)),))

    def test_resource_bounds_and_capacity(self):
        seq = gamma_sequential(DNF((2, 3), 4))
        resources = ResourceArray((1,), (machine('small', capacity=8, chunk=2),))
        with self.assertRaisesRegex(ValueError, 'capacity'):
            rectangular_lift(seq, resources, (Rectangle((0, 0), (2, 3), (0,)),))
        with self.assertRaises(IndexError):
            resources.psi((1,))
        with self.assertRaises(ValueError):
            ResourceArray((2, 2), (machine('only'),))

    def test_empty_and_scalar_domains(self):
        empty = gamma_sequential(DNF((2, 0, 4), 4))
        resource = ResourceArray((1,), (machine('cpu'),))
        self.assertTrue(rectangular_lift(empty, resource, ()).verify())
        scalar = gamma_sequential(DNF((), 8))
        result = rectangular_lift(scalar, resource, (Rectangle((), (), (0,)),))
        self.assertEqual(result.mapped_indices(), (((), (0,), 0),))

    def test_invalid_rectangle(self):
        seq = gamma_sequential(DNF((2, 2), 4))
        resources = ResourceArray((1,), (machine('cpu'),))
        with self.assertRaises(ValueError):
            rectangular_lift(seq, resources, (Rectangle((0, 0), (3, 2), (0,)),))


if __name__ == '__main__':
    unittest.main()
