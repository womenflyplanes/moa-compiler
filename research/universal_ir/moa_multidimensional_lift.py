"""MoA v3.3: additive rectangular dimension lifting of an immutable DNF.

No psi rewriting or optimization is performed. Coordinates are in the original
DNF domain; rectangular partitions are mapped to a shaped machine array.
"""
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Tuple


def _shape(shape):
    if not isinstance(shape, tuple) or any(type(d) is not int or d < 0 for d in shape):
        raise ValueError('shape must be a tuple of nonnegative integers')
    return shape


def _indices(shape):
    return product(*(range(d) for d in shape))


@dataclass(frozen=True)
class ResourceArray:
    """Shaped array of existing machine records, in row-major resource order."""
    shape: Tuple[int, ...]
    machines: Tuple[object, ...]

    def __post_init__(self):
        _shape(self.shape)
        if len(self.machines) != prod(self.shape):
            raise ValueError('resource array cardinality does not match shape')
        if any(not hasattr(m, 'capacity_bytes') or not hasattr(m, 'max_chunk_elements') for m in self.machines):
            raise TypeError('machines must provide v3 MachineArray capacity fields')

    def psi(self, index):
        if not isinstance(index, tuple) or len(index) != len(self.shape):
            raise IndexError(index)
        offset = 0
        for i, dim in zip(index, self.shape):
            if type(i) is not int or i < 0 or i >= dim:
                raise IndexError(index)
            offset = offset * dim + i
        return self.machines[offset]

    def indices(self):
        return tuple(_indices(self.shape))


@dataclass(frozen=True)
class Rectangle:
    """Half-open data-domain bounds, assigned to a machine-resource coordinate."""
    lower: Tuple[int, ...]
    upper: Tuple[int, ...]
    resource_index: Tuple[int, ...]

    @property
    def elements(self):
        return prod(b - a for a, b in zip(self.lower, self.upper))

    def contains(self, index):
        return all(a <= i < b for a, b, i in zip(self.lower, self.upper, index))


@dataclass(frozen=True)
class MultidimensionalONF:
    sequential: object  # v3 SequentialONF; its DNF is immutable
    resources: ResourceArray
    rectangles: Tuple[Rectangle, ...]

    def verify(self):
        shape = self.sequential.dnf.shape
        _shape(shape)
        if not isinstance(self.rectangles, tuple):
            raise ValueError('rectangles must be a tuple')
        for r in self.rectangles:
            if len(r.lower) != len(shape) or len(r.upper) != len(shape):
                raise ValueError('rectangle rank differs from DNF rank')
            if any(type(a) is not int or type(b) is not int or not (0 <= a <= b <= d)
                   for a, b, d in zip(r.lower, r.upper, shape)):
                raise ValueError('invalid rectangle bounds')
            machine = self.resources.psi(r.resource_index)
            if r.elements > machine.max_chunk_elements or r.elements * self.sequential.dnf.dtype_bytes > machine.capacity_bytes:
                raise ValueError('rectangle exceeds machine capacity')

        # Exact coverage proof for axis-aligned rectangles using boundary cells.
        # Every rectangle's membership is constant within each boundary cell.
        if prod(shape) == 0:
            if any(r.elements for r in self.rectangles):
                raise ValueError('nonempty rectangle in empty DNF domain')
            return True
        cuts = [sorted({0, d} | {v for r in self.rectangles for v in (r.lower[k], r.upper[k])})
                for k, d in enumerate(shape)]
        for cell in product(*(range(len(c) - 1) for c in cuts)):
            point = tuple(cuts[k][j] for k, j in enumerate(cell))
            matches = sum(r.contains(point) for r in self.rectangles)
            if matches != 1:
                raise ValueError(f'DNF index cell at {point} covered {matches} times')
        return True

    def mapped_indices(self):
        """Return (DNF index, resource coordinate, gamma address) in DNF order."""
        self.verify()
        result = []
        for index in _indices(self.sequential.dnf.shape):
            rectangle = next(r for r in self.rectangles if r.contains(index))
            result.append((index, rectangle.resource_index, self.sequential.address(index)))
        return tuple(result)


def rectangular_lift(sequential, resources, rectangles):
    result = MultidimensionalONF(sequential, resources, tuple(rectangles))
    result.verify()
    return result
